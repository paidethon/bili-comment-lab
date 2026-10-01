"""Analyze step: agent pack -> cached map-reduce -> reports.

Pipeline (never feeds all comments to the model at once):

    agent parts -> [map: extract per part]  (cached, resumable)
                -> [topic aggregation]      (cached)
                -> [reduce: final report]   (cached)
                -> data/reports/<id>/ + data/agent/<id>/analysis.json

Layer 1 (extractive) results always carry comment_ids; the Layer 2 summary is
only written in the reduce step, constrained to those ids. LLM config follows
the documented precedence — settings file (data/webui/ai_settings.json) over
environment (.env LLM_*); without either the pipeline still writes
statistics-based reports and marks the LLM parts as NOT RUN.

Deterministic guarantees added on top of the LLM layers (see textstats.py):

- cited comment_ids are validated against the dataset; unknown ids are
  dropped and counted in qa.json instead of reaching the report;
- a comment id appears under at most ONE of 支持/反对/中立 (LLM conflicts are
  resolved by textual overlap; ties go to 中立);
- feedback unrelated to the core controversy is split into an off_topic list
  instead of posing as 支持/反对 of it;
- emotion counts come from a lexicon over EVERY comment, so 正/负/中 sums to
  the dataset total and is reproducible without the LLM.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient, LLMConfigError
from .provider import build_client
from .textstats import resolve_stances, sentiment_counts

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

SUPPORT_LABEL = "支持"


def load_id_map(data_root: Path, video_id: str) -> Dict[int, Dict[str, Any]]:
    """comment_id -> content/likes/created/level from the normalized dataset
    (the same records the agent pack was built from)."""
    path = data_root / "normalized" / video_id / "comments.jsonl"
    id_map: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return id_map
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            id_map[int(r["comment_id"])] = r
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return id_map


def _merge_ids(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge extractive groups by point text across chunks, unioning ids."""
    merged: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        point = str(g.get("point") or g.get("question") or g.get("suggestion") or "").strip()
        if not point:
            continue
        key = point
        bucket = merged.setdefault(key, {"point": point, "comment_ids": []})
        for cid in g.get("comment_ids", []):
            if cid not in bucket["comment_ids"]:
                bucket["comment_ids"].append(cid)
    return sorted(merged.values(), key=lambda g: -len(g["comment_ids"]))


def _validate_ids(groups: List[Dict[str, Any]], id_map: Dict[int, Dict[str, Any]],
                  qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Drop comment ids that do not exist in the dataset (LLM hallucinations
    never reach the report) and count them in qa."""
    out: List[Dict[str, Any]] = []
    for g in groups:
        valid, bad = [], []
        for cid in g.get("comment_ids", []):
            try:
                cidi = int(cid)
            except (TypeError, ValueError):
                bad.append(cid)
                continue
            (valid.append(cidi) if cidi in id_map else bad.append(cidi))
        if bad:
            qa["invalid_ids"].extend(bad)
        if valid or not g.get("comment_ids"):
            out.append(dict(g, comment_ids=valid))
    return out


def extractive_only_analysis(pack_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Statistics-derived analysis used when no LLM is configured."""
    stats = json.loads((pack_dir / "statistics.json").read_text(encoding="utf-8"))
    return {
        "layer1_only": True,
        "llm_run": False,
        "statistics": stats,
        "note": "LLM 未配置：仅生成统计与高赞代表评论（extractive-only）。在网页「设置 → AI / NLP」或 .env 中配置后重跑 analyze。",
    }


def analyze(
    video_id: str,
    data_root: Path,
    budget_tokens: Optional[int] = None,
    log=lambda m: None,
) -> Dict[str, Any]:
    pack_dir = data_root / "agent" / video_id
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_dir = data_root / "cache"
    id_map = load_id_map(data_root, video_id)

    qa: Dict[str, Any] = {
        "video_id": video_id,
        "total_comments": len(id_map),
        "invalid_ids": [],          # cited by LLM, absent from dataset
        "stance_conflicts": 0,      # same id under >1 stance
        "conflict_resolved": [],
        "off_topic_groups": 0,
        "off_topic_ids": 0,
        "emotions_reconciled": True,
    }

    llm: Optional[LLMClient]
    try:
        llm = build_client(data_root)
        llm.cache_dir = cache_dir
    except ValueError as err:
        try:
            llm = LLMClient(cache_dir=cache_dir)
        except LLMConfigError:
            log(f"LLM 未配置（{err}）—— 生成 extractive-only 结果")
            result = extractive_only_analysis(pack_dir, manifest)
            result["qa"] = dict(qa, note="extractive-only：无 LLM 引用需要校验")
            _write_reports(video_id, data_root, result, manifest, id_map)
            return result

    ctx = budget_tokens or min(24000, llm.context_length // 2)

    # ---- map: extract per part (cached) --------------------------------
    extract_prompt = PROMPTS_DIR / "extract-v1.md"
    chunk_results: List[Dict[str, Any]] = []
    for part in manifest["parts"]:
        content = (pack_dir / part["file"]).read_text(encoding="utf-8")
        if len(content) > int(ctx * 1.6):
            log(f"警告：{part['file']} 超出预算，建议减小 prepare 的 budget_tokens")
        log(f"map: {part['file']} ({part['comments']} comments)")
        out = llm.cached_chat("extract", extract_prompt, content, response_json=True)
        log(f"  {'cache hit' if out['cached'] else 'llm ok'}: {part['file']}")
        chunk_results.append(out["result"])

    # ---- merge extractive results (deterministic, no LLM) --------------
    topics_flat: Dict[str, Dict[str, Any]] = {}
    support, oppose, neutral = [], [], []
    questions, suggestions, notable = [], [], []
    for res in chunk_results:
        for t in res.get("topics", []):
            bucket = topics_flat.setdefault(
                t.get("name", "未分类"),
                {"name": t.get("name", "未分类"), "comment_ids": [], "chunks": 0},
            )
            bucket["chunks"] += 1
            for cid in t.get("comment_ids", []):
                if cid not in bucket["comment_ids"]:
                    bucket["comment_ids"].append(cid)
        opinions = res.get("opinions", {})
        support += opinions.get("support", [])
        oppose += opinions.get("oppose", [])
        neutral += opinions.get("neutral", [])
        questions += res.get("questions", [])
        suggestions += res.get("suggestions", [])
        notable += res.get("notable", [])

    # deterministic guarantees: id validity -> stance partition -> off-topic
    for t in topics_flat.values():
        t["comment_ids"] = _validate_ids([t], id_map, qa)[0]["comment_ids"]
    merged_opinions = {
        "support": _merge_ids(support),
        "oppose": _merge_ids(oppose),
        "neutral": _merge_ids(neutral),
    }
    for key in list(merged_opinions):
        merged_opinions[key] = _validate_ids(merged_opinions[key], id_map, qa)
    merged_opinions, stance_qa = resolve_stances(merged_opinions, id_map)
    qa.update(stance_qa)
    questions = _validate_ids(_merge_ids(questions), id_map, qa)
    suggestions = _validate_ids(_merge_ids(suggestions), id_map, qa)
    notable = [n for n in notable
               if isinstance(n, dict)
               and _as_int(n.get("comment_id")) in id_map]

    # emotions: deterministic lexicon over EVERY comment (sums to total),
    # replacing per-chunk LLM estimates that never reconciled
    records = list(id_map.values())
    emotions = sentiment_counts(records)

    merged = {
        "topics": sorted(topics_flat.values(), key=lambda t: -len(t["comment_ids"])),
        "opinions": merged_opinions,
        "questions": questions,
        "suggestions": suggestions,
        "notable": notable,
        "emotions": {k: emotions[k] for k in ("positive", "negative", "neutral")},
    }

    # ---- topic aggregation (cached) -------------------------------------
    topic_prompt = PROMPTS_DIR / "topic-v1.md"
    topic_payload = json.dumps(merged["topics"], ensure_ascii=False)
    topic_out = llm.cached_chat("topic", topic_prompt, topic_payload, response_json=True)
    log(f"topic aggregation: {'cache hit' if topic_out['cached'] else 'llm ok'}")
    global_topics = topic_out["result"].get("topics", merged["topics"])

    # ---- reduce (cached) -------------------------------------------------
    stats = json.loads((pack_dir / "statistics.json").read_text(encoding="utf-8"))
    stats = dict(stats)
    stats["sentiment"] = emotions          # the report cites THESE numbers
    reduce_prompt = PROMPTS_DIR / "reduce-v2.md"
    reduce_payload = json.dumps(
        {
            "video": (pack_dir / "video.md").read_text(encoding="utf-8"),
            "statistics": stats,
            "topics": global_topics,
            "merged": merged,
            "qa": {k: v for k, v in qa.items()
                   if k in ("invalid_ids", "stance_conflicts",
                            "off_topic_ids", "total_comments")},
        },
        ensure_ascii=False,
    )
    reduce_out = llm.cached_chat("reduce", reduce_prompt, reduce_payload, response_json=True)
    log(f"reduce: {'cache hit' if reduce_out['cached'] else 'llm ok'}")

    result = {
        "layer1_only": False,
        "llm_run": True,
        "model": llm.model,
        "statistics": stats,
        "global_topics": global_topics,
        "layer1": merged,
        "report": reduce_out["result"],
        "qa": qa,
        "run_metadata": {
            "prompt_versions": {
                "extract": "extract-v1",
                "topic": "topic-v1",
                "reduce": "reduce-v2",
            },
            "model": llm.model,
            "cache_dir": str(cache_dir),
        },
    }
    (pack_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_reports(video_id, data_root, result, manifest, id_map)
    return result


# ---- report writing -----------------------------------------------------

def _header(video_id: str, manifest: Dict[str, Any], result: Dict[str, Any],
            stats: Dict[str, Any]) -> str:
    sent = stats.get("sentiment") or {}
    sent_line = ""
    if sent:
        sent_line = (f"\n- 情绪计数（词典法，全部 {sent.get('total', 0)} 条："
                     f"正面 {sent.get('positive', 0)} / 负面 {sent.get('negative', 0)}"
                     f" / 中性 {sent.get('neutral', 0)}，三类之和=总数）")
    return (
        f"# 分析报告 — {video_id}\n\n"
        f"- 生成范围：{manifest.get('total_comments')} 条评论"
        f"（一级 {stats.get('level1', 0)} / 楼中楼 {stats.get('level2', 0)}"
        f" / 置顶 {stats.get('pinned', 0)}）\n"
        f"- 模型：{result.get('model', '未使用')}\n"
        f"- 方式：{'map-reduce（Layer1 抽取 + Layer2 总结）' if result.get('llm_run') else 'extractive-only（未配置 LLM）'}"
        f"{sent_line}\n"
    )


def _write_reports(
    video_id: str, data_root: Path, result: Dict[str, Any],
    manifest: Dict[str, Any], id_map: Dict[int, Dict[str, Any]],
) -> None:
    """One merged report (single header, representative comments WITH their
    original text) plus the individual section files the UI tabs read.
    qa.json lands next to them so every number is checkable."""
    out = data_root / "reports" / video_id
    out.mkdir(parents=True, exist_ok=True)
    stats = result.get("statistics", {})
    report = result.get("report") or {}
    header = _header(video_id, manifest, result, stats)

    sections = {
        "summary": "## 摘要\n\n" + (report.get("summary") or result.get("note", "")),
        "topics": report.get("topics_md") or _fallback_topics(result),
        "opinions": report.get("opinions_md") or _fallback_opinions(result),
        "questions": report.get("questions_md") or _fallback_list(result, "questions", "question", "问题"),
        "suggestions": report.get("suggestions_md") or _fallback_list(result, "suggestions", "suggestion", "建议"),
        "representative-comments": _representative_md(result, id_map),
    }
    qa = result.get("qa") or {}
    qa_md = _qa_md(qa, stats)
    for name, body in sections.items():
        (out / f"{name}.md").write_text(body.strip() + "\n", encoding="utf-8")
    merged = header + "\n" + "\n\n".join(
        sections[n] for n in ("summary", "topics", "opinions", "questions",
                              "suggestions", "representative-comments"))
    if qa_md:
        merged += "\n\n" + qa_md
    (out / "report.md").write_text(merged + "\n", encoding="utf-8")
    (out / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if qa:
        (out / "qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def _qa_md(qa: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """Human-readable tail of the merged report: the deterministic checks the
    numbers must satisfy, and what the pipeline did about violations."""
    lines = ["## 质检与口径", ""]
    total = qa.get("total_comments")
    sent = stats.get("sentiment") or {}
    if sent and total:
        lines.append(
            f"- 情绪计数对账：正面 {sent['positive']} + 负面 {sent['negative']}"
            f" + 中性 {sent['neutral']} = {sent['total']}（数据集总数 {total}）✓")
    invalid = qa.get("invalid_ids") or []
    lines.append(
        f"- 引用 ID 校验：模型引用中 {len(invalid)} 个 ID 不存在于数据集"
        f"（已剔除{'：' + '、'.join(map(str, invalid[:5])) if invalid else ''}）")
    lines.append(
        f"- 立场一致性：{qa.get('stance_conflicts', 0)} 条评论曾被同时列入"
        f"支持与反对，已按文本重合度归入单一立场（平票归中立）")
    lines.append(
        f"- 论题外分流：{qa.get('off_topic_ids', 0)} 条与核心争议无关的反馈"
        f"（夸 UP、纯倾诉等）已移出支持/反对清单，单列为「其他反馈」")
    lines.append("- 口径：本报告基于抓取快照的全体评论；条数以 metadata/manifest 为准，"
                 "所有观点均带 comment_id 可回溯原文")
    return "\n".join(lines)


def _fallback_topics(result: Dict[str, Any]) -> str:
    lines = ["## 主要讨论主题（extractive-only，按评论数排序）", ""]
    for t in result.get("layer1", {}).get("topics", [])[:15]:
        lines.append(f"- **{t['name']}** ≈{len(t['comment_ids'])} 条 (comments: {', '.join(map(str, t['comment_ids'][:8]))})")
    return "\n".join(lines)


def _fallback_opinions(result: Dict[str, Any]) -> str:
    ops = result.get("layer1", {}).get("opinions", {})
    lines = ["## 观点分布（extractive-only）", ""]
    for label, key in [("支持", "support"), ("反对", "oppose"), ("中立/补充", "neutral"),
                       ("其他反馈", "off_topic")]:
        lines.append(f"### {label}")
        for g in ops.get(key, [])[:10]:
            lines.append(f"- {g['point']} (comments: {', '.join(map(str, g['comment_ids'][:6]))})")
        lines.append("")
    return "\n".join(lines)


def _fallback_list(result: Dict[str, Any], key: str, field: str, title: str) -> str:
    lines = [f"## {title}（extractive-only）", ""]
    for g in result.get("layer1", {}).get(key, [])[:15]:
        lines.append(f"- {g.get(field, '')} (comments: {', '.join(map(str, g.get('comment_ids', [])[:6]))})")
    return "\n".join(lines)


def _representative_md(result: Dict[str, Any],
                       id_map: Dict[int, Dict[str, Any]]) -> str:
    """代表性评论 with the promised 原文: every listed id is looked up in the
    dataset and quoted verbatim (truncated at 160 chars for readability)."""
    lines = ["## 代表性评论（原文保留，带 comment_id）", ""]
    items = [n for n in result.get("layer1", {}).get("notable", [])[:10]
             if isinstance(n, dict)]
    if not items:
        top = (result.get("statistics") or {}).get("top_liked", [])[:10]
        for t in top:
            row = id_map.get(_as_int(t.get("comment_id")), {})
            content = row.get("content") or t.get("excerpt", "")
            lines.append(f"### 评论 {t['comment_id']}\n\n"
                         f"点赞：{t.get('like_count', 0)}\n\n> {_quote(content)}\n")
        return "\n".join(lines)
    for it in items:
        cid = _as_int(it.get("comment_id"))
        row = id_map.get(cid, {})
        content = row.get("content") or "（原文缺失：该 ID 不在本数据集中）"
        likes = row.get("like_count", "—")
        created = row.get("created_at", "")
        lines.append(
            f"### 评论 {cid}\n\n"
            f"理由：{it.get('why', '')} · 赞 {likes} · {created}\n\n"
            f"> {_quote(content)}\n")
    return "\n".join(lines)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _quote(text: str, limit: int = 160) -> str:
    flat = re.sub(r"\s+", " ", str(text)).strip()
    if len(flat) > limit:
        flat = flat[:limit] + "…"
    return flat
