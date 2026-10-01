"""Prepare step: normalized JSONL -> agent-ready pack (data/agent/<id>/).

The pack is what an AI Agent reads. It never contains the full raw dataset:
comments are split into sized Markdown parts with complete, unmodified
originals, plus a manifest and basic statistics.

    agent/<video_id>/
      manifest.json      part list, counts, prompt/model metadata
      video.md           video metadata (title/owner/description)
      part-0001.md ...   comment blocks, complete originals
      statistics.json    aggregate numbers (no LLM involved)
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .textstats import sentiment_counts

# Rough Chinese-aware budget: ~1.6 chars per token, keep 30% headroom so a
# part plus its prompt always fits a small-context model.
CHARS_PER_TOKEN = 1.6
BUDGET_TOKENS_DEFAULT = 24000

def quote_text(text: str) -> str:
    """Quote a comment verbatim in Markdown without breaking block quotes.

    Originals must stay complete and un-polished; backslashes only protect
    the Markdown structure, never the content itself.
    """
    text = text.replace("\r\n", "\n").strip("\n")
    lines = text.split("\n") if text else [""]
    return "\n".join("> " + line for line in lines)


def chunk_records(
    records: List[Dict[str, Any]], budget_tokens: int = BUDGET_TOKENS_DEFAULT
) -> List[List[Dict[str, Any]]]:
    """Split comments into chunks under a per-chunk token budget.

    A single oversized comment always gets its own chunk (never dropped).
    """
    budget_chars = int(budget_tokens * CHARS_PER_TOKEN)
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for r in records:
        size = len(r.get("content", "")) + 160  # record framing overhead
        if current and current_chars + size > budget_chars:
            chunks.append(current)
            current, current_chars = [], 0
        current.append(r)
        current_chars += size
    if current:
        chunks.append(current)
    return chunks


def _format_record(r: Dict[str, Any], index: int) -> str:
    kind = "置顶一级评论" if r.get("is_pinned") else (
        "一级评论" if r.get("level") == 1 else "楼中楼回复"
    )
    parent = r.get("parent_comment_id")
    parts = [f"## Comment {r['comment_id']}"]
    parts.append(f"时间：{r.get('created_at') or '未知'}")
    parts.append(f"点赞：{r.get('like_count', 0)}")
    if r.get("reply_count"):
        parts.append(f"回复数：{r['reply_count']}")
    parts.append(f"类型：{kind}")
    if parent is not None:
        parts.append(f"Parent: {parent}")
    if r.get("is_pinned") and r.get("level") == 2:
        parts.append("（置顶评论的楼中楼）")
    parts.append("")
    parts.append("原文：")
    parts.append(quote_text(r.get("content", "")))
    return "\n".join(parts)


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    likes = [r["like_count"] for r in records]
    times = sorted(r["created_at"] for r in records if r["created_at"])
    top_liked = sorted(records, key=lambda r: -r["like_count"])[:10]
    authors = Counter(r["author_hash"] for r in records if r["author_hash"])
    return {
        "total_comments": len(records),
        "level1": sum(1 for r in records if r["level"] == 1),
        "level2": sum(1 for r in records if r["level"] == 2),
        "pinned": sum(1 for r in records if r["is_pinned"]),
        "unique_authors": len(authors),
        "author_repeat_max": max(authors.values()) if authors else 0,
        "like_total": sum(likes),
        "like_max": max(likes) if likes else 0,
        "like_median": (lambda s: s[len(s) // 2] if s else 0)(sorted(likes)),
        "time_first": times[0] if times else None,
        "time_last": times[-1] if times else None,
        "sentiment": sentiment_counts(records),
        "top_liked": [
            {
                "comment_id": r["comment_id"],
                "like_count": r["like_count"],
                "level": r["level"],
                "excerpt": r["content"][:60],
            }
            for r in top_liked
        ],
    }


def prepare(
    normalized_dir: Path,
    video_id: str,
    budget_tokens: int = BUDGET_TOKENS_DEFAULT,
) -> Dict[str, Any]:
    records = [
        json.loads(line)
        for line in (normalized_dir / "comments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    meta_file = normalized_dir.parent.parent / "raw" / video_id / "meta.json"
    meta = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    out_dir = normalized_dir.parent.parent / "agent" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # video.md
    info = meta.get("target_info") or {}
    video_md = [
        f"# {info.get('title') or video_id}",
        "",
        f"- video_id: `{video_id}`",
        f"- UP主: {info.get('owner') or '未知'}",
        f"- 发布时间: {info.get('pubdate') or '未知'}",
        f"- 描述: {info.get('desc') or '（无）'}",
        f"- 抓取时间: {meta.get('crawled_at', '未知')}",
        f"- 来源: {meta.get('source_url', '')}",
    ]
    (out_dir / "video.md").write_text("\n".join(video_md) + "\n", encoding="utf-8")

    # parts
    chunks = chunk_records(records, budget_tokens)
    part_files = []
    for i, chunk in enumerate(chunks, 1):
        name = f"part-{i:04d}.md"
        body = [f"# Comments part {i:04d} / {len(chunks):04d}", ""]
        body += [_format_record(r, i) for r in chunk]
        (out_dir / name).write_text("\n\n".join(body) + "\n\n", encoding="utf-8")
        part_files.append(
            {
                "file": name,
                "comments": len(chunk),
                "approx_tokens": int(sum(len(r.get("content", "")) + 160 for r in chunk) / CHARS_PER_TOKEN),
            }
        )

    stats = compute_statistics(records)
    (out_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "video_id": video_id,
        "generated_from": {
            "raw": f"data/raw/{video_id}/comments.jsonl",
            "normalized": f"data/normalized/{video_id}/comments.jsonl",
        },
        "schema_fields": [
            "video_id", "comment_id", "parent_comment_id", "content",
            "created_at", "like_count", "reply_count", "is_pinned",
            "level", "author_hash",
        ],
        "privacy": "author identities are salted hashes; no uid/ip/avatar kept",
        "total_comments": len(records),
        "parts": part_files,
        "budget_tokens_per_part": budget_tokens,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
