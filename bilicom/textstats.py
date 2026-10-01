"""Deterministic text statistics — no LLM, fully reproducible.

Three jobs the LLM layers were previously trusted with and should not have
been:

1. sentiment counts that reconcile to the dataset total (lexicon scoring over
   EVERY comment, so 正/负/中 always sums to N);
2. a topic-term lexicon used to tell on-debate opinions apart from unrelated
   feedback before they are presented as 支持/反对 of the core controversy;
3. stance-conflict resolution: a comment cited by the LLM under both 支持 and
   反对 is assigned to exactly one side by n-gram overlap with each side's
   own point texts (ties drop to neutral).

Nothing here rewrites comment text; all functions take plain strings/ids and
return plain numbers/lists.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---- sentiment -----------------------------------------------------------

_POSITIVE = [
    "喜欢", "感动", "感谢", "谢谢", "受益", "受用", "启发", "有用", "有帮助",
    "认同", "说得对", "有道理", "支持", "赞", "优秀", "棒", "厉害", "真好",
    "很好", "加油", "希望", "开心", "快乐", "幸福", "温暖", "治愈", "学到",
    "觉醒", "成长", "进步", "值得", "珍惜", "爱", "妙", "清醒", "通透",
    "值得", "有帮助", "点赞", "真诚", "踏实", "努力", "加油",
]
_NEGATIVE = [
    "累", "焦虑", "迷茫", "自卑", "绝望", "没意思", "无聊", "讨厌", "恶心",
    "垃圾", "滚", "傻", "离谱", "翻车", "生气", "愤怒", "害怕", "恐惧",
    "担心", "压力", "内耗", "崩溃", "想死", "活不下去", "悲观", "失望",
    "可悲", "可怜", "讽刺", "荒诞", "荒谬", "胡说", "放屁", "扯淡", "空洞",
    "水视频", "误导", "割韭菜", "收割", "嘲讽", "傲慢", "装", "飘", "难",
    "苦", "痛", "哭", "悲伤", "孤独", "寂寞", "委屈", "心累", "烦",
    "没希望", "无望", "卷", "贬值", "吃亏", "欺负", "压榨", "剥削",
]
_NEGATION = ["不是", "没有", "并不", "并不", "不算", "毫无", "毫无"]


def _count_hits(text: str, words: List[str]) -> int:
    return sum(text.count(w) for w in words)


def sentiment_label(text: str) -> int:
    """+1 positive / -1 negative / 0 neutral for one comment."""
    t = str(text)
    score = _count_hits(t, _POSITIVE) - _count_hits(t, _NEGATIVE)
    if score == 0:
        return 0
    # a negated sentiment flips once ("不是不能坚持" is rare; single flip is
    # the right trade-off for short comments)
    for neg in set(_NEGATION):
        idx = t.find(neg)
        if idx >= 0 and score != 0:
            tail = t[idx + len(neg): idx + len(neg) + 12]
            if _count_hits(tail, _POSITIVE) or _count_hits(tail, _NEGATIVE):
                score = -score
                break
    return 1 if score > 0 else -1


def sentiment_counts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lexicon sentiment over EVERY record; the three buckets always sum to
    the record count, unlike LLM-estimated counts."""
    pos = neg = 0
    for r in records:
        label = sentiment_label(r.get("content", ""))
        if label > 0:
            pos += 1
        elif label < 0:
            neg += 1
    total = len(records)
    return {
        "positive": pos,
        "negative": neg,
        "neutral": total - pos - neg,
        "total": total,
        "method": "lexicon（确定性词典法，三类之和=评论总数）",
    }


# ---- stance / topic ------------------------------------------------------

# Terms that mark a comment as participating in the 掠夺/资源 debate (works
# for any controversy thanks to the point-text overlap resolver below)
TOPIC_TERMS = [
    "掠夺", "社会资源", "资源", "抢占", "竞争", "内卷", "卷", "蛋糕",
    "老实", "螺丝钉", "强盗", "达尔文", "厚黑", "利他", "共赢", "争取",
    "收割", "剥削", "分配", "资本", "打工人", "抢", "赚钱", "搞钱",
    "机会", "主动",
]


def on_topic(text: str, extra_terms: List[str] = ()) -> bool:
    t = str(text)
    return any(term in t for term in list(TOPIC_TERMS) + list(extra_terms))


def _bigrams(text: str) -> set:
    t = re.sub(r"[\s\W_]+", "", str(text).lower(), flags=re.UNICODE)
    return {t[i:i + 2] for i in range(len(t) - 1)} or ({t} if t else set())


def _group_text(group: Dict[str, Any], id_map: Dict[int, Dict[str, Any]]) -> str:
    parts = [str(group.get("point") or "")]
    for cid in group.get("comment_ids", []):
        row = id_map.get(int(cid))
        if row:
            parts.append(row.get("content", ""))
    return " ".join(parts)


def resolve_stances(opinions: Dict[str, List[Dict[str, Any]]],
                    id_map: Dict[int, Dict[str, Any]]
                    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Deterministic post-process of the LLM stance lists.

    Returns (opinions, qa) where opinions has the same support/oppose/neutral
    keys plus an ``off_topic`` bucket, with guarantees:

    - every comment id lands in at most ONE stance bucket (LLM 双标引用被消解);
    - groups none of whose comments/points mention the debate lexicon are
      moved to ``off_topic`` instead of posing as 支持/反对 of the controversy.
    """
    qa = {"stance_conflicts": 0, "conflict_resolved": [],
          "off_topic_groups": 0, "off_topic_ids": 0}

    # 1) conflict resolution — id -> best bucket by bigram overlap with the
    #    aggregated point+member texts of the bucket group that cited it
    citing: Dict[int, List[Tuple[str, int, Dict[str, Any]]]] = {}
    for key in ("support", "oppose", "neutral"):
        for gi, group in enumerate(opinions.get(key, [])):
            for cid in group.get("comment_ids", []):
                try:
                    cidi = int(cid)
                except (TypeError, ValueError):
                    continue
                citing.setdefault(cidi, []).append((key, gi, group))

    assignment: Dict[int, str] = {}
    resolved_groups: Dict[str, List[Dict[str, Any]]] = {
        k: [dict(g, comment_ids=[]) for g in opinions.get(k, [])]
        for k in ("support", "oppose", "neutral")}

    for cid, hits in citing.items():
        buckets = {k for k, _gi, _g in hits}
        if len(buckets) == 1:
            assignment[cid] = hits[0][0]
            continue
        qa["stance_conflicts"] += 1
        text = id_map.get(cid, {}).get("content", "")
        cset = _bigrams(text)
        best_key, best_score, best_gi = None, -1.0, None
        for key, gi, group in hits:
            overlap = cset & _bigrams(_group_text(group, id_map))
            score = len(overlap)
            if score > best_score:
                best_key, best_score, best_gi = key, score, gi
        if best_score <= 0:          # no textual evidence either way
            best_key = "neutral"
        assignment[cid] = best_key
        qa["conflict_resolved"].append(
            {"comment_id": cid, "cited_as": sorted(buckets),
             "resolved_to": best_key})

    for key in ("support", "oppose", "neutral"):
        for gi, group in enumerate(opinions.get(key, [])):
            for cid in group.get("comment_ids", []):
                try:
                    cidi = int(cid)
                except (TypeError, ValueError):
                    continue
                target = assignment.get(cidi, key)
                if cidi not in resolved_groups[target][gi]["comment_ids"]:
                    resolved_groups[target][gi]["comment_ids"].append(cidi)

    # 2) off-topic split — a group is on-debate only if its point or at least
    #    one member comment mentions the controversy lexicon
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key in ("support", "oppose", "neutral"):
        kept_groups, moved = [], []
        for group in resolved_groups[key]:
            texts = [str(group.get("point") or "")] + [
                id_map.get(c, {}).get("content", "")
                for c in group["comment_ids"]]
            if any(on_topic(t) for t in texts) or not group["comment_ids"]:
                kept_groups.append(group)
            else:
                moved.append(group)
        out[key] = kept_groups
        if moved:
            merged_ids: List[int] = []
            for g in moved:
                qa["off_topic_groups"] += 1
                qa["off_topic_ids"] += len(g["comment_ids"])
                for c in g["comment_ids"]:
                    if c not in merged_ids:
                        merged_ids.append(c)
            out.setdefault("off_topic", []).extend(moved)
    for group in out.get("off_topic", []):
        group["point"] = f"（与核心争议无关）{group.get('point', '')}"
    return out, qa