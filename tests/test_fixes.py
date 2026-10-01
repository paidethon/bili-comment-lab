"""Regression tests for the 2026-09-30 quality fixes:

- analysis-view cleaning (reply prefixes / emote tags) and latin term handling
- emote-only comments bucketed instead of shattering clusters
- quality_score defined for every row (0 点赞 ≠ 0 分)
- snapshot ingest (DB index matches the raw file exactly)
- thread fields ride along on refined rows
- report QA: hallucinated ids dropped, stance conflicts resolved,
  off-topic feedback split out, sentiment counts reconcile
- bundle privacy scrubs @nicknames inside content
"""
from __future__ import annotations

import json

from bilicom import refine, store, textstats
from bilicom.analyze import load_id_map
from bilicom.refine import RefineParams, clean_for_analysis, content_terms
from bilicom.textstats import resolve_stances, sentiment_counts


# ---- text cleaning -------------------------------------------------------

def test_clean_strips_reply_prefix_and_emotes():
    assert clean_for_analysis("回复 @某人：说得对[给力]") == "说得对"
    assert clean_for_analysis("[微笑]") == ""
    assert "[给心心]" not in clean_for_analysis("好文[给心心][给心心]")


def test_content_terms_no_latin_letter_bigrams():
    terms = content_terms("some random english words here")
    assert all(len(t) >= 2 and t.isalnum() for t in terms)
    # whole words survive; no cross-word letter bigrams like 'so'/'om'
    assert set(terms) == {"some", "random", "english", "words", "here"}
    assert set(content_terms("社会资源")) == {"社会", "会资", "资源"}


def test_emote_only_comments_leave_content_terms_empty():
    assert content_terms("回复 @user：[doge_金箍][doge_金箍]") == []


# ---- sentiment -----------------------------------------------------------

def test_sentiment_counts_reconcile():
    records = [{"content": c} for c in [
        "很有帮助，学到了", "累死了，不想干了", "今天吃了顿饭",
        "谢谢分享", "真的吗？", "焦虑", "不错不错", "还行吧", " pushing on",
        "加油！", "无聊", "喜欢"]]
    counts = sentiment_counts(records)
    assert counts["total"] == len(records)
    assert counts["positive"] + counts["negative"] + counts["neutral"] == len(records)
    assert counts["method"].startswith("lexicon")


# ---- stance resolution ---------------------------------------------------

def _id_map(*pairs):
    return {cid: {"comment_id": cid, "content": text} for cid, text in pairs}


def test_resolve_stances_conflict_goes_to_one_side():
    id_map = _id_map((1, "我觉得改变自己就要掠夺社会资源，要主动抢占机会"),
                     (2, "掠夺这个词是形容强盗的，用词不当"))
    opinions = {
        "support": [{"point": "要主动掠夺社会资源", "comment_ids": [1]}],
        "oppose": [{"point": "up讲得很不对，逻辑有问题 掠夺说法误导", "comment_ids": [1]},
                   {"point": "用词不当，掠夺是强盗词", "comment_ids": [2]}],
        "neutral": [],
    }
    out, qa = resolve_stances(opinions, id_map)
    sides = [k for k in ("support", "oppose") if any(
        1 in g["comment_ids"] for g in out[k])]
    assert sides == ["support"]          # conflict resolved to the better match
    assert qa["stance_conflicts"] == 1
    assert qa["conflict_resolved"][0]["resolved_to"] == "support"


def test_resolve_stances_off_topic_split():
    id_map = _id_map((9, "up主声音真好听，爱了爱了"),
                     (10, "加油，相信未来会变好"))
    opinions = {
        "support": [{"point": "喜欢up主", "comment_ids": [9]},
                    {"point": "要主动争取机会掠夺资源", "comment_ids": [10]}],
        "oppose": [], "neutral": [],
    }
    out, qa = resolve_stances(opinions, id_map)
    assert all(9 not in g["comment_ids"] for g in out["support"])
    assert any(9 in g["comment_ids"] for g in out["off_topic"])
    assert qa["off_topic_ids"] == 1


# ---- refine internals ----------------------------------------------------

def test_refine_run_buckets_emotes_and_scores(tmp_path, monkeypatch):
    from conftest import VIDEO
    raw = tmp_path / "data" / "raw" / VIDEO
    raw.mkdir(parents=True)
    rows = []
    for i, text in enumerate([
        "怎样才能做到掠夺社会资源呢？求解",
        "掠夺社会资源说白了就是主动争取机会",
        "掠夺到底什么意思，求大佬科普一波",
        "[微笑]", "[微笑]", "[给心心]",
        "顶", "ding", "顶一下",
    ]):
        rows.append(dict(comment_id=1000 + i, root_id=1000 + i,
                         parent_id=None, is_reply=False, user_id=i,
                         content=text, like_count=i * 7, reply_count=0,
                         ctime=1700000000 + i, is_pinned=False))
    with (raw / "comments.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (raw / "meta.json").write_text(json.dumps(
        {"video_id": VIDEO, "target_info": {"title": "t"}}, ensure_ascii=False))
    from bilicom.normalize import normalize
    normalize(raw, VIDEO)

    conn = store.connect(tmp_path / "data")
    store.ingest_video(conn, raw, VIDEO)
    stats = refine.run_refine(conn, tmp_path / "data", "run-fix-1", VIDEO,
                              RefineParams(), log=lambda *_: None)
    st = stats["stats"]
    assert st["emote_only"] == 2   # [微笑] pair exact-deduped, then bucketed
    members = store.run_cluster_members(conn, "run-fix-1", None, False)
    assert members, "refined rows must exist"
    for m in members:
        assert 0.0 < m["quality_score"] <= 1.0      # defined for EVERY row
    emote_cluster = [c for c in store.run_clusters(conn, "run-fix-1")
                     if c["label"] == "纯表情/打卡"]
    assert emote_cluster and emote_cluster[0]["size"] == 2
    labels = [c["label"] for c in store.run_clusters(conn, "run-fix-1")]
    assert "纯表情/打卡" in labels
    topic = [c for c in store.run_clusters(conn, "run-fix-1")
             if "掠夺" in (c["label"] + "".join(c["keywords"]))]
    assert topic, "topic cluster must mention 掠夺 in label/keywords"


def test_run_cluster_members_carries_thread_fields(lab_root):
    from conftest import VIDEO
    conn = store.connect(lab_root / "data")
    store.ingest_video(conn, lab_root / "data" / "raw" / VIDEO, VIDEO)
    refine.run_refine(conn, lab_root / "data", "run-fix-2", VIDEO,
                      RefineParams(), log=lambda *_: None)
    members = store.run_cluster_members(conn, "run-fix-2", None, False)
    assert members
    for m in members:
        assert "root_id" in m and "parent_id" in m
        if m["level"] == 2:
            assert m["root_id"] is not None
    for m in members:
        expected = "representative" if m["is_representative"] else "cluster_member"
        assert m["selection_reason"] == expected


# ---- ingest snapshot semantics ------------------------------------------

def test_ingest_deletes_stale_rows(tmp_path):
    from conftest import VIDEO
    raw = tmp_path / "data" / "raw" / VIDEO
    raw.mkdir(parents=True)
    with (raw / "comments.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(dict(
            comment_id=1, root_id=1, parent_id=None, is_reply=False,
            user_id=1, content="第一版", like_count=1, reply_count=0,
            ctime=1700000000, is_pinned=False), ensure_ascii=False) + "\n")
    conn = store.connect(tmp_path / "data")
    store.ingest_video(conn, raw, VIDEO)
    # an older crawl left an extra row in the index
    conn.execute(
        "INSERT INTO comments (video_id, comment_id, root_id, parent_id, level,"
        " content, content_sha) VALUES (?,?,1,1,1,'旧爬取残留','sha-old')",
        (VIDEO, 999))
    conn.commit()
    out = store.ingest_video(conn, raw, VIDEO)
    assert out["stale_removed"] == 1
    left = conn.execute(
        "SELECT COUNT(*) FROM comments WHERE video_id=?", (VIDEO,)).fetchone()[0]
    assert left == 1


# ---- report assembly -----------------------------------------------------


def _import_webapp():
    """webapp.app runs load_env() at import, which would leak the real .env
    (LLM_* etc.) into the test process and flip later LLM-detection tests —
    snapshot and restore the environment around the import."""
    import os
    snapshot = dict(os.environ)
    try:
        import webapp.app as webapp_module
        return webapp_module
    finally:
        os.environ.clear()
        os.environ.update(snapshot)

def test_strip_report_header():
    _strip_report_header = _import_webapp()._strip_report_header
    section = ("# 分析报告 — BVx\n\n- 生成范围：10 条评论（一级 5 / 楼中楼 5 / 置顶 0）\n"
               "- 模型：m\n- 方式：map-reduce\n\n## 摘要\n\n正文")
    assert _strip_report_header(section).startswith("## 摘要")
    assert "# 分析报告" not in _strip_report_header(section)


def test_load_id_map_ignores_bad_lines(tmp_path):
    d = tmp_path / "normalized" / "BVx"
    d.mkdir(parents=True)
    (d / "comments.jsonl").write_text(
        '{"comment_id": 1, "content": "a"}\n{broken}\n'
        '{"comment_id": "x", "content": "b"}\n', encoding="utf-8")
    m = load_id_map(tmp_path, "BVx")
    assert list(m) == [1]


def test_privacy_scrub_nicknames_in_content():
    _privacy_scrub = _import_webapp()._privacy_scrub
    row = {"content": "回复 @某位用户：你说得对", "username": "u", "mid": "42"}
    out = _privacy_scrub(row)
    assert "某位用户" not in out["content"]
    assert out["content"].startswith("回复 @匿名")
    assert out["username"] == "" and out["mid"] == ""
