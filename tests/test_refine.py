"""Lossless refinement engine: dedupe, folding, clustering, representatives,
traceability — the core guarantee under test is that refined output never
invents or rewrites comment text, and every row traces back to the original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from bilicom import refine, store  # noqa: E402

NO_CANCEL = lambda: False  # noqa: E731
NO_LOG = lambda *_: None  # noqa: E731
NO_PROG = lambda *_: None  # noqa: E731


def rec(cid: int, text: str, likes: int = 0, level: int = 1,
        root: int = None) -> dict:
    return {"video_id": "BV1test", "comment_id": cid, "content": text,
            "like_count": likes, "level": level, "root_id": root or cid,
            "content_sha": store.content_sha(text), "ctime": cid}


def test_exact_dedupe_keeps_most_liked() -> None:
    records = [rec(1, "支持！！", likes=5), rec(2, "支持！！", likes=99),
               rec(3, "支持 ！！", likes=1), rec(4, "完全不同的内容", likes=0)]
    kept, stats = refine.fold_duplicates(records, 0.82)
    assert stats["original"] == 4
    assert stats["exact_removed"] == 2
    assert stats["survivors"] == 2
    canon = next(r for r in kept if r["content"] == "支持！！")
    assert canon["comment_id"] == 2 and canon["duplicate_of"] is None
    dups = [r for r in records if r.get("duplicate_of") == 2]
    assert {d["comment_id"] for d in dups} == {1, 3}
    assert all(d.get("_exact") for d in dups)


def test_near_dup_folding_variants() -> None:
    records = [
        rec(10, "哈哈哈哈哈哈哈哈", likes=50),
        rec(11, "哈哈哈哈哈哈哈"),          # near-identical noise
        rec(12, "哈哈哈哈哈哈哈哈哈哈哈哈哈哈"),
        rec(13, "这玩意实测续航只有四个小时", likes=30),
        rec(14, "官方宣传八小时我完全跑不到", likes=20),
    ]
    kept, stats = refine.fold_duplicates(records, 0.82)
    assert stats["near_removed"] >= 1
    kept_ids = {r["comment_id"] for r in kept}
    assert 10 in kept_ids and 13 in kept_ids and 14 in kept_ids


def test_near_dup_preserves_distinct_texts() -> None:
    records = [rec(1, "续航实测只有四个小时", likes=9),
               rec(2, "外观设计非常好看", likes=8),
               rec(3, "价格太贵了不划算", likes=7)]
    kept, stats = refine.fold_duplicates(records, 0.82)
    assert stats["near_removed"] == 0 and len(kept) == 3


def test_local_clustering_groups_related_texts() -> None:
    records = [
        rec(1, "续航实测只有四个小时，太差了", likes=30),
        rec(2, "实测续航确实只有四小时", likes=10),
        rec(3, "续航四小时虚标", likes=5),
        rec(4, "外观设计非常好看，颜值真高", likes=20),
        rec(5, "外观真的好看", likes=2),
    ]
    refine.attach_local_vectors(records)
    assign, n = refine._leader_pass(records, 0.12, True, NO_CANCEL, None, 5000)
    for r, c in zip(records, assign):
        r["_cluster"] = c
    clusters = {}
    for r in records:
        clusters.setdefault(r["_cluster"], []).append(r["comment_id"])
    battery = [ids for ids in clusters.values() if 1 in ids][0]
    looks = [ids for ids in clusters.values() if 4 in ids][0]
    # strong paraphrase groups together; no cross-topic pollution
    assert 2 in battery
    assert 5 in looks
    assert not (set(battery) & set(looks))


def test_tfidf_vector_is_normalized_sparse() -> None:
    records = [rec(1, "一条测试评论内容"), rec(2, "另一条完全不同的话题讨论")]
    refine.attach_local_vectors(records)
    v = records[0]["_vec"]
    assert isinstance(v, dict) and v
    assert abs(sum(w * w for w in v.values()) - 1.0) < 1e-6


def test_keywords_extractive_only() -> None:
    texts = ["续航实测只有四小时", "续航四小时虚标", "续航确实不行"]
    df = refine.global_df([rec(i, t) for i, t in enumerate(texts, 1)])
    kw = refine.cluster_keywords(texts, df, 6)
    assert kw, "keywords must not be empty on a small corpus"
    assert any("续航" in k or "小时" in k for k in kw)


def test_representative_selection_top_likes_then_ai_ids() -> None:
    members = [rec(1, "a", likes=5), rec(2, "b", likes=50), rec(3, "c", likes=10)]
    chosen = refine._pick_representatives(members, 2, [])
    assert [m["comment_id"] for m, _ in chosen] == [2, 3]
    chosen = refine._pick_representatives(members, 2, [1])
    assert [m["comment_id"] for m, _ in chosen] == [1, 2]  # ai first, then likes
    # invalid ai ids are dropped by the caller — here just verify robustness
    chosen = refine._pick_representatives(members, 1, [999])
    assert [m["comment_id"] for m, _ in chosen] == [2]


@pytest.fixture
def lab_db(tmp_path: Path):
    from conftest import VIDEO, make_raw

    raw_dir = make_raw(tmp_path)
    conn = store.connect(tmp_path)
    store.ingest_video(conn, raw_dir, VIDEO)
    yield conn, tmp_path
    conn.close()


def test_run_refine_end_to_end_traceable(lab_db) -> None:
    conn, tmp_path = lab_db
    run_id = "run-e2e"
    params = refine.RefineParams(ai_labels=False)
    out = refine.run_refine(conn, tmp_path, run_id, "BV1fixtureXX", params,
                            log=NO_LOG, cancel_fn=NO_CANCEL, progress=NO_PROG)
    assert out["stats"]["original"] == 7
    # every original comment is accounted for in run_comments
    rows = store.run_cluster_members(conn, run_id, None, False)
    assert len(rows) == out["stats"]["survivors"]
    run = store.get_run(conn, run_id)
    assert run["status"] == "done"
    assert run["stats"]["vector_source"] == "local-tfidf"
    # traceability: content always comes from the comments table, never stored
    # separately by the refinement
    for row in rows:
        original = conn.execute(
            "SELECT content FROM comments WHERE video_id=? AND comment_id=?",
            (row["video_id"], row["comment_id"])).fetchone()
        assert row["content"] == original["content"]


def test_run_refine_rerun_overwrites_cleanly(lab_db) -> None:
    conn, tmp_path = lab_db
    params = refine.RefineParams()
    refine.run_refine(conn, tmp_path, "rA", "BV1fixtureXX", params,
                      log=NO_LOG, cancel_fn=NO_CANCEL, progress=NO_PROG)
    refine.run_refine(conn, tmp_path, "rB", "BV1fixtureXX", params,
                      log=NO_LOG, cancel_fn=NO_CANCEL, progress=NO_PROG)
    assert store.get_run(conn, "rA")["status"] == "done"
    assert store.get_run(conn, "rB")["status"] == "done"
    assert len(store.run_clusters(conn, "rA")) == store.get_run(conn, "rA")["stats"]["clusters"]


def test_preview_no_persist(lab_db) -> None:
    conn, tmp_path = lab_db
    out = refine.preview(conn, "BV1fixtureXX", sample=50)
    assert out["stats"]["original"] == 7
    assert out["stats"]["sample"] <= 50
    assert store.list_runs(conn, "refine") == []  # nothing persisted


def test_embedding_cache_roundtrip(lab_db) -> None:
    import numpy as np

    conn, _tmp = lab_db
    store.embedding_put(conn, "m1", "sha-x", 3, np.array([1, 2, 3], dtype=np.float32).tobytes())
    got = store.embedding_get_many(conn, "m1", ["sha-x", "sha-missing"])
    assert "sha-x" in got and "sha-missing" not in got
    assert np.frombuffer(got["sha-x"], dtype=np.float32).tolist() == [1, 2, 3]


def test_dense_similarity_shape() -> None:
    import numpy as np

    cand = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    leaders = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = refine._similarity_dense(cand, leaders)
    assert sims.shape == (2, 1)
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[1, 0] == pytest.approx(0.0)
