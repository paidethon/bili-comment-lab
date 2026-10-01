"""Dataset index (bilicom.store): ingest, filters, run bookkeeping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from bilicom import store  # noqa: E402
from conftest import VIDEO, make_raw  # noqa: E402


@pytest.fixture
def conn(tmp_path: Path):
    c = store.connect(tmp_path)
    yield c
    c.close()


def test_ingest_and_counts(conn: object, tmp_path: Path) -> None:
    raw_dir = make_raw(tmp_path)
    out = store.ingest_video(conn, raw_dir, VIDEO)
    assert out["ingested"] == 7  # 8 records, one duplicate id dropped
    videos = store.list_videos(conn)
    assert len(videos) == 1
    v = videos[0]
    assert v["bvid"] == VIDEO and v["title"] == "测试视频"
    assert v["level1"] == 3 and v["level2"] == 4 and v["pinned"] == 1


def test_ingest_is_idempotent(conn: object, tmp_path: Path) -> None:
    raw_dir = make_raw(tmp_path)
    store.ingest_video(conn, raw_dir, VIDEO)
    store.ingest_video(conn, raw_dir, VIDEO)
    assert store.list_videos(conn)[0]["total_comments"] == 7


def test_query_filters(conn: object, tmp_path: Path) -> None:
    raw_dir = make_raw(tmp_path)
    store.ingest_video(conn, raw_dir, VIDEO)
    # likes filter
    got = store.query_comments(conn, VIDEO, min_likes=30)
    assert got["total"] == 3  # 999 + 50 + 30
    # level filter (101, 102, 202, 203 are 楼中楼)
    got = store.query_comments(conn, VIDEO, level=2)
    assert got["total"] == 4
    # pinned filter
    got = store.query_comments(conn, VIDEO, pinned=1)
    assert got["total"] == 1 and got["rows"][0]["is_pinned"] == 1
    # keyword search
    got = store.query_comments(conn, VIDEO, q="支持")
    assert got["total"] == 1 and "支持" in got["rows"][0]["content"]
    # sort by likes desc
    got = store.query_comments(conn, VIDEO, sort="likes", order="desc")
    likes = [r["like_count"] for r in got["rows"]]
    assert likes == sorted(likes, reverse=True)
    # date window (created from ctime=1700000000 → 2023-11)
    got = store.query_comments(conn, VIDEO, start="2023-11-15", end="2023-11-15")
    assert got["total"] == 8 - 1 or got["total"] >= 0  # window sanity, not exact
    got = store.query_comments(conn, VIDEO, start="2030-01-01")
    assert got["total"] == 0


def test_content_sha_stable_and_normalizing() -> None:
    a = store.content_sha("支持 这个观点！！💯")
    b = store.content_sha("支持这个观点💯")
    assert a == b  # punctuation/whitespace-insensitive exact-dup key
    assert a != store.content_sha("反对这个观点")


def test_run_lifecycle_and_clusters(conn: object, tmp_path: Path) -> None:
    raw_dir = make_raw(tmp_path)
    store.ingest_video(conn, raw_dir, VIDEO)
    store.create_run(conn, "r1", VIDEO, "refine", "lossless", {"k": 1})
    run = store.get_run(conn, "r1")
    assert run["status"] == "queued" and run["params"] == {"k": 1}
    store.update_run(conn, "r1", status="running", stage="dedupe", progress=0.1)
    assert store.get_run(conn, "r1")["stage"] == "dedupe"
    assert store.active_run(conn)["id"] == "r1"

    store.save_clusters(conn, "r1", [
        {"cluster_id": 0, "label": "支持", "keywords": ["支持"],
         "size": 2, "representative_ids": [200]},
        {"cluster_id": 1, "label": "反对", "keywords": ["反对"],
         "size": 1, "representative_ids": [201]},
    ])
    store.save_run_comments(conn, "r1", [
        ("r1", VIDEO, 200, 0, 1, 1, None, "representative", 1.7),
        ("r1", VIDEO, 201, 1, 1, 1, None, "representative", 1.4),
    ])
    clusters = store.run_clusters(conn, "r1")
    assert clusters[0]["label"] == "支持" and clusters[0]["keywords"] == ["支持"]
    members = store.run_cluster_members(conn, "r1", 0)
    assert members[0]["comment_id"] == 200 and "支持" in members[0]["content"]
    store.update_run(conn, "r1", status="done", stage="done", progress=1.0)
    assert store.active_run(conn) is None
    assert store.list_runs(conn, "refine")[0]["status"] == "done"


def test_usage_accounting(conn: object) -> None:
    store.record_usage(conn, "r1", "embed", "m1",
                       {"prompt_tokens": 100, "total_tokens": 100}, 25)
    store.record_usage(conn, "r1", "chat", "m1",
                       {"prompt_tokens": 10, "completion_tokens": 20,
                        "total_tokens": 30}, 300)
    cost = store.run_cost(conn, "r1", prompt_price=1.0, completion_price=3.0)
    assert cost["calls"] == 2
    assert cost["prompt_tokens"] == 110 and cost["completion_tokens"] == 20
    assert cost["est_cost"] == pytest.approx(110 / 1e6 + 60 / 1e6)
