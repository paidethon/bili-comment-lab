"""Shared fixtures: synthetic raw comment datasets covering every case the
task requires — roots, 楼中楼, pinned comments + their replies, duplicates,
emoji/UTF-8, malformed lines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

VIDEO = "BV1fixtureXX"


def make_raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw" / VIDEO
    raw_dir.mkdir(parents=True)
    records = [
        # pinned root + its 楼中楼
        dict(comment_id=100, root_id=100, parent_id=None, is_reply=False,
             user_id=7, content="置顶评论：欢迎讨论 📌",
             like_count=999, reply_count=2, ctime=1700000000, is_pinned=True),
        dict(comment_id=101, root_id=100, parent_id=100, is_reply=True,
             user_id=8, content="置顶楼中楼A", like_count=5, reply_count=0, ctime=1700000100),
        dict(comment_id=102, root_id=100, parent_id=101, is_reply=True,
             user_id=9, content="回复置顶楼中楼A的楼中楼", like_count=1, reply_count=0, ctime=1700000200),
        # normal roots
        dict(comment_id=200, root_id=200, parent_id=None, is_reply=False,
             user_id=8, content="支持这个观点 💯", like_count=50, reply_count=1, ctime=1700001000),
        dict(comment_id=201, root_id=201, parent_id=None, is_reply=False,
             user_id=10, content="反对，数据不对", like_count=30, reply_count=0, ctime=1700002000),
        # duplicate id (must be dropped)
        dict(comment_id=200, root_id=200, parent_id=None, is_reply=False,
             user_id=8, content="支持这个观点 💯", like_count=50, reply_count=1, ctime=1700001000),
        # nested reply with emoji + newlines
        dict(comment_id=202, root_id=200, parent_id=200, is_reply=True,
             user_id=9, content="楼中楼：line1\nline2 🎉 <script>alert(1)</script>",
             like_count=2, reply_count=0, ctime=1700003000),
        # orphan reply (parent missing from dataset)
        dict(comment_id=203, root_id=404, parent_id=404, is_reply=True,
             user_id=11, content="孤儿楼中楼", like_count=0, reply_count=0, ctime=1700004000),
    ]
    with (raw_dir / "comments.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write("{not valid json}\n")  # malformed line

    meta = {
        "video_id": VIDEO, "oid": 42, "content_type": 1,
        "total_comments": len(records),
        "target_info": {"title": "测试视频", "owner": "UP主", "desc": "描述"},
    }
    (raw_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return raw_dir


@pytest.fixture
def lab_root(tmp_path: Path) -> Path:
    """Raw fixture data plus the normalized/agent artifacts derived from it."""
    from bilicom.normalize import normalize
    from bilicom.prepare import prepare

    root = tmp_path
    make_raw(root)
    normalize(root / "data" / "raw" / VIDEO, VIDEO)
    prepare(root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
    return root
