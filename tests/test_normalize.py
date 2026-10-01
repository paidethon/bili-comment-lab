"""Unit tests — normalize schema, dedupe, parent/child, pinned, hashing."""
from __future__ import annotations

import json

from bilicom.normalize import (
    author_hash,
    dedupe,
    normalize,
    normalize_record,
)

from conftest import VIDEO


def read_normalized(root):
    f = root / "data" / "normalized" / VIDEO / "comments.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_schema_fields(lab_root):
    recs = read_normalized(lab_root)
    assert recs
    for r in recs:
        assert set(r) == {
            "video_id", "comment_id", "parent_comment_id", "content",
            "created_at", "like_count", "reply_count", "is_pinned",
            "level", "author_hash",
        }


def test_dedupe_keeps_first(lab_root):
    recs = read_normalized(lab_root)
    ids = [r["comment_id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicates must be removed"
    assert 200 in ids


def test_parent_child_relations(lab_root):
    recs = {r["comment_id"]: r for r in read_normalized(lab_root)}
    assert recs[101]["parent_comment_id"] == 100      # 楼中楼 direct child
    assert recs[102]["parent_comment_id"] == 101      # nested answer keeps real parent
    assert recs[200]["parent_comment_id"] is None     # level-1 has no parent
    assert recs[101]["level"] == 2 and recs[200]["level"] == 1


def test_pinned_flag_preserved(lab_root):
    recs = {r["comment_id"]: r for r in read_normalized(lab_root)}
    assert recs[100]["is_pinned"] is True
    # a reply inside a pinned thread is a normal reply, not itself pinned
    assert recs[101]["is_pinned"] is False
    assert recs[200]["is_pinned"] is False


def test_malformed_line_skipped(lab_root):
    recs = read_normalized(lab_root)
    assert len(recs) == 7  # 8 raw records: 1 malformed line skipped, 1 duplicate dropped


def test_author_hash_stable_and_salted():
    h1 = author_hash(12345, "salt-a")
    h2 = author_hash(12345, "salt-a")
    h3 = author_hash(12345, "salt-b")
    assert h1 == h2 and h1 != h3
    assert len(h1) == 16
    assert "12345" not in h1


def test_same_user_same_hash_across_records(lab_root):
    recs = {r["comment_id"]: r for r in read_normalized(lab_root)}
    assert recs[200]["author_hash"] == recs[101]["author_hash"]  # both user_id=8
    assert recs[100]["author_hash"] != recs[200]["author_hash"]


def test_no_private_fields_leak(lab_root):
    raw_text = (lab_root / "data" / "normalized" / VIDEO / "comments.jsonl").read_text("utf-8")
    for banned in ("user_id", "username", "ip_location", "avatar", "user_level"):
        assert f'"{banned}"' not in raw_text


def test_orphan_reply_counted(lab_root):
    stats = json.loads((lab_root / "data" / "normalized" / VIDEO / "stats.json").read_text())
    assert stats["orphan_replies"] == 1
    assert stats["pinned"] == 1
    assert stats["malformed_lines"] == 1


def test_dedupe_helper():
    rows = [{"comment_id": 1}, {"comment_id": 1}, {"comment_id": 2}]
    assert [r["comment_id"] for r in dedupe(rows)] == [1, 2]
