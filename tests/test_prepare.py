"""Unit tests — agent pack: chunking, budgets, Markdown quoting, emoji/UTF-8."""
from __future__ import annotations

import json

from bilicom.prepare import chunk_records, compute_statistics, prepare, quote_text

from conftest import VIDEO


def test_quote_text_preserves_verbatim():
    original = "line1\nline2 🎉 <script>alert(1)</script>"
    quoted = quote_text(original)
    for token in ("line1", "line2", "🎉", "<script>alert(1)</script>"):
        assert token in quoted
    assert all(line.startswith(">") for line in quoted.splitlines())


def test_quote_text_empty():
    assert quote_text("") == "> "


def test_chunk_respects_budget():
    recs = [{"content": "x" * 1000} for _ in range(50)]
    chunks = chunk_records(recs, budget_tokens=1000)
    assert len(chunks) >= 2
    for c in chunks:
        size = sum(len(r["content"]) + 160 for r in c)
        assert size <= 1000 * 1.6 + 1000  # one record may exceed, never two


def test_oversized_single_comment_gets_own_chunk():
    recs = [{"content": "a" * 5000}, {"content": "small"}, {"content": "b" * 5000}]
    chunks = chunk_records(recs, budget_tokens=500)
    assert len(chunks) == 3
    assert len(chunks[1]) == 1 and chunks[1][0]["content"] == "small"


def test_chunk_no_comment_dropped():
    recs = [{"content": f"c{i}" * 50} for i in range(120)]
    total = sum(len(c) for c in chunk_records(recs, budget_tokens=800))
    assert total == 120


def test_statistics_numbers(lab_root):
    prepare(lab_root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
    stats = json.loads(
        (lab_root / "data" / "agent" / VIDEO / "statistics.json").read_text("utf-8")
    )
    assert stats["total_comments"] == 7
    assert stats["pinned"] == 1
    assert stats["like_max"] == 999
    assert stats["top_liked"][0]["comment_id"] == 100


def test_agent_pack_files(lab_root):
    manifest = prepare(lab_root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
    d = lab_root / "data" / "agent" / VIDEO
    for name in ["manifest.json", "video.md", "statistics.json"]:
        assert (d / name).exists()
    assert manifest["total_comments"] == 7
    for part in manifest["parts"]:
        text = (d / part["file"]).read_text("utf-8")
        assert "## Comment" in text


def test_pack_contains_complete_originals(lab_root):
    prepare(lab_root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
    text = (lab_root / "data" / "agent" / VIDEO / "part-0001.md").read_text("utf-8")
    assert "置顶评论：欢迎讨论 📌" in text          # emoji preserved
    assert "<script>alert(1)</script>" in text    # original not sanitized away
    assert "line1" in text and "line2" in text    # multiline preserved
