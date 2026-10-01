"""Unit tests — LLM cache keys, loose JSON parsing, prompt versioning."""
from __future__ import annotations

import json

import pytest

from bilicom.llm import LLMClient, _parse_json_loose
from bilicom.prepare import compute_statistics

from conftest import VIDEO


def make_client(tmp_path, model="flash-test"):
    return LLMClient(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model=model,
        cache_dir=tmp_path / "cache",
    )


def test_cache_key_changes_with_model(tmp_path):
    a = make_client(tmp_path, "model-a").cache_key("extract", "content")
    b = make_client(tmp_path, "model-b").cache_key("extract", "content")
    assert a != b


def test_cache_key_changes_with_content(tmp_path):
    c = make_client(tmp_path)
    assert c.cache_key("extract", "a") != c.cache_key("extract", "b")


def test_cache_key_changes_with_kind(tmp_path):
    c = make_client(tmp_path)
    assert c.cache_key("extract", "a") != c.cache_key("topic", "a")


def test_cache_roundtrip_and_metadata(tmp_path):
    c = make_client(tmp_path)
    key = c.cache_key("extract", "hello")
    c._cache_put(key, {"model": c.model, "prompt_version": "extract-v1",
                       "timestamp": "t", "input_hash": "h", "result": {"x": 1}})
    hit = c._cache_get(key)
    assert hit is not None
    assert hit["result"] == {"x": 1}
    assert hit["prompt_version"] == "extract-v1"


def test_parse_json_loose_variants():
    assert _parse_json_loose('{"a": 1}') == {"a": 1}
    assert _parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_loose('Sure! Here is: {"a": {"b": 2}} ok') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        _parse_json_loose("no json here")


def test_extractive_only_reports_written(lab_root, tmp_path):
    """Without LLM env the analyze step must still produce full report set."""
    from bilicom.prepare import prepare
    from bilicom.analyze import analyze, extractive_only_analysis

    manifest = prepare(lab_root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
    pack_dir = lab_root / "data" / "agent" / VIDEO
    result = extractive_only_analysis(pack_dir, manifest)
    from bilicom.analyze import _write_reports, load_id_map
    _write_reports(VIDEO, lab_root / "data", result, manifest,
                   load_id_map(lab_root / "data", VIDEO))

    reports = lab_root / "data" / "reports" / VIDEO
    for name in ["summary.md", "topics.md", "opinions.md", "questions.md",
                 "suggestions.md", "representative-comments.md", "statistics.json"]:
        assert (reports / name).exists(), f"missing {name}"
    text = (reports / "topics.md").read_text("utf-8")
    assert "extractive-only" in text or "未配置" in text


def test_statistics_no_private_data(lab_root):
    stats = compute_statistics([])
    assert stats["total_comments"] == 0
    payload = json.dumps(stats)
    assert "user_id" not in payload
