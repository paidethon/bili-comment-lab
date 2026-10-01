"""Normalize step: raw crawler JSONL -> the stable analysis schema.

Raw files under data/raw/ are ground truth and are never modified. Normalized
output lives in data/normalized/<id>/comments.jsonl with one record per comment:

    {
      "video_id", "comment_id", "parent_comment_id", "content",
      "created_at", "like_count", "reply_count", "is_pinned",
      "level", "author_hash"
    }

Privacy: raw user ids are replaced by a salted hash (BILI_HASH_SALT from the
environment, or a file-local default salt). Nothing else about the author is
kept — no nickname, uid, ip location, avatar or level.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FIELDS = [
    "video_id",
    "comment_id",
    "parent_comment_id",
    "content",
    "created_at",
    "like_count",
    "reply_count",
    "is_pinned",
    "level",
    "author_hash",
]


def author_hash(user_id: Any, salt: str) -> str:
    """Deterministic salted hash so the same commenter can be traced without
    exposing their uid."""
    if user_id in (None, "", 0):
        return ""
    digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    return digest[:16]


def _to_iso(ctime: Any) -> str:
    try:
        return (
            datetime.fromtimestamp(int(ctime), tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds")
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _parent_of(record: Dict[str, Any]) -> Optional[int]:
    """Parent id for 楼中楼 replies; None for level-1 comments.

    Upstream keeps root_id == rpid for roots; a reply's parent may point at
    an intermediate reply inside the same thread (nested answers), which is
    the relationship worth preserving.
    """
    if record.get("is_reply"):
        parent = record.get("parent_id") or record.get("root_id")
        if parent and parent != record.get("comment_id"):
            return int(parent)
    return None


def normalize_record(
    raw: Dict[str, Any], video_id: str, salt: str
) -> Dict[str, Any]:
    parent = _parent_of(raw)
    return {
        "video_id": video_id,
        "comment_id": int(raw["comment_id"]),
        "parent_comment_id": parent,
        "content": str(raw.get("content", "") or ""),
        "created_at": _to_iso(raw.get("ctime")),
        "like_count": int(raw.get("like_count") or 0),
        "reply_count": int(raw.get("reply_count") or 0),
        "is_pinned": bool(raw.get("is_pinned")),
        "level": 2 if raw.get("is_reply") else 1,
        "author_hash": author_hash(raw.get("user_id"), salt),
    }


def default_salt() -> str:
    return os.environ.get("BILI_HASH_SALT", "bilicom-default-salt")


def dedupe(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the first occurrence of each comment_id, preserving order."""
    seen: set = set()
    out = []
    for r in records:
        cid = r.get("comment_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


def normalize(
    raw_dir: Path,
    video_id: str,
    salt: Optional[str] = None,
) -> Dict[str, Any]:
    raw_file = raw_dir / "comments.jsonl"
    records = []
    bad = 0
    for line in raw_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(normalize_record(json.loads(line), video_id, salt or default_salt()))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            bad += 1
    records = dedupe(records)
    # stable, human-friendly order: level-1 by likes desc, then their replies
    roots = sorted(
        [r for r in records if r["level"] == 1],
        key=lambda r: -r["like_count"],
    )
    children = [r for r in records if r["level"] == 2]
    ordered = dedupe(roots + children)

    out_dir = raw_dir.parent.parent / "normalized" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "comments.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    orphan = sum(
        1
        for r in ordered
        if r["parent_comment_id"] is not None
        and r["parent_comment_id"] not in {x["comment_id"] for x in ordered}
    )
    stats = {
        "video_id": video_id,
        "normalized": len(ordered),
        "input_lines": len(ordered) + bad,
        "malformed_lines": bad,
        "level1": sum(1 for r in ordered if r["level"] == 1),
        "level2": sum(1 for r in ordered if r["level"] == 2),
        "pinned": sum(1 for r in ordered if r["is_pinned"]),
        "orphan_replies": orphan,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
