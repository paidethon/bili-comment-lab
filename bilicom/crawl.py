"""Crawl step: fetch comments (including pinned ones) into data/raw/<id>/comments.jsonl.

Uses the vendored BilibiliCrawler package for all Bilibili API access
(wbi signing, pagination, adaptive rate limiting). Adds what upstream lacks:
pinned ("置顶") comments are read from the first page's ``top_replies`` /
``top`` fields — the same approach MediaCrawler uses — and their 楼中楼 is
crawled like any other root comment.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bilibili_crawler.api.bilibili_api import BilibiliAPI
from bilibili_crawler.crawler.comment_crawler import CommentCrawler

logger = logging.getLogger("bilicom.crawl")


class CrawlAborted(Exception):
    """Raised when the site answers with captcha/412/risk-control signals.

    Policy (task rules): stop crawling immediately, never try to evade.
    """


@dataclass
class CrawlResult:
    video_id: str
    oid: int
    content_type: str
    out_dir: Path
    comments: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.comments)


def _looks_like_risk_control(payload: Dict[str, Any]) -> bool:
    code = payload.get("code")
    if code in (-412, -352, -401):
        return True
    msg = str(payload.get("message") or "")
    return any(k in msg for k in ("验证码", "风控", "请求被拦截", "ganming"))


def fetch_pinned_comments(api: BilibiliAPI, oid: int, type_id: int) -> List[Dict[str, Any]]:
    """Extract pinned comments from the first page payload.

    Bilibili exposes them under ``data.top_replies`` (current) and sometimes
    ``data.top``; each entry looks like a regular reply object plus
    ``reply_control.is_up_top / is_admin_top`` flags.
    """
    payload = api.get_comments(oid, page=1, mode=3, type_id=type_id, next_page=0)
    if not payload or _looks_like_risk_control(payload):
        raise CrawlAborted(f"risk control on first page: {payload}")
    data = (payload or {}).get("data") or {}
    pinned: List[Dict[str, Any]] = []
    seen = set()
    for source_key in ("top_replies", "top"):
        entries = data.get(source_key) or []
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rpid = entry.get("rpid")
            if not rpid or rpid in seen:
                continue
            seen.add(rpid)
            entry.setdefault("reply_control", {})
            entry["reply_control"]["is_up_top"] = True  # normalise the flag
            pinned.append(entry)
    return pinned


def _is_pinned(raw: Dict[str, Any]) -> bool:
    rc = raw.get("reply_control") or {}
    return bool(rc.get("is_up_top") or rc.get("is_admin_top"))


def list_uploader_videos(mid: int, limit: int = 10,
                         cookie: Optional[str] = None) -> List[Dict[str, Any]]:
    """List an uploader's recent videos (space arc search, wbi-signed via the
    vendored API layer). Used by the webapp for UP 主批量分析 — returns
    [{bvid, title, created, play}] newest first. No risk-control evasion: a
    412 raises CrawlAborted like every other call."""
    api = BilibiliAPI()
    if cookie:
        api.set_cookie(cookie)
    payload = api._request_wbi(
        "https://api.bilibili.com/x/space/wbi/arc/search",
        {"mid": int(mid), "pn": 1, "ps": min(max(1, limit), 30),
         "order": "pubdate"})
    if not payload or _looks_like_risk_control(payload):
        raise CrawlAborted(f"uploader video list blocked: {payload}")
    if payload.get("code") != 0:
        raise CrawlAborted(f"uploader video list failed: code={payload.get('code')}"
                           f" {payload.get('message')}")
    vlist = ((payload.get("data") or {}).get("list") or {}).get("vlist") or []
    out = []
    for v in vlist[:limit]:
        out.append({"bvid": v.get("bvid"), "title": v.get("title"),
                    "created": v.get("created"), "play": v.get("play")})
    return out


def crawl(
    url_or_id: str,
    out_root: Path,
    cookie: Optional[str] = None,
    include_replies: bool = True,
    max_pages: int = 100,
    mode: int = 3,
    limit: Optional[int] = None,
) -> CrawlResult:
    """Crawl comments for a video/dynamic/article and write the raw dataset.

    Raw output is immutable ground truth for every later step.
    """
    api = BilibiliAPI()
    if cookie:
        api.set_cookie(cookie)

    crawler = CommentCrawler(progress_callback=lambda m: logger.info(m), api=api)

    target = crawler.resolve_target(url_or_id)
    if not target or target.oid is None:
        raise CrawlAborted(f"cannot resolve target: {url_or_id}")

    video_id = getattr(target, "bvid", None) or getattr(target, "video_id", None) or f"oid{target.oid}"

    # Upstream crawl (handles wbi, pagination, 楼中楼, dedupe, rate limiting).
    comments: List[Dict[str, Any]] = []
    try:
        comments = crawler.crawl_comments(
            url_or_id, include_replies=include_replies, max_pages=max_pages, mode=mode
        )
    except Exception as err:  # CrawlError carries partial results
        partial = getattr(err, "args", [None])
        if len(partial) > 1 and isinstance(partial[1], list):
            comments = partial[1]
            logger.error("crawl failed after %s comments: %s", len(comments), err)
            if not comments:
                raise
        else:
            raise

    # Pinned comments: upstream misses them entirely (only reads data.replies).
    pinned_raw = fetch_pinned_comments(api, target.oid, target.content_type)
    known_ids = {c.get("comment_id") for c in comments}
    for entry in pinned_raw:
        comment = crawler._process_comment(entry, target.oid, is_reply=False)
        comment["is_pinned"] = True
        if comment["comment_id"] not in known_ids:
            comments.insert(0, comment)
            known_ids.add(comment["comment_id"])
        if include_replies:
            # reuse upstream 楼中楼 logic for the pinned thread
            replies = crawler._crawl_single_reply(entry, target.oid)
            for r in replies:
                r["root_is_pinned"] = True
                if r.get("comment_id") not in known_ids:
                    comments.append(r)
                    known_ids.add(r["comment_id"])
    # flag pinned roots that were already picked up by the normal loop
    pinned_ids = {e.get("rpid") for e in pinned_raw}
    for c in comments:
        if c.get("comment_id") in pinned_ids:
            c["is_pinned"] = True

    if limit is not None:
        comments = comments[:limit]

    out_dir = out_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "comments.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "video_id": video_id,
        "oid": target.oid,
        "content_type": target.content_type,
        "source_url": url_or_id,
        "crawl_mode": mode,
        "max_pages": max_pages,
        "include_replies": include_replies,
        "total_comments": len(comments),
        "root_comments": sum(1 for c in comments if not c.get("is_reply")),
        "reply_comments": sum(1 for c in comments if c.get("is_reply")),
        "pinned_comments": sum(1 for c in comments if c.get("is_pinned")),
        "target_info": crawler.target_info,
        "reply_warnings": crawler.reply_warnings,
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream": "BilibiliCrawler (MIT) + bilicom pinned patch",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return CrawlResult(
        video_id=video_id,
        oid=target.oid,
        content_type=target.content_type,
        out_dir=out_dir,
        comments=comments,
        meta=meta,
    )
