"""bili — command line entry point.

    ./bili crawl  "BILIBILI_URL"          抓取评论 -> data/raw/
    ./bili prepare "BV_ID"                生成 Agent Pack -> data/agent/
    ./bili analyze "BV_ID"                map-reduce AI 分析 -> data/reports/
    ./bili all    "BILIBILI_URL"          全流程

Options common to crawl/all: --max-pages N (0=all), --mode 2|3 (热度|时间),
--no-replies, --limit N (test helper), --cookie "SESSDATA=...".
prepare/analyze: --budget N (tokens per part/charge).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

DATA_ROOT = LAB / "data"


def load_env() -> None:
    """Minimal .env loader (no dotenv dependency); real env wins."""
    env_file = LAB / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _resolve_video_id(arg: str) -> str:
    """Accept a BV id, a path containing data/raw/<id>, or a full URL."""
    from bilibili_crawler.utils.helpers import parse_input

    if (DATA_ROOT / "raw" / arg / "comments.jsonl").exists():
        return arg
    for child in (DATA_ROOT / "raw").glob("*/comments.jsonl"):
        if arg in child.parent.name or arg in child.read_text(encoding="utf-8")[:4000]:
            return child.parent.name
    parsed = parse_input(arg)
    if parsed and parsed.bvid:
        return parsed.bvid
    raise SystemExit(f"无法定位视频 ID: {arg}")


def cmd_crawl(args: argparse.Namespace) -> str:
    from bilicom.crawl import crawl

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = crawl(
        args.url,
        out_root=DATA_ROOT / "raw",
        cookie=args.cookie or os.environ.get("BILIBILI_COOKIE"),
        include_replies=not args.no_replies,
        max_pages=args.max_pages,
        mode=args.mode,
        limit=args.limit,
    )
    print(f"[crawl] {result.total} 条评论 -> {result.out_dir}")
    pinned = sum(1 for c in result.comments if c.get("is_pinned"))
    print(f"[crawl] 一级 {result.meta['root_comments']} / 楼中楼 {result.meta['reply_comments']} / 置顶 {pinned}")
    if result.meta.get("reply_warnings"):
        print(f"[crawl] 楼中楼警告 {len(result.meta['reply_warnings'])} 条（记录在 meta.json）")
    return result.video_id


def cmd_prepare(args: argparse.Namespace) -> str:
    from bilicom.normalize import normalize
    from bilicom.prepare import prepare

    video_id = _resolve_video_id(args.video)
    stats = normalize(DATA_ROOT / "raw" / video_id, video_id)
    print(f"[normalize] {stats}")
    manifest = prepare(DATA_ROOT / "normalized" / video_id, video_id, budget_tokens=args.budget)
    print(f"[prepare] {manifest['total_comments']} 条 -> {DATA_ROOT / 'agent' / video_id}")
    return video_id


def cmd_analyze(args: argparse.Namespace) -> str:
    from bilicom.normalize import normalize
    from bilicom.analyze import analyze

    video_id = _resolve_video_id(args.video)
    stats = normalize(DATA_ROOT / "raw" / video_id, video_id)
    print(f"[normalize] {stats}")
    result = analyze(video_id, DATA_ROOT, budget_tokens=args.budget, log=print)
    reports = DATA_ROOT / "reports" / video_id
    print(f"[analyze] 报告 -> {reports}")
    print(f"[analyze] llm_run={result.get('llm_run')} model={result.get('model', '-')}")
    return video_id


def cmd_all(args: argparse.Namespace) -> None:
    video_id = cmd_crawl(args)
    ns = argparse.Namespace(video=video_id, budget=args.budget)
    cmd_prepare(ns)
    cmd_analyze(ns)


def main(argv=None) -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="bili", description="Bilibili 评论提取/AI 提炼工具链")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common_crawl(p):
        p.add_argument("url", help="B 站视频/动态/文章 URL 或 BV 号")
        p.add_argument("--max-pages", type=int, default=100, help="一级评论最大页数，0=全量")
        p.add_argument("--mode", type=int, default=3, choices=[2, 3], help="2=热度 3=时间")
        p.add_argument("--no-replies", action="store_true", help="不抓楼中楼")
        p.add_argument("--limit", type=int, default=None, help="测试用：限制总条数")
        p.add_argument("--cookie", default=None, help="B 站 Cookie（缺省读 BILIBILI_COOKIE 环境变量）")

    p_crawl = sub.add_parser("crawl", help="抓取评论")
    common_crawl(p_crawl)

    p_all = sub.add_parser("all", help="抓取+清洗+打包+分析")
    common_crawl(p_all)
    p_all.add_argument("--budget", type=int, default=24000, help="每部分 token 预算")

    p_prep = sub.add_parser("prepare", help="normalize + Agent Pack")
    p_prep.add_argument("video", help="BV 号或 data/raw 下的目录名")
    p_prep.add_argument("--budget", type=int, default=24000)

    p_ana = sub.add_parser("analyze", help="normalize + map-reduce 分析")
    p_ana.add_argument("video", help="BV 号或 data/raw 下的目录名")
    p_ana.add_argument("--budget", type=int, default=None, help="每部分上下文预算（tokens）")

    args = parser.parse_args(argv)
    if args.cmd == "crawl":
        cmd_crawl(args)
    elif args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
