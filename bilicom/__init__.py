"""bilicom — Bilibili comment pipeline: crawl -> normalize -> agent pack -> AI analysis.

Layered on top of the vendored BilibiliCrawler package (MIT) which provides the
Bilibili API client (wbi signing) and comment crawling. This package adds:
pinned-comment capture, a normalized schema, agent-ready packs and a cached
map-reduce LLM analysis.
"""
__version__ = "1.0.0"
