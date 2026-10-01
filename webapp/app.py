"""bilicom-web — Web entry for the Bilibili comment lab.

Binds 127.0.0.1 by default; for public access put it behind a reverse proxy
(Caddy/Nginx) with TLS + basic auth. Reuses the bilicom pipeline functions
in-process: no shell, no user-controlled commands, no secrets returned to
clients.

Pages: 提取 (crawl/batch) → 分析 (lossless refine + AI report) →
数据集 (browse/filter/import/export) → 历史任务 → 设置 (AI provider).

Run: ./venv/bin/python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8765
(systemd unit: bilicom-web.service)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

LAB = Path(__file__).resolve().parent.parent
DATA_ROOT = LAB / "data"
STATIC = Path(__file__).resolve().parent / "static"
JOBS_FILE = DATA_ROOT / "webui" / "jobs.json"

MAX_PAGES = 100          # hard cap per job (matches CLI default)
MAX_LIMIT = 2000         # hard cap on total comments per job
MAX_EXPORT_BYTES = 80 * 1024 * 1024
MAX_IMPORT_BYTES = 60 * 1024 * 1024
SUBMIT_RATE = 10         # jobs per hour per IP
BATCH_MAX = 10           # videos per batch submission
REPORT_FILES = ["summary", "topics", "opinions", "questions", "suggestions",
                "representative-comments"]

log = logging.getLogger("bilicom-web")

os.chdir(LAB)
from bilicom.cli import load_env  # noqa: E402

load_env()

from bilicom import provider, refine, store  # noqa: E402

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
# report/comment JSON and CSV exports compress well; gzip level 1 keeps CPU flat
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_submit_times: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=64))
_busy = threading.Semaphore(1)          # crawl/analyze jobs, one at a time
_refine_busy = threading.Semaphore(1)   # refine runs, one at a time
_run_events: Dict[str, threading.Event] = {}

_BV_RE = re.compile(r"^BV[0-9A-Za-z]{8,12}$")
_VIDEO_RE = re.compile(
    r"^https?://(?:www\.|m\.)?bilibili\.com/video/(BV[0-9A-Za-z]{8,12})(?:[/?#]|$)")
_SPACE_RE = re.compile(r"^https?://space\.bilibili\.com/(\d+)(?:[/?#]|$)")
_MID_RE = re.compile(r"^\d{4,16}$")
_BILI_HOSTS = {"b23.tv", "www.bilibili.com", "m.bilibili.com", "bilibili.com"}
_STATUS_ZH = {
    "queued": "等待", "crawling": "抓取中", "preparing": "整理中",
    "analyzing": "分析中", "done": "完成", "failed": "失败",
    "cancelled": "已取消", "running": "运行中",
}
_RUN_STATUS_ZH = {
    "queued": "等待", "running": "运行中", "done": "完成", "failed": "失败",
    "cancelled": "已取消", "interrupted": "中断（可续跑）",
}


def _scrub(text: str) -> str:
    """Never leak cookie material into job errors shown in the UI."""
    return re.sub(r"(SESSDATA|bili_jct|DedeUserID|Authorization|Bearer)=[^;&\s]+",
                  r"\1=***", str(text))


def _db() -> Any:
    return store.connect(DATA_ROOT)


def _resolve_url(user_input: str) -> str:
    """Whitelist-only URL/BV resolution. SSRF-safe: only bilibili hosts,
    manual redirects, max 3 hops, final URL must be a /video/BV page."""
    text = (user_input or "").strip()
    if _BV_RE.fullmatch(text):
        return f"https://www.bilibili.com/video/{text}"
    if not re.fullmatch(r"https?://[A-Za-z0-9.\-/?#%&=_:~]+", text):
        raise ValueError("仅支持 Bilibili 视频 URL 或 BV 号")
    url = text
    for _ in range(3):
        host = re.match(r"https?://([^/]+)/?", url).group(1).lower()
        if host not in _BILI_HOSTS:
            raise ValueError("URL 域名不在白名单内（仅允许 bilibili.com / b23.tv）")
        m = _VIDEO_RE.match(url)
        if m:
            return f"https://www.bilibili.com/video/{m.group(1)}"
        if not host.endswith("b23.tv"):
            raise ValueError("无法从该 URL 解析出 BV 号")
        resp = requests.get(url, allow_redirects=False, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        loc = resp.headers.get("location", "")
        if not loc or resp.status_code not in (301, 302, 303, 307, 308):
            raise ValueError("短链接解析失败")
        url = requests.compat.urljoin(url, loc)
    raise ValueError("重定向次数过多")


def _save_jobs() -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(_jobs.values()), ensure_ascii=False), "utf-8")
    tmp.replace(JOBS_FILE)


def _load_jobs() -> None:
    if JOBS_FILE.exists():
        try:
            for job in json.loads(JOBS_FILE.read_text("utf-8")):
                if job.get("status") in ("queued", "crawling", "preparing",
                                         "analyzing"):
                    job["status"], job["error"] = "failed", "服务重启，任务中断"
                _jobs[job["id"]] = job
        except Exception:
            log.warning("jobs.json unreadable, starting fresh")
    # refine runs interrupted by a restart stay resumable, not failed
    try:
        with _db() as conn:
            conn.execute(
                "UPDATE runs SET status='interrupted' WHERE kind='refine'"
                " AND status IN ('queued','running')")
            conn.commit()
    except Exception:
        pass


def _set(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs[job_id].update(fields, updated_at=time.time())
        _save_jobs()


def _job_cancelled(job_id: str) -> bool:
    with _lock:
        return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def _run_job(job_id: str, url: str, include_replies: bool, max_pages: int) -> None:
    with _busy:
        try:
            if _job_cancelled(job_id):
                _set(job_id, status="cancelled")
                return
            _set(job_id, status="crawling")
            from bilicom.crawl import crawl

            result = crawl(
                url, out_root=DATA_ROOT / "raw",
                cookie=os.environ.get("BILIBILI_COOKIE"),
                include_replies=include_replies, max_pages=max_pages, mode=3,
            )
            bvid = result.video_id
            _set(job_id, bvid=bvid, total=result.total,
                 meta_root=result.meta.get("root_comments", 0),
                 meta_reply=result.meta.get("reply_comments", 0))
            _finish_pipeline(job_id, bvid)
        except Exception as exc:  # noqa: BLE001 — surfaced to the job row
            log.exception("job %s failed", job_id)
            with _lock:
                cancelled = _jobs.get(job_id, {}).get("cancel_requested")
            if cancelled:
                _set(job_id, status="cancelled")
            else:
                _set(job_id, status="failed", error=_scrub(exc)[:500])


def _finish_pipeline(job_id: str, bvid: str) -> None:
    """normalize → agent pack → AI report (map-reduce), shared by crawl and
    analyze-only jobs. Ingests the raw data into the lab.db index."""
    if _job_cancelled(job_id):
        _set(job_id, status="cancelled")
        return
    _set(job_id, status="preparing")
    from bilicom.normalize import normalize
    from bilicom.prepare import prepare

    stats = normalize(DATA_ROOT / "raw" / bvid, bvid)
    prepare(DATA_ROOT / "normalized" / bvid, bvid, budget_tokens=24000)
    _set(job_id, normalized=stats.get("normalized"))
    with _db() as conn:
        store.ingest_video(conn, DATA_ROOT / "raw" / bvid, bvid)

    if _job_cancelled(job_id):
        _set(job_id, status="cancelled")
        return
    _set(job_id, status="analyzing")
    from bilicom.analyze import analyze

    outcome = analyze(bvid, DATA_ROOT, budget_tokens=24000, log=lambda *_: None)
    _set(job_id, status="done",
         llm_run=bool(outcome.get("llm_run")),
         model=outcome.get("model") or "")


def _run_analyze_only(job_id: str, bvid: str) -> None:
    """AI 综合报告 for an already-crawled dataset (重新分析, no recrawl)."""
    with _busy:
        try:
            if not (DATA_ROOT / "raw" / bvid / "comments.jsonl").exists():
                raise RuntimeError(f"本地没有 {bvid} 的原始数据")
            _finish_pipeline(job_id, bvid)
        except Exception as exc:  # noqa: BLE001
            log.exception("analyze job %s failed", job_id)
            _set(job_id, status="failed", error=_scrub(exc)[:500])


def _run_refine(run_id: str, bvid: str, params: refine.RefineParams) -> None:
    with _refine_busy:
        event = _run_events.setdefault(run_id, threading.Event())
        conn = _db()

        def cancel_fn() -> bool:
            if event.is_set():
                return True
            row = conn.execute(
                "SELECT cancel_requested FROM runs WHERE id=?",
                (run_id,)).fetchone()
            return bool(row and row["cancel_requested"])

        def log_sink(message: str) -> None:
            log.info("refine %s: %s", run_id, message)
            store.update_run(conn, run_id, last_log=str(message)[:200])

        try:
            refine.run_refine(conn, DATA_ROOT, run_id, bvid, params,
                              log=log_sink, cancel_fn=cancel_fn,
                              progress=lambda p, s: store.update_run(
                                  conn, run_id, progress=round(p, 3), stage=s))
        except refine.RefineCancelled:
            store.update_run(conn, run_id, status="cancelled", stage="",
                             error="用户取消")
        except Exception as exc:  # noqa: BLE001
            log.exception("refine run %s failed", run_id)
            store.update_run(conn, run_id, status="failed",
                             error=_scrub(exc)[:500])
        finally:
            _run_events.pop(run_id, None)
            conn.close()


# ---- jobs API -----------------------------------------------------------

def _job_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: job.get(k) for k in
           ("id", "kind", "input", "bvid", "status", "error", "created_at",
            "updated_at", "total", "normalized", "llm_run", "model",
            "meta_root", "meta_reply", "batch_id")}
    out.setdefault("kind", "crawl")
    out["status_zh"] = _STATUS_ZH.get(job.get("status"), job.get("status"))
    return out


def _public(job_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return _job_payload(job)


def _new_job(user_input: str, url: str, kind: str = "crawl",
             batch_id: str = "") -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = {"id": job_id, "kind": kind, "input": user_input[:200], "url": url,
           "bvid": re.search(r"(BV[0-9A-Za-z]{8,12})", url).group(1)
           if kind == "analyze" and re.search(r"(BV[0-9A-Za-z]{8,12})", url)
           else None,
           "status": "queued", "error": None, "created_at": now,
           "updated_at": now, "cancel_requested": False,
           "batch_id": batch_id}
    with _lock:
        _jobs[job_id] = job
        _save_jobs()
    return job


def _submit_budget(ip: str, n: int = 1) -> None:
    now = time.time()
    times = _submit_times[ip]
    while times and now - times[0] > 3600:
        times.popleft()
    if len(times) + n > SUBMIT_RATE:
        raise HTTPException(429, "提交过于频繁，每小时最多 %d 个任务" % SUBMIT_RATE)


@app.post("/api/jobs")
async def create_job(request: Request) -> JSONResponse:
    ip = request.client.host if request.client else "?"
    _submit_budget(ip)
    if _busy._value == 0:
        raise HTTPException(409, "已有任务在运行，请等待完成后再提交")
    body = await request.json()
    try:
        url = _resolve_url(str(body.get("url", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    include_replies = bool(body.get("include_replies", True))
    try:
        max_pages = int(body.get("max_pages", 20))
    except (TypeError, ValueError):
        max_pages = 20
    max_pages = max(1, min(max_pages, MAX_PAGES))

    job = _new_job(str(body.get("url", "")), url)
    _submit_times[ip].append(time.time())
    threading.Thread(target=_run_job, name=f"job-{job['id']}",
                     args=(job["id"], url, include_replies, max_pages),
                     daemon=True).start()
    return JSONResponse(_public(job["id"]), status_code=202)


@app.post("/api/jobs/batch")
async def create_batch(request: Request) -> JSONResponse:
    """批量提交：多个 BV/URL，一行的 space.bilibili.com/<mid> 或纯数字 mid
    会被展开为该 UP 主最近视频（最多 BATCH_MAX 个）。"""
    ip = request.client.host if request.client else "?"
    body = await request.json()
    lines = [ln.strip() for ln in re.split(r"[\s,;]+", str(body.get("urls", "")))
             if ln.strip()]
    if not lines:
        raise HTTPException(400, "未提供任何 URL/BV 号")
    include_replies = bool(body.get("include_replies", True))
    try:
        max_pages = max(1, min(int(body.get("max_pages", 20)), MAX_PAGES))
    except (TypeError, ValueError):
        max_pages = 20

    targets: List[str] = []      # resolved video URLs
    uploader_errors: List[str] = []
    for line in lines[:20]:
        m = _SPACE_RE.match(line)
        if not m and _MID_RE.match(line):
            m = None
            try:
                from bilicom.crawl import list_uploader_videos

                vids = list_uploader_videos(int(line), BATCH_MAX,
                                            os.environ.get("BILIBILI_COOKIE"))
                targets += [f"https://www.bilibili.com/video/{v['bvid']}"
                            for v in vids if v.get("bvid")]
                continue
            except Exception as exc:  # noqa: BLE001
                uploader_errors.append(f"UP主 {line}: {_scrub(exc)[:120]}")
                continue
        if m:
            try:
                from bilicom.crawl import list_uploader_videos

                vids = list_uploader_videos(int(m.group(1)), BATCH_MAX,
                                            os.environ.get("BILIBILI_COOKIE"))
                targets += [f"https://www.bilibili.com/video/{v['bvid']}"
                            for v in vids if v.get("bvid")]
                continue
            except Exception as exc:  # noqa: BLE001
                uploader_errors.append(f"UP主 {m.group(1)}: {_scrub(exc)[:120]}")
                continue
        try:
            targets.append(_resolve_url(line))
        except ValueError as exc:
            uploader_errors.append(f"{line[:40]}: {exc}")
    targets = list(dict.fromkeys(targets))[:BATCH_MAX]
    if not targets:
        raise HTTPException(400, "没有可用的视频目标：" +
                            "；".join(uploader_errors)[:300])

    _submit_budget(ip, len(targets))
    batch_id = uuid.uuid4().hex[:8]
    created = []
    for url in targets:
        job = _new_job(url, url, batch_id=batch_id)
        _submit_times[ip].append(time.time())
        threading.Thread(target=_run_job, name=f"job-{job['id']}",
                         args=(job["id"], url, include_replies, max_pages),
                         daemon=True).start()
        created.append(job["id"])
    return JSONResponse({"jobs": created, "batch_id": batch_id,
                         "warnings": uploader_errors}, status_code=202)


@app.get("/api/jobs")
async def list_jobs() -> List[Dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: -j["created_at"])
    return [_job_payload(j) for j in jobs[:50]]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    return _public(job_id)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    _public(job_id)
    _set(job_id, cancel_requested=True)
    with _lock:
        status = _jobs[job_id]["status"]
    if status == "queued":
        _set(job_id, status="cancelled")
    return {"ok": True, "status": status}


@app.post("/api/analyze/{bvid}")
async def analyze_existing(bvid: str, request: Request) -> JSONResponse:
    """对已抓取的数据集重新跑 AI 综合报告（模式③，不重爬）。"""
    ip = request.client.host if request.client else "?"
    _submit_budget(ip)
    if _busy._value == 0:
        raise HTTPException(409, "已有任务在运行，请等待完成后再提交")
    if not _BV_RE.fullmatch(bvid):
        raise HTTPException(400, "BV 号不合法")
    job = _new_job(bvid, f"analyze:{bvid}", kind="analyze")
    threading.Thread(target=_run_analyze_only, name=f"job-{job['id']}",
                     args=(job["id"], bvid), daemon=True).start()
    return JSONResponse(_public(job["id"]), status_code=202)


# ---- datasets API -------------------------------------------------------

@app.get("/api/datasets")
async def datasets() -> List[Dict[str, Any]]:
    with _db() as conn:
        return store.list_videos(conn)


@app.post("/api/datasets/ingest")
async def ingest_datasets() -> Dict[str, Any]:
    """(Re)index every raw dataset into lab.db — also fixes a DB that predates
    a manually dropped file."""
    with _db() as conn:
        return {"ingested": store.ingest_all(conn, DATA_ROOT)}


@app.get("/api/datasets/{bvid}/comments")
async def dataset_comments(bvid: str, level: int = 0, min_likes: int = 0,
                           min_replies: int = 0, q: str = "", start: str = "",
                           end: str = "", pinned: int = -1, sort: str = "likes",
                           order: str = "desc", page: int = 1,
                           page_size: int = 50) -> Dict[str, Any]:
    with _db() as conn:
        if not store.get_video(conn, bvid):
            raise HTTPException(404, "数据集不存在")
        return store.query_comments(conn, bvid, level=level,
                                    min_likes=min_likes, min_replies=min_replies,
                                    q=q[:100], start=start[:10], end=end[:10],
                                    pinned=pinned, sort=sort, order=order,
                                    page=page, page_size=page_size)


_IMPORT_ALIASES = {
    "comment_id": ("comment_id", "rpid", "id", "ID"),
    "content": ("content", "message", "text", "comment", "评论", "内容"),
    "like_count": ("like_count", "like", "likes", "点赞"),
    "reply_count": ("reply_count", "rcount", "replies", "回复数"),
    "ctime": ("ctime", "created_at", "time", "时间"),
    "created_at": ("created_at", "ctime_text", "时间"),
    "username": ("username", "uname", "name", "用户名", "昵称"),
    "mid": ("mid", "user_id", "uid", "用户ID"),
    "root_id": ("root_id", "root", "root_rpid"),
    "parent_id": ("parent_id", "parent", "parent_rpid"),
    "is_reply": ("is_reply", "reply", "level"),
    "is_pinned": ("is_pinned", "pinned", "置顶"),
}


def _map_import_row(row: Dict[str, Any], seq: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for field, names in _IMPORT_ALIASES.items():
        for n in names:
            key = n.lower()
            if key in lower and lower[key] not in ("", None):
                out[field] = lower[key]
                break
    try:
        out["comment_id"] = int(out.get("comment_id") or -seq)
    except (TypeError, ValueError):
        out["comment_id"] = -seq
    content = str(out.get("content") or "")
    level = out.get("is_reply")
    # "level" semantics: 1=root, 2=楼中楼; boolean aliases stay as-is
    if isinstance(level, str):
        is_reply = level.strip() == "2"
    else:
        is_reply = bool(level)
    ctime = out.get("ctime")
    try:
        if isinstance(ctime, str) and "-" in ctime:
            ctime = int(time.mktime(time.strptime(
                ctime.strip()[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")))
        else:
            ctime = int(float(ctime or 0)) or None
    except (TypeError, ValueError):
        ctime = None
    return {"comment_id": out["comment_id"],
            "root_id": out.get("root_id") or (out["comment_id"] if not is_reply else None),
            "parent_id": out.get("parent_id") or out.get("root_id"),
            "is_reply": is_reply, "is_pinned": bool(out.get("is_pinned")),
            "content": content,
            "like_count": int(out.get("like_count") or 0),
            "reply_count": int(out.get("reply_count") or 0),
            "ctime": ctime, "created_at": str(out.get("created_at") or ""),
            "username": str(out.get("username") or ""),
            "mid": str(out.get("mid") or "")}


@app.post("/api/datasets/import")
async def import_dataset(request: Request, name: str = "") -> Dict[str, Any]:
    """Import an existing JSONL / JSON / CSV comment dump as a new dataset.

    The body is the raw file; format is sniffed. Column names are mapped
    flexibly (comment_id/rpid/id, content/message/text, ...). Missing ids get
    stable negative ids. The import lands in data/raw/<import-…>/ exactly like
    a crawl, so every later step (normalize/refine/export) works on it.
    """
    body = await request.body()
    if not body:
        raise HTTPException(400, "空文件")
    if len(body) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "文件过大（>60MB）")
    text = body.decode("utf-8-sig", errors="replace")
    rows: List[Dict[str, Any]] = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            rows = [r for r in json.loads(text) if isinstance(r, dict)]
        except json.JSONDecodeError as err:
            raise HTTPException(400, f"JSON 解析失败：{err}") from None
    elif stripped.startswith("{"):
        # either JSONL or a single object / {"comments": [...]}
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise HTTPException(400, "JSONL 行解析失败") from None
            if isinstance(obj, dict):
                if isinstance(obj.get("comments"), list):
                    rows += [r for r in obj["comments"] if isinstance(r, dict)]
                else:
                    rows.append(obj)
    else:
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(r) for r in reader]
    if not rows:
        raise HTTPException(400, "未解析出任何评论行")
    if len(rows) > MAX_LIMIT * 50:
        raise HTTPException(413, "评论行数过多（>100k）")

    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", name.strip())[:60]
    video_id = safe_name or time.strftime("import-%Y%m%d-%H%M%S")
    raw_dir = DATA_ROOT / "raw" / video_id
    n = 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "comments.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            rec = _map_import_row(row, i)
            if not rec["content"]:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    meta = {"video_id": video_id, "source_url": "import", "source": "import",
            "title": name or video_id, "total_comments": n,
            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "upstream": "webapp import"}
    (raw_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    from bilicom.normalize import normalize
    from bilicom.prepare import prepare

    stats = normalize(raw_dir, video_id)
    prepare(DATA_ROOT / "normalized" / video_id, video_id, budget_tokens=24000)
    with _db() as conn:
        ingest = store.ingest_video(conn, raw_dir, video_id, source="import")
    return {"video_id": video_id, "imported": n, "normalize": stats,
            "ingested": ingest.get("ingested")}


# ---- refine API ---------------------------------------------------------

@app.get("/api/refine/preview")
async def refine_preview(bvid: str, sample: int = 100,
                         near_dup_threshold: float = 0.82) -> Dict[str, Any]:
    if bvid != "all" and not _BV_RE.fullmatch(bvid):
        raise HTTPException(400, "BV 号不合法")
    try:
        with _db() as conn:
            return refine.preview(conn, bvid, sample=sample,
                                  near_dup_threshold=near_dup_threshold)
    except RuntimeError as err:
        raise HTTPException(409, str(err)) from None


@app.post("/api/refine/runs")
async def refine_start(request: Request) -> JSONResponse:
    body = await request.json()
    bvid = str(body.get("bvid", "")).strip()
    if bvid != "all" and not _BV_RE.fullmatch(bvid):
        raise HTTPException(400, "BV 号不合法（或填 all 处理全部数据集）")
    with _db() as conn:
        if bvid != "all" and not store.get_video(conn, bvid):
            raise HTTPException(404, "数据集不存在（先提取或导入）")
        if store.active_run(conn):
            raise HTTPException(409, "已有精炼任务在运行")
    params = refine.RefineParams(
        near_dup_threshold=float(body.get("near_dup_threshold", 0.82)),
        cluster_threshold=float(body.get("cluster_threshold", 0.86)),
        top_n_representatives=int(body.get("top_n_representatives", 3)),
        ai_labels=bool(body.get("ai_labels", False)),
        model=(str(body.get("model", "")).strip() or None),
        embed_model=(str(body.get("embed_model", "")).strip() or None),
        batch_size=int(body.get("batch_size", 64)),
    )
    run_id = uuid.uuid4().hex[:12]
    with _db() as conn:
        store.create_run(conn, run_id, bvid, "refine",
                         "lossless_ai" if params.ai_labels else "lossless",
                         vars(params), model=params.model or "",
                         embed_model=params.embed_model or "")
    threading.Thread(target=_run_refine, name=f"refine-{run_id}",
                     args=(run_id, bvid, params), daemon=True).start()
    return JSONResponse({"run_id": run_id}, status_code=202)


@app.get("/api/refine/runs")
async def refine_runs(bvid: str = "") -> List[Dict[str, Any]]:
    with _db() as conn:
        runs = store.list_runs(conn, "refine")
    if bvid:
        runs = [r for r in runs if r.get("video_id") == bvid]
    for r in runs:
        r["status_zh"] = _RUN_STATUS_ZH.get(r["status"], r["status"])
    return runs


@app.get("/api/refine/runs/{run_id}")
async def refine_run_detail(run_id: str) -> Dict[str, Any]:
    with _db() as conn:
        run = store.get_run(conn, run_id)
        if not run:
            raise HTTPException(404, "运行不存在")
        clusters = store.run_clusters(conn, run_id)
        cfg = provider.load_settings(DATA_ROOT)
        cost = store.run_cost(conn, run_id,
                              float(cfg.get("prompt_price") or 0),
                              float(cfg.get("completion_price") or 0))
    run["status_zh"] = _RUN_STATUS_ZH.get(run["status"], run["status"])
    return {"run": run, "clusters": clusters[:200], "cluster_total": len(clusters),
            "cost": cost}


@app.get("/api/refine/runs/{run_id}/comments")
async def refine_run_comments(run_id: str, cluster_id: int = -1,
                              rep_only: int = 0, page: int = 1,
                              page_size: int = 50) -> Dict[str, Any]:
    with _db() as conn:
        rows = store.run_cluster_members(conn, run_id,
                                         None if cluster_id < 0 else cluster_id,
                                         bool(rep_only))
    total = len(rows)
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    return {"total": total, "page": page, "page_size": page_size,
            "rows": rows[(page - 1) * page_size: page * page_size]}


@app.get("/api/refine/runs/{run_id}/duplicates")
async def refine_run_duplicates(run_id: str, comment_id: int) -> List[Dict[str, Any]]:
    with _db() as conn:
        return store.run_duplicates_of(conn, run_id, comment_id)


@app.post("/api/refine/runs/{run_id}/cancel")
async def refine_cancel(run_id: str) -> Dict[str, Any]:
    with _db() as conn:
        run = store.get_run(conn, run_id)
        if not run:
            raise HTTPException(404, "运行不存在")
        store.update_run(conn, run_id, cancel_requested=1)
    _run_events.setdefault(run_id, threading.Event()).set()
    return {"ok": True}


@app.post("/api/refine/runs/{run_id}/resume")
async def refine_resume(run_id: str) -> JSONResponse:
    with _db() as conn:
        run = store.get_run(conn, run_id)
        if not run:
            raise HTTPException(404, "运行不存在")
        if run["status"] not in ("failed", "cancelled", "interrupted"):
            raise HTTPException(409, "该运行不可续跑（状态：%s）" % run["status"])
        if store.active_run(conn):
            raise HTTPException(409, "已有精炼任务在运行")
        params = refine.RefineParams(**{
            k: v for k, v in (run.get("params") or {}).items()
            if k in refine.RefineParams.__dataclass_fields__})
        store.update_run(conn, run_id, status="queued", error=None,
                         cancel_requested=0)
    _run_events.pop(run_id, None)
    threading.Thread(target=_run_refine, name=f"refine-{run_id}",
                     args=(run_id, run["video_id"], params),
                     daemon=True).start()
    return JSONResponse({"ok": True, "run_id": run_id}, status_code=202)


@app.post("/api/refine/runs/{run_id}/rerun")
async def refine_rerun(run_id: str) -> JSONResponse:
    with _db() as conn:
        run = store.get_run(conn, run_id)
        if not run:
            raise HTTPException(404, "运行不存在")
        if store.active_run(conn):
            raise HTTPException(409, "已有精炼任务在运行")
        params = refine.RefineParams(**{
            k: v for k, v in (run.get("params") or {}).items()
            if k in refine.RefineParams.__dataclass_fields__})
        new_id = uuid.uuid4().hex[:12]
        store.create_run(conn, new_id, run["video_id"], "refine", run["mode"],
                         vars(params), model=params.model or "",
                         embed_model=params.embed_model or "")
    threading.Thread(target=_run_refine, name=f"refine-{new_id}",
                     args=(new_id, run["video_id"], params),
                     daemon=True).start()
    return JSONResponse({"run_id": new_id}, status_code=202)


@app.get("/api/refine/latest")
async def refine_latest(bvid: str) -> Dict[str, Any]:
    with _db() as conn:
        for run in store.list_runs(conn, "refine"):
            if run["video_id"] == bvid and run["status"] == "done":
                return {"run_id": run["id"]}
    raise HTTPException(404, "该数据集暂无精炼结果")


# ---- AI provider settings API ------------------------------------------

@app.get("/api/settings/ai")
async def settings_get() -> Dict[str, Any]:
    return provider.public_settings(provider.load_settings(DATA_ROOT))


@app.post("/api/settings/ai")
async def settings_set(request: Request) -> Dict[str, Any]:
    body = await request.json()
    cfg = provider.save_settings(DATA_ROOT, body)
    return provider.public_settings(cfg)


def _client_for_test(body: Dict[str, Any], purpose: str = "chat") -> Any:
    """Build a one-off client from request overrides falling back to stored
    settings. Masked/empty key means 'use stored key'."""
    cfg = provider.load_settings(DATA_ROOT)
    base_url = str(body.get("base_url") or cfg["base_url"]).strip().rstrip("/")
    model = str(body.get("model") or "").strip() or (
        cfg.get("embed_model") if purpose == "embed" else "") or cfg["model"]
    api_key = str(body.get("api_key") or "").strip()
    if not api_key or api_key.startswith("sk-••••") or api_key.startswith("••••"):
        api_key = cfg["api_key"]
    if not base_url:
        raise ValueError("未配置 API 地址")
    if not model:
        raise ValueError("未配置模型名")
    if not api_key and not provider.is_local(base_url):
        raise ValueError("该 API 地址需要 API Key（本地 Ollama 等可留空）")
    from bilicom.llm import LLMClient

    return LLMClient(base_url=base_url, api_key=api_key, model=model,
                     context_length=int(cfg.get("context_length") or 32000))


@app.post("/api/settings/ai/test")
async def settings_test(request: Request) -> Dict[str, Any]:
    body = await request.json()
    try:
        client = _client_for_test(body)
    except ValueError as err:
        return {"ok": False, "error": str(err)}
    started = time.monotonic()
    try:
        reply = client.chat("You are a connection test. Reply with the single word: pong",
                            "ping", temperature=0.0, retries=1, kind="test")
        latency = int((time.monotonic() - started) * 1000)
        return {"ok": True, "model": client.model, "latency_ms": latency,
                "reply": (reply or "").strip()[:40]}
    except Exception as exc:  # noqa: BLE001 — error text must not leak the key
        return {"ok": False, "error": _scrub(str(exc))[:300]}


@app.get("/api/settings/ai/models")
async def settings_models() -> Dict[str, Any]:
    try:
        client = _client_for_test({})
        models = client.list_models(timeout=20)
        return {"ok": True, "models": models}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "models": [], "error": _scrub(str(exc))[:300]}


# ---- exports ------------------------------------------------------------

def _file_guard(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "文件不存在")
    if path.stat().st_size > MAX_EXPORT_BYTES:
        raise HTTPException(413, "文件过大")
    return path


def _zip_dir(path: Path, bvid: str) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.stat().st_size <= MAX_EXPORT_BYTES:
                zf.write(f, f.relative_to(path.parent))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{bvid}-export.zip"'})


def _latest_refine_run(conn, bvid: str) -> Optional[Dict[str, Any]]:
    for run in store.list_runs(conn, "refine"):
        if run["video_id"] in (bvid, "all") and run["status"] == "done":
            return run
    return None


def _strip_report_header(text: str) -> str:
    """Section files written by older analyze versions each carry the same
    report header; drop it so a bundle concatenation shows it once."""
    lines = text.splitlines()
    out: List[str] = []
    skipping = lines and lines[0].startswith("# 分析报告 —")
    for i, line in enumerate(lines):
        if skipping and (line.startswith("# 分析报告 —")
                         or line.startswith("- 生成范围：")
                         or line.startswith("- 模型：")
                         or line.startswith("- 方式：")):
            continue
        if skipping and not out and line.strip() == "":
            continue
        out.append(line)
    return "\n".join(out).strip()


def _refined_rows(conn, run_id: str) -> List[Dict[str, Any]]:
    return store.run_cluster_members(conn, run_id, None, False)


def _rows_to_csv(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])
    return out.getvalue()


REFINED_FIELDS = ["video_id", "comment_id", "root_id", "parent_id",
                  "cluster_id", "cluster_label", "is_representative",
                  "duplicate_of", "selection_reason", "quality_score",
                  "content", "like_count", "reply_count", "level",
                  "created_at", "is_pinned"]

# at-replies inside content leak nicknames even in privacy mode
_AT_NICK = re.compile(r"回复\s*@\S{1,40}\s*([:：])")


def _scrub_names_in_text(text: str) -> str:
    """Privacy mode: replace @nicknames in reply prefixes with a placeholder.
    Thread relations stay traceable via root_id/parent_id columns."""
    return _AT_NICK.sub(r"回复 @匿名\1 ", str(text or ""))


def _privacy_scrub(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: ("" if k in ("username", "mid", "ip_location") else v)
           for k, v in row.items()}
    if out.get("content"):
        out["content"] = _scrub_names_in_text(out["content"])
    return out


def _refined_markdown(bvid: str, title: str, run: Dict[str, Any],
                      clusters: List[Dict[str, Any]],
                      rows: List[Dict[str, Any]]) -> str:
    stats = run.get("stats") or {}
    algo = "本地 TF-IDF 向量" if stats.get("vector_source") != "api" else \
        f"AI 嵌入（{stats.get('embed_model') or run.get('embed_model') or ''}）"
    lines = [f"# 精炼评论 — {title}（{bvid}）", "",
             f"- 运行：`{run['id']}` · 模式：{run.get('mode')}"
             f" · 模型：{run.get('model') or '未使用（纯本地算法）'}"
             f" · 向量：{algo}", ""]
    if stats:
        lines += [f"- 原始 {stats.get('original')} → 去重后 {stats.get('survivors')}"
                  f"（完全重复 -{stats.get('exact_removed', 0)}，"
                  f"近重复折叠 -{stats.get('near_removed', 0)}）"
                  f" → 聚类 {stats.get('clusters')} → 代表 {stats.get('representatives')}"
                  f"（压缩率 {stats.get('compression', 0):.0%}）",
                  f"- quality_score：0–1 综合分（55% 内容信号 + 45% 对数化点赞），"
                  f"每行必有值；{stats.get('emote_only', 0)} 条纯表情/打卡归入独立主题", ""]
    by_cluster: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cluster.setdefault(r.get("cluster_id"), []).append(r)
    for c in clusters:
        members = by_cluster.get(c["cluster_id"], [])
        reps = [m for m in members if m.get("is_representative")]
        lines.append(f"## {c['label']}（{c['size']} 条）")
        if c.get("keywords"):
            lines.append(f"关键词：{'、'.join(c['keywords'])}")
        lines.append("")
        for m in reps or members[:3]:
            lines.append(f"> {m['content']}".replace("\n", "\n> "))
            lines.append(f"— comment_id {m['comment_id']}，赞 {m['like_count']}"
                         f"（{m['created_at']}）")
            lines.append("")
        folded = c["size"] - len(reps or members[:3])
        if folded > 0:
            lines.append(f"*簇内其余 {folded} 条同主题评论已折叠，"
                         f"完整原文见 refined/comments.jsonl（按 cluster_id 筛选）。*")
    return "\n".join(lines)


@app.get("/api/export/refine/{run_id}/{fmt}")
async def export_refined(run_id: str, fmt: str) -> Response:
    """精炼评论导出（与原始评论完全分开）。每行都带 comment_id 可追溯。"""
    with _db() as conn:
        run = store.get_run(conn, run_id)
        if not run:
            raise HTTPException(404, "运行不存在")
        clusters = store.run_clusters(conn, run_id)
        rows = _refined_rows(conn, run_id)
    label_of = {c["cluster_id"]: c["label"] for c in clusters}
    for r in rows:
        r["cluster_label"] = label_of.get(r.get("cluster_id"), "")
    fname = f"refined-{run['video_id']}-{run_id}"
    if fmt == "jsonl":
        text = "\n".join(json.dumps(
            {k: r.get(k) for k in REFINED_FIELDS}, ensure_ascii=False)
            for r in rows)
        return Response(text, media_type="application/x-ndjson",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{fname}.jsonl"'})
    if fmt == "csv":
        csv_text = _rows_to_csv(rows, REFINED_FIELDS)
        return Response("\ufeff" + csv_text,  # BOM so Excel reads UTF-8
                        media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{fname}.csv"'})
    if fmt == "md":
        with _db() as conn:
            video = store.get_video(conn, run["video_id"]) if run["video_id"] != "all" else None
        title = (video["title"] if video else run["video_id"]) or run["video_id"]
        md = _refined_markdown(run["video_id"], title, run, clusters, rows)
        return Response(md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{fname}.md"'})
    raise HTTPException(400, "未知导出格式")


@app.get("/api/export/bundle/{bvid}")
async def export_bundle(bvid: str, privacy: int = 1) -> Response:
    """「导出全部数据」：metadata + raw(jsonl/csv/json) + refined + analysis
    + manifest 的单个 ZIP。privacy=1 时隐藏用户名/UID/属地（分享数据集用）。"""
    if bvid != "all" and not _BV_RE.fullmatch(bvid):
        raise HTTPException(400, "BV 号不合法")
    targets = [bvid]
    with _db() as conn:
        if bvid == "all":
            targets = [v["bvid"] for v in store.list_videos(conn)]
        elif not store.get_video(conn, bvid):
            raise HTTPException(404, "数据集不存在")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest: Dict[str, Any] = {"generated_at": datetime.now(
            timezone.utc).isoformat(), "privacy": bool(privacy),
            "videos": {}}
        for vid in targets:
            raw_dir = DATA_ROOT / "raw" / vid
            raw_file = raw_dir / "comments.jsonl"
            if not raw_file.exists():
                continue
            meta = _read_json(raw_dir / "meta.json") or {}
            raw_rows: List[Dict[str, Any]] = []
            for line in raw_file.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if privacy:
                raw_rows = [_privacy_scrub(r) for r in raw_rows]
            folder = vid if len(targets) > 1 else ""
            def add(name: str, data: Any, raw_bytes: bool = False) -> None:
                arc = "/".join(p for p in (folder, name) if p)
                zf.writestr(arc, data if not raw_bytes else data)

            add("metadata.json", json.dumps({
                "video_id": vid, "source_url": meta.get("source_url"),
                "title": (meta.get("target_info") or {}).get("title") or vid,
                "owner": (meta.get("target_info") or {}).get("owner"),
                "crawled_at": meta.get("crawled_at"),
                "counts": {"total": len(raw_rows),
                           "root": meta.get("root_comments"),
                           "reply": meta.get("reply_comments"),
                           "pinned": meta.get("pinned_comments")},
                "pipeline": "bili-comment-lab",
            }, ensure_ascii=False, indent=2))
            add("raw/comments.jsonl",
                "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows))
            fields = ["comment_id", "root_id", "parent_id", "is_reply",
                      "is_pinned", "content", "like_count", "reply_count",
                      "ctime", "username", "mid", "ip_location"]
            if privacy:
                fields = [f for f in fields
                          if f not in ("username", "mid", "ip_location")]
            add("raw/comments.csv", "\ufeff" + _rows_to_csv(raw_rows, fields))
            add("raw/comments.json",
                json.dumps(raw_rows, ensure_ascii=False, indent=1))
            # refined layer from the latest done run covering this video
            with _db() as conn:
                run = _latest_refine_run(conn, vid)
                db_count = conn.execute(
                    "SELECT COUNT(*) FROM comments WHERE video_id=?",
                    (vid,)).fetchone()[0]
                if run:
                    clusters = store.run_clusters(conn, run["id"])
                    rows = _refined_rows(conn, run["id"])
                else:
                    clusters, rows = [], []
            label_of = {c["cluster_id"]: c["label"] for c in clusters}
            for r in rows:
                r["cluster_label"] = label_of.get(r.get("cluster_id"), "")
            warnings: List[str] = []
            if db_count != len(raw_rows):
                warnings.append(
                    f"数据库索引 {db_count} 条与 raw 文件 {len(raw_rows)} 条不一致，"
                    f"请在数据集页重新索引（ingest）后再精炼/导出")
            if run:
                rstats = run.get("stats") or {}
                if rstats.get("original") not in (None, len(raw_rows)):
                    warnings.append(
                        f"精炼运行 {run['id']} 基于旧快照（{rstats.get('original')} 条，"
                        f"当前 {len(raw_rows)} 条），建议重新精炼")
            if run:
                add("refined/comments.jsonl", "\n".join(json.dumps(
                    {k: r.get(k) for k in REFINED_FIELDS}, ensure_ascii=False)
                    for r in rows))
                add("refined/comments.csv", "\ufeff" + _rows_to_csv(
                    rows, REFINED_FIELDS))
                with _db() as conn:
                    video = store.get_video(conn, vid) if vid != "all" else None
                title = (video["title"] if video else vid) or vid
                add("refined/comments.md",
                    _refined_markdown(vid, title, run, clusters, rows))
                add("analysis/topics.json", json.dumps(
                    [{"cluster_id": c["cluster_id"], "label": c["label"],
                      "size": c["size"], "keywords": c["keywords"],
                      "representative_ids": c["representative_ids"]}
                     for c in clusters], ensure_ascii=False, indent=1))
                add("analysis/keywords.json", json.dumps(
                    {str(c["cluster_id"]): c["keywords"] for c in clusters},
                    ensure_ascii=False, indent=1))
                add("analysis/clusters.json", json.dumps(rows, ensure_ascii=False))
            # merged AI report: single header (report.md is written merged;
            # legacy per-section files get their repeated headers stripped)
            report_md = DATA_ROOT / "reports" / vid / "report.md"
            if report_md.exists():
                add("analysis/ai_report.md", report_md.read_text("utf-8"))
            elif (DATA_ROOT / "reports" / vid / "summary.md").exists():
                parts = [_strip_report_header(p.read_text("utf-8"))
                         for name in REPORT_FILES
                         for p in [DATA_ROOT / "reports" / vid / f"{name}.md"]
                         if p.exists()]
                add("analysis/ai_report.md", "\n\n".join(parts))
            qa_file = DATA_ROOT / "reports" / vid / "qa.json"
            if qa_file.exists():
                add("analysis/qa_report.json", qa_file.read_text("utf-8"))
            manifest["videos"][vid] = {
                "comments": len(raw_rows),
                "refined_run": run["id"] if run else None,
                "refined_stats": (run or {}).get("stats") or {},
                "warnings": warnings,
            }
        add("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    fname = f"{bvid}_export_{time.strftime('%Y-%m-%d')}.zip"
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{fname}"'})


# ---- uploader (UP 主批量) ------------------------------------------------

@app.get("/api/uploader")
async def uploader(mid: str = "") -> Dict[str, Any]:
    if not _MID_RE.fullmatch(mid.strip()):
        raise HTTPException(400, "需要 UP 主的纯数字 mid（space.bilibili.com/<mid>）")
    try:
        from bilicom.crawl import list_uploader_videos

        vids = list_uploader_videos(int(mid), 30,
                                    os.environ.get("BILIBILI_COOKIE"))
        return {"ok": True, "videos": vids}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "videos": [], "error": _scrub(str(exc))[:300]}


# ---- legacy job result / exports ----------------------------------------

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str) -> Dict[str, Any]:
    job = _public(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "任务尚未完成")
    bvid = job["bvid"]
    meta = _read_json(DATA_ROOT / "raw" / bvid / "meta.json") or {}
    target = meta.get("target_info") or {}
    stats = _read_json(DATA_ROOT / "normalized" / bvid / "stats.json") or {}
    rstats = _read_json(DATA_ROOT / "reports" / bvid / "statistics.json") or {}
    analysis = _read_json(DATA_ROOT / "agent" / bvid / "analysis.json") or {}
    # reconciled lexicon sentiment over ALL comments (sums to the total);
    # fall back to counting stance groups only for pre-fix datasets
    sentiment = rstats.get("sentiment") or {
        k: len(v) for k, v in (analysis.get("opinions") or {}).items()
        if isinstance(v, list)}
    qa = _read_json(DATA_ROOT / "reports" / bvid / "qa.json") or {}
    reports = {}
    for name in REPORT_FILES:
        p = DATA_ROOT / "reports" / bvid / f"{name}.md"
        if p.exists():
            reports[name] = p.read_text("utf-8")[:60000]
    return {"job": job, "title": target.get("title") or meta.get("title") or bvid,
            "owner": target.get("owner") or "", "bvid": bvid,
            "stats": stats, "report_stats": rstats, "sentiment": sentiment,
            "qa": qa,
            "top_liked": (rstats.get("top_liked") or [])[:8],
            "reports": reports, "llm_run": bool(job.get("llm_run"))}


@app.get("/api/jobs/{job_id}/export/{fmt}")
async def export(job_id: str, fmt: str) -> Response:
    job = _public(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "任务尚未完成")
    bvid = job["bvid"]
    if fmt == "jsonl":
        p = _file_guard(DATA_ROOT / "normalized" / bvid / "comments.jsonl")
        return FileResponse(p, filename=f"{bvid}-normalized.jsonl",
                            media_type="application/x-ndjson")
    if fmt == "raw-jsonl":
        p = _file_guard(DATA_ROOT / "raw" / bvid / "comments.jsonl")
        return FileResponse(p, filename=f"{bvid}-raw.jsonl",
                            media_type="application/x-ndjson")
    if fmt == "csv":
        src = _file_guard(DATA_ROOT / "normalized" / bvid / "comments.jsonl")
        out = io.StringIO()
        rows = [json.loads(line) for line in src.read_text("utf-8").splitlines()
                if line.strip()][:MAX_LIMIT]
        if rows:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return Response(out.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{bvid}.csv"'})
    if fmt == "markdown":
        merged = DATA_ROOT / "reports" / bvid / "report.md"
        if merged.exists():
            md = merged.read_text("utf-8")
        else:
            parts = [f"# {bvid} 评论分析报告\n"]
            for name in REPORT_FILES:
                p = DATA_ROOT / "reports" / bvid / f"{name}.md"
                if p.exists():
                    parts.append(_strip_report_header(p.read_text("utf-8")))
            md = "\n\n".join(parts)
        return Response(md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{bvid}-report.md"'})
    if fmt == "agent":
        p = DATA_ROOT / "agent" / bvid
        if not p.is_dir():
            raise HTTPException(404, "Agent Pack 不存在")
        return _zip_dir(p, bvid)
    if fmt == "report":
        p = DATA_ROOT / "reports" / bvid
        if not p.is_dir():
            raise HTTPException(404, "报告目录不存在")
        return _zip_dir(p, bvid)
    if fmt == "bundle":
        return await export_bundle(bvid)
    raise HTTPException(400, "未知导出格式")


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"ok": "1", "time": datetime.now(timezone.utc).isoformat()}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

_load_jobs()
