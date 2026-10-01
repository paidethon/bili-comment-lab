"""SQLite dataset layer: a queryable index over the immutable raw JSONL.

data/raw/<id>/comments.jsonl stays the single source of truth and is never
written to by this module. The DB (data/lab.db) is a derived index used by the
webapp for dataset browsing, filtering, lossless refinement and cost tracking.

Comment identity is (video_id, comment_id): bilibili rpid is globally unique,
but imported datasets may carry arbitrary or missing ids. Refined results
always reference rows by (video_id, comment_id) so every refined comment can
be traced back to its original text.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_NAME = "lab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  bvid TEXT PRIMARY KEY,
  aid INTEGER,
  title TEXT DEFAULT '',
  owner TEXT DEFAULT '',
  owner_mid INTEGER,
  source_url TEXT DEFAULT '',
  source TEXT DEFAULT 'crawl',
  crawled_at TEXT DEFAULT '',
  raw_dir TEXT DEFAULT '',
  total_comments INTEGER DEFAULT 0,
  ingested_at REAL
);
CREATE TABLE IF NOT EXISTS comments (
  video_id TEXT NOT NULL,
  comment_id INTEGER NOT NULL,
  root_id INTEGER,
  parent_id INTEGER,
  level INTEGER NOT NULL DEFAULT 1,
  is_pinned INTEGER NOT NULL DEFAULT 0,
  content TEXT NOT NULL DEFAULT '',
  content_sha TEXT NOT NULL DEFAULT '',
  like_count INTEGER NOT NULL DEFAULT 0,
  reply_count INTEGER NOT NULL DEFAULT 0,
  ctime INTEGER,
  created_at TEXT DEFAULT '',
  username TEXT DEFAULT '',
  mid TEXT DEFAULT '',
  ip_location TEXT DEFAULT '',
  PRIMARY KEY (video_id, comment_id)
);
CREATE INDEX IF NOT EXISTS idx_comments_video
  ON comments(video_id, level, like_count DESC);
CREATE INDEX IF NOT EXISTS idx_comments_root ON comments(root_id);
CREATE INDEX IF NOT EXISTS idx_comments_sha ON comments(content_sha);
CREATE TABLE IF NOT EXISTS embeddings (
  content_sha TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,
  created_at TEXT,
  PRIMARY KEY (content_sha, model)
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  video_id TEXT DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'refine',
  mode TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'queued',
  stage TEXT DEFAULT '',
  progress REAL DEFAULT 0,
  params_json TEXT DEFAULT '{}',
  stats_json TEXT DEFAULT '{}',
  model TEXT DEFAULT '',
  embed_model TEXT DEFAULT '',
  error TEXT,
  last_log TEXT DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  created_at REAL,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS run_comments (
  run_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  comment_id INTEGER NOT NULL,
  cluster_id INTEGER,
  is_representative INTEGER NOT NULL DEFAULT 0,
  selected INTEGER NOT NULL DEFAULT 1,
  duplicate_of INTEGER,
  selection_reason TEXT DEFAULT '',
  quality_score REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, video_id, comment_id)
);
CREATE INDEX IF NOT EXISTS idx_rc_cluster ON run_comments(run_id, cluster_id);
CREATE TABLE IF NOT EXISTS clusters (
  run_id TEXT NOT NULL,
  cluster_id INTEGER NOT NULL,
  label TEXT DEFAULT '',
  keywords_json TEXT DEFAULT '[]',
  size INTEGER NOT NULL DEFAULT 0,
  representative_ids_json TEXT DEFAULT '[]',
  PRIMARY KEY (run_id, cluster_id)
);
CREATE TABLE IF NOT EXISTS ai_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  ts REAL,
  kind TEXT DEFAULT '',
  model TEXT DEFAULT '',
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  latency_ms INTEGER DEFAULT 0
);
"""


def connect(data_root: Path) -> sqlite3.Connection:
    data_root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_root / DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # pragmas tuned for this workload: many short reads (UI browsing),
    # occasional batch writes (ingest/refine). NORMAL sync is safe with WAL;
    # mmap saves a copy on hot read paths; small page cache keeps RSS flat.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-2000")          # 2 MB page cache
    conn.execute("PRAGMA mmap_size=134217728")       # 128 MB read map
    conn.executescript(SCHEMA)
    return conn


def content_sha(text: str) -> str:
    """Hash of the *normalized* comment text: exact-duplicate key.

    Normalization only affects dedup keying, never stored content.
    """
    norm = re.sub(r"[\s\W_]+", "", text.lower(), flags=re.UNICODE)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _row_from_raw(raw: Dict[str, Any], video_id: str) -> Optional[Tuple]:
    cid = raw.get("comment_id") or raw.get("rpid")
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return None
    is_reply = bool(raw.get("is_reply"))
    level = 2 if is_reply else 1
    root = raw.get("root_id") or raw.get("root_rpid")
    parent = raw.get("parent_id") or raw.get("parent_rpid")
    if is_reply:
        root = int(root) if root else cid
        parent = int(parent) if parent else root
        if parent == cid:
            parent = root
    else:
        root = parent = cid
    ctime = raw.get("ctime")
    try:
        ctime = int(ctime) if ctime not in (None, "", 0) else None
    except (TypeError, ValueError):
        ctime = None
    created = raw.get("created_at") or raw.get("ctime_text") or ""
    if not created and ctime:
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctime))
    content = str(raw.get("content") or raw.get("message") or "")
    return (
        video_id, cid, root, parent, level, 1 if raw.get("is_pinned") else 0,
        content, content_sha(content),
        int(raw.get("like_count") or raw.get("like") or 0),
        int(raw.get("reply_count") or raw.get("rcount") or 0),
        ctime, created,
        str(raw.get("username") or raw.get("uname") or ""),
        str(raw.get("mid") or raw.get("user_id") or ""),
        str(raw.get("ip_location") or ""),
    )


def ingest_video(conn: sqlite3.Connection, raw_dir: Path, video_id: str,
                 source: str = "crawl") -> Dict[str, Any]:
    """(Re)ingest one raw dataset into the index. Idempotent."""
    raw_file = raw_dir / "comments.jsonl"
    meta: Dict[str, Any] = {}
    meta_file = raw_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    target = meta.get("target_info") or {}
    rows: List[Tuple] = []
    bad = 0
    if raw_file.exists():
        for line in raw_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _row_from_raw(json.loads(line), video_id)
                if row:
                    rows.append(row)
            except (json.JSONDecodeError, TypeError, ValueError):
                bad += 1
    # keep the last occurrence per (video_id, comment_id) — matching the
    # INSERT OR REPLACE semantics below; raw feeds may repeat an rpid
    by_id: Dict[Tuple[str, int], Tuple] = {}
    for row in rows:
        by_id[(row[0], row[1])] = row
    rows = list(by_id.values())
    if not rows:
        return {"video_id": video_id, "ingested": 0, "malformed": bad,
                "stale_removed": 0}

    conn.executemany(
        "INSERT OR REPLACE INTO comments (video_id, comment_id, root_id, parent_id,"
        " level, is_pinned, content, content_sha, like_count, reply_count, ctime,"
        " created_at, username, mid, ip_location)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    # snapshot semantics: the raw JSONL is the single source of truth, so the
    # derived index must match it exactly — drop rows left over from older
    # crawls that are no longer in this file (prevents refined exports and
    # raw exports from disagreeing about the same video)
    keep_ids = [r[1] for r in rows]
    placeholder = ",".join("?" * len(keep_ids))
    cur = conn.execute(
        f"DELETE FROM comments WHERE video_id=? AND comment_id NOT IN ({placeholder})",
        [video_id] + keep_ids)
    stale_removed = cur.rowcount
    owner = target.get("owner") or meta.get("owner") or ""
    owner_mid = target.get("owner_mid") or meta.get("owner_mid")
    conn.execute(
        "INSERT OR REPLACE INTO videos (bvid, aid, title, owner, owner_mid,"
        " source_url, source, crawled_at, raw_dir, total_comments, ingested_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (video_id, target.get("aid") or meta.get("oid"),
         target.get("title") or meta.get("title") or video_id, owner,
         int(owner_mid) if owner_mid else None,
         meta.get("source_url") or "", source,
         meta.get("crawled_at") or "", str(raw_dir), len(rows), time.time()))
    conn.commit()
    return {"video_id": video_id, "ingested": len(rows), "malformed": bad,
            "stale_removed": stale_removed}


def ingest_all(conn: sqlite3.Connection, data_root: Path) -> List[Dict[str, Any]]:
    out = []
    for raw_dir in sorted((data_root / "raw").iterdir()):
        if raw_dir.is_dir() and (raw_dir / "comments.jsonl").exists():
            out.append(ingest_video(conn, raw_dir, raw_dir.name))
    return out


def get_video(conn: sqlite3.Connection, bvid: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM videos WHERE bvid=?", (bvid,)).fetchone()


def list_videos(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT v.*, "
        " (SELECT COUNT(*) FROM comments c WHERE c.video_id=v.bvid AND c.level=1) AS level1,"
        " (SELECT COUNT(*) FROM comments c WHERE c.video_id=v.bvid AND c.level=2) AS level2,"
        " (SELECT COUNT(*) FROM comments c WHERE c.video_id=v.bvid AND c.is_pinned=1) AS pinned,"
        " (SELECT COUNT(DISTINCT r.id) FROM runs r WHERE r.video_id=v.bvid"
        "  AND r.kind='refine' AND r.status='done') AS refine_runs"
        " FROM videos v ORDER BY v.ingested_at DESC").fetchall()
    return [dict(r) for r in rows]


SORTS = {
    "likes": "like_count {o}, comment_id {o}",
    "time": "ctime {o}, comment_id {o}",
    "replies": "reply_count {o}, like_count {o}",
}


def query_comments(conn: sqlite3.Connection, video_id: str, *, level: int = 0,
                   min_likes: int = 0, min_replies: int = 0, q: str = "",
                   start: str = "", end: str = "", pinned: int = -1,
                   sort: str = "likes", order: str = "desc",
                   page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    """Filtered comment listing. Server-side so the browser never loads a
    multi-MB JSONL just to page through it."""
    where = ["video_id=?"]
    args: List[Any] = [video_id]
    if level in (1, 2):
        where.append("level=?")
        args.append(level)
    if pinned in (0, 1):
        where.append("is_pinned=?")
        args.append(pinned)
    if min_likes:
        where.append("like_count>=?")
        args.append(min_likes)
    if min_replies:
        where.append("reply_count>=?")
        args.append(min_replies)
    if q:
        where.append("content LIKE ?")
        args.append(f"%{q}%")
    if start:
        where.append("ctime>=?")
        args.append(_parse_date(start, False))
    if end:
        where.append("ctime<=?")
        args.append(_parse_date(end, True))
    sql_where = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) FROM comments WHERE {sql_where}", args).fetchone()[0]
    clause = SORTS.get(sort, SORTS["likes"]).format(
        o="ASC" if order == "asc" else "DESC")
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    rows = conn.execute(
        f"SELECT * FROM comments WHERE {sql_where} ORDER BY {clause}"
        " LIMIT ? OFFSET ?",
        args + [page_size, (page - 1) * page_size]).fetchall()
    return {"total": total, "page": page, "page_size": page_size,
            "rows": [dict(r) for r in rows]}


def _parse_date(text: str, end_of_day: bool) -> int:
    text = str(text).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    try:
        if m:
            day = time.mktime(time.strptime(text, "%Y-%m-%d"))
            return int(day) + (86399 if end_of_day else 0)
        return int(float(text))
    except (ValueError, OverflowError):
        return 0


def comments_for_video(conn: sqlite3.Connection, video_id: str
                       ) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM comments WHERE video_id=?"
        " ORDER BY level ASC, like_count DESC", (video_id,)).fetchall()
    return [dict(r) for r in rows]


# ---- refine runs ------------------------------------------------------

def create_run(conn: sqlite3.Connection, run_id: str, video_id: str, kind: str,
               mode: str, params: Dict[str, Any], model: str = "",
               embed_model: str = "") -> None:
    now = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO runs (id, video_id, kind, mode, status, stage,"
        " progress, params_json, model, embed_model, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, video_id, kind, mode, "queued", "", 0,
         json.dumps(params, ensure_ascii=False), model, embed_model, now, now))
    conn.commit()


def update_run(conn: sqlite3.Connection, run_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE runs SET {cols} WHERE id=?",
                 list(fields.values()) + [run_id])
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    try:
        out["params"] = json.loads(out.pop("params_json") or "{}")
    except json.JSONDecodeError:
        out["params"] = {}
    try:
        out["stats"] = json.loads(out.pop("stats_json") or "{}")
    except json.JSONDecodeError:
        out["stats"] = {}
    return out


def list_runs(conn: sqlite3.Connection, kind: str = "refine",
              limit: int = 100) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, video_id, kind, mode, status, stage, progress, model,"
        " embed_model, error, created_at, updated_at, params_json, stats_json"
        " FROM runs WHERE kind=? ORDER BY created_at DESC LIMIT ?",
        (kind, limit)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["params"] = json.loads(d.pop("params_json") or "{}")
        except json.JSONDecodeError:
            d["params"] = {}
        try:
            d["stats"] = json.loads(d.pop("stats_json") or "{}")
        except json.JSONDecodeError:
            d["stats"] = {}
        out.append(d)
    return out


def active_run(conn: sqlite3.Connection, kind: str = "refine") -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id FROM runs WHERE kind=? AND status IN ('queued','running')"
        " ORDER BY created_at DESC LIMIT 1", (kind,)).fetchone()
    return get_run(conn, row["id"]) if row else None


def save_run_comments(conn: sqlite3.Connection, run_id: str,
                      rows: List[Tuple]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO run_comments (run_id, video_id, comment_id,"
        " cluster_id, is_representative, selected, duplicate_of,"
        " selection_reason, quality_score) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def save_clusters(conn: sqlite3.Connection, run_id: str,
                  clusters: List[Dict[str, Any]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO clusters (run_id, cluster_id, label,"
        " keywords_json, size, representative_ids_json) VALUES (?,?,?,?,?,?)",
        [(run_id, c["cluster_id"], c.get("label", ""),
          json.dumps(c.get("keywords", []), ensure_ascii=False),
          c.get("size", 0),
          json.dumps(c.get("representative_ids", []), ensure_ascii=False))
         for c in clusters])
    conn.commit()


def run_clusters(conn: sqlite3.Connection, run_id: str,
                 min_size: int = 1) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM clusters WHERE run_id=? AND size>=? ORDER BY size DESC",
        (run_id, min_size)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["keywords"] = json.loads(d.pop("keywords_json") or "[]")
        except json.JSONDecodeError:
            d["keywords"] = []
        try:
            d["representative_ids"] = json.loads(
                d.pop("representative_ids_json") or "[]")
        except json.JSONDecodeError:
            d["representative_ids"] = []
        out.append(d)
    return out


def run_cluster_members(conn: sqlite3.Connection, run_id: str, cluster_id: int,
                        representatives_only: bool = False) -> List[Dict[str, Any]]:
    """Cluster members with original text joined in — traceability read path.
    Thread fields (root_id/parent_id) ride along so refined exports keep the
    楼中楼 conversation structure."""
    sql = (
        "SELECT rc.video_id, rc.comment_id, c.root_id, c.parent_id,"
        " rc.cluster_id, rc.is_representative,"
        " rc.selected, rc.duplicate_of, rc.selection_reason, rc.quality_score,"
        " c.content, c.like_count, c.reply_count, c.level, c.created_at,"
        " c.is_pinned, o.content AS original_content, o.like_count AS original_likes"
        " FROM run_comments rc JOIN comments c"
        "   ON c.video_id=rc.video_id AND c.comment_id=rc.comment_id"
        " LEFT JOIN comments o"
        "   ON o.video_id=rc.video_id AND o.comment_id=rc.duplicate_of"
        " WHERE rc.run_id=? AND rc.selected=1")
    args: List[Any] = [run_id]
    if cluster_id is not None and cluster_id >= 0:
        sql += " AND rc.cluster_id=?"
        args.append(cluster_id)
    if representatives_only:
        sql += " AND rc.is_representative=1"
    sql += " ORDER BY c.like_count DESC"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def run_duplicates_of(conn: sqlite3.Connection, run_id: str,
                      comment_id: int) -> List[Dict[str, Any]]:
    """Folded duplicates pointing at a kept comment (聚类展开)."""
    rows = conn.execute(
        "SELECT rc.video_id, rc.comment_id, rc.duplicate_of, c.content,"
        " c.like_count, c.level, c.created_at"
        " FROM run_comments rc JOIN comments c"
        "   ON c.video_id=rc.video_id AND c.comment_id=rc.comment_id"
        " WHERE rc.run_id=? AND rc.selected=0 AND rc.duplicate_of=?"
        " ORDER BY c.like_count DESC", (run_id, comment_id)).fetchall()
    return [dict(r) for r in rows]


def record_usage(conn: sqlite3.Connection, run_id: str, kind: str, model: str,
                 usage: Dict[str, Any], latency_ms: int = 0) -> None:
    conn.execute(
        "INSERT INTO ai_usage (run_id, ts, kind, model, prompt_tokens,"
        " completion_tokens, total_tokens, latency_ms) VALUES (?,?,?,?,?,?,?,?)",
        (run_id, time.time(), kind, model,
         int(usage.get("prompt_tokens") or 0),
         int(usage.get("completion_tokens") or 0),
         int(usage.get("total_tokens") or usage.get("prompt_tokens", 0)) or 0,
         int(latency_ms or 0)))
    conn.commit()


def run_cost(conn: sqlite3.Connection, run_id: str,
             prompt_price: float = 0.0, completion_price: float = 0.0
             ) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*) AS calls,"
        " COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
        " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
        " COALESCE(SUM(total_tokens),0) AS total_tokens,"
        " COALESCE(AVG(latency_ms),0) AS avg_latency"
        " FROM ai_usage WHERE run_id=?", (run_id,)).fetchone()
    cost = (row["prompt_tokens"] / 1e6 * prompt_price
            + row["completion_tokens"] / 1e6 * completion_price)
    out = dict(row)
    out["est_cost"] = round(cost, 6)
    return out


def embedding_get_many(conn: sqlite3.Connection, model: str,
                       shas: List[str]) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for i in range(0, len(shas), 500):
        chunk = shas[i:i + 500]
        q = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT content_sha, vec FROM embeddings WHERE model=?"
                f" AND content_sha IN ({q})", [model] + chunk):
            out[row["content_sha"]] = row["vec"]
    return out


def embedding_put(conn: sqlite3.Connection, model: str, sha: str, dim: int,
                  vec: bytes) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (content_sha, model, dim, vec, created_at)"
        " VALUES (?,?,?,?,?)", (sha, model, dim, vec, time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    conn.commit()
