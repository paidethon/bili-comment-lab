"""Lossless refinement engine: 精炼，但不改变评论原话。

The pipeline is extractive by construction — AI text generation is only ever
used for *metadata* (cluster labels / keywords / picking ids), never for
comment text. Every refined row points back at (video_id, comment_id) and the
displayed text is always read from the store.

    raw comments (immutable)
        → exact dedupe            (normalized-text sha)
        → near-dup folding        (char 3-gram MinHash LSH + Jaccard verify)
        → semantic vectors        (OpenAI-compatible /embeddings, cached;
                                   offline fallback: local TF-IDF vectors)
        → leader clustering       (cosine / Jaccard-estimate threshold)
        → representatives         (top-N per cluster by likes, original text)
        → optional AI labeling    (JSON in/out; ids validated against members)

Nothing in this module ever writes to data/raw/. Refined output lives in the
lab.db run tables; exports materialize from there.
"""
from __future__ import annotations

import json
import math
import random
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import store
from .llm import LLMClient
from .provider import build_client

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

SRC_API = "api"
SRC_LOCAL = "local-tfidf"

_SIG_PERMS = 64          # minhash signature length
_SHINGLE_K = 3           # char shingle size
_BANDS, _ROWS = 16, 4    # LSH: 16 bands x 4 rows
_MERSENNE = (1 << 61) - 1
_RNG_SEED = 0x5EED
_LEADER_CAP_API = 6000
_LEADER_CAP_LOCAL = 30000


class RefineCancelled(Exception):
    pass


@dataclass
class RefineParams:
    near_dup_threshold: float = 0.82     # jaccard on char shingles
    cluster_threshold: float = 0.86      # cosine (api) — mapped for local space
    top_n_representatives: int = 3
    ai_labels: bool = False              # mode ①: AI picks ids/labels only
    model: Optional[str] = None          # task-level model override
    embed_model: Optional[str] = None
    batch_size: int = 64
    sample: Optional[int] = None         # preview: cluster a random sample
    max_ai_label_clusters: int = 300

    def sanitized(self) -> "RefineParams":
        p = RefineParams(**vars(self))
        p.near_dup_threshold = min(0.99, max(0.5, float(p.near_dup_threshold)))
        p.cluster_threshold = min(0.99, max(0.5, float(p.cluster_threshold)))
        p.top_n_representatives = min(20, max(1, int(p.top_n_representatives)))
        p.batch_size = min(256, max(8, int(p.batch_size)))
        if p.sample is not None:
            p.sample = min(5000, max(10, int(p.sample)))
        return p

    def local_threshold(self) -> float:
        """The local (no-API) path clusters on char-bigram TF-IDF cosine;
        map the user's embedding threshold into that space monotonically
        (0.86 → ~0.16, calibrated on Chinese short comments)."""
        return max(0.08, round(self.cluster_threshold - 0.7, 2))


# ---- text processing ---------------------------------------------------

_PUNCT = re.compile(r"[\s\W_]+|[\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F]+", re.UNICODE)
# "回复 @昵称 ：" reply prefix — analysis-only noise (the thread relation lives
# in root_id/parent_id); never touches stored content
_REPLY_PREFIX = re.compile(r"^\s*回复\s*@\S{1,40}\s*[:：]\s*")
# bilibili emote tags like [给心心] / [doge_金箍]
_EMOTE = re.compile(r"\[[^\[\]\n]{1,16}\]")


def clean_for_analysis(text: str) -> str:
    """Analysis view of a comment: reply prefixes and emote tags removed.

    Lossless rule: the stored/DISPLAYED content is never modified; this view
    feeds vectorization, keywording, dedupe shingles and AI payloads so that
    emoji trends and @nicknames cannot dominate topic clustering.
    """
    t = _REPLY_PREFIX.sub("", str(text))
    t = _EMOTE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def emote_tags(text: str) -> List[str]:
    return _EMOTE.findall(str(text))


def norm_text(text: str) -> str:
    return _PUNCT.sub("", clean_for_analysis(text))


def shingles(text: str) -> frozenset:
    t = norm_text(text)
    if len(t) <= _SHINGLE_K:
        return frozenset([t] if t else [])
    return frozenset(t[i:i + _SHINGLE_K] for i in range(len(t) - _SHINGLE_K + 1))


_PERM_A = np.array([random.Random(_RNG_SEED + i).randint(1, _MERSENNE - 1)
                    for i in range(_SIG_PERMS)], dtype=np.uint64)
_PERM_B = np.array([random.Random(_RNG_SEED + 10_000 + i).randint(0, _MERSENNE - 1)
                    for i in range(_SIG_PERMS)], dtype=np.uint64)


def minhash_sig(shs: frozenset) -> np.ndarray:
    if not shs:
        return np.zeros(_SIG_PERMS, dtype=np.uint64)
    h = np.array([zlib.crc32(s.encode("utf-8")) for s in shs], dtype=np.uint64)
    sig = ((h[:, None] * _PERM_A[None, :] + _PERM_B[None, :]) % _MERSENNE).min(axis=0)
    return sig.astype(np.uint64)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a) + len(b) - len(a & b))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---- stage 1/2: dedupe + near-dup folding ------------------------------

def fold_duplicates(records: List[Dict[str, Any]], near_threshold: float
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Exact dedupe by normalized text, then near-dup folding via MinHash LSH.

    Returns (kept survivors, funnel stats). Folded records carry
    duplicate_of=<canonical comment_id> and a _exact marker; canonical = most
    liked, ties broken by earliest comment id.
    """
    exact: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        exact.setdefault(r["content_sha"], []).append(r)

    survivors: List[Dict[str, Any]] = []
    exact_removed = 0
    for group in exact.values():
        group.sort(key=lambda r: (-r["like_count"], r.get("ctime") or 0,
                                  r["comment_id"]))
        canon, rest = group[0], group[1:]
        canon["duplicate_of"] = None
        survivors.append(canon)
        for dup in rest:
            dup["duplicate_of"] = canon["comment_id"]
            dup["_exact"] = True
        exact_removed += len(rest)

    # near-dup LSH over survivors; tiny texts carry no reliable shingle signal
    # and stay as-is
    index: List[Tuple[int, frozenset]] = []
    for i, r in enumerate(survivors):
        shs = shingles(r["content"])
        if shs and len(r["content"]) >= 4:
            index.append((i, shs))

    uf = UnionFind(len(survivors))
    buckets: Dict[bytes, List[int]] = {}
    sigs: Dict[int, np.ndarray] = {}
    for pos, (i, shs) in enumerate(index):
        sig = minhash_sig(shs)
        sigs[i] = sig
        for b in range(_BANDS):
            key = sig[b * _ROWS:(b + 1) * _ROWS].tobytes()
            buckets.setdefault(key, []).append(pos)

    near_groups = 0
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        # greedy within bucket: compare against already-kept group heads only,
        # which stays linear even when one shingle set attracts thousands
        heads: List[Tuple[int, frozenset]] = []
        for pos in bucket:
            i, shs = index[pos]
            hit = None
            for h_i, h_shs in heads:
                if jaccard(shs, h_shs) >= near_threshold:
                    hit = h_i
                    break
            if hit is None:
                heads.append((i, shs))
            else:
                uf.union(i, hit)

    near_removed = 0
    comp: Dict[int, List[int]] = {}
    for pos, (i, _shs) in enumerate(index):
        comp.setdefault(uf.find(i), []).append(i)
    for members in comp.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: (-survivors[i]["like_count"],
                                    survivors[i].get("ctime") or 0,
                                    survivors[i]["comment_id"]))
        canon = survivors[members[0]]
        for i in members[1:]:
            r = survivors[i]
            r["duplicate_of"] = canon["comment_id"]
            near_removed += 1
        near_groups += 1

    kept = [r for r in survivors if r.get("duplicate_of") is None]
    stats = {
        "original": len(records),
        "exact_groups": len(exact),
        "exact_removed": exact_removed,
        "near_groups": near_groups,
        "near_removed": near_removed,
        "survivors": len(kept),
    }
    return kept, stats


# ---- stage 3: vectors ---------------------------------------------------

def _vec_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _embed_text(r: Dict[str, Any]) -> str:
    """Text actually sent to the embedding API (cleaned view), keyed by the
    hash of that same text so the cache stays content-addressed."""
    return clean_for_analysis(r["content"]) or r["content"]


def embed_records(conn, records: List[Dict[str, Any]], client: LLMClient,
                  embed_model: str, batch_size: int,
                  log: Callable[[str], None],
                  progress: Callable[[float, str], None],
                  cancel_fn: Callable[[], bool]) -> None:
    """Fill r['_vec'] for every record via the API with a content-addressed
    DB cache (checkpoint/resume: re-running never re-pays cached texts)."""
    keys = {r["comment_id"]: store.content_sha(_embed_text(r)) for r in records}
    cached = store.embedding_get_many(conn, embed_model, list(set(keys.values())))
    pending = [r for r in records if keys[r["comment_id"]] not in cached]
    log(f"embeddings: {len(records) - len(pending)} cached, {len(pending)} to fetch")
    total = len(pending)
    for i in range(0, total, batch_size):
        if cancel_fn():
            raise RefineCancelled()
        batch = pending[i:i + batch_size]
        vectors, _usage = client.embeddings([_embed_text(r) for r in batch],
                                            model=embed_model)
        for r, vec in zip(batch, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            r["_vec"] = arr
            store.embedding_put(conn, embed_model, keys[r["comment_id"]],
                                len(arr), _vec_bytes(arr))
        done = min(i + batch_size, total)
        log(f"embeddings: {done}/{total}")
        progress(0.2 + 0.5 * done / max(1, total), "embedding")
    for r in records:
        if "_vec" in r:
            continue
        blob = cached.get(keys[r["comment_id"]])
        if blob is None:
            raise RuntimeError(f"embedding cache miss: {r['comment_id']}")
        r["_vec"] = np.frombuffer(blob, dtype=np.float32)


def content_terms(text: str) -> List[str]:
    """Sorted unique terms — the local vector space. CJK contributes char
    bigrams (the sweet spot for short Chinese comments); latin/digit runs
    contribute whole lowercase words (≥2 chars) instead of letter bigrams,
    so English prose can no longer spawn 'ar'/'oge'-style junk clusters."""
    t = clean_for_analysis(text).lower()
    terms = set()
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", t):
        if re.fullmatch(r"[a-z0-9]+", run):
            if len(run) >= 2 and not run.isdigit():
                terms.add(run)
            continue
        if len(run) == 1:
            terms.add(run)
            continue
        for i in range(len(run) - 1):
            terms.add(run[i:i + 2])
    return sorted(terms)


def keyword_terms(text: str) -> set:
    """CJK 2-4 grams + latin words for extractive keyword display. 4-grams
    let whole phrases like 社会资源 beat their own fragments as labels."""
    terms = set(content_terms(text))
    t = clean_for_analysis(text).lower()
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", t):
        if re.fullmatch(r"[a-z0-9]+", run):
            continue                      # whole word already in content_terms
        for n in (3, 4):                  # trigrams/4-grams only; bigrams exist
            for i in range(len(run) - n + 1):
                terms.add(run[i:i + n])
    return terms


def attach_local_vectors(records: List[Dict[str, Any]]) -> None:
    """Offline fallback vectors: binary TF-IDF over char bigrams + trigrams,
    l2-normalized, stored as sparse {term_id: weight} dicts. Groups comments
    that share content vocabulary even when the wording differs; deep
    paraphrases with zero shared n-grams need the API embedding path."""
    df: Dict[str, int] = {}
    terms_of: List[List[str]] = []
    for r in records:
        terms = content_terms(r["content"])
        terms_of.append(terms)
        for term in terms:
            df[term] = df.get(term, 0) + 1
    n = max(1, len(records))
    vocab: Dict[str, int] = {}
    idf: Dict[int, float] = {}
    for term, count in df.items():
        vocab[term] = len(vocab)
        idf[vocab[term]] = math.log((n + 1) / (count + 0.5))
    for r, terms in zip(records, terms_of):
        vec: Dict[int, float] = {}
        for term in terms:
            tid = vocab[term]
            vec[tid] = vec.get(tid, 0.0) + idf[tid]
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        r["_vec"] = {tid: w / norm for tid, w in vec.items()}


# ---- stage 4: leader clustering ----------------------------------------

def _similarity_dense(cand: np.ndarray, leaders: np.ndarray) -> np.ndarray:
    """Cosine of each candidate row against each leader row: (B,D)x(L,D)."""
    cn = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-9)
    ln = leaders / np.maximum(np.linalg.norm(leaders, axis=1, keepdims=True), 1e-9)
    return cn @ ln.T


def _leader_pass_dense(records: List[Dict[str, Any]], threshold: float,
                       cancel_fn: Callable[[], bool],
                       progress: Optional[Callable[[float, str], None]],
                       batch_size: int) -> Tuple[List[int], int]:
    """Greedy leader clustering over dense API embedding vectors, streaming
    from a like-ordered record list so leaders (and cluster ids) are
    deterministic. Guarded by a leader cap: past the cap, low-similarity
    records join their best existing leader."""
    n = len(records)
    if n == 0:
        return [], 0
    dim = records[0]["_vec"].shape[0]
    leaders = np.zeros((0, dim), dtype=np.float32)
    cap_hit = False
    assignments: List[int] = []
    for start in range(0, n, batch_size):
        if cancel_fn():
            raise RefineCancelled()
        batch = records[start:start + batch_size]
        cand = np.stack([r["_vec"] for r in batch]).astype(np.float32)
        if leaders.shape[0]:
            sims = _similarity_dense(cand, leaders)
            best_idx = sims.argmax(axis=1)
            best = sims[np.arange(len(batch)), best_idx]
        else:
            best_idx = np.zeros(len(batch), dtype=int)
            best = np.full(len(batch), -1.0, dtype=np.float32)
        for j, r in enumerate(batch):
            if best[j] >= threshold and leaders.shape[0]:
                assignments.append(int(best_idx[j]))
            elif leaders.shape[0] < _LEADER_CAP_API:
                assignments.append(leaders.shape[0])
                leaders = np.vstack([leaders, cand[j:j + 1]])
            else:
                cap_hit = True
                assignments.append(int(best_idx[j]))
        if progress:
            progress((start + len(batch)) / n, "clustering")
    return assignments, leaders.shape[0]


def _leader_pass_sparse(records: List[Dict[str, Any]], threshold: float,
                        cancel_fn: Callable[[], bool],
                        progress: Optional[Callable[[float, str], None]]
                        ) -> Tuple[List[int], int]:
    """Same greedy pass for the local TF-IDF path, via an inverted index over
    term ids (only leaders are indexed), so it stays linear-ish without numpy
    matrix blowups. Vectors are l2-normalized dicts {term_id: weight}."""
    postings: Dict[int, List[Tuple[int, float]]] = {}
    leader_norms: List[float] = []
    assignments: List[int] = []
    cap_hit = False
    for idx, r in enumerate(records):
        if cancel_fn() and idx % 1000 == 0:
            raise RefineCancelled()
        vec: Dict[int, float] = r["_vec"]
        best_li, best_sim = -1, 0.0
        if vec:
            acc: Dict[int, float] = {}
            for tid, w in vec.items():
                for li, lw in postings.get(tid, ()):
                    acc[li] = acc.get(li, 0.0) + w * lw
            for li, dot in acc.items():
                sim = dot / (leader_norms[li] or 1.0)
                if sim > best_sim:
                    best_li, best_sim = li, sim
        if best_sim >= threshold and best_li >= 0:
            assignments.append(best_li)
        elif len(leader_norms) < _LEADER_CAP_LOCAL:
            assignments.append(len(leader_norms))
            leader_norms.append(math.sqrt(sum(w * w for w in vec.values())) or 1e-9)
            for tid, w in vec.items():
                postings.setdefault(tid, []).append((len(leader_norms) - 1, w))
        else:
            cap_hit = True
            assignments.append(max(best_li, 0))
        if progress and idx % 2000 == 0:
            progress(idx / max(1, len(records)), "clustering")
    return assignments, len(leader_norms)


def choose_threshold(records: List[Dict[str, Any]], base: float,
                     local_space: bool, log: Callable[[str], None]) -> float:
    """If the corpus fragments into singletons, lower the threshold stepwise
    (high threshold = everything is a singleton). Bounded, and only ever
    relaxes downward. Stops once at most ~35% of the sample stands alone —
    a corpus where most comments form no group means the threshold is still
    too sharp for short-text similarity."""
    if len(records) <= 800:          # tiny sets: cluster directly at base
        return base
    sample = records[:3000]
    eff = base
    floor = 0.05 if local_space else 0.45
    for _ in range(8):
        assign, _n = _leader_pass(sample, eff, local_space, lambda: False, None)
        alone = len(set(assign)) / len(sample)
        if alone <= 0.35:
            return eff
        eff = max(floor, round(eff - (0.04 if local_space else 0.06), 2))
        log(f"clustering: sample fragmented ({len(set(assign))}/{len(sample)}),"
            f" threshold -> {eff:.2f}")
    return eff


def _leader_pass(records: List[Dict[str, Any]], threshold: float,
                 local_space: bool, cancel_fn: Callable[[], bool],
                 progress: Optional[Callable[[float, str], None]] = None,
                 batch_size: int = 1000) -> Tuple[List[int], int]:
    if local_space:
        return _leader_pass_sparse(records, threshold, cancel_fn, progress)
    return _leader_pass_dense(records, threshold, cancel_fn, progress, batch_size)


def cluster_records(records: List[Dict[str, Any]], threshold: float,
                    local_space: bool, log: Callable[[str], None],
                    progress: Callable[[float, str], None],
                    cancel_fn: Callable[[], bool], batch_size: int = 1000
                    ) -> Tuple[int, bool, float]:
    """Cluster in place (r['_cluster']); returns (cluster count, cap_hit,
    effective threshold after adaptive relaxation)."""
    eff = choose_threshold(records, threshold, local_space, log)
    if eff != threshold:
        log(f"clustering: effective threshold {eff}")
    assign, n_leaders = _leader_pass(records, eff, local_space, cancel_fn,
                                     progress, batch_size)
    for r, c in zip(records, assign):
        r["_cluster"] = c
    return n_leaders, False, eff


# ---- stage 5: keywords + optional AI labeling ---------------------------

_STOP = {"的", "了", "是", "我", "你", "他", "她", "这", "那", "都", "也", "就",
         "不", "人", "啊", "吧", "吗", "呢", "有", "在", "和", "一个", "什么",
         "没有", "我们", "你们", "他们", "自己", "觉得", "感觉", "现在", "这个",
         "那个", "还是", "但是", "可以", "哈哈", "哈哈哈", "哈哈哈哈", "笑死",
         "确实", "回复", "如果", "因为", "所以", "就是", "真的", "可能", "视频",
         "我觉得", "知道", "个视频", "的好", "不是", "一下", "自己", "他们",
         "没有", "什么", "这个", "那个", "还是", "时候", "现在", "可以",
         "这样", "那样", "有点", "一个", "已经", "应该", "这么", "那么"}


def global_df(records: List[Dict[str, Any]]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for r in records:
        for g in content_terms(r["content"]):
            df[g] = df.get(g, 0) + 1
    return df


def cluster_keywords(texts: List[str], df: Dict[str, int], n_docs: int,
                     limit: int = 6) -> List[str]:
    """Extractive keywords: char n-grams concentrated in this cluster but
    rare corpus-wide, longest-first with substring suppression. Falls back to
    the cluster's top raw-count terms so a keyword row is never empty."""
    if not texts:
        return []
    local: Dict[str, int] = {}
    for t in texts:
        for g in keyword_terms(t):
            local[g] = local.get(g, 0) + 1
    size = len(texts)
    scored = []
    for g, cnt in local.items():
        if cnt < max(2, math.ceil(0.15 * size)):
            continue
        # the corpus-wide rarity filter only makes sense on a meaningful corpus
        if n_docs >= 50:
            corpus_rate = df.get(g, 0) / max(1, n_docs)
            if corpus_rate > 0.25:
                continue
        if g in _STOP or len(g) < 2:
            continue
        scored.append((g, cnt * (1.0 - df.get(g, 0) / max(1, n_docs))))
    scored.sort(key=lambda kv: (-kv[1], -len(kv[0])))
    if not scored:
        scored = sorted(((g, c) for g, c in local.items()
                         if c >= 2 and g not in _STOP and len(g) >= 2),
                        key=lambda kv: (-kv[1], -len(kv[0])))
        if not scored:
            scored = sorted(((g, c) for g, c in local.items() if len(g) >= 2),
                            key=lambda kv: (-kv[1], -len(kv[0])))
    out: List[str] = []
    for g, _s in scored:
        if any(g in kept or kept in g for kept in out):
            continue
        out.append(g)
        if len(out) >= limit:
            break
    return out


def ai_label_clusters(client: LLMClient, clusters: List[Dict[str, Any]],
                      max_clusters: int, log: Callable[[str], None],
                      progress: Callable[[float, str], None],
                      cancel_fn: Callable[[], bool]) -> None:
    """Mode ① AI pass: the model sees original comments with ids and returns
    ONLY {topic, keywords, representative_ids}. Returned text never becomes a
    comment; ids not in the cluster are dropped. Cached per (model, content)."""
    prompt_file = PROMPTS_DIR / "refine-label-v1.md"
    targets = [c for c in clusters if c["size"] >= 2][:max_clusters]
    for n, c in enumerate(targets, 1):
        if cancel_fn():
            raise RefineCancelled()
        members = c["members"]
        payload = [{"id": r["comment_id"],
                    "text": clean_for_analysis(r["content"]) or r["content"],
                    "likes": r["like_count"]} for r in members[:40]]
        content = json.dumps(
            {"suggested_keywords": c["keywords"], "comments": payload},
            ensure_ascii=False)
        out = client.cached_chat("refine-label", prompt_file, content,
                                 response_json=True)
        result = out.get("result") or {}
        label = str(result.get("topic") or "").strip()[:40]
        kws = [str(k).strip()[:20] for k in (result.get("keywords") or [])[:8]
               if str(k).strip()]
        valid_ids = {r["comment_id"] for r in members}
        ai_ids = [int(i) for i in (result.get("representative_ids") or [])
                  if str(i).lstrip("-").isdigit() and int(i) in valid_ids]
        if label:
            c["label"] = label
        if kws:
            c["keywords"] = kws
        c["ai_representative_ids"] = ai_ids
        log(f"ai label {n}/{len(targets)}: {label or c['keywords'][:2]}"
            f"{' (cache)' if out.get('cached') else ''}")
        progress(0.95 * n / max(1, len(targets)), "labeling")


# ---- orchestration ------------------------------------------------------

def _pick_representatives(members: List[Dict[str, Any]], top_n: int,
                          ai_ids: List[int]) -> List[Tuple[Dict[str, Any], str]]:
    chosen: List[Tuple[Dict[str, Any], str]] = []
    seen = set()
    by_id = {m["comment_id"]: m for m in members}
    for cid in ai_ids:
        m = by_id.get(cid)
        if m and cid not in seen:
            chosen.append((m, "ai_selected"))
            seen.add(cid)
    for m in sorted(members, key=lambda r: (-r["like_count"],
                                            r.get("ctime") or 0,
                                            r["comment_id"])):
        if len(chosen) >= top_n:
            break
        if m["comment_id"] in seen:
            continue
        chosen.append((m, "top_likes"))
        seen.add(m["comment_id"])
    return chosen


def run_refine(conn, data_root: Path, run_id: str, bvid: str,
               params: RefineParams, log: Callable[[str], None] = print,
               cancel_fn: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[float, str], None]] = None
               ) -> Dict[str, Any]:
    """Execute one refinement run end-to-end. Deterministic stages recompute
    from the immutable index; API work (embeddings, labels) is cached, so a
    resumed run never re-pays for completed calls."""
    params = params.sanitized()
    cancel_fn = cancel_fn or (lambda: False)
    progress = progress or (lambda p, s: None)
    # direct callers (CLI/tests) may not have pre-created the row
    if store.get_run(conn, run_id) is None:
        store.create_run(conn, run_id, bvid, "refine",
                         "lossless_ai" if params.ai_labels else "lossless",
                         vars(params))
    store.update_run(conn, run_id, status="running", stage="loading",
                     progress=0.02, error=None)

    if bvid and bvid != "all":
        videos = [bvid]
    else:
        videos = [v["bvid"] for v in store.list_videos(conn)]
    records: List[Dict[str, Any]] = []
    for v in videos:
        records.extend(store.comments_for_video(conn, v))
    if not records:
        raise RuntimeError("数据集为空：先提取或导入评论")

    progress(0.05, "dedupe")
    store.update_run(conn, run_id, stage="dedupe")
    kept, fold_stats = fold_duplicates(records, params.near_dup_threshold)
    log(f"dedupe: {fold_stats}")

    if params.sample and len(kept) > params.sample:
        kept = random.Random(42).sample(kept, params.sample)

    # comments whose analysis view is empty (pure emoji / 打卡) carry no
    # topical signal: park them in a dedicated bucket instead of letting each
    # become its own leader and shattering the cluster space
    emote_bucket = [r for r in kept if not clean_for_analysis(r["content"])]
    kept = [r for r in kept if clean_for_analysis(r["content"])]
    if emote_bucket:
        log(f"emote bucket: {len(emote_bucket)} pure-emoji/check-in comments")

    # vectors — API embeddings with graceful local fallback
    store.update_run(conn, run_id, stage="embedding")
    embed_model = ""
    try:
        client = build_client(data_root, model=params.model, purpose="embed")
    except ValueError as err:
        client = None
        log(f"AI 嵌入不可用（{err}），使用本地 TF-IDF 向量")
    source = SRC_LOCAL
    if client is not None:
        embed_model = params.embed_model or client.model
        try:
            embed_records(conn, kept, client, embed_model, params.batch_size,
                          log, progress, cancel_fn)
            source = SRC_API
        except RefineCancelled:
            raise
        except Exception as err:  # provider without /embeddings etc.
            log(f"AI 嵌入失败（{err}），回退本地向量")
            embed_model = ""
            for r in kept:
                r.pop("_vec", None)
    if source == SRC_LOCAL:
        attach_local_vectors(kept)
        progress(0.6, "embedding")

    progress(0.7, "clustering")
    store.update_run(conn, run_id, stage="clustering")
    local_space = source == SRC_LOCAL
    eff_threshold = (params.local_threshold() if local_space
                     else params.cluster_threshold)
    n_clusters, _cap, eff_threshold = cluster_records(
        kept, eff_threshold, local_space, log, progress, cancel_fn,
        params.batch_size * 8)
    log(f"clusters: {n_clusters} (threshold {eff_threshold}, {source})")

    progress(0.95, "keywords")
    store.update_run(conn, run_id, stage="keywords")
    df = global_df(kept)
    n_docs = max(1, len(kept))
    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for r in kept:
        by_cluster.setdefault(r["_cluster"], []).append(r)
    if emote_bucket:
        by_cluster[max(by_cluster, default=-1) + 1] = emote_bucket
    clusters = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        is_emote = members is emote_bucket
        clusters.append({
            "cluster_id": cid,
            "size": len(members),
            "members": members,
            "keywords": [] if is_emote else cluster_keywords(
                [m["content"] for m in members], df, n_docs),
            "label": "纯表情/打卡" if is_emote else "",
            "ai_representative_ids": [],
        })

    if params.ai_labels and clusters:
        try:
            chat_client = build_client(data_root, model=params.model)
        except ValueError as err:
            chat_client = None
            log(f"AI 标注跳过：{err}")
        if chat_client is not None:
            store.update_run(conn, run_id, stage="labeling")
            ai_label_clusters(chat_client,
                              [c for c in clusters if not c["label"]],
                              params.max_ai_label_clusters, log, progress,
                              cancel_fn)

    # persist clusters + representatives + folded duplicates
    progress(0.98, "saving")
    store.update_run(conn, run_id, stage="saving")

    _max_like = max((r["like_count"] or 0) for r in records) if records else 0

    def quality(r: Dict[str, Any]) -> float:
        """Composite 0-1 score, defined for every row (0 点赞不再是 0 分):
        55% content signal + 45% engagement (log-scaled likes)."""
        text = clean_for_analysis(r["content"])
        n = len(text)
        if n <= 1:
            content = 0.1
        elif n <= 5:
            content = 0.3
        elif n <= 15:
            content = 0.55
        elif n <= 40:
            content = 0.75
        else:
            content = 0.9
        if re.search(r"？|\?|怎么|如何|为什么|求|建议|应该", text):
            content = min(1.0, content + 0.1)
        if re.search(r"(.)\1{6,}", text):      # 凿凿凿凿… spam
            content *= 0.5
        eng = 0.0
        if _max_like > 0:
            eng = math.log10((r["like_count"] or 0) + 1) / math.log10(_max_like + 1)
        return round(0.55 * content + 0.45 * eng, 3)

    cluster_rows = []
    rep_rows: List[Tuple] = []
    for c in clusters:
        reps = _pick_representatives(c["members"], params.top_n_representatives,
                                     c["ai_representative_ids"])
        rep_ids = [m["comment_id"] for m, _why in reps]
        label = c["label"]
        if not label:
            # a 1-3 letter latin fragment makes an unreadable label for a
            # CJK corpus — prefer the first keyword that isn't one
            label = next((k for k in c["keywords"]
                          if not re.fullmatch(r"[a-z0-9]{1,3}", k)), None)
            label = label or "杂项"
        c["label"] = label
        c["representative_ids"] = rep_ids
        cluster_rows.append({"cluster_id": c["cluster_id"], "label": label,
                             "keywords": c["keywords"], "size": c["size"],
                             "representative_ids": rep_ids})
        rep_set = set(rep_ids)
        for m in c["members"]:
            is_rep = 1 if m["comment_id"] in rep_set else 0
            rep_rows.append((run_id, m["video_id"], m["comment_id"],
                             c["cluster_id"], is_rep, 1, None,
                             ("representative" if is_rep else
                              "cluster_member"), quality(m)))
    for r in records:
        if r.get("duplicate_of") is not None:
            rep_rows.append((run_id, r["video_id"], r["comment_id"], None, 0,
                             0, r["duplicate_of"],
                             "exact_duplicate" if r.get("_exact") else
                             "near_duplicate", quality(r)))
    conn.execute("DELETE FROM run_comments WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM clusters WHERE run_id=?", (run_id,))
    store.save_clusters(conn, run_id, cluster_rows)
    store.save_run_comments(conn, run_id, rep_rows)

    n_reps = sum(len(c["representative_ids"]) for c in clusters)
    stats = dict(fold_stats)
    stats.update({
        "embedded": len(kept) + len(emote_bucket),
        "vector_source": source,
        "embed_model": embed_model,
        "cluster_threshold": eff_threshold,
        "clusters": n_clusters,
        "multi_clusters": sum(1 for c in clusters if c["size"] >= 2),
        "representatives": n_reps,
        "compression": round(1 - n_reps / max(1, stats["original"]), 4),
        "sampled": bool(params.sample),
        "emote_only": len(emote_bucket),
        "quality_score": "0-1 composite (55% content signal + 45% log-likes)",
    })
    store.update_run(conn, run_id, status="done", stage="done", progress=1.0,
                     stats_json=json.dumps(stats, ensure_ascii=False))
    return {"stats": stats, "clusters": len(clusters)}


def preview(conn, bvid: str, sample: int = 100,
            near_dup_threshold: float = 0.82) -> Dict[str, Any]:
    """Dry-run the dedupe stages + local clustering on a sample — no AI cost,
    nothing persisted. Gives the operator an estimate before the real run."""
    if bvid and bvid != "all":
        videos = [bvid]
    else:
        videos = [v["bvid"] for v in store.list_videos(conn)]
    records: List[Dict[str, Any]] = []
    for v in videos:
        records.extend(store.comments_for_video(conn, v))
    if not records:
        raise RuntimeError("数据集为空")
    kept, fold_stats = fold_duplicates(records, near_dup_threshold)
    sample_records = kept if len(kept) <= sample else random.Random(42).sample(
        kept, sample)
    attach_local_vectors(sample_records)
    n_clusters = 0
    example_groups: List[Dict[str, Any]] = []
    if sample_records:
        assign, n_clusters = _leader_pass(sample_records, 0.36, True,
                                          lambda: False, None, 5000)
        for r, c in zip(sample_records, assign):
            r["_cluster"] = c
        by_cluster: Dict[int, List[Dict[str, Any]]] = {}
        for r in sample_records:
            by_cluster.setdefault(r["_cluster"], []).append(r)
        multi = sorted((c for c in by_cluster.values() if len(c) >= 2),
                       key=len, reverse=True)
        for members in multi[:8]:
            members.sort(key=lambda r: -r["like_count"])
            example_groups.append({
                "size": len(members),
                "comments": [{"comment_id": m["comment_id"],
                              "content": m["content"][:120],
                              "like_count": m["like_count"]}
                             for m in members[:6]],
            })
    est = dict(fold_stats)
    est["sample"] = len(sample_records)
    est["sample_clusters"] = n_clusters
    est["note"] = "预览使用本地词汇聚类估算；正式运行配置 AI 嵌入后语义聚类更准"
    return {"stats": est, "example_groups": example_groups,
            "removed_preview": [
                {"comment_id": r["comment_id"], "content": r["content"][:80],
                 "duplicate_of": r["duplicate_of"]}
                for r in records if r.get("duplicate_of")][:15]}
