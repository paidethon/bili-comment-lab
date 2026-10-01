"""Minimal OpenAI-compatible client (no vendor SDK, just requests).

Configured via environment (or .env loaded by the CLI), or built directly by
bilicom.provider from the web settings file:

    LLM_BASE_URL      e.g. https://api.example.com/v1
    LLM_API_KEY       optional for local endpoints (Ollama /v1)
    LLM_MODEL         flash/mini class models are the default target
    LLM_CONTEXT_LENGTH  context window in tokens (used for chunk budgets)

Also covers /v1/embeddings (semantic clustering) and /v1/models (settings
page model picker). Token usage flows through the on_usage hook so the
webapp can accumulate per-run cost statistics.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

# one pooled session for the process: LLM calls reuse TCP+TLS connections
# instead of a fresh handshake per request
_SESSION = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)

PROMPT_VERSIONS = {
    "extract": "extract-v1",
    "topic": "topic-v1",
    "reduce": "reduce-v2",
    "refine-label": "refine-label-v1",
}


class LLMConfigError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        context_length: Optional[int] = None,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.context_length = int(
            context_length or os.environ.get("LLM_CONTEXT_LENGTH", "32000"))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # usage sink: fn(kind, model, usage_dict, latency_ms)
        self.on_usage: Optional[Callable[[str, str, Dict[str, Any], int], None]] = None
        if not (self.base_url and self.model):
            raise LLMConfigError(
                "LLM_BASE_URL / LLM_MODEL must be set (see .env.example)"
            )

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    # ---- cache -------------------------------------------------------
    def cache_key(self, kind: str, content: str) -> str:
        """Cache key covers content, model, prompt version and kind so that
        switching model or prompt transparently invalidates old results."""
        payload = json.dumps(
            {
                "kind": kind,
                "prompt_version": PROMPT_VERSIONS.get(kind, "unknown"),
                "model": self.model,
                "content_sha": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        d = self.cache_dir / self.model.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        if path:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ---- raw completion ----------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        response_json: bool = False,
        retries: int = 3,
        kind: str = "chat",
    ) -> str:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_json:
            body["response_format"] = {"type": "json_object"}
        last_err: Optional[str] = None
        for attempt in range(retries):
            try:
                started = time.monotonic()
                resp = _SESSION.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=180,
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(2**attempt * 3)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._report_usage(kind, data.get("usage") or {},
                                   int((time.monotonic() - started) * 1000))
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as err:
                last_err = repr(err)
                time.sleep(2**attempt * 3)
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")

    def _report_usage(self, kind: str, usage: Dict[str, Any], latency_ms: int) -> None:
        if self.on_usage and isinstance(usage, dict) and usage:
            try:
                self.on_usage(kind, self.model, usage, latency_ms)
            except Exception:  # usage accounting must never break a run
                pass

    # ---- embeddings (semantic clustering / dedup) ----------------------
    def embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        retries: int = 3,
    ) -> tuple[List[List[float]], Dict[str, Any]]:
        """Batch embeddings via the OpenAI-compatible /embeddings endpoint.
        Returns (vectors in input order, usage dict)."""
        body = {"model": model or self.model, "input": texts}
        last_err: Optional[str] = None
        for attempt in range(retries):
            try:
                started = time.monotonic()
                resp = _SESSION.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=body,
                    timeout=timeout or 120,
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(2**attempt * 3)
                    continue
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
                vectors = [item["embedding"] for item in items]
                usage = data.get("usage") or {}
                self._report_usage("embed", usage,
                                   int((time.monotonic() - started) * 1000))
                return vectors, usage
            except (requests.RequestException, KeyError, TypeError, ValueError) as err:
                last_err = repr(err)
                time.sleep(2**attempt * 3)
        raise RuntimeError(f"embeddings request failed after {retries} attempts: {last_err}")

    def list_models(self, timeout: Optional[int] = None) -> List[str]:
        """GET /models — for the settings page model picker."""
        resp = _SESSION.get(f"{self.base_url}/models", headers=self._headers(),
                            timeout=timeout or 30)
        resp.raise_for_status()
        data = resp.json()
        return sorted(str(m["id"]) for m in data.get("data", []) if m.get("id"))

    # ---- cached, prompt-versioned call --------------------------------
    def cached_chat(
        self, kind: str, prompt_file: Path, content: str, response_json: bool = False
    ) -> Dict[str, Any]:
        """Prompt text lives in prompts/<kind>-vN.md and is version-pinned;
        both the prompt body and the input content hash go into the cache key,
        so re-runs after a failure only pay for the missing chunks."""
        system = prompt_file.read_text(encoding="utf-8")
        key = self.cache_key(kind, system + "\n\x00\n" + content)
        hit = self._cache_get(key)
        if hit is not None:
            return {"cached": True, **hit}
        raw = self.chat(system, content, response_json=response_json)
        value: Dict[str, Any]
        if response_json:
            value = _parse_json_loose(raw)
        else:
            value = {"text": raw}
        record = {
            "model": self.model,
            "prompt_version": PROMPT_VERSIONS.get(kind, "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "result": value,
        }
        self._cache_put(key, record)
        return {"cached": False, **record}


def _parse_json_loose(raw: str) -> Dict[str, Any]:
    """Tolerate models that wrap JSON in prose or code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
