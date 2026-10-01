"""AI provider settings: server-side config for any OpenAI-compatible API.

The entry point is the web UI (设置 → AI / NLP) but the API key itself lives
only in a 0600 file on the server (data/webui/ai_settings.json). The browser
receives a masked value (sk-••••R9xP) and can never read the key back; keys
must never appear in logs, exports, job records or run metadata.

Precedence: settings file > environment (.env LLM_*). Works with any
OpenAI-compatible endpoint (OpenAI, DeepSeek, GLM, Kimi, ...); local stacks
(Ollama / llama.cpp / vLLM) expose the same /v1 interface — for localhost
base URLs the API key may be empty.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, Optional

from .llm import LLMClient

DEFAULTS: Dict[str, Any] = {
    "base_url": "",
    "api_key": "",
    "model": "",
    "embed_model": "",
    "temperature": 0.0,
    "concurrency": 4,
    "batch_size": 100,
    "timeout": 120,
    "context_length": 32000,
    "prompt_price": 0.0,       # per 1M tokens, for cost estimation only
    "completion_price": 0.0,
}

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")


def settings_file(data_root: Path) -> Path:
    d = data_root / "webui"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ai_settings.json"


def load_settings(data_root: Path) -> Dict[str, Any]:
    """Environment (.env LLM_*) seeds the defaults; a saved settings file is
    the source of truth once present (empty string = explicit override)."""
    cfg = dict(DEFAULTS)
    env = os.environ
    for env_key, cfg_key in (("LLM_BASE_URL", "base_url"), ("LLM_API_KEY", "api_key"),
                             ("LLM_MODEL", "model")):
        if env.get(env_key):
            cfg[cfg_key] = env[env_key]
    if env.get("LLM_CONTEXT_LENGTH"):
        try:
            cfg["context_length"] = int(env["LLM_CONTEXT_LENGTH"])
        except ValueError:
            pass
    path = settings_file(data_root)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        for key in DEFAULTS:
            if key in stored and stored[key] is not None:
                cfg[key] = stored[key]
    return cfg


def save_settings(data_root: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a settings patch; secrets never round-trip through the client,
    so an empty api_key value means "keep the stored key"."""
    path = settings_file(data_root)
    current: Dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    for key in DEFAULTS:
        if key not in patch:
            continue
        value = patch[key]
        if key == "api_key" and (value is None or str(value).startswith("sk-••••")):
            continue  # masked echo back — keep stored secret untouched
        if key in ("temperature", "prompt_price", "completion_price"):
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
        elif key in ("concurrency", "batch_size", "timeout", "context_length"):
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        else:
            value = str(value).strip().rstrip("/")
        current[key] = value
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        path.parent.chmod(stat.S_IRWXU)          # 0700
    except OSError:
        pass
    return load_settings(data_root)


def masked_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:3]}••••••••{key[-4:]}"


def public_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Client-safe view: masked key only, plus flags the UI needs."""
    out = {k: cfg.get(k) for k in DEFAULTS}
    out["api_key"] = masked_key(cfg.get("api_key", ""))
    out["has_key"] = bool(cfg.get("api_key"))
    out["configured"] = bool(cfg.get("base_url") and cfg.get("model"))
    out["source"] = "file"
    return out


def is_local(base_url: str) -> bool:
    try:
        host = base_url.split("//", 1)[-1].split("/", 1)[0].lower()
    except IndexError:
        return False
    return any(host == h or host.startswith(h + ":") or host.split(":")[0] == h
               for h in LOCAL_HOSTS)


def build_client(data_root: Path, model: Optional[str] = None,
                 purpose: str = "chat") -> LLMClient:
    """LLMClient from stored settings.

    purpose='embed' prefers embed_model, falling back to the chat model.
    Local endpoints (Ollama etc.) may omit the API key. Raises ValueError
    with a user-facing message when unusable.
    """
    cfg = load_settings(data_root)
    base_url = cfg["base_url"]
    if not base_url:
        raise ValueError("未配置 AI API 地址（设置 → AI / NLP）")
    use_model = (model or (cfg.get("embed_model") if purpose == "embed" else "")
                 or cfg["model"])
    if not use_model:
        raise ValueError("未配置模型名（设置 → AI / NLP → 获取模型列表）")
    api_key = cfg.get("api_key") or ""
    if not api_key and not is_local(base_url):
        raise ValueError("该 API 地址需要 API Key（本地 Ollama 等可留空）")
    return LLMClient(
        base_url=base_url, api_key=api_key, model=use_model,
        cache_dir=None, context_length=cfg.get("context_length", 32000),
    )


def configured(data_root: Path) -> bool:
    cfg = load_settings(data_root)
    return bool(cfg["base_url"] and cfg["model"])
