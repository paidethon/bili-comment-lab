"""AI provider settings: server-side secret handling. The API key must never
be exposed unmasked to the client, and a masked echo-back must not overwrite
the stored key."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from bilicom import provider  # noqa: E402


def test_defaults_without_file(tmp_path: Path) -> None:
    cfg = provider.load_settings(tmp_path)
    assert cfg["base_url"] == "" and cfg["temperature"] == 0.0


def test_save_and_mask_roundtrip(tmp_path: Path) -> None:
    provider.save_settings(tmp_path, {"base_url": "https://api.example.com/v1/",
                                      "api_key": "sk-secret-123456789",
                                      "model": "test-model",
                                      "batch_size": "64"})
    cfg = provider.load_settings(tmp_path)
    assert cfg["base_url"] == "https://api.example.com/v1"  # trailing / stripped
    assert cfg["api_key"] == "sk-secret-123456789"
    assert cfg["batch_size"] == 64

    pub = provider.public_settings(cfg)
    assert pub["api_key"] == "sk-••••••••6789"
    assert "sk-secret" not in str(pub)
    assert pub["configured"] is True


def test_masked_patch_does_not_overwrite_key(tmp_path: Path) -> None:
    provider.save_settings(tmp_path, {"api_key": "sk-real-secret-value"})
    provider.save_settings(tmp_path, {"api_key": "sk-••••••••alue"})  # UI echo
    assert provider.load_settings(tmp_path)["api_key"] == "sk-real-secret-value"


def test_settings_file_permissions(tmp_path: Path) -> None:
    import stat

    path = provider.settings_file(tmp_path)
    provider.save_settings(tmp_path, {"api_key": "sk-x", "model": "m"})
    mode = path.stat().st_mode & 0o777
    assert mode == stat.S_IRUSR | stat.S_IWUSR  # 0600


def test_is_local_detection() -> None:
    assert provider.is_local("http://localhost:11434/v1")
    assert provider.is_local("http://127.0.0.1:8080/v1")
    assert not provider.is_local("https://api.deepseek.com/v1")


def test_build_client_requires_key_for_remote(tmp_path: Path) -> None:
    provider.save_settings(tmp_path, {"base_url": "https://api.example.com/v1",
                                      "model": "m"})
    with pytest.raises(ValueError, match="API Key"):
        provider.build_client(tmp_path)
    provider.save_settings(tmp_path, {"base_url": "http://localhost:11434/v1",
                                      "model": "ollama-model"})
    client = provider.build_client(tmp_path)  # local: no key needed
    assert client.model == "ollama-model"
    assert client._headers() == {}


def test_build_client_embed_purpose(tmp_path: Path) -> None:
    provider.save_settings(tmp_path, {"base_url": "http://localhost:11434/v1",
                                      "model": "chat-m",
                                      "embed_model": "bge-m3"})
    client = provider.build_client(tmp_path, purpose="embed")
    assert client.model == "bge-m3"
    client = provider.build_client(tmp_path, model="task-m")
    assert client.model == "task-m"  # task-level override wins
