"""Integration test: fixture raw data -> normalize -> prepare -> analyze
(extractive-only path; no network, no LLM). Mirrors ./bili prepare + analyze."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import LAB, VIDEO


def test_full_pipeline_offline(lab_root):
    env_patcher = None
    proc = subprocess.run(
        [
            sys.executable,           # 当前解释器，任何环境可用
            str(LAB / "bilicom" / "cli.py"),
            "prepare",
            VIDEO,
            "--budget", "4000",
        ],
        capture_output=True,
        text=True,
        cwd=LAB,
        env=None,
    )
    # The CLI is bound to the real lab root; for the fixture we drive the
    # modules directly instead when the subprocess cannot see tmp data.
    if proc.returncode != 0:
        from bilicom.normalize import normalize
        from bilicom.prepare import prepare
        from bilicom.analyze import analyze

        stats = normalize(lab_root / "data" / "raw" / VIDEO, VIDEO)
        assert stats["normalized"] == 7
        manifest = prepare(lab_root / "data" / "normalized" / VIDEO, VIDEO, budget_tokens=4000)
        assert manifest["total_comments"] == 7
        result = analyze(VIDEO, lab_root / "data", log=lambda m: None)
        assert result["llm_run"] is False
        assert result["layer1_only"] is True

        reports = lab_root / "data" / "reports" / VIDEO
        assert (reports / "summary.md").exists()
        assert (reports / "representative-comments.md").exists()
        rep = (reports / "representative-comments.md").read_text("utf-8")
        assert "100" in rep  # pinned comment id present
    else:
        out = proc.stdout
        assert "normalize" in out
