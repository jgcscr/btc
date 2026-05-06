from __future__ import annotations

from pathlib import Path

import pytest

import src.runtime.cadence_support as cadence_support


def test_execute_weekly_cadence_runs_reliability_then_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(cadence_support, "resolve_latest_trustworthy_run_id", lambda: "run-123")

    cadence_support.execute_cadence(
        "weekly",
        python_bin="python",
        repo_root=tmp_path,
        run_command=lambda command: commands.append(list(command)),
    )

    assert commands[0][:4] == ["python", "-m", "src.scripts.run_reliability_pipeline", "--config"]
    assert commands[1][:3] == ["python", "-m", "src.scripts.run_refresh_and_predict"]


def test_execute_cadence_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported cadence"):
        cadence_support.execute_cadence("unknown", python_bin="python", repo_root=tmp_path, run_command=lambda _cmd: None)