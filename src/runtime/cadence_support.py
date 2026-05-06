from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

from src.runtime.reliability_registry import ReliabilityRunRegistry


def resolve_latest_trustworthy_run_id() -> str:
    run_id = ReliabilityRunRegistry().resolve_latest_trustworthy_run_id()
    if not run_id:
        raise RuntimeError(
            "No trustworthy reliability run found under artifacts/reliability. "
            "Restore the deployed artifacts bundle before running cadence."
        )
    return run_id


def ensure_runtime_directories(repo_root: Path) -> None:
    for relative in ("data/spot_klines", "artifacts/predictions", "artifacts/monitoring"):
        (repo_root / relative).mkdir(parents=True, exist_ok=True)


def build_reliability_command(python_bin: str, config_path: str) -> list[str]:
    return [
        python_bin,
        "-m",
        "src.scripts.run_reliability_pipeline",
        "--config",
        config_path,
        "--continue-on-promotion-fail",
    ]


def build_refresh_command(python_bin: str, run_id: str) -> list[str]:
    return [
        python_bin,
        "-m",
        "src.scripts.run_refresh_and_predict",
        "--config",
        "configs/run_refresh_and_predict.shadow_simplified.yaml",
        "--targets",
        "0.25,1,4,8,12",
        "--thresholds-json",
        f"artifacts/reliability/{run_id}/summary/calibrated_thresholds.json",
        "--platt-calibration",
        f"artifacts/reliability/{run_id}/summary/platt_calibration.json",
        "--write-artifacts",
    ]


def build_shadow_command(python_bin: str, run_id: str) -> list[str]:
    return [
        python_bin,
        "-m",
        "src.scripts.run_shadow_profile_comparison",
        "--lhs-config",
        "configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml",
        "--rhs-config",
        "configs/run_refresh_and_predict.shadow_chop_suppression.yaml",
        "--lhs-label",
        "shadow_direction_enhanced_relaxed_chop",
        "--rhs-label",
        "shadow_chop_suppression",
        "--targets",
        "0.25,1,4,8,12",
        "--thresholds-json",
        f"artifacts/reliability/{run_id}/summary/calibrated_thresholds.json",
        "--platt-calibration",
        f"artifacts/reliability/{run_id}/summary/platt_calibration.json",
        "--restore-latest-to",
        "rhs",
    ]


def execute_cadence(
    cadence: str,
    *,
    python_bin: str,
    repo_root: Path,
    run_command: Callable[[Sequence[str]], None] | None = None,
) -> None:
    run = run_command or (lambda command: subprocess.run(list(command), check=True))
    ensure_runtime_directories(repo_root)

    if cadence == "weekly":
        run(build_reliability_command(python_bin, "configs/reliability_workflow.runtime.yaml"))
    elif cadence == "monthly":
        run(build_reliability_command(python_bin, "configs/reliability_workflow.default.yaml"))

    run_id = resolve_latest_trustworthy_run_id()
    if cadence in {"daily", "weekly", "monthly"}:
        run(build_refresh_command(python_bin, run_id))
        return
    if cadence == "shadow":
        run(build_shadow_command(python_bin, run_id))
        return
    raise ValueError(f"Unsupported cadence: {cadence}")