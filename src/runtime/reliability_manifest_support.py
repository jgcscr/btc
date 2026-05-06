from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime.reliability_workflow_common import StepResult


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_cadence_plan_payload(*, cadence_cfg: Mapping[str, Any], summary_dir: Path) -> dict[str, Any]:
    return {
        "generated_at": utc_timestamp(),
        "monthly_retrain_day": int(cadence_cfg.get("monthly_retrain_day", 1)),
        "weekly_recalibration_weekday": str(cadence_cfg.get("weekly_recalibration_weekday", "mon")),
        "trigger_file": str(summary_dir / "reliability_triggers.json"),
        "notes": "Trigger immediate retrain when reliability_triggers.global_trigger = true.",
    }


def write_cadence_plan(*, cadence_cfg: Mapping[str, Any], summary_dir: Path) -> Path:
    cadence_path = summary_dir / "cadence_plan.json"
    cadence_path.write_text(
        json.dumps(build_cadence_plan_payload(cadence_cfg=cadence_cfg, summary_dir=summary_dir), indent=2),
        encoding="utf-8",
    )
    return cadence_path


def build_workflow_manifest(
    *,
    config_path: str,
    run_dir: Path,
    run_profile_id: str,
    run_profile_name: str,
    steps: Sequence[StepResult],
    cadence_path: Path,
) -> dict[str, Any]:
    return {
        "generated_at": utc_timestamp(),
        "config": str(config_path),
        "profile": {
            "id": run_profile_id,
            "name": run_profile_name,
        },
        "run_dir": str(run_dir),
        "steps": [
            {
                "name": step.name,
                "returncode": step.returncode,
                "log": str(step.log_path),
                "command": step.command,
            }
            for step in steps
        ],
        "cadence_plan": str(cadence_path),
    }


def write_workflow_manifest(
    *,
    summary_dir: Path,
    config_path: str,
    run_dir: Path,
    run_profile_id: str,
    run_profile_name: str,
    steps: Sequence[StepResult],
    cadence_path: Path,
) -> Path:
    manifest_path = summary_dir / "workflow_manifest.json"
    manifest_path.write_text(
        json.dumps(
            build_workflow_manifest(
                config_path=config_path,
                run_dir=run_dir,
                run_profile_id=run_profile_id,
                run_profile_name=run_profile_name,
                steps=steps,
                cadence_path=cadence_path,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path