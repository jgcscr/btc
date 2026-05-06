from __future__ import annotations

import json
from pathlib import Path

from src.runtime.reliability_manifest_support import (
    build_cadence_plan_payload,
    build_workflow_manifest,
    write_cadence_plan,
    write_workflow_manifest,
)
from src.runtime.reliability_workflow_common import StepResult


def test_build_cadence_plan_payload_uses_defaults(tmp_path: Path) -> None:
    payload = build_cadence_plan_payload(cadence_cfg={}, summary_dir=tmp_path)

    assert payload["monthly_retrain_day"] == 1
    assert payload["weekly_recalibration_weekday"] == "mon"
    assert payload["trigger_file"].endswith("reliability_triggers.json")


def test_write_workflow_manifest_serializes_steps(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    cadence_path = write_cadence_plan(cadence_cfg={"monthly_retrain_day": 5}, summary_dir=summary_dir)
    steps = [
        StepResult(
            name="quality_eval",
            command=["python", "-m", "quality"],
            returncode=0,
            log_path=summary_dir / "quality.log",
        )
    ]

    manifest = build_workflow_manifest(
        config_path="configs/reliability_workflow.runtime.yaml",
        run_dir=tmp_path / "run-1",
        run_profile_id="runtime",
        run_profile_name="Runtime",
        steps=steps,
        cadence_path=cadence_path,
    )

    assert manifest["profile"]["id"] == "runtime"
    assert manifest["steps"][0]["name"] == "quality_eval"

    manifest_path = write_workflow_manifest(
        summary_dir=summary_dir,
        config_path="configs/reliability_workflow.runtime.yaml",
        run_dir=tmp_path / "run-1",
        run_profile_id="runtime",
        run_profile_name="Runtime",
        steps=steps,
        cadence_path=cadence_path,
    )

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["cadence_plan"] == str(cadence_path)
    assert written["steps"][0]["command"] == ["python", "-m", "quality"]