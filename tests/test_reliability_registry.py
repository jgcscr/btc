from __future__ import annotations

import json

from src.runtime.reliability_registry import ReliabilityRunRegistry


def test_reliability_registry_records_latest_trustworthy_run(tmp_path) -> None:
    summary_dir = tmp_path / "artifacts" / "reliability" / "reliability-20260422T010000-abcd1234" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "edge_trustworthiness.json").write_text(
        json.dumps({"edge_trustworthy": True}),
        encoding="utf-8",
    )

    registry = ReliabilityRunRegistry(tmp_path / "artifacts")
    payload = registry.record_workflow_manifest(
        {
            "run_dir": str(tmp_path / "artifacts" / "reliability" / "reliability-20260422T010000-abcd1234"),
        }
    )

    assert payload is not None
    assert payload["run_id"] == "reliability-20260422T010000-abcd1234"
    assert registry.resolve_latest_trustworthy_run_id(allow_scan_fallback=False) == payload["run_id"]


def test_reliability_registry_falls_back_to_scan_when_registry_missing(tmp_path) -> None:
    summary_dir = tmp_path / "artifacts" / "reliability" / "reliability-20260422T020000-efgh5678" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "edge_trustworthiness.json").write_text(
        json.dumps({"edge_trustworthy": True}),
        encoding="utf-8",
    )
    (summary_dir / "calibrated_thresholds.json").write_text("{}", encoding="utf-8")
    (summary_dir / "platt_calibration.json").write_text("{}", encoding="utf-8")

    registry = ReliabilityRunRegistry(tmp_path / "artifacts")

    assert registry.resolve_latest_trustworthy_run_id() == "reliability-20260422T020000-efgh5678"