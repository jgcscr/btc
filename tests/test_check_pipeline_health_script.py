from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.check_pipeline_health import (
    AlertOptions,
    ArtifactPolicy,
    MonitoringConfig,
    VendorStatus,
    run_check,
)


def _write_artifact(path: Path, timestamp: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_at": timestamp.isoformat()}), encoding="utf-8")


def test_run_check_emits_alert_payload_for_degraded_warning(tmp_path) -> None:
    artifact_root = tmp_path / "monitoring"
    artifact_root.mkdir()
    alpha_path = artifact_root / "alpha.json"
    stale_time = datetime.now(timezone.utc) - timedelta(hours=5)
    _write_artifact(alpha_path, stale_time)

    policy = ArtifactPolicy(
        name="alpha",
        path=str(alpha_path),
        staleness_hours=1.0,
        vendor_status=VendorStatus(state="degraded", reason="vendor outage"),
    )
    config = MonitoringConfig([policy])
    alert_path = tmp_path / "alert-warning.json"
    started_at = datetime.now(timezone.utc)

    exit_code = run_check(
        artifact_root=artifact_root,
        staleness_hours=1.0,
        missing_ratio_limit=0.01,
        config=config,
        alert=AlertOptions(emit_json=False, output_path=alert_path),
        run_metadata={"job_id": "job-123"},
        started_at=started_at,
    )

    assert exit_code == 0
    payload = json.loads(alert_path.read_text())
    assert payload["status"] == "warning"
    issue = payload["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["vendor_status"]["state"] == "degraded"
    run_meta = payload["run"]
    assert run_meta["job_id"] == "job-123"
    assert run_meta["duration_seconds"] >= 0.0


def test_run_check_marks_critical_failures_in_alert_payload(tmp_path) -> None:
    artifact_root = tmp_path / "monitoring-critical"
    artifact_root.mkdir()
    beta_path = artifact_root / "beta.json"
    stale_time = datetime.now(timezone.utc) - timedelta(hours=6)
    _write_artifact(beta_path, stale_time)

    policy = ArtifactPolicy(name="beta", path=str(beta_path), staleness_hours=1.0)
    config = MonitoringConfig([policy])
    alert_path = tmp_path / "alert-critical.json"

    exit_code = run_check(
        artifact_root=artifact_root,
        staleness_hours=1.0,
        missing_ratio_limit=0.01,
        config=config,
        alert=AlertOptions(emit_json=False, output_path=alert_path),
        run_metadata={"job_id": "critical-job"},
        started_at=datetime.now(timezone.utc),
    )

    assert exit_code == 1
    payload = json.loads(alert_path.read_text())
    assert payload["status"] == "critical"
    issue = payload["issues"][0]
    assert issue["severity"] == "critical"
    assert issue["artifact"] == "beta.json"
