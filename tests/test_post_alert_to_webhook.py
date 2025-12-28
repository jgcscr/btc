from __future__ import annotations

import json
import sys

from src.scripts import post_alert_to_webhook


def test_build_message_includes_job_and_issue_details() -> None:
    payload = {
        "status": "critical",
        "run": {"job_id": "job-77", "duration_seconds": 3.5},
        "issues": [
            {"artifact": "alpha.json", "severity": "critical", "issues": ["stale"]},
            {"artifact": "beta.json", "severity": "warning", "issues": ["vendor"]},
        ],
    }

    message = post_alert_to_webhook._build_message(payload)

    assert "[CRITICAL] pipeline health" in message
    assert "job=job-77" in message
    assert "duration=3.50s" in message
    assert "CRITICAL alpha.json" in message


def test_main_honors_dry_run(monkeypatch, tmp_path, capsys) -> None:
    payload = {
        "status": "warning",
        "issues": [{"artifact": "alpha.json", "severity": "warning", "issues": ["stale"]}],
    }
    alert_path = tmp_path / "alert.json"
    alert_path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "post_alert_to_webhook.py",
            "--alert-path",
            str(alert_path),
            "--webhook-url",
            "https://example.com/webhook",
            "--dry-run",
        ],
    )

    exit_code = post_alert_to_webhook.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "\"text\"" in out
