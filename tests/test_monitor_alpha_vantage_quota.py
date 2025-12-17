import json
from pathlib import Path

from src.scripts import monitor_alpha_vantage_quota as monitor

def test_monitor_handles_missing_file(monkeypatch, tmp_path, capsys):
    missing_path = tmp_path / "missing_usage.json"
    monkeypatch.setenv("ALPHA_VANTAGE_USAGE_PATH", str(missing_path))
    monkeypatch.delenv("ALPHA_VANTAGE_ALERT_THRESHOLD", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_ALERT_DATE", raising=False)

    exit_code = monitor.run_monitor()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Usage file not found at {missing_path}" in captured.out

def test_monitor_reports_summary_below_threshold(monkeypatch, tmp_path, capsys):
    usage_path = tmp_path / "usage.json"
    usage_data = {
        "2025-01-01": {
            "PRIMARY": {
                "calls": 120,
                "rate_limit_hits": 0,
                "last_updated": "2025-01-01T12:00:00+00:00",
            }
        }
    }
    usage_path.write_text(json.dumps(usage_data), encoding="utf-8")

    monkeypatch.setenv("ALPHA_VANTAGE_USAGE_PATH", str(usage_path))
    monkeypatch.setenv("ALPHA_VANTAGE_ALERT_THRESHOLD", "180")
    monkeypatch.setenv("ALPHA_VANTAGE_ALERT_DATE", "2025-01-01")

    exit_code = monitor.run_monitor()

    assert exit_code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip())
    assert summary["date"] == "2025-01-01"
    assert summary["threshold"] == 180.0
    assert summary["keys"]["PRIMARY"]["calls"] == 120.0
    assert summary["keys"]["PRIMARY"]["remaining"] == 60.0
    assert "All keys remain under threshold." in summary["message"]

def test_monitor_emits_alert_above_threshold(monkeypatch, tmp_path, capsys):
    usage_path = tmp_path / "usage.json"
    usage_data = {
        "2025-01-01": {
            "PRIMARY": {
                "calls": 220,
                "rate_limit_hits": 0,
                "last_updated": "2025-01-01T23:30:00+00:00",
            },
            "SECONDARY": {
                "calls": 90,
                "rate_limit_hits": 1,
                "last_updated": "2025-01-01T22:00:00+00:00",
            },
        }
    }
    usage_path.write_text(json.dumps(usage_data), encoding="utf-8")

    monkeypatch.setenv("ALPHA_VANTAGE_USAGE_PATH", str(usage_path))
    monkeypatch.setenv("ALPHA_VANTAGE_ALERT_THRESHOLD", "200")
    monkeypatch.setenv("ALPHA_VANTAGE_ALERT_DATE", "2025-01-01")

    exit_code = monitor.run_monitor()

    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    alert = json.loads(lines[0])
    assert alert["key"] == "PRIMARY"
    assert alert["calls"] == 220.0
    assert alert["threshold"] == 200.0
    assert alert["exceeded_by"] == 20.0
    assert alert["last_updated"] == "2025-01-01T23:30:00+00:00"
