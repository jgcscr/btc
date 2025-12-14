import json
from pathlib import Path

import pytest

from src.scripts import metrics_diff


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def test_metrics_diff_within_tolerance(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"

    baseline_payload = [
        {
            "window": "w1",
            "hit_rate": 0.50,
            "cum_ret": 0.10,
            "max_drawdown": 0.020,
            "n_trades": 100,
            "sharpe_like": 0.30,
        }
    ]
    new_payload = [
        {
            "window": "w1",
            "hit_rate": 0.51,
            "cum_ret": 0.14,
            "max_drawdown": 0.025,
            "n_trades": 103,
            "sharpe_like": 0.32,
        }
    ]

    _write_json(baseline_path, baseline_payload)
    _write_json(new_path, new_payload)

    baseline_entries = metrics_diff.load_entries(baseline_path)
    new_entries = metrics_diff.load_entries(new_path)

    index_field = metrics_diff.detect_index_field(baseline_entries)
    baseline_map = metrics_diff.build_mapping(baseline_entries, index_field)
    new_map = metrics_diff.build_mapping(new_entries, index_field)

    violations, missing, extra = metrics_diff.compare_metrics(baseline_map, new_map)

    assert violations == []
    assert missing == []
    assert extra == []


def test_metrics_diff_flags_violation(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"

    baseline_payload = [
        {
            "window": "w1",
            "hit_rate": 0.60,
            "cum_ret": 0.20,
            "max_drawdown": 0.010,
            "n_trades": 80,
            "sharpe_like": 0.40,
        }
    ]
    new_payload = [
        {
            "window": "w1",
            "hit_rate": 0.56,
            "cum_ret": 0.24,
            "max_drawdown": 0.030,
            "n_trades": 90,
            "sharpe_like": 0.36,
        }
    ]

    _write_json(baseline_path, baseline_payload)
    _write_json(new_path, new_payload)

    index_field = "window"
    baseline_map = metrics_diff.build_mapping(metrics_diff.load_entries(baseline_path), index_field)
    new_map = metrics_diff.build_mapping(metrics_diff.load_entries(new_path), index_field)

    violations, missing, extra = metrics_diff.compare_metrics(baseline_map, new_map)

    assert missing == []
    assert extra == []
    assert any(v.metric == "hit_rate" for v in violations)
    assert any(v.metric == "max_drawdown" for v in violations)


def test_metrics_diff_update_overwrites_baseline(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    new_path = tmp_path / "new.json"

    baseline_payload = [{"window": "w1", "hit_rate": 0.5}]
    new_payload = [{"window": "w1", "hit_rate": 0.52}]

    _write_json(baseline_path, baseline_payload)
    _write_json(new_path, new_payload)

    exit_code = metrics_diff.main([
        "--baseline",
        str(baseline_path),
        "--new",
        str(new_path),
        "--update",
    ])

    assert exit_code == 0
    updated = json.loads(baseline_path.read_text())
    assert updated == new_payload
