from __future__ import annotations

import json
import pandas as pd

from src.scripts.monitor_live_signals import build_summary, compute_metric_summary, take_window


def _baseline_stats(mean: float = 0.0, std: float = 1.0) -> dict[str, object]:
    return {
        "mean": mean,
        "std": std,
        "min": -1.0,
        "max": 1.0,
        "quantiles": {
            "0.10": -0.8,
            "0.25": -0.5,
            "0.50": 0.0,
            "0.75": 0.5,
            "0.90": 0.8,
        },
    }


def test_take_window_returns_tail() -> None:
    df = pd.DataFrame({"value": range(5)})
    window = take_window(df, 3)
    assert len(window) == 3
    assert list(window["value"]) == [2, 3, 4]


def test_compute_metric_summary_flags_alert() -> None:
    series = pd.Series([10.0, 11.0, 12.0])
    stats = compute_metric_summary("p_up", series, _baseline_stats(), alert_threshold=2.0)
    assert stats["alert"] is True
    assert stats["z_score"] > 2.0


def test_build_summary_marks_missing_columns() -> None:
    df = pd.DataFrame({"p_up": [0.4, 0.5]})
    baseline = {
        "row_count": 2,
        "columns": {
            "p_up": _baseline_stats(0.5, 0.1),
            "signal_ensemble": _baseline_stats(0.5, 0.1),
        },
    }
    summary = build_summary(df, baseline, alert_threshold=2.0)
    assert summary["metrics"]["signal_ensemble"]["error"] == "missing_column"
    assert summary["metrics"]["p_up"]["alert"] is False
    assert summary["alerts"] == []
