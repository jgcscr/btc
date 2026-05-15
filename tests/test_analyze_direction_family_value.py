from __future__ import annotations

import pandas as pd

from src.scripts.analyze_direction_family_value import analyze_direction_family_value


def test_analyze_direction_family_value_reports_family_and_leave_one_out_metrics() -> None:
    frame = pd.DataFrame(
        {
            "ts": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
            ],
            "ret_1h": [0.01, -0.02, 0.015, -0.01],
            "y_true": [1, 0, 1, 0],
            "p_up": [0.7, 0.3, 0.8, 0.25],
            "p_up_xgb": [0.8, 0.25, 0.82, 0.2],
            "p_up_lgbm": [0.75, 0.35, 0.77, 0.3],
            "p_up_transformer": [0.62, 0.42, 0.7, 0.38],
            "p_up_regime_logit": [0.55, 0.45, 0.6, 0.4],
            "regime_state": ["trend_ignition", "chop", "trend_ignition", "chop"],
            "fold": [0, 0, 1, 1],
        }
    )

    summary = analyze_direction_family_value(frame, trade_band=0.05, fee_bps=3.0)

    assert summary["rows"] == 4
    assert summary["family_members"]["tree"] == ["p_up_lgbm", "p_up_xgb"]
    assert summary["family_members"]["attention"] == ["p_up_transformer"]
    assert summary["family_members"]["regime"] == ["p_up_regime_logit"]
    assert "leave_one_out" in summary["families"]["tree"]
    assert "delta_vs_component_baseline" in summary["families"]["tree"]["leave_one_out"]
    assert summary["families"]["tree"]["regime_stability"]["available"] is True
    assert summary["families"]["tree"]["leave_one_out"]["fold_summary"]["available"] is True
    assert summary["recommendations"]["most_incremental_family_by_brier"]["family"] in {"tree", "attention", "regime"}