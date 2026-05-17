from __future__ import annotations

import pandas as pd
import pytest

from src.runtime.prediction_pipeline import _apply_derivatives_shadow_probability_adjustment


def test_apply_derivatives_shadow_probability_adjustment_uses_futures_basis() -> None:
    adjusted_probability, payload = _apply_derivatives_shadow_probability_adjustment(
        probability=0.72,
        close=100.0,
        row_features=pd.Series({"fut_close": 101.0}),
        horizon_label="4h",
        regime_state="neutral",
        ret_pred=0.002,
        signal_dir_only=1,
        trade_decision_policy={
            "derivatives_shadow_adjustment": {
                "enabled": True,
                "mode": "futures_basis_crowding_penalty",
                "horizons": ["4h"],
                "regime_states": ["neutral"],
                "min_abs_basis_bps": 8.0,
                "max_abs_ret_pred": 0.01,
                "strength": 0.5,
            }
        },
        coerce_row_value=lambda value: None if value is None else float(value),
    )

    assert adjusted_probability < 0.72
    assert payload["applied"] is True
    assert payload["basis_bps"] == pytest.approx(100.0)
    assert payload["reason"] == "futures_basis_crowding_penalty"


def test_apply_derivatives_shadow_probability_adjustment_skips_when_unscoped() -> None:
    adjusted_probability, payload = _apply_derivatives_shadow_probability_adjustment(
        probability=0.72,
        close=100.0,
        row_features=pd.Series({"fut_close": 101.0}),
        horizon_label="1h",
        regime_state="trend_ignition",
        ret_pred=0.002,
        signal_dir_only=1,
        trade_decision_policy={
            "derivatives_shadow_adjustment": {
                "enabled": True,
                "mode": "futures_basis_crowding_penalty",
                "horizons": ["4h"],
                "regime_states": ["neutral"],
                "min_abs_basis_bps": 8.0,
                "max_abs_ret_pred": 0.01,
                "strength": 0.5,
            }
        },
        coerce_row_value=lambda value: None if value is None else float(value),
    )

    assert adjusted_probability == 0.72
    assert payload["applied"] is False
    assert payload["reason"] == "horizon_not_scoped"
