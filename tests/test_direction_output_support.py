from __future__ import annotations

from src.runtime.direction_output_support import (
    apply_probability_calibration,
    resolve_direction_output_policy,
    resolve_trade_probability_for_horizon,
)


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return float(value)


def _direction_from_ret_pred(value):
    if value is None:
        return "neutral"
    numeric = float(value)
    if numeric > 0:
        return "up"
    if numeric < 0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close, projected_price):
    if close is None or projected_price is None:
        return "neutral"
    current = float(close)
    projected = float(projected_price)
    if projected > current:
        return "up"
    if projected < current:
        return "down"
    return "neutral"


def _direction_from_probability(value, *, neutral_band=0.02):
    if value is None:
        return "neutral"
    numeric = float(value)
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def test_resolve_direction_output_policy_parses_weight_specs_and_horizon_maps() -> None:
    policy = resolve_direction_output_policy(
        {
            "enabled": True,
            "horizons": [1, "4"],
            "neutral_band": 0.02,
            "neutral_band_by_horizon": {"1": 0.03},
            "probability_shrinkage": {
                "enabled": True,
                "horizons": [1],
                "regimes": ["neutral"],
                "default_strength": 0.25,
                "strength_by_horizon": {"1": 0.4},
                "bypass_edge": 0.15,
            },
            "marginal_rerank": {
                "enabled": True,
                "horizons": [1],
                "lower": 0.49,
                "upper": 0.61,
                "min_component_count": 2,
                "weight_specs": {
                    "default": "gru:1.5,lstm:1.0,cnn_lstm:0.0",
                },
            },
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert policy["horizons"] == [1.0, 4.0]
    assert policy["neutral_band_by_horizon"] == {1.0: 0.03}
    assert policy["probability_shrinkage"]["strength_by_horizon"] == {1.0: 0.4}
    assert policy["marginal_rerank"]["enabled"] is True
    assert policy["marginal_rerank"]["weight_specs"] == {"default": {"gru": 1.5, "lstm": 1.0}}


def test_resolve_trade_probability_for_horizon_uses_base_calibration_when_regime_flips_consensus() -> None:
    probability, calibration_key, calibration_used_regime_key, guard_payload = resolve_trade_probability_for_horizon(
        platt_calibration={
            "1h": {"method": "isotonic", "x": [0.46, 0.46], "y": [0.46, 0.46]},
            "1h@trend_ignition": {"method": "isotonic", "x": [0.46, 0.46], "y": [0.52, 0.52]},
        },
        label="1h",
        regime_state="trend_ignition",
        raw_probability=0.46,
        close=100.0,
        projected_price=99.0,
        ret_pred=-0.01,
        neutral_band=0.02,
        regime_calibration_min_platt_slope=0.05,
        apply_probability_calibration=apply_probability_calibration,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
    )

    assert probability == 0.46
    assert calibration_key == "1h"
    assert calibration_used_regime_key is False
    assert guard_payload is not None
    assert guard_payload["applied"] is True
    assert guard_payload["fallback_source"] == "base_horizon_calibration"