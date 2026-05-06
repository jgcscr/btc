from __future__ import annotations

from src.runtime.forecast_direction_support import (
    coerce_result_horizon,
    derive_probability_alignment_features,
    direction_from_probability,
    direction_from_projected_price,
    direction_from_ret_pred,
    direction_vote,
    resolve_direction_signal_for_horizon,
)


def test_direction_helpers_cover_neutral_and_polarity_cases() -> None:
    assert coerce_result_horizon("4") == 4.0
    assert coerce_result_horizon("bad") is None
    assert coerce_result_horizon("2", normalize_horizon_value=lambda value: round(float(value), 1)) == 2.0
    assert direction_vote({"direction_next": "up"}) == "up"
    assert direction_from_ret_pred(0.1) == "up"
    assert direction_from_ret_pred(-0.1) == "down"
    assert direction_from_ret_pred(None) == "neutral"
    assert direction_from_projected_price(100.0, 101.0) == "up"
    assert direction_from_projected_price(100.0, 99.0) == "down"
    assert direction_from_projected_price(0.0, 99.0) == "neutral"
    assert direction_from_probability(0.6, neutral_band=0.02) == "up"
    assert direction_from_probability(0.4, neutral_band=0.02) == "down"
    assert direction_from_probability(0.51, neutral_band=0.02) == "neutral"


def test_resolve_direction_signal_prefers_forecast_consensus_when_calibrated_side_disagrees() -> None:
    signal = resolve_direction_signal_for_horizon(
        raw_probability=0.49,
        calibrated_probability=0.48,
        threshold=0.55,
        close=100.0,
        projected_price=101.0,
        ret_pred=0.02,
        calibration_key="1h",
        calibration_used_regime_key=False,
    )

    assert signal == 1


def test_derive_probability_alignment_features_reports_consensus_and_mismatches() -> None:
    payload = derive_probability_alignment_features(
        close=100.0,
        projected_price=102.0,
        ret_pred=0.03,
        raw_probability=0.35,
        resolved_probability=0.7,
        direction="up",
        neutral_band=0.02,
        probability_guard={"applied": True},
        calibration_used_regime_key=True,
    )

    assert payload["forecast_consensus_side"] == "up"
    assert payload["raw_p_up_side"] == "down"
    assert payload["resolved_p_up_side"] == "up"
    assert payload["raw_p_up_direction_mismatch"] == 1.0
    assert payload["probability_calibration_guard_applied"] == 1.0
    assert payload["probability_calibration_used_regime_key"] == 1.0