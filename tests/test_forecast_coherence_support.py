from __future__ import annotations

import pytest

from src.runtime.forecast_coherence_support import (
    apply_forecast_coherence_policy,
    coherence_weight_multiplier,
    forecast_coherence_excluded,
    resolve_forecast_coherence_policy,
)


def _coerce_result_horizon(value):
    return None if value is None else float(value)


def _direction_vote(entry):
    return str(entry.get("direction") or "neutral")


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
    projected = float(projected_price)
    current = float(close)
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


def _finite_float_or_none(value):
    return None if value is None else float(value)


def _append_gate_trace(entry, *, stage, reason, triggered, blocking):
    entry.setdefault("gate_trace", []).append(
        {
            "stage": stage,
            "reason": reason,
            "triggered": triggered,
            "blocking": blocking,
        }
    )


def test_resolve_forecast_coherence_policy_normalizes_horizons_and_defaults() -> None:
    policy = resolve_forecast_coherence_policy(
        {
            "enabled": True,
            "horizons": ["4", 1],
            "consensus_relief_horizons": ["8", 4],
            "p_up_neutral_band": 0.03,
            "min_p_up_edge": 0.07,
            "exclude_blocked_horizons_from_voting": False,
        },
        normalize_horizon_value=lambda value: float(value),
    )

    assert policy["enabled"] is True
    assert policy["horizons"] == [1.0, 4.0]
    assert policy["consensus_relief_horizons"] == [4.0, 8.0]
    assert policy["p_up_neutral_band"] == 0.03
    assert policy["min_p_up_edge"] == 0.07
    assert policy["exclude_blocked_horizons_from_voting"] is False
    assert policy["block_on_direction_ret_mismatch"] is True


def test_apply_forecast_coherence_policy_blocks_mismatched_entry() -> None:
    summary = {
        "4h": {
            "horizon_hours": 4,
            "direction": "up",
            "ret_pred": -0.03,
            "close": 100.0,
            "projected_price": 98.0,
            "p_up": 0.30,
            "trade_action": "long",
            "signal_ensemble": 1,
            "direction_next_display": "up",
            "direction_output": {"direction": "up"},
            "trade_decision": {"triggered": True},
        }
    }

    result = apply_forecast_coherence_policy(
        summary,
        {
            "enabled": True,
            "horizons": [4.0],
            "min_p_up_edge": 0.05,
            "min_abs_ret_pred": 0.01,
        },
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
        append_gate_trace=_append_gate_trace,
    )

    entry = result["4h"]
    coherence = entry["forecast_coherence"]
    assert coherence["triggered"] is True
    assert coherence["reasons"] == [
        "direction_ret_mismatch",
        "direction_projected_price_mismatch",
    ]
    assert coherence["exclude_from_voting"] is True
    assert forecast_coherence_excluded(entry) is True
    assert entry["trade_action"] == "hold"
    assert entry["signal_ensemble"] == 0
    assert entry["direction_next_display"] == "neutral"
    assert entry["direction_output"]["direction"] == "neutral"
    assert entry["trade_decision"]["blocked"] is True
    assert entry["trade_decision"]["blocking_reason"] == "forecast_coherence_gate"
    assert entry["gate_trace"] == [
        {
            "stage": "forecast_coherence",
            "reason": "direction_ret_mismatch|direction_projected_price_mismatch",
            "triggered": True,
            "blocking": True,
        }
    ]


def test_apply_forecast_coherence_policy_marks_low_trust_on_low_edge_mismatch() -> None:
    summary = {
        "1h": {
            "horizon_hours": 1,
            "direction": "up",
            "ret_pred": 0.02,
            "close": 100.0,
            "projected_price": 101.0,
            "p_up": 0.48,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "trade_decision": {},
        }
    }

    result = apply_forecast_coherence_policy(
        summary,
        {
            "enabled": True,
            "horizons": [1.0],
            "p_up_neutral_band": 0.01,
            "min_p_up_edge": 0.05,
            "min_abs_ret_pred": 0.01,
        },
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
        append_gate_trace=_append_gate_trace,
    )

    entry = result["1h"]
    coherence = entry["forecast_coherence"]
    assert coherence["triggered"] is False
    assert coherence["low_trust"] is True
    assert coherence["advisory_reasons"] == ["low_edge_p_up_ret_mismatch"]
    assert coherence["exclude_from_voting"] is True
    assert entry["trade_decision"]["forecast_coherence_low_trust"] is True
    assert entry["trade_decision"]["forecast_coherence_low_trust_reasons"] == ["low_edge_p_up_ret_mismatch"]
    assert entry["gate_trace"] == [
        {
            "stage": "forecast_coherence",
            "reason": "low_edge_p_up_ret_mismatch",
            "triggered": True,
            "blocking": False,
        }
    ]


def test_apply_forecast_coherence_policy_applies_consensus_relief_without_exclusion() -> None:
    summary = {
        "8h": {
            "horizon_hours": 8,
            "direction": "up",
            "ret_pred": 0.03,
            "close": 100.0,
            "projected_price": 102.0,
            "p_up": 0.42,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "trade_decision": {},
        }
    }

    result = apply_forecast_coherence_policy(
        summary,
        {
            "enabled": True,
            "horizons": [8.0],
            "min_p_up_edge": 0.05,
            "min_abs_ret_pred": 0.01,
            "allow_consensus_p_up_ret_relief": True,
            "consensus_relief_horizons": [8.0],
            "consensus_relief_max_p_up_edge": 0.1,
            "exclude_blocked_horizons_from_voting": True,
            "consensus_relief_exclude_from_voting": False,
        },
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
        append_gate_trace=_append_gate_trace,
    )

    entry = result["8h"]
    coherence = entry["forecast_coherence"]
    assert coherence["triggered"] is False
    assert coherence["consensus_relief_applied"] is True
    assert coherence["advisory_reasons"] == ["consensus_p_up_ret_mismatch_relief"]
    assert coherence["exclude_from_voting"] is False
    assert forecast_coherence_excluded(entry) is False
    assert entry["trade_decision"]["forecast_coherence_low_trust"] is True


def test_coherence_weight_multiplier_applies_low_trust_penalty_and_conflict_penalty() -> None:
    multiplier = coherence_weight_multiplier(
        {
            "ret_pred": 0.03,
            "close": 100.0,
            "projected_price": 103.0,
            "p_up": 0.35,
            "forecast_coherence": {
                "low_trust": True,
                "ret_pred_side": "up",
                "projected_price_side": "up",
                "p_up_side": "down",
            },
        },
        horizon=4.0,
        policy={
            "coherence_weighting": {
                "enabled": True,
                "by_horizon": {"4.0": 1.0},
                "min_multiplier": 0.1,
                "low_trust_penalty": 0.25,
                "p_up_conflict_penalty": 0.2,
                "consensus_bonus": 0.1,
            }
        },
        lookup_horizon_value=lambda mapping, horizon, default: mapping.get(f"{horizon:.1f}", default),
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
    )

    assert multiplier == pytest.approx(0.6)