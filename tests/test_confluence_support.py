from __future__ import annotations

from src.runtime.confluence_support import apply_confluence_policy, resolve_confluence_policy


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return float(value)


def _coerce_result_horizon(value):
    return None if value is None else float(value)


def _direction_vote(entry):
    return str(entry.get("direction_next_display") or entry.get("direction") or "neutral")


def _append_gate_trace(entry, *, stage, reason, triggered, blocking):
    entry.setdefault("gate_trace", []).append(
        {
            "stage": stage,
            "reason": reason,
            "triggered": triggered,
            "blocking": blocking,
        }
    )


def test_resolve_confluence_policy_preserves_horizon_specific_overrides() -> None:
    policy = resolve_confluence_policy(
        {
            "enabled": True,
            "min_support_ratio": 0.66,
            "min_support_ratio_by_horizon": {"4": 1.0},
            "min_aligned_horizons_by_horizon": {"4": 3},
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert policy["min_support_ratio"] == 0.66
    assert policy["min_support_ratio_by_horizon"] == {4.0: 1.0}
    assert policy["min_aligned_horizons_by_horizon"] == {4.0: 3.0}


def test_apply_confluence_policy_blocks_insufficient_support() -> None:
    summary = {
        "4h": {
            "horizon_hours": 4.0,
            "direction_next_display": "up",
            "trade_action": "long",
            "signal_ensemble": 1,
        },
        "8h": {
            "horizon_hours": 8.0,
            "direction_next_display": "down",
            "trade_action": "short",
            "signal_ensemble": 1,
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next_display": "down",
            "trade_action": "short",
            "signal_ensemble": 1,
        },
    }
    policy = resolve_confluence_policy(
        {
            "enabled": True,
            "short_horizons": [4.0],
            "mid_horizons": [8.0, 12.0],
            "min_support_ratio": 0.66,
            "min_support_ratio_by_horizon": {"4": 1.0},
            "min_aligned_horizons": 2,
            "min_aligned_horizons_by_horizon": {"4": 3},
            "require_mid_term_alignment": True,
            "min_mid_term_ratio": 0.66,
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    updated = apply_confluence_policy(
        summary,
        policy,
        forecast_coherence_excluded=lambda entry: False,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        lookup_horizon_value=lambda mapping, horizon, default: mapping.get(horizon, default),
        append_gate_trace=_append_gate_trace,
    )

    assert updated["4h"]["trade_action"] == "hold"
    assert updated["4h"]["confluence"]["triggered"] is True
    assert "aligned_horizons_below_min" in updated["4h"]["confluence"]["reasons"]
    assert "support_ratio_below_min" in updated["4h"]["confluence"]["reasons"]
    assert updated["8h"]["trade_action"] == "short"
