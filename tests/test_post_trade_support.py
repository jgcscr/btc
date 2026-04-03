from __future__ import annotations

from src.runtime.post_trade_support import (
    apply_abstention_policy,
    apply_post_trade_gates,
    apply_uncertainty_abstention,
    resolve_abstention_expected_value,
    resolve_abstention_policy,
    resolve_abstention_policy_for_horizon,
    resolve_uncertainty_policy,
    resolve_uncertainty_settings,
)


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return float(value)


def _coerce_result_horizon(value):
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


def test_resolve_abstention_policy_for_horizon_prefers_regime_override() -> None:
    policy = resolve_abstention_policy(
        {
            "enabled": True,
            "min_confidence": 0.2,
            "thresholds_by_horizon_regime": {
                "8": {
                    "neutral": {
                        "min_confidence": 0.55,
                        "min_abs_expected_value": 0.01,
                    }
                }
            },
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    scoped = resolve_abstention_policy_for_horizon(
        policy,
        horizon=8.0,
        regime_state="neutral",
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert scoped["min_confidence"] == 0.55
    assert scoped["min_abs_expected_value"] == 0.01


def test_apply_post_trade_gates_uses_trade_decision_expected_net_for_abstention() -> None:
    summary = {
        "4h": {
            "horizon_hours": 4.0,
            "regime_state": "neutral",
            "trade_action": "long",
            "signal_ensemble": 1,
            "confidence_score": 0.9,
            "confidence_min": 0.2,
            "p_up": 0.8,
            "expected_value": 0.02,
            "trade_decision": {"expected_net": 0.005, "expected_net_valid": True},
            "p_up_components": {"a": 0.8, "b": 0.82, "c": 0.79},
        }
    }
    abstention_policy = resolve_abstention_policy(
        {"enabled": True, "min_abs_expected_value": 0.01},
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )
    uncertainty_policy = resolve_uncertainty_policy(
        {"enabled": False},
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    result = apply_post_trade_gates(
        summary,
        confidence_min=0.2,
        abstention_policy=abstention_policy,
        uncertainty_policy=uncertainty_policy,
        default_fee_bps=10.0,
        default_slippage_bps=5.0,
        regime_neutral="neutral",
        append_gate_trace=_append_gate_trace,
        resolve_abstention_expected_value=resolve_abstention_expected_value,
        resolve_abstention_policy_for_horizon=lambda policy, horizon, regime_state: resolve_abstention_policy_for_horizon(
            policy,
            horizon=horizon,
            regime_state=regime_state,
            normalize_horizon_value=_normalize_horizon_value,
        ),
        apply_abstention_policy=apply_abstention_policy,
        apply_uncertainty_abstention=lambda **kwargs: apply_uncertainty_abstention(
            **kwargs,
            resolve_uncertainty_settings=lambda policy, horizon, regime_state: resolve_uncertainty_settings(
                policy,
                horizon=horizon,
                regime_state=regime_state,
                normalize_horizon_value=_normalize_horizon_value,
            ),
        ),
        coerce_result_horizon=_coerce_result_horizon,
    )

    entry = result["4h"]
    assert entry["trade_action"] == "hold"
    assert entry["signal_ensemble"] == 0
    assert entry["abstention"]["triggered"] is True
    assert entry["abstention"]["reason"] == "expected_value_below_abs_floor"
    assert entry["abstention"]["expected_value_used"] == 0.005
    assert entry["abstention"]["expected_value_source"] == "trade_decision_expected_net"


def test_apply_post_trade_gates_blocks_on_uncertainty_interval() -> None:
    summary = {
        "8h": {
            "horizon_hours": 8.0,
            "regime_state": "neutral",
            "trade_action": "long",
            "signal_ensemble": 1,
            "confidence_score": 0.9,
            "p_up": 0.7,
            "expected_value": 0.03,
            "p_up_components": {"a": 0.1, "b": 0.9, "c": 0.85},
        }
    }
    abstention_policy = resolve_abstention_policy(
        {"enabled": False},
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )
    uncertainty_policy = resolve_uncertainty_policy(
        {
            "enabled": True,
            "alpha": 0.2,
            "hold_prob_center": 0.5,
            "max_interval_width": 0.5,
            "require_center_cross": False,
            "thresholds_by_horizon_regime": {"8": {"neutral": {"max_interval_width": 0.6}}},
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    result = apply_post_trade_gates(
        summary,
        confidence_min=0.2,
        abstention_policy=abstention_policy,
        uncertainty_policy=uncertainty_policy,
        default_fee_bps=10.0,
        default_slippage_bps=5.0,
        regime_neutral="neutral",
        append_gate_trace=_append_gate_trace,
        resolve_abstention_expected_value=resolve_abstention_expected_value,
        resolve_abstention_policy_for_horizon=lambda policy, horizon, regime_state: resolve_abstention_policy_for_horizon(
            policy,
            horizon=horizon,
            regime_state=regime_state,
            normalize_horizon_value=_normalize_horizon_value,
        ),
        apply_abstention_policy=apply_abstention_policy,
        apply_uncertainty_abstention=lambda **kwargs: apply_uncertainty_abstention(
            **kwargs,
            resolve_uncertainty_settings=lambda policy, horizon, regime_state: resolve_uncertainty_settings(
                policy,
                horizon=horizon,
                regime_state=regime_state,
                normalize_horizon_value=_normalize_horizon_value,
            ),
        ),
        coerce_result_horizon=_coerce_result_horizon,
    )

    entry = result["8h"]
    assert entry["trade_action"] == "hold"
    assert entry["signal_ensemble"] == 0
    assert entry["abstention"]["reason"] == "uncertainty_interval_too_wide"
    assert entry["uncertainty"]["available"] is True
    assert entry["uncertainty"]["effective_policy"]["max_interval_width"] == 0.6
