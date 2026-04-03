from __future__ import annotations

import json

import pytest

from src.runtime.trade_decision_support import (
    apply_trade_decision_model,
    apply_trade_decision_stage,
    resolve_trade_decision_policy,
    resolve_trade_decision_threshold,
)


def _finite_float_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _finite_float(value, default):
    numeric = _finite_float_or_none(value)
    return default if numeric is None else numeric


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return float(value)


def _parse_horizon_label(value):
    return float(str(value).rstrip("h"))


def _format_horizon_label(value):
    return f"{int(value)}h" if float(value).is_integer() else f"{value}h"


def _sigmoid(value):
    return 1.0 / (1.0 + pow(2.718281828459045, -float(value)))


def _append_gate_trace(entry, *, stage, reason, triggered, blocking):
    entry.setdefault("gate_trace", []).append(
        {
            "stage": stage,
            "reason": reason,
            "triggered": triggered,
            "blocking": blocking,
        }
    )


def test_resolve_trade_decision_policy_loads_model_and_horizon_overrides(tmp_path) -> None:
    model_path = tmp_path / "trade_decision_model.json"
    model_path.write_text(
        json.dumps({"threshold": 0.61, "feature_columns": ["p_up"], "coefficients": [1.0], "intercept": 0.0}),
        encoding="utf-8",
    )
    messages: list[str] = []

    policy = resolve_trade_decision_policy(
        {
            "enabled": True,
            "model_path": str(model_path),
            "thresholds_by_horizon_regime": {"4": {"neutral": {"threshold": 0.7}}},
        },
        finite_float_or_none=_finite_float_or_none,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=messages.append,
    )

    assert policy["enabled"] is True
    assert policy["threshold"] == 0.61
    assert policy["thresholds_by_horizon_regime"] == {4.0: {"neutral": 0.7}}
    assert messages == []


def test_resolve_trade_decision_threshold_prefers_horizon_regime_override() -> None:
    threshold, source = resolve_trade_decision_threshold(
        {
            "threshold": 0.55,
            "thresholds_by_horizon_regime": {4.0: {"neutral": 0.67}},
        },
        horizon_label="4h",
        regime_state="neutral",
        normalize_horizon_value=_normalize_horizon_value,
        parse_horizon_label=_parse_horizon_label,
        format_horizon_label=_format_horizon_label,
    )

    assert threshold == 0.67
    assert source == "4h@neutral"


def test_apply_trade_decision_model_allows_raw_ev_fallback_when_no_positive_oof_bin() -> None:
    payload = apply_trade_decision_model(
        result={
            "p_up": 0.8,
            "raw_p_up": 0.8,
            "ret_pred": 0.02,
            "expected_value": 0.02,
            "signal_dir_only": 1,
            "volatility": {"snapshot": {}},
        },
        horizon_label="4h",
        regime_state="neutral",
        residual_std=0.1,
        policy={
            "enabled": True,
            "threshold": 0.5,
            "use_oof_expected_value": False,
            "replace_threshold_rule": True,
            "require_direction_ret_alignment": True,
            "enforce_positive_oof_envelope": True,
            "block_when_no_positive_oof_bin": True,
            "allow_raw_ev_fallback_when_no_positive_oof_bin": True,
            "raw_ev_fallback_quantile": 0.9,
            "raw_ev_fallback_min_edge_over_fee": 0.0,
            "positive_oof_min_samples": 1,
            "model": {
                "feature_columns": ["p_up"],
                "coefficients": [10.0],
                "intercept": -5.0,
                "oof_expected_value": {
                    "bins": [{"p_min": 0.0, "p_max": 1.0, "mean_ret_net": -0.01, "samples": 10}],
                },
                "raw_ev_fallback": {"quantiles": {"q90": 0.01}},
            },
        },
        fee_bps=0.0,
        slippage_bps=0.0,
        regime_trend="trend",
        regime_neutral="neutral",
        regime_chop="chop",
        resolve_trade_decision_threshold=lambda policy, **kwargs: resolve_trade_decision_threshold(
            policy,
            normalize_horizon_value=_normalize_horizon_value,
            parse_horizon_label=_parse_horizon_label,
            format_horizon_label=_format_horizon_label,
            **kwargs,
        ),
        sigmoid=_sigmoid,
        finite_float_or_none=_finite_float_or_none,
        finite_float=_finite_float,
    )

    assert payload["triggered"] is True
    assert payload["raw_ev_fallback_pass"] is True
    assert payload["raw_ev_fallback_threshold"] == 0.01
    assert payload["proposed_trade_action"] == "long"


def test_apply_trade_decision_stage_blocks_on_upstream_reasons() -> None:
    summary = {
        "4h": {
            "regime_state": "neutral",
            "signal_ensemble": 0,
            "trade_action": "hold",
            "forecast_coherence": {"triggered": True},
        }
    }

    result = apply_trade_decision_stage(
        summary,
        {"4h": {"residual_std": 0.2}},
        {"replace_threshold_rule": True},
        default_residual_std=0.1,
        regime_neutral="neutral",
        default_fee_bps=10.0,
        default_slippage_bps=5.0,
        apply_trade_decision_model=lambda **kwargs: {
            "enabled": True,
            "triggered": True,
            "proposed_signal_ensemble": 1,
            "proposed_trade_action": "long",
        },
        upstream_trade_gate_reasons=lambda entry: ["forecast_coherence_gate"],
        append_gate_trace=_append_gate_trace,
    )

    payload = result["4h"]["trade_decision"]
    assert payload["triggered"] is False
    assert payload["blocked"] is True
    assert payload["blocking_reason"] == "forecast_coherence_gate"
    assert result["4h"]["gate_trace"] == [
        {
            "stage": "trade_decision",
            "reason": "forecast_coherence_gate",
            "triggered": True,
            "blocking": True,
        }
    ]