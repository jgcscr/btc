import json

from src.runtime.summary_support import (
    apply_prompt_trust_degradation,
    apply_short_term_downtrend_fail_safe,
    build_stub_summary,
    build_prompt_forecast_clause,
    build_execution_prior_summary,
    build_blocked_trade_analytics,
    build_degradation_monitoring,
    build_prompt_ready_summary,
    build_runtime_degradation_monitoring,
    build_runtime_prompt_ready_summary,
    coerce_result_horizon,
    confidence_level_from_score,
    finite_float_or_none,
    format_usd_value,
    prompt_direction_label,
    prompt_effective_direction,
    prompt_entry_vetoed_for_preferred_horizon,
    resolve_degradation_monitoring_policy,
    select_prompt_candidate_entries,
    select_prompt_preferred_entry,
    suppress_long_bias_for_short_term_downtrend,
    write_prediction_summary,
)
from src.runtime.horizon_support import horizon_sort_key


def test_build_stub_summary_preserves_thresholds_and_dry_run_defaults():
    summary = build_stub_summary(
        [4.0, 1.0],
        0.61,
        0.02,
        close=100000.0,
        ts_iso="2026-04-02T00:00:00+00:00",
        thresholds_by_horizon={4.0: {"p_up_min": 0.7, "ret_min": 0.03, "volatility_ceiling": 1.5}},
        normalize_horizon_value=lambda value: float(value),
        format_horizon_label=lambda value: f"{int(value)}h",
        resolve_thresholds_for_horizon=lambda horizon, default_p_up, default_ret, overrides: dict(
            (overrides or {}).get(horizon, {"p_up_min": default_p_up, "ret_min": default_ret})
        ),
        confidence_min_default=0.0,
        regime_neutral="neutral",
    )

    assert list(summary.keys()) == ["1h", "4h"]
    assert summary["1h"]["execution_plan"]["status"] == "dry_run"
    assert summary["1h"]["thresholds"]["p_up_min_effective"] == 0.61
    assert summary["1h"]["thresholds"]["ret_min_effective"] == 0.02
    assert summary["4h"]["volatility"]["ceiling"] == 1.5
    assert summary["4h"]["thresholds"]["adaptive_scale"] == 1.0
    assert summary["4h"]["direction_only_fallback"]["reason"] == "dry_run"


def _select_prompt_preferred_entry(summary):
    return "4h", summary["4h"], {"conflict_present": True, "support_horizons": ["1h", "4h"]}


def _horizon_sort_key(label):
    return {"1h": 1, "4h": 4, "8h": 8}.get(label, 99)


def _build_prompt_forecast_clause(label, entry):
    return f"{label}:{entry['trade_action']}"


def _prompt_effective_direction(entry):
    direction_display = str(entry.get("direction_next_display") or "neutral").lower()
    if entry.get("forecast_coherence", {}).get("triggered") and direction_display == "neutral":
        return str(entry.get("direction_next") or "neutral").lower()
    return direction_display


def _prompt_direction_label(direction):
    return {"up": "Long", "down": "Short"}.get(direction, "Neutral")


def _confidence_level_from_score(score):
    if float(score or 0.0) >= 0.7:
        return "High"
    return "Medium"


def _finite_float_or_none(value):
    return None if value is None else float(value)


def _format_usd_value(value):
    if value is None:
        return None
    return f"${float(value):,.2f}"


def _resolve_degradation_monitoring_policy(policy):
    resolved = {
        "enabled": True,
        "lookback_snapshots": 5,
        "min_snapshots": 2,
        "min_ready_ratio": 0.4,
        "max_blocked_ratio": 0.8,
        "min_confidence": 0.5,
        "min_expected_net": 0.0,
        "min_directional_samples": 2,
        "max_long_wrong_ratio": 0.65,
        "max_long_wrong_streak": 2,
    }
    if policy:
        resolved.update(policy)
    return resolved


def test_build_prompt_ready_summary_returns_compact_operator_blockers():
    summary = {
        "1h": {
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.62,
            "execution_plan": {
                "status": "waiting_pullback",
                "reason": "await_pullback_entry_zone",
                "pending_trade_action": "buy",
            },
            "forecast_coherence": {"triggered": False},
        },
        "4h": {
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.81,
            "entry_price": 100000,
            "stop_loss": 99000,
            "take_profit": 102500,
            "risk_reward_ratio": 2.5,
            "execution_plan": {
                "status": "ready",
                "reason": "pass",
                "pending_trade_action": "buy",
                "disagreement_severity": {"score": 0.12, "triggered": False},
            },
            "forecast_coherence": {"triggered": False},
        },
    }

    result = build_prompt_ready_summary(
        summary,
        select_prompt_preferred_entry=_select_prompt_preferred_entry,
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )

    assert result["market_outlook_strategy"]["selected_direction"] == "Long"
    assert result["market_outlook_strategy"]["preferred_horizon"] == "4h"
    assert result["operator_summary_compact"]["recommended_operator_action"] == "enter_now"
    assert result["trade_execution_plan_usd"]["entry_point"] == 100000.0
    assert "wins side arbitration" in result["analysis_summary"]["rationale"]


def test_extracted_prompt_formatting_helpers_preserve_legacy_behavior():
    entry = {
        "direction_next_display": "neutral",
        "direction_next": "up",
        "projected_high": 101250,
        "projected_low": 99500,
        "forecast_coherence": {"triggered": True},
        "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate"},
    }

    assert confidence_level_from_score(0.7, finite_float_or_none=_finite_float_or_none) == "High"
    assert prompt_direction_label("down") == "Short"
    assert format_usd_value(100000, finite_float_or_none=_finite_float_or_none) == "$100,000.00"
    assert prompt_effective_direction(entry) == "up"
    assert build_prompt_forecast_clause("4h", entry, finite_float_or_none=_finite_float_or_none) == (
        "4h: up, projected range $99,500.00 to $101,250.00 (coherence blocked)"
    )


def test_runtime_summary_wrappers_use_builtin_helpers() -> None:
    summary = {
        "1h": {
            "horizon_hours": 1.0,
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.61,
            "execution_plan": {
                "status": "ready",
                "reason": "pass",
                "pending_trade_action": "buy",
                "confluence_tier": "high",
                "execution_alignment_ratio": 0.8,
                "bias_alignment_ratio": 0.75,
                "execution_score": 0.8,
                "bias_score": 0.7,
            },
            "forecast_coherence": {"triggered": False},
        },
        "4h": {
            "horizon_hours": 4.0,
            "trade_action": "hold",
            "direction_next_display": "neutral",
            "confidence_score": 0.4,
            "execution_plan": {
                "status": "rejected",
                "reason": "forecast_coherence_gate",
                "confluence_tier": "low",
                "execution_alignment_ratio": 0.2,
                "bias_alignment_ratio": 0.2,
                "execution_score": 0.2,
                "bias_score": 0.2,
            },
            "forecast_coherence": {"triggered": True, "reasons": ["forecast_coherence_gate"]},
            "trade_decision": {"expected_net": -1.0},
        },
    }

    prompt_payload = build_runtime_prompt_ready_summary(summary, horizon_sort_key=horizon_sort_key)
    degradation_payload = build_runtime_degradation_monitoring(
        [{"predictions": summary}],
        {"enabled": True, "min_snapshots": 1},
        horizon_sort_key=horizon_sort_key,
    )

    assert prompt_payload["market_outlook_strategy"]["preferred_horizon"] == "1h"
    assert degradation_payload["enabled"] is True
    assert degradation_payload["by_horizon"]["1h"]["samples"] == 1


def test_prompt_entry_vetoed_for_preferred_horizon_blocks_flip_divergence_and_hard_reasons() -> None:
    blocked_entry = {
        "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate"},
        "forecast_coherence": {"triggered": True},
    }
    flip_entry = {
        "execution_plan": {"status": "rejected", "reason": "low_execution_confluence"},
        "raw_p_up": 0.38,
        "p_up": 0.79,
    }

    assert prompt_entry_vetoed_for_preferred_horizon(blocked_entry, finite_float_or_none=_finite_float_or_none) is True
    assert prompt_entry_vetoed_for_preferred_horizon(flip_entry, finite_float_or_none=_finite_float_or_none) is True


def test_select_prompt_preferred_entry_suppresses_long_bias_when_short_term_stack_is_weak() -> None:
    summary = {
        "15m": {
            "horizon_hours": 0.25,
            "direction_next_display": "down",
            "close": 100.0,
            "projected_price": 99.0,
            "confidence_score": 0.8,
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "confluence_tier": "low"},
            "forecast_coherence": {"triggered": False},
        },
        "1h": {
            "horizon_hours": 1.0,
            "direction_next_display": "neutral",
            "close": 100.0,
            "projected_price": 99.2,
            "confidence_score": 0.4,
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "confluence_tier": "low"},
            "forecast_coherence": {"triggered": False},
        },
        "4h": {
            "horizon_hours": 4.0,
            "direction_next_display": "up",
            "raw_p_up": 0.41,
            "p_up": 0.68,
            "close": 100.0,
            "projected_low": 98.5,
            "confidence_score": 0.7,
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "confluence_tier": "medium"},
            "forecast_coherence": {"triggered": False},
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next_display": "up",
            "raw_p_up": 0.39,
            "p_up": 0.82,
            "close": 100.0,
            "projected_low": 97.5,
            "confidence_score": 0.8,
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "confluence_tier": "medium"},
            "forecast_coherence": {"triggered": False},
        },
    }

    assert suppress_long_bias_for_short_term_downtrend(summary, finite_float_or_none=_finite_float_or_none) is True
    preferred_label, preferred_entry, side_profile = select_prompt_preferred_entry(
        summary,
        coerce_result_horizon=coerce_result_horizon,
        finite_float_or_none=_finite_float_or_none,
    )

    assert preferred_label == "15m"
    assert preferred_entry is not None
    assert prompt_effective_direction(preferred_entry) == "down"
    assert side_profile is not None


def test_apply_short_term_downtrend_fail_safe_blocks_long_entries() -> None:
    summary = {
        "15m": {
            "horizon_hours": 0.25,
            "direction_next_display": "down",
            "close": 100.0,
            "projected_price": 99.0,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "side": "short"},
            "gate_trace": [],
        },
        "1h": {
            "horizon_hours": 1.0,
            "direction_next_display": "neutral",
            "close": 100.0,
            "projected_price": 99.2,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "side": "short"},
            "gate_trace": [],
        },
        "4h": {
            "horizon_hours": 4.0,
            "direction_next_display": "up",
            "raw_p_up": 0.41,
            "p_up": 0.68,
            "close": 100.0,
            "projected_low": 98.5,
            "trade_action": "long",
            "signal_ensemble": 1,
            "position_size": 0.7,
            "direction_output": {"direction": "up", "probability": 0.68},
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "ready", "reason": "pass", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next_display": "up",
            "raw_p_up": 0.39,
            "p_up": 0.82,
            "close": 100.0,
            "projected_low": 97.5,
            "trade_action": "long",
            "signal_ensemble": 1,
            "position_size": 0.9,
            "direction_output": {"direction": "up", "probability": 0.82},
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "waiting_pullback", "reason": "pass", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
    }

    updated = apply_short_term_downtrend_fail_safe(summary)

    assert updated["4h"]["trade_action"] == "hold"
    assert updated["4h"]["signal_ensemble"] == 0
    assert updated["4h"]["position_size"] == 0.0
    assert updated["4h"]["direction_next_display"] == "neutral"
    assert updated["4h"]["direction_output"]["direction"] == "neutral"
    assert updated["4h"]["direction_output"]["downtrend_fail_safe_override"]["raw_direction"] == "up"
    assert updated["4h"]["execution_plan"]["status"] == "rejected"
    assert updated["4h"]["execution_plan"]["reason"] == "short_term_downtrend_fail_safe"
    assert updated["4h"]["execution_plan"]["pending_trade_action"] == "hold"
    assert updated["4h"]["downtrend_fail_safe"]["applied"] is True
    assert updated["4h"]["gate_trace"][-1]["stage"] == "downtrend_fail_safe"
    assert updated["12h"]["trade_action"] == "hold"
    assert updated["12h"]["direction_next_display"] == "neutral"
    assert updated["12h"]["direction_output"]["direction"] == "neutral"
    assert updated["12h"]["execution_plan"]["reason"] == "short_term_downtrend_fail_safe"

    prompt_summary = build_prompt_ready_summary(
        updated,
        select_prompt_preferred_entry=lambda payload: select_prompt_preferred_entry(
            payload,
            coerce_result_horizon=coerce_result_horizon,
            finite_float_or_none=_finite_float_or_none,
        ),
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )

    assert prompt_summary["market_outlook_strategy"]["selected_direction"] == "Short"
    assert prompt_summary["market_outlook_strategy"]["preferred_horizon"] == "15m"


def test_apply_short_term_downtrend_fail_safe_handles_1h_bearish_reversal_with_4h_coherence_block() -> None:
    summary = {
        "15m": {
            "horizon_hours": 0.25,
            "direction_next_display": "up",
            "close": 100.0,
            "projected_price": 101.0,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "rejected", "reason": "low_execution_confluence", "side": "long"},
            "gate_trace": [],
        },
        "1h": {
            "horizon_hours": 1.0,
            "direction_next_display": "down",
            "close": 100.0,
            "projected_price": 98.0,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "rejected", "reason": "bias_direction_conflict", "side": "short"},
            "gate_trace": [],
        },
        "4h": {
            "horizon_hours": 4.0,
            "direction_next_display": "up",
            "raw_p_up": 0.49,
            "p_up": 0.12,
            "close": 100.0,
            "projected_low": 97.5,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "direction_output": {"direction": "up", "probability": 0.12},
            "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
            "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
        "8h": {
            "horizon_hours": 8.0,
            "direction_next_display": "up",
            "raw_p_up": 0.78,
            "p_up": 0.66,
            "close": 100.0,
            "projected_low": 99.2,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "direction_output": {"direction": "up", "probability": 0.66},
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "rejected", "reason": "insufficient_mfe_headroom", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next_display": "up",
            "raw_p_up": 0.48,
            "p_up": 0.67,
            "close": 100.0,
            "projected_low": 98.8,
            "trade_action": "hold",
            "signal_ensemble": 0,
            "direction_output": {"direction": "up", "probability": 0.67},
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
    }

    assert suppress_long_bias_for_short_term_downtrend(summary, finite_float_or_none=_finite_float_or_none) is True

    updated = apply_short_term_downtrend_fail_safe(summary)

    assert updated["4h"]["direction_next_display"] == "neutral"
    assert updated["4h"]["direction_output"]["direction"] == "neutral"
    assert updated["8h"]["direction_next_display"] == "neutral"
    assert updated["8h"]["direction_output"]["direction"] == "neutral"
    assert updated["12h"]["direction_next_display"] == "neutral"
    assert updated["12h"]["execution_plan"]["reason"] == "short_term_downtrend_fail_safe"

    prompt_summary = build_prompt_ready_summary(
        updated,
        select_prompt_preferred_entry=lambda payload: select_prompt_preferred_entry(
            payload,
            coerce_result_horizon=coerce_result_horizon,
            finite_float_or_none=_finite_float_or_none,
        ),
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )

    assert prompt_summary["market_outlook_strategy"]["selected_direction"] == "Neutral"
    assert prompt_summary["market_outlook_strategy"]["preferred_horizon"] is None


def test_prompt_summary_neutralizes_confluence_gated_long_bias_during_short_term_downtrend() -> None:
    summary = {
        "15m": {
            "horizon_hours": 0.25,
            "direction_next_display": "down",
            "close": 100.0,
            "projected_price": 98.5,
            "confidence_score": 0.6,
            "execution_plan": {"status": "rejected", "reason": "bias_direction_conflict", "side": "short"},
            "forecast_coherence": {"triggered": False},
        },
        "1h": {
            "horizon_hours": 1.0,
            "direction_next_display": "neutral",
            "direction_next": "down",
            "close": 100.0,
            "projected_price": 99.0,
            "confidence_score": 0.55,
            "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate", "side": "short"},
            "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
        },
        "4h": {
            "horizon_hours": 4.0,
            "direction_next_display": "up",
            "raw_p_up": 0.72,
            "p_up": 0.81,
            "close": 100.0,
            "projected_low": 99.6,
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "take_profit": 105.0,
            "risk_reward_ratio": 1.67,
            "confidence_score": 0.7,
            "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate", "pending_trade_action": "long", "side": "long"},
            "forecast_coherence": {"triggered": False},
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next_display": "up",
            "raw_p_up": 0.84,
            "p_up": 0.88,
            "close": 100.0,
            "projected_low": 99.4,
            "entry_price": 100.0,
            "stop_loss": 96.0,
            "take_profit": 108.0,
            "risk_reward_ratio": 2.0,
            "confidence_score": 0.8,
            "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate", "pending_trade_action": "long", "side": "long"},
            "forecast_coherence": {"triggered": False},
        },
    }

    assert suppress_long_bias_for_short_term_downtrend(summary, finite_float_or_none=_finite_float_or_none) is True

    preferred_label, preferred_entry, side_profile = select_prompt_preferred_entry(
        summary,
        coerce_result_horizon=coerce_result_horizon,
        finite_float_or_none=_finite_float_or_none,
    )

    assert preferred_label is None
    assert preferred_entry is None
    assert side_profile is None

    result = build_prompt_ready_summary(
        summary,
        select_prompt_preferred_entry=lambda payload: select_prompt_preferred_entry(
            payload,
            coerce_result_horizon=coerce_result_horizon,
            finite_float_or_none=_finite_float_or_none,
        ),
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )

    assert result["market_outlook_strategy"]["selected_direction"] == "Neutral"
    assert result["market_outlook_strategy"]["preferred_horizon"] is None
    assert result["operator_summary_compact"]["market_bias"] == "Neutral"
    assert "short_term_downtrend_fail_safe" in result["operator_summary_compact"]["caution_flags"]


def test_runtime_scalar_helpers_reject_non_finite_values() -> None:
    assert finite_float_or_none("1.25") == 1.25
    assert finite_float_or_none("nan") is None
    assert coerce_result_horizon(4) == 4.0
    assert coerce_result_horizon(0) is None
    assert resolve_degradation_monitoring_policy({"lookback_snapshots": 1, "min_confidence": 2.0}) == {
        "enabled": False,
        "lookback_snapshots": 3,
        "min_snapshots": 10,
        "min_ready_ratio": 0.1,
        "max_blocked_ratio": 0.85,
        "min_expected_net": 0.0,
        "min_confidence": 1.0,
        "min_directional_samples": 3,
        "max_long_wrong_ratio": 0.65,
        "max_long_wrong_streak": 3,
    }


def test_build_degradation_monitoring_flags_recent_long_miss_streak() -> None:
    history = [
        {
            "predictions": {
                "4h": {
                    "close": 100.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.7,
                    "trade_decision": {"expected_net": 0.2},
                    "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate"},
                }
            }
        },
        {
            "predictions": {
                "4h": {
                    "close": 95.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.68,
                    "trade_decision": {"expected_net": 0.15},
                    "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate"},
                }
            }
        },
        {
            "predictions": {
                "4h": {
                    "close": 90.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.66,
                    "trade_decision": {"expected_net": 0.1},
                    "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate"},
                }
            }
        },
    ]

    payload = build_degradation_monitoring(
        history,
        policy={"enabled": True, "min_snapshots": 2, "min_directional_samples": 2, "max_long_wrong_streak": 2},
        resolve_degradation_monitoring_policy=_resolve_degradation_monitoring_policy,
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )

    assert payload["by_horizon"]["4h"]["long_wrong_count"] == 2
    assert payload["by_horizon"]["4h"]["long_wrong_streak"] == 2
    assert "recent_long_miss_streak" in payload["by_horizon"]["4h"]["reasons"]


def test_apply_prompt_trust_degradation_neutralizes_long_bias_after_recent_miss_streak() -> None:
    prompt_summary = {
        "market_outlook_strategy": {
            "selected_direction": "Long",
            "preferred_horizon": "4h",
            "confidence_level": "High",
            "tradeable": True,
            "execution_state": "ready",
            "pending_trade_action": "long",
        },
        "analysis_summary": {
            "rationale": "Preferred horizon 4h carries the strongest post-policy bias.",
            "blocking_factors": [],
        },
        "operator_summary_compact": {
            "market_bias": "Long",
            "recommended_operator_action": "bias_only",
            "primary_blocker": None,
            "caution_flags": [],
        },
    }

    updated = apply_prompt_trust_degradation(
        prompt_summary,
        {
            "by_horizon": {
                "4h": {
                    "reasons": ["recent_long_miss_rate_high", "recent_long_miss_streak"],
                }
            }
        },
    )

    assert updated["market_outlook_strategy"]["selected_direction"] == "Neutral"
    assert updated["market_outlook_strategy"]["execution_state"] == "degraded_trust_hold"
    assert updated["operator_summary_compact"]["recommended_operator_action"] == "hold"
    assert "recent_long_miss_streak" in updated["analysis_summary"]["blocking_factors"]


def test_build_blocked_trade_analytics_counts_rejections_and_gates():
    summary = {
        "1h": {
            "trade_action": "hold",
            "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate"},
            "gate_trace": [
                {"stage": "forecast_coherence", "reason": "forecast_coherence_gate", "triggered": True},
                {"stage": "entry", "reason": "await_pullback_entry_zone", "triggered": False},
            ],
        },
        "4h": {
            "trade_action": "up",
            "execution_plan": {"status": "ready", "reason": "pass"},
            "gate_trace": [],
        },
    }

    result = build_blocked_trade_analytics(summary)

    assert result["blocked_total"] == 1
    assert result["ready_total"] == 1
    assert result["gate_stage_counts"]["forecast_coherence"] == 1
    assert result["reason_counts"]["forecast_coherence_gate"] == 1


def test_build_execution_prior_summary_counts_provenance_sources():
    summary = {
        "1h": {
            "execution_prior_provenance": {
                "analytics_source": "backtest_proxy",
                "stop_source": "atr_structure",
                "target_source": "analytics_mfe",
            }
        },
        "4h": {
            "execution_prior_provenance": {
                "analytics_source": "backtest_proxy",
                "stop_source": "atr_structure",
                "target_source": "existing_or_projection",
            }
        },
    }

    result = build_execution_prior_summary(summary)

    assert result["analytics_source_counts"] == {"backtest_proxy": 2}
    assert result["stop_source_counts"] == {"atr_structure": 2}
    assert result["target_source_counts"] == {"analytics_mfe": 1, "existing_or_projection": 1}


def test_select_prompt_candidate_entries_filters_subhour_directional_when_hourly_exists():
    summary = {
        "15m": {
            "horizon_hours": 0.25,
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.9,
            "execution_plan": {"status": "ready", "reason": "pass", "confluence_tier": "high"},
        },
        "1h": {
            "horizon_hours": 1.0,
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.6,
            "execution_plan": {"status": "waiting_pullback", "reason": "await_pullback_entry_zone", "confluence_tier": "medium"},
        },
        "4h": {
            "horizon_hours": 4.0,
            "trade_action": "hold",
            "direction_next_display": "neutral",
            "confidence_score": 0.5,
            "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate", "confluence_tier": "low"},
        },
    }

    result = select_prompt_candidate_entries(
        summary,
        coerce_result_horizon=lambda value: None if value is None else float(value),
        finite_float_or_none=_finite_float_or_none,
    )

    labels = [label for _rank, label, _entry in result]
    assert "15m" not in labels
    assert labels == ["1h", "4h"]


def test_select_prompt_preferred_entry_prefers_side_with_more_ready_support():
    summary = {
        "1h": {
            "horizon_hours": 1.0,
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.61,
            "execution_plan": {
                "status": "ready",
                "reason": "pass",
                "confluence_tier": "high",
                "execution_alignment_ratio": 0.8,
                "bias_alignment_ratio": 0.75,
                "execution_score": 0.8,
                "bias_score": 0.7,
            },
        },
        "4h": {
            "horizon_hours": 4.0,
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.72,
            "execution_plan": {
                "status": "waiting_pullback",
                "reason": "await_pullback_entry_zone",
                "confluence_tier": "medium",
                "execution_alignment_ratio": 0.7,
                "bias_alignment_ratio": 0.8,
                "execution_score": 0.72,
                "bias_score": 0.82,
            },
        },
        "8h": {
            "horizon_hours": 8.0,
            "trade_action": "down",
            "direction_next_display": "down",
            "confidence_score": 0.8,
            "execution_plan": {
                "status": "ready",
                "reason": "pass",
                "confluence_tier": "high",
                "execution_alignment_ratio": 0.78,
                "bias_alignment_ratio": 0.7,
                "execution_score": 0.78,
                "bias_score": 0.71,
            },
        },
    }

    label, entry, side_profile = select_prompt_preferred_entry(
        summary,
        coerce_result_horizon=lambda value: None if value is None else float(value),
        finite_float_or_none=_finite_float_or_none,
    )

    assert label == "1h"
    assert entry is summary["1h"]
    assert side_profile["side"] == "up"
    assert side_profile["conflict_present"] is True
    assert side_profile["support_count"] == 2


def test_write_prediction_summary_appends_history_and_degradation(tmp_path):
    latest_path = tmp_path / "latest.json"
    history_path = tmp_path / "history.json"
    summary = {
        "4h": {
            "trade_action": "up",
            "direction_next_display": "up",
            "confidence_score": 0.8,
            "entry_price": 100000,
            "stop_loss": 99000,
            "take_profit": 102500,
            "risk_reward_ratio": 2.5,
            "trade_decision": {"expected_net": 15.0},
            "execution_plan": {
                "status": "ready",
                "reason": "pass",
                "pending_trade_action": "buy",
                "disagreement_severity": {"score": 0.0, "triggered": False},
            },
            "forecast_coherence": {"triggered": False},
            "gate_trace": [],
        }
    }

    printed = []
    payload = write_prediction_summary(
        summary,
        degradation_policy={"enabled": True, "min_snapshots": 1},
        latest_prediction_path=latest_path,
        history_prediction_path=history_path,
        build_prompt_ready_summary_fn=lambda data: build_prompt_ready_summary(
            data,
            select_prompt_preferred_entry=_select_prompt_preferred_entry,
            horizon_sort_key=_horizon_sort_key,
            finite_float_or_none=_finite_float_or_none,
        ),
        build_blocked_trade_analytics_fn=build_blocked_trade_analytics,
        build_degradation_monitoring_fn=lambda history, policy: build_degradation_monitoring(
            history,
            policy=policy,
            resolve_degradation_monitoring_policy=_resolve_degradation_monitoring_policy,
            horizon_sort_key=_horizon_sort_key,
            finite_float_or_none=_finite_float_or_none,
        ),
        print_fn=printed.append,
    )

    assert latest_path.exists()
    assert history_path.exists()
    history = json.loads(history_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_path.read_text(encoding="utf-8"))

    assert len(history) == 1
    assert payload["degradation_monitoring"]["enabled"] is True
    assert latest["execution_prior_summary"]["analytics_source_counts"] == {}
    assert latest["prompt_ready_summary"]["market_outlook_strategy"]["selected_direction"] == "Long"
    assert printed
