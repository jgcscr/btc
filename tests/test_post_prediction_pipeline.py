from __future__ import annotations

from src.runtime.post_prediction_pipeline import apply_post_prediction_policies


def test_apply_post_prediction_policies_runs_enabled_stages_in_order() -> None:
    calls: list[str] = []
    summary = {"4h": {"value": 1}}
    contexts = {"4h": {"prepared": object()}}

    def forecast_stage(payload, policy):
        calls.append(f"forecast:{policy['enabled']}")
        payload["4h"]["forecast"] = True
        return payload

    def confluence_stage(payload, policy):
        calls.append(f"confluence:{policy['enabled']}")
        payload["4h"]["confluence"] = True
        return payload

    def trust_stage(payload, policy):
        calls.append(f"trust:{policy['enabled']}")
        payload["4h"]["trust"] = True
        return payload

    def trade_decision_stage(payload, execution_contexts, policy):
        calls.append(f"trade:{bool(execution_contexts)}")
        payload["4h"]["trade"] = policy["enabled"]
        return payload

    def post_trade_stage(payload, confidence_min, abstention_policy, uncertainty_policy):
        calls.append(f"post:{confidence_min}")
        payload["4h"]["post"] = abstention_policy["enabled"] and uncertainty_policy["enabled"]
        return payload

    def execution_stage(payload, execution_contexts, policy):
        calls.append(f"execution:{policy['enabled']}")
        payload["4h"]["execution"] = True
        return payload

    result = apply_post_prediction_policies(
        summary,
        contexts,
        forecast_coherence_policy={"enabled": True},
        trust_hardening_policy={"enabled": True},
        confluence_policy={"enabled": True},
        trade_decision_policy={"enabled": True},
        confidence_min=0.25,
        abstention_policy={"enabled": True},
        uncertainty_policy={"enabled": True},
        execution_policy={"enabled": True},
        apply_forecast_coherence_policy=forecast_stage,
        apply_trust_hardening_stage=trust_stage,
        apply_confluence_policy=confluence_stage,
        apply_trade_decision_stage=trade_decision_stage,
        apply_post_trade_gates=post_trade_stage,
        apply_execution_policy=execution_stage,
    )

    assert calls == [
        "forecast:True",
        "trust:True",
        "confluence:True",
        "trade:True",
        "post:0.25",
        "execution:True",
    ]
    assert result["4h"]["forecast"] is True
    assert result["4h"]["trust"] is True
    assert result["4h"]["confluence"] is True
    assert result["4h"]["trade"] is True
    assert result["4h"]["post"] is True
    assert result["4h"]["execution"] is True


def test_apply_post_prediction_policies_skips_disabled_optional_stages() -> None:
    calls: list[str] = []
    summary = {"1h": {}}

    def forecast_stage(payload, policy):
        calls.append("forecast")
        return payload

    def confluence_stage(payload, policy):
        calls.append("confluence")
        return payload

    def trust_stage(payload, policy):
        calls.append("trust")
        return payload

    def trade_decision_stage(payload, execution_contexts, policy):
        calls.append("trade")
        return payload

    def post_trade_stage(payload, confidence_min, abstention_policy, uncertainty_policy):
        calls.append("post")
        return payload

    def execution_stage(payload, execution_contexts, policy):
        calls.append("execution")
        return payload

    apply_post_prediction_policies(
        summary,
        {},
        forecast_coherence_policy={"enabled": False},
        trust_hardening_policy={"enabled": False},
        confluence_policy={"enabled": False},
        trade_decision_policy={"enabled": True},
        confidence_min=0.0,
        abstention_policy={"enabled": False},
        uncertainty_policy={"enabled": False},
        execution_policy={"enabled": False},
        apply_forecast_coherence_policy=forecast_stage,
        apply_trust_hardening_stage=trust_stage,
        apply_confluence_policy=confluence_stage,
        apply_trade_decision_stage=trade_decision_stage,
        apply_post_trade_gates=post_trade_stage,
        apply_execution_policy=execution_stage,
    )

    assert calls == ["trade", "post"]


def test_apply_post_prediction_policies_applies_downtrend_fail_safe_after_execution() -> None:
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
            "close": 100.0,
            "projected_low": 98.5,
            "trade_action": "long",
            "signal_ensemble": 1,
            "position_size": 0.5,
            "forecast_coherence": {"triggered": False},
            "execution_plan": {"status": "ready", "reason": "pass", "pending_trade_action": "long", "side": "long"},
            "gate_trace": [],
        },
    }

    result = apply_post_prediction_policies(
        summary,
        {"4h": {"prepared": object()}},
        forecast_coherence_policy={"enabled": False},
        trust_hardening_policy={"enabled": False},
        confluence_policy={"enabled": False},
        trade_decision_policy={"enabled": True},
        confidence_min=0.0,
        abstention_policy={"enabled": False},
        uncertainty_policy={"enabled": False},
        execution_policy={"enabled": True},
        apply_forecast_coherence_policy=lambda payload, policy: payload,
        apply_trust_hardening_stage=lambda payload, policy: payload,
        apply_confluence_policy=lambda payload, policy: payload,
        apply_trade_decision_stage=lambda payload, execution_contexts, policy: payload,
        apply_post_trade_gates=lambda payload, confidence_min, abstention_policy, uncertainty_policy: payload,
        apply_execution_policy=lambda payload, execution_contexts, policy: payload,
    )

    assert result["4h"]["trade_action"] == "hold"
    assert result["4h"]["signal_ensemble"] == 0
    assert result["4h"]["execution_plan"]["reason"] == "short_term_downtrend_fail_safe"
