from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from src.utils.component_diversity_support import (
    component_feature_column_names,
    pairwise_feature_column_names,
)


PredictionResult = Dict[str, Any]


def build_prediction_result(
    *,
    signal: Mapping[str, Any],
    label: str,
    horizon: float,
    signal_ts: str,
    close: float,
    p_up: float,
    raw_p_up: float,
    ret_pred: float,
    trend_prob: float,
    ignition_state: int,
    cooldown_active: bool,
    signal_ensemble: int,
    signal_dir_only: int,
    confidence_score: float,
    position_size: float,
    confidence_min: float,
    confidence_min_source: str,
    position_size_cap: float,
    stop_loss_price: float,
    take_profit_price: float,
    expected_value: float,
    horizon_thresholds: Mapping[str, Any],
    regime_state: str,
    calibration_key: str | None,
    calibration_used_regime_key: bool,
    probability_guard: Mapping[str, Any] | None,
    regime_weight_policy: Mapping[str, Any] | None,
    target_projection: Mapping[str, Any] | None,
    volatility_payload: Mapping[str, Any],
    volatility_flag: bool,
    forecast_coherence_policy: Mapping[str, Any],
    direction_output_policy: Mapping[str, Any],
    direction_output_scoped: bool,
    trade_decision_policy: Mapping[str, Any],
    abstention_policy: Mapping[str, Any],
    direction_fallback_policy: Mapping[str, Any] | None,
    trend_payload: Mapping[str, Any] | None,
    target_range_policy: Mapping[str, Any] | None,
    regime_score: float | None,
    adaptive_scale: float,
    horizon_p_up: float,
    horizon_ret: float,
    row_features: Any,
    optional_feature_fields: Sequence[str],
    project_price: Callable[[float, float], float],
    get_active_regime_weight_override: Callable[[str, float, Mapping[str, Any] | None], Any],
    derive_probability_alignment_features: Callable[..., Mapping[str, Any]],
    build_direction_output: Callable[..., Mapping[str, Any]],
    apply_target_range_overrides: Callable[[float, float, Mapping[str, Any], float, int], tuple[Mapping[str, Any], float, float]],
    evaluate_direction_only_fallback: Callable[..., tuple[Mapping[str, Any], bool]],
    finite_float_or_none: Callable[[Any], float | None],
    coerce_row_value: Callable[[Any], float | None],
) -> tuple[PredictionResult, bool]:
    projected_price = project_price(close, ret_pred)
    result: PredictionResult = {
        "timestamp": signal_ts,
        "horizon_hours": horizon,
        "close": close,
        "entry_price": close,
        "p_up": p_up,
        "p_trend_ignition": trend_prob,
        "ignition_state": ignition_state,
        "ignition_cooldown_active": cooldown_active if trend_payload else False,
        "ret_pred": ret_pred,
        "projected_price": projected_price,
        "signal_ensemble": signal_ensemble,
        "signal_dir_only": signal_dir_only,
        "direction_next": "up" if signal_dir_only == 1 else "down",
        "trade_action": (
            "long" if signal_ensemble == 1 and signal_dir_only == 1 else
            "short" if signal_ensemble == 1 and signal_dir_only == 0 else
            "hold"
        ),
        "confidence_score": confidence_score,
        "position_size": position_size,
        "confidence_min": confidence_min,
        "confidence_min_source": confidence_min_source,
        "position_size_cap": position_size_cap,
        "p_up_components": signal.get("p_up_components", {}),
        "direction_ensemble": signal.get("direction_ensemble", {}),
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "expected_value": expected_value,
        "thresholds": dict(horizon_thresholds),
        "regime_state": regime_state,
        "probability_calibration": {
            "requested_key": f"{label}@{regime_state}",
            "applied_key": calibration_key,
            "used_regime_key": calibration_used_regime_key,
            "fallback_to_base": bool(calibration_key) and not calibration_used_regime_key,
            "raw_probability": float(raw_p_up),
            "resolved_probability": float(p_up),
            "absolute_gap": float(abs(float(p_up) - float(raw_p_up))),
            "forecast_alignment_guard": probability_guard,
        },
        "regime_weight_overrides": get_active_regime_weight_override(regime_state, horizon, regime_weight_policy),
        "projected_high": target_projection.get("projected_high") if target_projection else None,
        "projected_low": target_projection.get("projected_low") if target_projection else None,
        "projected_high_confidence": target_projection.get("projected_high_confidence", 0.0) if target_projection else 0.0,
        "projected_low_confidence": target_projection.get("projected_low_confidence", 0.0) if target_projection else 0.0,
        "projected_high_rmse": target_projection.get("projected_high_rmse") if target_projection else None,
        "projected_low_rmse": target_projection.get("projected_low_rmse") if target_projection else None,
        "projected_high_residual_std": target_projection.get("projected_high_residual_std") if target_projection else None,
        "projected_low_residual_std": target_projection.get("projected_low_residual_std") if target_projection else None,
        "volatility": volatility_payload,
        "volatility_flag": bool(volatility_flag),
        "gate_trace": [],
    }
    if isinstance(signal.get("derivatives_shadow_adjustment"), Mapping):
        result["derivatives_shadow_adjustment"] = dict(signal.get("derivatives_shadow_adjustment") or {})
    probability_alignment_features = derive_probability_alignment_features(
        close=close,
        projected_price=float(result["projected_price"]),
        ret_pred=ret_pred,
        raw_probability=float(raw_p_up),
        resolved_probability=float(p_up),
        direction=str(result["direction_next"]),
        neutral_band=float(forecast_coherence_policy.get("p_up_neutral_band", 0.02) or 0.02),
        probability_guard=probability_guard if isinstance(probability_guard, Mapping) else None,
        calibration_used_regime_key=bool(calibration_used_regime_key),
    )
    result.update(probability_alignment_features)
    result["probability_calibration"].update(
        {
            "raw_side": probability_alignment_features["raw_p_up_side"],
            "resolved_side": probability_alignment_features["resolved_p_up_side"],
            "ret_pred_side": probability_alignment_features["ret_pred_side"],
            "projected_price_side": probability_alignment_features["projected_price_side"],
            "forecast_consensus_side": probability_alignment_features["forecast_consensus_side"],
            "guard_applied": bool(probability_alignment_features["probability_calibration_guard_applied"]),
        }
    )
    direction_output = build_direction_output(
        enabled=bool(direction_output_policy.get("enabled", False)),
        scoped=direction_output_scoped,
        label=label,
        regime_state=regime_state,
        signal_dir_only=signal_dir_only,
        raw_probability=raw_p_up,
        trade_probability=p_up,
        ret_pred=ret_pred,
        close=close,
        projected_price=projected_price,
        p_up_components=signal.get("p_up_components", {}),
        policy=direction_output_policy,
    )
    result["direction_output"] = direction_output
    result["direction_next_display"] = direction_output.get("direction", result["direction_next"])

    for field in [
        *component_feature_column_names(),
        *pairwise_feature_column_names(),
        "direction_ensemble_selected_count",
        "direction_ensemble_rejected_count",
        "direction_ensemble_missing_preferred_group_count",
    ]:
        if field in signal:
            value = finite_float_or_none(signal.get(field))
            result[field] = 0.0 if value is None else float(value)

    for field in optional_feature_fields:
        if field in row_features.index:
            value = coerce_row_value(row_features.get(field))
            result[field] = value
    if regime_score is not None:
        result["regime_score"] = regime_score
    result["thresholds"]["p_up_min_effective"] = horizon_p_up
    result["thresholds"]["ret_min_effective"] = horizon_ret
    result["thresholds"]["adaptive_scale"] = adaptive_scale

    overrides_payload: Mapping[str, Any] = {"stop_loss": None, "take_profit": None}
    if target_projection and target_range_policy and target_range_policy.get("enabled"):
        overrides_payload, updated_stop, updated_take = apply_target_range_overrides(
            result["stop_loss"],
            result["take_profit"],
            target_projection,
            float(target_range_policy.get("override_ratio", 0.01)),
            int(result["signal_dir_only"]),
        )
        result["stop_loss"] = updated_stop
        result["take_profit"] = updated_take
    result["target_range_overrides"] = dict(overrides_payload)
    entry_price = float(result["entry_price"])
    stop_loss = float(result["stop_loss"])
    take_profit = float(result["take_profit"])
    downside = abs(entry_price - stop_loss)
    upside = abs(take_profit - entry_price)
    result["risk_reward_ratio"] = (upside / downside) if downside > 0 else None

    fallback_info, fallback_triggered = evaluate_direction_only_fallback(
        direction_fallback_policy,
        p_up=p_up,
        signal_dir_only=int(signal_dir_only),
        expected_value=expected_value,
        projected_price=result["projected_price"],
        signal_ts=signal_ts,
        trend_prob=trend_prob,
        trend_threshold=float(trend_payload.get("threshold")) if trend_payload else None,
    )
    result["direction_only_fallback"] = fallback_info
    result["trade_decision"] = {
        "enabled": bool(trade_decision_policy.get("enabled", False)),
        "triggered": False,
        "reason": "pending_stage",
    }
    result["confidence_filter_triggered"] = False
    result["abstention"] = {
        "enabled": bool(abstention_policy.get("enabled", False)),
        "triggered": False,
        "reason": "pending_stage",
    }
    result["uncertainty"] = {"available": False, "reason": "pending_stage"}
    return result, fallback_triggered
