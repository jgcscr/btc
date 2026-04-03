from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping

import numpy as np


SummaryPayload = Dict[str, Dict[str, Any]]


def resolve_abstention_policy(
    config: Mapping[str, Any] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    cfg = config or {}
    thresholds_by_horizon_regime: Dict[float, Dict[str, Dict[str, Any]]] = {}
    raw_thresholds = cfg.get("thresholds_by_horizon_regime") if isinstance(cfg.get("thresholds_by_horizon_regime"), Mapping) else {}
    for raw_horizon, raw_regimes in raw_thresholds.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None or not isinstance(raw_regimes, Mapping):
            continue
        resolved_regimes: Dict[str, Dict[str, Any]] = {}
        for raw_regime, raw_values in raw_regimes.items():
            if not isinstance(raw_values, Mapping):
                continue
            resolved_values = {
                key: value
                for key, value in {
                    "min_confidence": (
                        max(0.0, min(1.0, float(raw_values.get("min_confidence"))))
                        if raw_values.get("min_confidence") is not None
                        else None
                    ),
                    "min_abs_expected_value": (
                        max(float(raw_values.get("min_abs_expected_value")), 0.0)
                        if raw_values.get("min_abs_expected_value") is not None
                        else None
                    ),
                    "min_edge_over_fee": (
                        max(float(raw_values.get("min_edge_over_fee")), 0.0)
                        if raw_values.get("min_edge_over_fee") is not None
                        else None
                    ),
                    "require_positive_ev": (
                        bool(raw_values.get("require_positive_ev"))
                        if raw_values.get("require_positive_ev") is not None
                        else None
                    ),
                    "hold_prob_center": (
                        max(0.0, min(1.0, float(raw_values.get("hold_prob_center"))))
                        if raw_values.get("hold_prob_center") is not None
                        else None
                    ),
                    "hold_prob_band": (
                        max(0.0, min(0.5, float(raw_values.get("hold_prob_band"))))
                        if raw_values.get("hold_prob_band") is not None
                        else None
                    ),
                }.items()
                if value is not None
            }
            if resolved_values:
                resolved_regimes[str(raw_regime).strip().lower()] = resolved_values
        if resolved_regimes:
            thresholds_by_horizon_regime[normalize_horizon_value(horizon)] = resolved_regimes
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "min_confidence": max(0.0, min(1.0, float(cfg.get("min_confidence") or 0.0))),
        "min_abs_expected_value": max(float(cfg.get("min_abs_expected_value") or 0.0), 0.0),
        "min_edge_over_fee": max(float(cfg.get("min_edge_over_fee") or 0.0), 0.0),
        "require_positive_ev": bool(cfg.get("require_positive_ev", False)),
        "hold_prob_center": max(0.0, min(1.0, float(cfg.get("hold_prob_center") or 0.5))),
        "hold_prob_band": max(0.0, min(0.5, float(cfg.get("hold_prob_band") or 0.0))),
        "thresholds_by_horizon_regime": thresholds_by_horizon_regime,
    }


def resolve_abstention_policy_for_horizon(
    policy: Mapping[str, Any],
    *,
    horizon: float | None,
    regime_state: str,
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    resolved = dict(policy or {})
    if horizon is None:
        return resolved
    overrides = resolved.get("thresholds_by_horizon_regime") if isinstance(resolved.get("thresholds_by_horizon_regime"), Mapping) else {}
    regime_map = overrides.get(normalize_horizon_value(horizon))
    if not isinstance(regime_map, Mapping):
        return resolved
    regime_key = str(regime_state).strip().lower()
    scoped = regime_map.get(regime_key)
    if scoped is None:
        scoped = regime_map.get("default")
    if not isinstance(scoped, Mapping):
        return resolved
    merged = dict(resolved)
    merged.update(scoped)
    return merged


def resolve_uncertainty_policy(
    config: Mapping[str, Any] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    cfg = config or {}
    alpha = float(cfg.get("alpha") or 0.2)
    alpha = max(0.01, min(0.49, alpha))
    thresholds_by_horizon_regime: Dict[float, Dict[str, Dict[str, Any]]] = {}
    raw_thresholds = cfg.get("thresholds_by_horizon_regime") if isinstance(cfg.get("thresholds_by_horizon_regime"), Mapping) else {}
    for raw_horizon, raw_regimes in raw_thresholds.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None or not isinstance(raw_regimes, Mapping):
            continue
        resolved_regimes: Dict[str, Dict[str, Any]] = {}
        for raw_regime, raw_values in raw_regimes.items():
            if not isinstance(raw_values, Mapping):
                continue
            resolved_regimes[str(raw_regime).strip().lower()] = {
                key: value
                for key, value in {
                    "alpha": (float(raw_values.get("alpha")) if raw_values.get("alpha") is not None else None),
                    "hold_prob_center": (
                        float(raw_values.get("hold_prob_center")) if raw_values.get("hold_prob_center") is not None else None
                    ),
                    "max_interval_width": (
                        float(raw_values.get("max_interval_width")) if raw_values.get("max_interval_width") is not None else None
                    ),
                    "require_center_cross": (
                        bool(raw_values.get("require_center_cross")) if raw_values.get("require_center_cross") is not None else None
                    ),
                    "min_component_count": (
                        int(float(raw_values.get("min_component_count"))) if raw_values.get("min_component_count") is not None else None
                    ),
                }.items()
                if value is not None
            }
        if resolved_regimes:
            thresholds_by_horizon_regime[normalize_horizon_value(horizon)] = resolved_regimes
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "alpha": alpha,
        "hold_prob_center": max(0.0, min(1.0, float(cfg.get("hold_prob_center") or 0.5))),
        "max_interval_width": max(float(cfg.get("max_interval_width") or 1.0), 0.0),
        "require_center_cross": bool(cfg.get("require_center_cross", True)),
        "min_component_count": max(int(float(cfg.get("min_component_count") or 3)), 1),
        "thresholds_by_horizon_regime": thresholds_by_horizon_regime,
    }


def resolve_uncertainty_settings(
    policy: Mapping[str, Any],
    *,
    horizon: float | None,
    regime_state: str,
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    resolved = {
        "alpha": float(policy.get("alpha", 0.2)),
        "hold_prob_center": float(policy.get("hold_prob_center", 0.5)),
        "max_interval_width": float(policy.get("max_interval_width", 1.0)),
        "require_center_cross": bool(policy.get("require_center_cross", True)),
        "min_component_count": int(policy.get("min_component_count", 3)),
    }
    if horizon is None:
        return resolved
    raw_overrides = policy.get("thresholds_by_horizon_regime") if isinstance(policy, Mapping) else None
    if not isinstance(raw_overrides, Mapping):
        return resolved
    horizon_overrides = raw_overrides.get(normalize_horizon_value(horizon))
    if not isinstance(horizon_overrides, Mapping):
        return resolved
    regime_overrides = horizon_overrides.get(str(regime_state).strip().lower())
    if not isinstance(regime_overrides, Mapping):
        return resolved
    resolved.update({key: value for key, value in regime_overrides.items() if value is not None})
    resolved["alpha"] = max(0.01, min(0.49, float(resolved.get("alpha", 0.2))))
    resolved["hold_prob_center"] = max(0.0, min(1.0, float(resolved.get("hold_prob_center", 0.5))))
    resolved["max_interval_width"] = max(float(resolved.get("max_interval_width", 1.0)), 0.0)
    resolved["min_component_count"] = max(int(resolved.get("min_component_count", 3)), 1)
    resolved["require_center_cross"] = bool(resolved.get("require_center_cross", True))
    return resolved


def apply_abstention_policy(
    *,
    trade_action: str,
    p_up: float,
    confidence_score: float,
    expected_value: float,
    fee_bps: float,
    slippage_bps: float,
    policy: Mapping[str, Any],
) -> tuple[bool, str]:
    if trade_action == "hold":
        return False, "already_hold"
    if not bool(policy.get("enabled", False)):
        return False, "disabled"

    min_confidence = float(policy.get("min_confidence", 0.0))
    if confidence_score < min_confidence:
        return True, "confidence_below_min"

    abs_ev_floor = float(policy.get("min_abs_expected_value", 0.0))
    if abs(expected_value) < abs_ev_floor:
        return True, "expected_value_below_abs_floor"

    if bool(policy.get("require_positive_ev", False)) and expected_value <= 0.0:
        return True, "non_positive_expected_value"

    edge_over_fee_floor = float(policy.get("min_edge_over_fee", 0.0))
    total_cost = max(fee_bps + slippage_bps, 0.0) / 10_000.0
    edge_over_fee = expected_value - total_cost
    if edge_over_fee < edge_over_fee_floor:
        return True, "edge_over_fee_below_min"

    hold_center = float(policy.get("hold_prob_center", 0.5))
    hold_band = float(policy.get("hold_prob_band", 0.0))
    if hold_band > 0.0 and abs(float(p_up) - hold_center) <= hold_band:
        return True, "probability_in_hold_band"

    return False, "pass"


def resolve_abstention_expected_value(
    expected_value: float,
    trade_decision: Mapping[str, Any] | None,
) -> tuple[float, str]:
    if isinstance(trade_decision, Mapping):
        expected_net = trade_decision.get("expected_net")
        expected_net_valid = bool(trade_decision.get("expected_net_valid", False))
        if expected_net_valid and expected_net is not None:
            try:
                resolved = float(expected_net)
            except (TypeError, ValueError):
                resolved = expected_value
            else:
                if math.isfinite(resolved):
                    return resolved, "trade_decision_expected_net"
    return expected_value, "raw_expected_value"


def apply_uncertainty_abstention(
    *,
    trade_action: str,
    p_up_components: Mapping[str, Any],
    horizon: float | None,
    regime_state: str,
    policy: Mapping[str, Any],
    resolve_uncertainty_settings: Callable[..., Dict[str, Any]],
) -> tuple[bool, str, Dict[str, Any]]:
    if trade_action == "hold":
        return False, "already_hold", {"available": False}
    if not bool(policy.get("enabled", False)):
        return False, "disabled", {"available": False}

    vals: list[float] = []
    for value in p_up_components.values():
        try:
            vals.append(float(value))
        except Exception:
            continue
    if len(vals) < int(policy.get("min_component_count", 3)):
        return False, "insufficient_components", {"available": False, "component_count": len(vals)}

    settings = resolve_uncertainty_settings(policy, horizon=horizon, regime_state=regime_state)
    arr = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    alpha = float(settings.get("alpha", 0.2))
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    width = hi - lo
    center = float(settings.get("hold_prob_center", 0.5))
    cross_center = bool(lo <= center <= hi)
    max_width = float(settings.get("max_interval_width", 1.0))
    too_wide = width > max_width

    should_abstain = False
    reason = "pass"
    if bool(settings.get("require_center_cross", True)) and cross_center:
        should_abstain = True
        reason = "uncertainty_interval_crosses_center"
    if too_wide:
        should_abstain = True
        reason = "uncertainty_interval_too_wide"

    return should_abstain, reason, {
        "available": True,
        "component_count": int(arr.size),
        "interval_low": lo,
        "interval_high": hi,
        "interval_width": width,
        "crosses_hold_center": cross_center,
        "effective_policy": settings,
    }


def apply_post_trade_gates(
    summary: SummaryPayload,
    *,
    confidence_min: float,
    abstention_policy: Mapping[str, Any],
    uncertainty_policy: Mapping[str, Any],
    default_fee_bps: float,
    default_slippage_bps: float,
    regime_neutral: str,
    append_gate_trace: Callable[..., None],
    resolve_abstention_expected_value: Callable[[float, Mapping[str, Any] | None], tuple[float, str]],
    resolve_abstention_policy_for_horizon: Callable[..., Dict[str, Any]],
    apply_abstention_policy: Callable[..., tuple[bool, str]],
    apply_uncertainty_abstention: Callable[..., tuple[bool, str, Dict[str, Any]]],
    coerce_result_horizon: Callable[[Any], float | None],
) -> SummaryPayload:
    for entry in summary.values():
        confidence_score = float(entry.get("confidence_score", 0.0))
        effective_confidence_min = max(0.0, min(1.0, float(entry.get("confidence_min", confidence_min))))
        if str(entry.get("trade_action", "hold")) != "hold" and confidence_score < effective_confidence_min:
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["confidence_filter_triggered"] = True
            append_gate_trace(
                entry,
                stage="confidence_filter",
                reason="confidence_below_min",
                triggered=True,
                blocking=True,
            )
        else:
            entry["confidence_filter_triggered"] = False

        expected_value = float(entry.get("expected_value", 0.0))
        abstention_expected_value, abstention_expected_value_source = resolve_abstention_expected_value(
            expected_value,
            entry.get("trade_decision") if isinstance(entry.get("trade_decision"), Mapping) else None,
        )
        effective_abstention_policy = resolve_abstention_policy_for_horizon(
            abstention_policy,
            horizon=coerce_result_horizon(entry.get("horizon_hours")),
            regime_state=str(entry.get("regime_state") or regime_neutral),
        )
        abstain, abstain_reason = apply_abstention_policy(
            trade_action=str(entry.get("trade_action", "hold")),
            p_up=float(entry.get("p_up", 0.0)),
            confidence_score=confidence_score,
            expected_value=abstention_expected_value,
            fee_bps=float(default_fee_bps),
            slippage_bps=float(default_slippage_bps),
            policy=effective_abstention_policy,
        )
        entry["abstention"] = {
            "enabled": bool(effective_abstention_policy.get("enabled", False)),
            "triggered": bool(abstain),
            "reason": abstain_reason,
            "expected_value_used": float(abstention_expected_value),
            "expected_value_source": abstention_expected_value_source,
        }
        if abstain:
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            append_gate_trace(
                entry,
                stage="abstention",
                reason=abstain_reason,
                triggered=True,
                blocking=True,
            )

        uncertainty_abstain, uncertainty_reason, uncertainty_payload = apply_uncertainty_abstention(
            trade_action=str(entry.get("trade_action", "hold")),
            p_up_components=entry.get("p_up_components", {}),
            horizon=coerce_result_horizon(entry.get("horizon_hours")),
            regime_state=str(entry.get("regime_state") or regime_neutral),
            policy=uncertainty_policy,
        )
        entry["uncertainty"] = uncertainty_payload
        if uncertainty_abstain:
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["abstention"] = {
                "enabled": True,
                "triggered": True,
                "reason": uncertainty_reason,
            }
            append_gate_trace(
                entry,
                stage="uncertainty",
                reason=uncertainty_reason,
                triggered=True,
                blocking=True,
            )
    return summary
