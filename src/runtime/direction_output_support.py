from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping

import numpy as np


def resolve_direction_output_policy(
    config: Mapping[str, Any] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    def _normalize_float_map(raw: Any, *, minimum: float = 0.0) -> Dict[float, float]:
        if not isinstance(raw, Mapping):
            return {}
        resolved: Dict[float, float] = {}
        for key, value in raw.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                resolved[horizon] = max(float(value), minimum)
            except (TypeError, ValueError):
                continue
        return resolved

    def _parse_weight_spec(spec: Any) -> Dict[str, float]:
        if isinstance(spec, Mapping):
            resolved: Dict[str, float] = {}
            for name, value in spec.items():
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    continue
                if weight > 0.0:
                    resolved[str(name)] = weight
            return resolved
        if spec is None:
            return {}
        resolved: Dict[str, float] = {}
        for raw_chunk in str(spec).split(","):
            chunk = raw_chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            raw_name, raw_value = chunk.split(":", 1)
            try:
                weight = float(raw_value.strip())
            except ValueError:
                continue
            if weight > 0.0:
                resolved[raw_name.strip()] = weight
        return resolved

    cfg = config or {}
    horizons = cfg.get("horizons") or [1.0]
    calibration_map = cfg.get("calibration_map") if isinstance(cfg.get("calibration_map"), Mapping) else {}
    marginal_rerank_cfg = cfg.get("marginal_rerank") if isinstance(cfg.get("marginal_rerank"), Mapping) else {}
    shrinkage_cfg = cfg.get("probability_shrinkage") if isinstance(cfg.get("probability_shrinkage"), Mapping) else {}
    marginal_weight_specs_raw = (
        marginal_rerank_cfg.get("weight_specs") if isinstance(marginal_rerank_cfg.get("weight_specs"), Mapping) else {}
    )
    marginal_weight_specs = {
        str(name): _parse_weight_spec(spec)
        for name, spec in marginal_weight_specs_raw.items()
    }
    marginal_horizons = marginal_rerank_cfg.get("horizons") or horizons
    lower = float(marginal_rerank_cfg.get("lower", 0.5) or 0.5)
    upper = float(marginal_rerank_cfg.get("upper", 0.6) or 0.6)
    if upper < lower:
        lower, upper = upper, lower
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "horizons": sorted({normalize_horizon_value(v) for v in horizons}),
        "neutral_band": max(float(cfg.get("neutral_band") or 0.0), 0.0),
        "neutral_band_by_horizon": _normalize_float_map(cfg.get("neutral_band_by_horizon"), minimum=0.0),
        "use_trade_probability_fallback": bool(cfg.get("use_trade_probability_fallback", True)),
        "calibration_path": str(cfg.get("calibration_path") or "") or None,
        "calibration_map": calibration_map,
        "probability_shrinkage": {
            "enabled": bool(shrinkage_cfg.get("enabled", False)),
            "horizons": sorted(
                {
                    normalize_horizon_value(v)
                    for v in (shrinkage_cfg.get("horizons") or horizons)
                }
            ),
            "regimes": {
                str(value).strip().lower()
                for value in (shrinkage_cfg.get("regimes") or [])
                if str(value).strip()
            },
            "default_strength": max(min(float(shrinkage_cfg.get("default_strength") or 0.0), 0.95), 0.0),
            "strength_by_horizon": _normalize_float_map(shrinkage_cfg.get("strength_by_horizon"), minimum=0.0),
            "bypass_edge": max(min(float(shrinkage_cfg.get("bypass_edge") or 1.0), 0.5), 0.0),
        },
        "marginal_rerank": {
            "enabled": bool(marginal_rerank_cfg.get("enabled", False)) and bool(marginal_weight_specs),
            "horizons": sorted({normalize_horizon_value(v) for v in marginal_horizons}),
            "lower": lower,
            "upper": upper,
            "min_component_count": max(int(marginal_rerank_cfg.get("min_component_count") or 2), 1),
            "use_raw_probability_gate": bool(marginal_rerank_cfg.get("use_raw_probability_gate", True)),
            "weight_specs": marginal_weight_specs,
        },
    }


def apply_probability_calibration(p: float, params: Mapping[str, Any]) -> float:
    p_clip = min(max(float(p), 1e-6), 1.0 - 1e-6)
    method = str(params.get("method", "platt")).lower()
    if method == "platt":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        logit = math.log(p_clip / (1.0 - p_clip))
        return float(1.0 / (1.0 + math.exp(-(a * logit + b))))
    if method == "beta":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", -1.0))
        c = float(params.get("c", 0.0))
        z = a * math.log(p_clip) + b * math.log(1.0 - p_clip) + c
        return float(1.0 / (1.0 + math.exp(-z)))
    if method == "isotonic":
        x = np.asarray(params.get("x", []), dtype=float)
        y = np.asarray(params.get("y", []), dtype=float)
        if x.size >= 2 and y.size == x.size:
            return float(np.interp(p_clip, x, y, left=y[0], right=y[-1]))
    return float(p_clip)


def resolve_probability_calibration(
    platt_calibration: Mapping[str, Mapping[str, Any]] | None,
    label: str,
    regime_state: str,
    *,
    regime_calibration_min_platt_slope: float,
) -> tuple[str | None, Mapping[str, Any] | None, bool]:
    if not platt_calibration:
        return None, None, False
    regime_key = f"{label}@{regime_state}"
    regime_params = platt_calibration.get(regime_key)
    if isinstance(regime_params, Mapping):
        method = str(regime_params.get("method", "platt")).lower()
        try:
            slope = float(regime_params.get("a", 1.0))
        except (TypeError, ValueError):
            slope = 1.0
        if method == "platt" and (abs(slope) < regime_calibration_min_platt_slope or slope <= 0.0):
            base_params = platt_calibration.get(label)
            if isinstance(base_params, Mapping):
                return label, base_params, False
        return regime_key, regime_params, True
    base_params = platt_calibration.get(label)
    if isinstance(base_params, Mapping):
        return label, base_params, False
    return None, None, False


def resolve_trade_probability_for_horizon(
    *,
    platt_calibration: Mapping[str, Mapping[str, Any]] | None,
    label: str,
    regime_state: str,
    raw_probability: float,
    close: float,
    projected_price: float,
    ret_pred: float,
    neutral_band: float = 0.02,
    regime_calibration_min_platt_slope: float,
    apply_probability_calibration: Callable[[float, Mapping[str, Any]], float],
    direction_from_ret_pred: Callable[[Any], str],
    direction_from_projected_price: Callable[[Any, Any], str],
    direction_from_probability: Callable[..., str],
) -> tuple[float, str | None, bool, Dict[str, Any] | None]:
    calibration_key, params, calibration_used_regime_key = resolve_probability_calibration(
        platt_calibration,
        label,
        regime_state,
        regime_calibration_min_platt_slope=regime_calibration_min_platt_slope,
    )
    probability = float(raw_probability)
    if isinstance(params, Mapping):
        probability = apply_probability_calibration(float(raw_probability), params)

    ret_side = direction_from_ret_pred(ret_pred)
    projected_side = direction_from_projected_price(close, projected_price)
    raw_side = direction_from_probability(raw_probability, neutral_band=neutral_band)
    calibrated_side = direction_from_probability(probability, neutral_band=neutral_band)
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else None
    guard_payload: Dict[str, Any] | None = None

    if (
        consensus_side is not None
        and calibration_key is not None
        and raw_side == consensus_side
        and calibrated_side != consensus_side
    ):
        resolved_probability = float(raw_probability)
        resolved_key: str | None = None
        resolved_used_regime_key = False
        fallback_source = "raw_probability"
        base_probability = None
        base_side = None

        if platt_calibration:
            base_params = platt_calibration.get(label)
            if isinstance(base_params, Mapping):
                base_probability = apply_probability_calibration(float(raw_probability), base_params)
                base_side = direction_from_probability(base_probability, neutral_band=neutral_band)
                if base_side == consensus_side:
                    resolved_probability = float(base_probability)
                    resolved_key = label
                    fallback_source = "base_horizon_calibration"

        guard_payload = {
            "applied": True,
            "reason": "calibration_conflicts_with_forecast_consensus",
            "forecast_side": consensus_side,
            "raw_side": raw_side,
            "regime_calibrated_side": calibrated_side,
            "original_applied_key": calibration_key,
            "original_used_regime_key": bool(calibration_used_regime_key),
            "fallback_source": fallback_source,
            "raw_probability": float(raw_probability),
            "regime_calibrated_probability": float(probability),
            "base_probability": None if base_probability is None else float(base_probability),
            "base_side": base_side,
            "resolved_probability": float(resolved_probability),
        }
        return resolved_probability, resolved_key, resolved_used_regime_key, guard_payload

    return probability, calibration_key, calibration_used_regime_key, guard_payload


def build_direction_output(
    *,
    enabled: bool,
    scoped: bool,
    label: str,
    regime_state: str,
    signal_dir_only: int,
    raw_probability: float,
    trade_probability: float,
    ret_pred: float | None,
    close: float | None,
    projected_price: float | None,
    p_up_components: Mapping[str, Any],
    policy: Mapping[str, Any],
    apply_probability_calibration: Callable[[float, Mapping[str, Any]], float],
    resolve_probability_calibration: Callable[[Mapping[str, Mapping[str, Any]] | None, str, str], tuple[str | None, Mapping[str, Any] | None, bool]],
    parse_horizon_label: Callable[[str], float],
    lookup_horizon_value: Callable[[Mapping[float, float], float, float], float],
    direction_from_ret_pred: Callable[[Any], str],
    direction_from_projected_price: Callable[[Any, Any], str],
    direction_from_probability: Callable[..., str],
) -> Dict[str, Any]:
    def _blend_probability_from_components(
        components: Mapping[str, Any],
        weights: Mapping[str, float],
        *,
        min_component_count: int,
    ) -> tuple[float | None, Dict[str, float]]:
        weighted_sum = 0.0
        total_weight = 0.0
        used: Dict[str, float] = {}
        for name, weight in weights.items():
            try:
                component_probability = float(components.get(name))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(component_probability):
                continue
            clipped_probability = min(max(component_probability, 0.0), 1.0)
            weighted_sum += clipped_probability * float(weight)
            total_weight += float(weight)
            used[str(name)] = clipped_probability
        if total_weight <= 0.0 or len(used) < int(min_component_count):
            return None, used
        return weighted_sum / total_weight, used

    base_direction = "up" if int(signal_dir_only) == 1 else "down"
    payload: Dict[str, Any] = {
        "enabled": bool(enabled),
        "evaluated": bool(enabled and scoped),
        "direction": base_direction,
        "probability": float(trade_probability),
        "raw_probability": float(raw_probability),
        "neutral_band": 0.0,
        "source": "trade_probability",
        "calibration": {
            "requested_key": f"{label}@{regime_state}",
            "applied_key": None,
            "used_regime_key": False,
            "fallback_to_trade_probability": True,
            "skipped_due_to_marginal_rerank": False,
        },
        "marginal_rerank": {
            "enabled": False,
            "applied": False,
            "weight_key": None,
            "band": None,
            "component_count": 0,
            "components_used": {},
        },
    }
    if not enabled or not scoped:
        return payload

    horizon_value = parse_horizon_label(label)
    neutral_band = lookup_horizon_value(
        policy.get("neutral_band_by_horizon", {}),
        horizon_value,
        float(policy.get("neutral_band", 0.0) or 0.0),
    )
    internal_direction = base_direction
    ret_side = direction_from_ret_pred(ret_pred)
    projected_side = direction_from_projected_price(close, projected_price)
    probability = float(trade_probability)
    source = "trade_probability"
    fallback_to_trade_probability = True
    calibration_key = None
    calibration_used_regime_key = False
    calibration_skipped_due_to_marginal_rerank = False
    calibration_map = policy.get("calibration_map") if isinstance(policy.get("calibration_map"), Mapping) else None
    if calibration_map:
        calibration_key, params, calibration_used_regime_key = resolve_probability_calibration(
            calibration_map,
            label,
            regime_state,
        )
        if isinstance(params, Mapping):
            probability = apply_probability_calibration(float(raw_probability), params)
            source = "direction_output_calibration"
            fallback_to_trade_probability = False
        elif not bool(policy.get("use_trade_probability_fallback", True)):
            probability = float(raw_probability)
            source = "raw_probability"
            fallback_to_trade_probability = False
    elif not bool(policy.get("use_trade_probability_fallback", True)):
        probability = float(raw_probability)
        source = "raw_probability"
        fallback_to_trade_probability = False

    marginal_rerank_policy = policy.get("marginal_rerank") if isinstance(policy.get("marginal_rerank"), Mapping) else {}
    marginal_horizons = set(marginal_rerank_policy.get("horizons", []))
    if bool(marginal_rerank_policy.get("enabled", False)) and horizon_value in marginal_horizons:
        gate_probability = float(raw_probability) if bool(marginal_rerank_policy.get("use_raw_probability_gate", True)) else float(probability)
        lower = float(marginal_rerank_policy.get("lower", 0.5) or 0.5)
        upper = float(marginal_rerank_policy.get("upper", 0.6) or 0.6)
        if lower <= gate_probability <= upper:
            weight_specs = marginal_rerank_policy.get("weight_specs") if isinstance(marginal_rerank_policy.get("weight_specs"), Mapping) else {}
            weight_key = regime_state if regime_state in weight_specs else "default"
            weights = weight_specs.get(weight_key) if isinstance(weight_specs.get(weight_key), Mapping) else {}
            reranked_probability, used_components = _blend_probability_from_components(
                p_up_components,
                weights,
                min_component_count=int(marginal_rerank_policy.get("min_component_count", 2) or 2),
            )
            payload["marginal_rerank"] = {
                "enabled": True,
                "applied": reranked_probability is not None,
                "weight_key": weight_key if weights else None,
                "band": {
                    "lower": lower,
                    "upper": upper,
                },
                "component_count": int(len(used_components)),
                "components_used": used_components,
            }
            if reranked_probability is not None:
                probability = float(reranked_probability)
                source = "direction_output_marginal_rerank"
                fallback_to_trade_probability = False
                calibration_key = None
                calibration_used_regime_key = False
                calibration_skipped_due_to_marginal_rerank = True

    shrinkage_policy = policy.get("probability_shrinkage") if isinstance(policy.get("probability_shrinkage"), Mapping) else {}
    shrinkage_payload = {
        "enabled": bool(shrinkage_policy.get("enabled", False)),
        "applied": False,
        "strength": 0.0,
        "bypass_edge": float(shrinkage_policy.get("bypass_edge", 1.0) or 1.0),
    }
    if bool(shrinkage_policy.get("enabled", False)):
        scoped_horizons = set(shrinkage_policy.get("horizons", []))
        scoped_regimes = set(shrinkage_policy.get("regimes", []))
        in_horizon_scope = (not scoped_horizons) or (horizon_value in scoped_horizons)
        in_regime_scope = (not scoped_regimes) or (regime_state in scoped_regimes)
        if in_horizon_scope and in_regime_scope:
            strength = lookup_horizon_value(
                shrinkage_policy.get("strength_by_horizon", {}),
                horizon_value,
                float(shrinkage_policy.get("default_strength", 0.0) or 0.0),
            )
            strength = max(min(float(strength), 0.95), 0.0)
            bypass_edge = max(min(float(shrinkage_policy.get("bypass_edge", 1.0) or 1.0), 0.5), 0.0)
            shrinkage_payload["strength"] = float(strength)
            shrinkage_payload["bypass_edge"] = float(bypass_edge)
            if strength > 0.0 and abs(float(probability) - 0.5) < bypass_edge:
                probability = 0.5 + (float(probability) - 0.5) * (1.0 - strength)
                source = "direction_output_probability_shrinkage"
                shrinkage_payload["applied"] = True

    payload.update(
        {
            "direction": direction_from_probability(probability, neutral_band=neutral_band),
            "probability": float(probability),
            "neutral_band": float(neutral_band),
            "source": source,
            "calibration": {
                "requested_key": f"{label}@{regime_state}",
                "applied_key": calibration_key,
                "used_regime_key": calibration_used_regime_key,
                "fallback_to_trade_probability": fallback_to_trade_probability,
                "skipped_due_to_marginal_rerank": calibration_skipped_due_to_marginal_rerank,
            },
            "probability_shrinkage": shrinkage_payload,
        }
    )

    display_direction = str(payload.get("direction", base_direction)).lower()
    internal_support = 0
    display_support = 0
    for side in (ret_side, projected_side):
        if side == internal_direction:
            internal_support += 1
        if side == display_direction:
            display_support += 1

    if (
        display_direction not in {"neutral", internal_direction}
        and internal_support > 0
        and display_support == 0
    ):
        payload["forecast_alignment_override"] = {
            "applied": True,
            "reason": "fallback_to_internal_forecast_alignment",
            "candidate_direction": display_direction,
            "internal_direction": internal_direction,
            "ret_pred_side": ret_side,
            "projected_price_side": projected_side,
        }
        payload["direction"] = internal_direction

    return payload