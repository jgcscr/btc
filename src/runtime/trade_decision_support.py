from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np


SummaryPayload = Dict[str, Dict[str, Any]]
ExecutionContexts = Mapping[str, Mapping[str, Any]]


def resolve_trade_decision_policy(
    config: Mapping[str, Any] | None,
    *,
    finite_float_or_none: Callable[[Any], float | None],
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    cfg = config or {}
    model_payload = None
    model_path = cfg.get("model_path")
    if model_path:
        path = Path(str(model_path)).expanduser()
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    model_payload = payload
            except Exception as exc:
                stderr_write(f"Warning: failed to parse trade decision model {path}: {exc}\n")
        else:
            stderr_write(f"Warning: trade decision model not found at {path}; policy disabled.\n")
    enabled = bool(cfg.get("enabled", False) and model_payload is not None)
    midband_veto_cfg = cfg.get("midband_veto") if isinstance(cfg.get("midband_veto"), Mapping) else {}
    weak_band_veto_cfg = cfg.get("weak_band_veto") if isinstance(cfg.get("weak_band_veto"), Mapping) else {}
    thresholds_by_horizon_regime: Dict[float, Dict[str, float]] = {}
    raw_thresholds = (
        cfg.get("thresholds_by_horizon_regime")
        if isinstance(cfg.get("thresholds_by_horizon_regime"), Mapping)
        else {}
    )
    for raw_horizon, raw_regimes in raw_thresholds.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None or not isinstance(raw_regimes, Mapping):
            continue
        resolved_regimes: Dict[str, float] = {}
        for raw_regime, raw_value in raw_regimes.items():
            regime_key = str(raw_regime).strip().lower()
            threshold_value: float | None = None
            if isinstance(raw_value, Mapping):
                threshold_value = finite_float_or_none(raw_value.get("threshold"))
            else:
                threshold_value = finite_float_or_none(raw_value)
            if threshold_value is None:
                continue
            resolved_regimes[regime_key] = max(0.0, min(1.0, float(threshold_value)))
        if resolved_regimes:
            thresholds_by_horizon_regime[normalize_horizon_value(horizon)] = resolved_regimes
    return {
        "enabled": enabled,
        "replace_threshold_rule": bool(cfg.get("replace_threshold_rule", True)),
        "require_direction_ret_alignment": bool(cfg.get("require_direction_ret_alignment", True)),
        "use_oof_expected_value": bool(cfg.get("use_oof_expected_value", True)),
        "oof_expected_value_mode": str(cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")).lower(),
        "enforce_positive_oof_envelope": bool(cfg.get("enforce_positive_oof_envelope", False)),
        "positive_oof_envelope_mode": str(cfg.get("positive_oof_envelope_mode", "strict_positive_bin")).lower(),
        "block_when_no_positive_oof_bin": bool(cfg.get("block_when_no_positive_oof_bin", True)),
        "positive_oof_min_samples": int(float(cfg.get("positive_oof_min_samples", 4))),
        "allow_raw_ev_fallback_when_no_positive_oof_bin": bool(cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)),
        "raw_ev_fallback_quantile": float(cfg.get("raw_ev_fallback_quantile", 0.9)),
        "raw_ev_fallback_min_edge_over_fee": float(cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0)),
        "threshold": float(cfg.get("threshold") if cfg.get("threshold") is not None else (model_payload or {}).get("threshold", 0.55)),
        "thresholds_by_horizon_regime": thresholds_by_horizon_regime,
        "min_expected_net": float(cfg.get("min_expected_net", 0.0)),
        "min_edge_over_fee": float(cfg.get("min_edge_over_fee", 0.0)),
        "midband_veto": {
            "enabled": bool(midband_veto_cfg.get("enabled", False)),
            "p_up_low": float(midband_veto_cfg.get("p_up_low", 0.55)),
            "p_up_high": float(midband_veto_cfg.get("p_up_high", 0.60)),
            "high_inclusive": bool(midband_veto_cfg.get("high_inclusive", False)),
            "min_abs_ret_pred": (
                float(midband_veto_cfg.get("min_abs_ret_pred"))
                if midband_veto_cfg.get("min_abs_ret_pred") is not None
                else None
            ),
            "max_abs_ret_pred": (
                float(midband_veto_cfg.get("max_abs_ret_pred"))
                if midband_veto_cfg.get("max_abs_ret_pred") is not None
                else None
            ),
            "regime_states": [
                str(value).strip().lower()
                for value in (
                    midband_veto_cfg.get("regime_states", [])
                    if isinstance(midband_veto_cfg.get("regime_states", []), list)
                    else []
                )
                if str(value).strip()
            ],
        },
        "weak_band_veto": {
            "enabled": bool(weak_band_veto_cfg.get("enabled", False)),
            "p_up_low": float(weak_band_veto_cfg.get("p_up_low", 0.55)),
            "p_up_high": float(weak_band_veto_cfg.get("p_up_high", 0.60)),
            "high_inclusive": bool(weak_band_veto_cfg.get("high_inclusive", False)),
        },
        "model": model_payload,
    }


def resolve_trade_decision_threshold(
    policy: Mapping[str, Any],
    *,
    horizon_label: str | None,
    regime_state: str,
    normalize_horizon_value: Callable[[float], float],
    parse_horizon_label: Callable[[str], float],
    format_horizon_label: Callable[[float], str],
) -> tuple[float, str]:
    base_threshold = max(0.0, min(1.0, float(policy.get("threshold", 0.55))))
    source = "default"
    if not horizon_label:
        return base_threshold, source

    try:
        horizon = normalize_horizon_value(parse_horizon_label(horizon_label))
    except (TypeError, ValueError):
        return base_threshold, source

    overrides = (
        policy.get("thresholds_by_horizon_regime")
        if isinstance(policy.get("thresholds_by_horizon_regime"), Mapping)
        else {}
    )
    regime_overrides = overrides.get(horizon)
    if not isinstance(regime_overrides, Mapping):
        return base_threshold, source

    regime_key = str(regime_state).strip().lower()
    override = regime_overrides.get(regime_key)
    if override is None:
        override = regime_overrides.get("default")
        if override is None:
            return base_threshold, source
        source = f"{format_horizon_label(horizon)}@default"
    else:
        source = f"{format_horizon_label(horizon)}@{regime_key}"
    return max(0.0, min(1.0, float(override))), source


def lookup_raw_ev_fallback_threshold(
    model: Mapping[str, Any],
    quantile: float,
    *,
    finite_float_or_none: Callable[[Any], float | None],
) -> float | None:
    payload = model.get("raw_ev_fallback") if isinstance(model, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    quantiles = payload.get("quantiles")
    if not isinstance(quantiles, Mapping):
        return None

    q = float(max(0.0, min(1.0, quantile)))
    key = f"q{int(round(q * 100))}"
    direct = quantiles.get(key)
    if direct is not None:
        return finite_float_or_none(direct)

    best_dist = float("inf")
    best_value: float | None = None
    for raw_key, raw_value in quantiles.items():
        if not isinstance(raw_key, str) or not raw_key.startswith("q"):
            continue
        try:
            candidate_quantile = float(raw_key[1:]) / 100.0
            dist = abs(candidate_quantile - q)
            if dist < best_dist:
                best_dist = dist
                best_value = finite_float_or_none(raw_value)
        except Exception:
            continue
    return best_value


def apply_trade_decision_model(
    *,
    result: Dict[str, Any],
    horizon_label: str | None = None,
    regime_state: str,
    residual_std: float,
    policy: Mapping[str, Any],
    fee_bps: float,
    slippage_bps: float,
    regime_trend: str,
    regime_neutral: str,
    regime_chop: str,
    resolve_trade_decision_threshold: Callable[[Mapping[str, Any]], tuple[float, str]] | Callable[..., tuple[float, str]],
    sigmoid: Callable[[float], float],
    finite_float_or_none: Callable[[Any], float | None],
    finite_float: Callable[[Any, float], float],
) -> Dict[str, Any]:
    if not policy or not bool(policy.get("enabled", False)):
        return {
            "enabled": bool(policy.get("enabled", False)) if isinstance(policy, Mapping) else False,
            "triggered": False,
            "reason": "disabled",
        }

    model = policy.get("model") if isinstance(policy, Mapping) else None
    if not isinstance(model, Mapping):
        return {"enabled": True, "triggered": False, "reason": "missing_model"}

    feature_names = [str(value) for value in model.get("feature_columns", [])] if isinstance(model.get("feature_columns"), list) else []
    coefficients = [float(value) for value in model.get("coefficients", [])] if isinstance(model.get("coefficients"), list) else []
    intercept = float(model.get("intercept", 0.0))
    if not feature_names or len(feature_names) != len(coefficients):
        return {"enabled": True, "triggered": False, "reason": "bad_model_shape"}

    vol_payload = result.get("volatility", {}) if isinstance(result.get("volatility"), Mapping) else {}
    vol_snapshot = vol_payload.get("snapshot", {}) if isinstance(vol_payload, Mapping) else {}

    feature_values: Dict[str, float] = {
        "p_up": float(result.get("p_up", 0.0)),
        "raw_p_up": float(result.get("raw_p_up", result.get("p_up", 0.0))),
        "ret_pred": float(result.get("ret_pred", 0.0)),
        "expected_value_proxy": float(result.get("p_up", 0.0)) * float(result.get("ret_pred", 0.0)),
        "abs_ret_pred": abs(float(result.get("ret_pred", 0.0))),
        "raw_calibrated_probability_gap": float(result.get("raw_calibrated_probability_gap", 0.0) or 0.0),
        "probability_alignment_gap": float(result.get("probability_alignment_gap", 0.0) or 0.0),
        "raw_p_up_ret_mismatch": float(result.get("raw_p_up_ret_mismatch", 0.0) or 0.0),
        "p_up_ret_mismatch": float(result.get("p_up_ret_mismatch", 0.0) or 0.0),
        "raw_p_up_direction_mismatch": float(result.get("raw_p_up_direction_mismatch", 0.0) or 0.0),
        "p_up_direction_mismatch": float(result.get("p_up_direction_mismatch", 0.0) or 0.0),
        "ret_projected_price_consensus": float(result.get("ret_projected_price_consensus", 0.0) or 0.0),
        "probability_calibration_guard_applied": float(result.get("probability_calibration_guard_applied", 0.0) or 0.0),
        "probability_calibration_used_regime_key": float(result.get("probability_calibration_used_regime_key", 0.0) or 0.0),
        "residual_std": float(residual_std),
        "confidence_score": float(result.get("confidence_score", 0.0)),
        "position_size": float(result.get("position_size", 0.0)),
        "volatility_realized_24h": float(vol_snapshot.get("volatility_realized_24h", 0.0) or 0.0),
        "volatility_ewm_24h": float(vol_snapshot.get("volatility_ewm_24h", 0.0) or 0.0),
        "volatility_garch_like": float(vol_snapshot.get("volatility_garch_like", 0.0) or 0.0),
        "range_expansion_1h": float(result.get("range_expansion_1h", 0.0) or 0.0),
        "distance_from_session_high_8h": float(result.get("distance_from_session_high_8h", 0.0) or 0.0),
        "distance_from_session_low_8h": float(result.get("distance_from_session_low_8h", 0.0) or 0.0),
        "vwap_deviation_8h": float(result.get("vwap_deviation_8h", 0.0) or 0.0),
        "momentum_slope_2h": float(result.get("momentum_slope_2h", 0.0) or 0.0),
        "momentum_slope_4h": float(result.get("momentum_slope_4h", 0.0) or 0.0),
        "confluence_support_ratio": float(result.get("confluence_support_ratio", 0.0) or 0.0),
        "confluence_short_term_ratio": float(result.get("confluence_short_term_ratio", 0.0) or 0.0),
        "confluence_mid_term_ratio": float(result.get("confluence_mid_term_ratio", 0.0) or 0.0),
        "confluence_direction_matches_dominant": float(result.get("confluence_direction_matches_dominant", 0.0) or 0.0),
        "incumbent_signal_reference": float(result.get("incumbent_signal_reference", 0.0) or 0.0),
        "candidate_only_reference": float(result.get("candidate_only_reference", 0.0) or 0.0),
        "candidate_incumbent_disagreement": float(result.get("candidate_incumbent_disagreement", 0.0) or 0.0),
        "regime_is_trend": 1.0 if regime_state == regime_trend else 0.0,
        "regime_is_neutral": 1.0 if regime_state == regime_neutral else 0.0,
        "regime_is_chop": 1.0 if regime_state == regime_chop else 0.0,
    }

    logit = intercept
    for name, coef in zip(feature_names, coefficients):
        logit += coef * float(feature_values.get(name, 0.0))
    trade_prob = sigmoid(logit)

    threshold, threshold_source = resolve_trade_decision_threshold(
        policy,
        horizon_label=horizon_label,
        regime_state=regime_state,
    )
    expected_net_raw = finite_float(result.get("expected_value", 0.0), 0.0)
    expected_net_oof = _lookup_oof_expected_net(model, trade_prob, finite_float_or_none=finite_float_or_none, finite_float=finite_float)
    expected_net_raw_calibrated = _lookup_raw_ev_expected_net(
        model,
        expected_net_raw,
        finite_float_or_none=finite_float_or_none,
    )
    use_oof_expected_value = bool(policy.get("use_oof_expected_value", True))
    oof_mode = str(policy.get("oof_expected_value_mode", "max_with_raw_calibrated")).lower()
    if oof_mode == "calibrated_only":
        expected_net = float(expected_net_raw_calibrated) if expected_net_raw_calibrated is not None else float(expected_net_raw)
    elif use_oof_expected_value and expected_net_oof is not None and oof_mode == "strict":
        expected_net = float(expected_net_oof)
    elif use_oof_expected_value and expected_net_oof is not None and oof_mode == "blend":
        expected_net = 0.5 * (float(expected_net_raw) + float(expected_net_oof))
    else:
        candidates = [float(expected_net_raw)]
        if use_oof_expected_value and expected_net_oof is not None:
            candidates.append(float(expected_net_oof))
        if expected_net_raw_calibrated is not None:
            candidates.append(float(expected_net_raw_calibrated))
        finite_candidates = [value for value in candidates if math.isfinite(value)]
        expected_net = max(finite_candidates) if finite_candidates else float("nan")
    expected_net_valid = math.isfinite(expected_net)
    fee_cost = (float(fee_bps) + float(slippage_bps)) / 10_000.0
    edge_over_fee = (expected_net - fee_cost) if expected_net_valid else float("-inf")
    ret_pred = finite_float(result.get("ret_pred", 0.0), 0.0)
    signal_dir_only = int(result.get("signal_dir_only", 0))
    aligned = ((signal_dir_only == 1 and ret_pred > 0.0) or (signal_dir_only == 0 and ret_pred < 0.0))

    trade_ok = trade_prob >= threshold
    if not expected_net_valid:
        trade_ok = False
    if expected_net < float(policy.get("min_expected_net", 0.0)):
        trade_ok = False
    if edge_over_fee < float(policy.get("min_edge_over_fee", 0.0)):
        trade_ok = False
    if bool(policy.get("require_direction_ret_alignment", True)) and not aligned:
        trade_ok = False

    envelope = _oof_positive_envelope_status(
        model,
        trade_prob,
        min_samples=int(policy.get("positive_oof_min_samples", 4)),
    )
    envelope_mode = str(policy.get("positive_oof_envelope_mode", "strict_positive_bin")).lower()
    raw_ev_fallback_threshold: float | None = None
    raw_ev_fallback_pass = False
    if bool(policy.get("enforce_positive_oof_envelope", False)) and envelope.get("available", False):
        has_positive = bool(envelope.get("has_positive_bin", False))
        in_positive = bool(envelope.get("in_positive_bin", False))
        matched_populated_bin = bool(envelope.get("matched_populated_bin", False))
        matched_positive_bin = bool(envelope.get("matched_positive_bin", False))
        if envelope_mode == "populated_bin_sign":
            if matched_populated_bin and not matched_positive_bin:
                trade_ok = False
        elif has_positive and not in_positive:
            trade_ok = False
        if (not has_positive) and bool(policy.get("block_when_no_positive_oof_bin", True)):
            allow_raw_fallback = bool(policy.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False))
            if allow_raw_fallback:
                fallback_threshold = lookup_raw_ev_fallback_threshold(
                    model,
                    quantile=float(policy.get("raw_ev_fallback_quantile", 0.9)),
                    finite_float_or_none=finite_float_or_none,
                )
                raw_edge_over_fee = float(expected_net_raw) - fee_cost
                min_raw_edge = float(policy.get("raw_ev_fallback_min_edge_over_fee", 0.0))
                raw_ev_fallback_threshold = None if fallback_threshold is None else float(fallback_threshold)
                fallback_pass = (
                    fallback_threshold is not None
                    and float(expected_net_raw) >= float(fallback_threshold)
                    and raw_edge_over_fee >= min_raw_edge
                )
                raw_ev_fallback_pass = bool(fallback_pass)
                if not fallback_pass:
                    trade_ok = False
            else:
                trade_ok = False

    weak_band_veto_cfg = policy.get("weak_band_veto") if isinstance(policy.get("weak_band_veto"), Mapping) else {}
    weak_band_veto_triggered = False
    weak_band_veto_reason = "disabled"
    if bool(weak_band_veto_cfg.get("enabled", False)) and trade_ok:
        p_up_low = float(weak_band_veto_cfg.get("p_up_low", 0.55))
        p_up_high = float(weak_band_veto_cfg.get("p_up_high", 0.60))
        high_inclusive = bool(weak_band_veto_cfg.get("high_inclusive", False))
        in_band = (feature_values["p_up"] >= p_up_low) and (
            feature_values["p_up"] <= p_up_high if high_inclusive else feature_values["p_up"] < p_up_high
        )
        if in_band:
            trade_ok = False
            weak_band_veto_triggered = True
            weak_band_veto_reason = "weak_band_veto"

    if bool(policy.get("replace_threshold_rule", True)):
        midband_veto_cfg = policy.get("midband_veto") if isinstance(policy.get("midband_veto"), Mapping) else {}
        midband_veto_triggered = False
        midband_veto_reason = "disabled"
        if bool(midband_veto_cfg.get("enabled", False)) and trade_ok:
            p_up_low = float(midband_veto_cfg.get("p_up_low", 0.55))
            p_up_high = float(midband_veto_cfg.get("p_up_high", 0.60))
            high_inclusive = bool(midband_veto_cfg.get("high_inclusive", False))
            regime_filters = [
                str(value).strip().lower()
                for value in (
                    midband_veto_cfg.get("regime_states", [])
                    if isinstance(midband_veto_cfg.get("regime_states", []), list)
                    else []
                )
                if str(value).strip()
            ]
            abs_ret_pred = abs(ret_pred)
            in_band = (feature_values["p_up"] >= p_up_low) and (
                feature_values["p_up"] <= p_up_high if high_inclusive else feature_values["p_up"] < p_up_high
            )
            if regime_filters and regime_state not in regime_filters:
                in_band = False
            if in_band:
                min_abs_ret_pred = midband_veto_cfg.get("min_abs_ret_pred")
                max_abs_ret_pred = midband_veto_cfg.get("max_abs_ret_pred")
                if min_abs_ret_pred is not None and abs_ret_pred < float(min_abs_ret_pred):
                    in_band = False
                if max_abs_ret_pred is not None and abs_ret_pred >= float(max_abs_ret_pred):
                    in_band = False
            if in_band:
                trade_ok = False
                midband_veto_triggered = True
                midband_veto_reason = "midband_veto"
    else:
        midband_veto_triggered = False
        midband_veto_reason = "replace_threshold_rule_disabled"

    proposed_signal_ensemble = int(trade_ok)
    proposed_trade_action = (
        "long" if proposed_signal_ensemble == 1 and signal_dir_only == 1 else
        "short" if proposed_signal_ensemble == 1 and signal_dir_only == 0 else
        "hold"
    )

    return {
        "enabled": True,
        "triggered": bool(trade_ok),
        "proposed_signal_ensemble": int(proposed_signal_ensemble),
        "proposed_trade_action": proposed_trade_action,
        "trade_probability": float(trade_prob),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "expected_net": (float(expected_net) if expected_net_valid else None),
        "expected_net_valid": bool(expected_net_valid),
        "expected_net_raw": float(expected_net_raw),
        "expected_net_raw_calibrated": None if expected_net_raw_calibrated is None else float(expected_net_raw_calibrated),
        "expected_net_oof": None if expected_net_oof is None else float(expected_net_oof),
        "oof_expected_value_mode": oof_mode,
        "use_oof_expected_value": use_oof_expected_value,
        "positive_oof_envelope": envelope,
        "positive_oof_envelope_mode": envelope_mode,
        "enforce_positive_oof_envelope": bool(policy.get("enforce_positive_oof_envelope", False)),
        "allow_raw_ev_fallback_when_no_positive_oof_bin": bool(
            policy.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)
        ),
        "raw_ev_fallback_quantile": float(policy.get("raw_ev_fallback_quantile", 0.9)),
        "raw_ev_fallback_threshold": raw_ev_fallback_threshold,
        "raw_ev_fallback_pass": bool(raw_ev_fallback_pass),
        "edge_over_fee": (float(edge_over_fee) if math.isfinite(edge_over_fee) else None),
        "direction_ret_aligned": bool(aligned),
        "replace_threshold_rule": bool(policy.get("replace_threshold_rule", True)),
        "weak_band_veto": {
            "enabled": bool((policy.get("weak_band_veto") or {}).get("enabled", False)) if isinstance(policy, Mapping) else False,
            "triggered": bool(weak_band_veto_triggered),
            "reason": weak_band_veto_reason,
        },
        "midband_veto": {
            "enabled": bool((policy.get("midband_veto") or {}).get("enabled", False)) if isinstance(policy, Mapping) else False,
            "triggered": bool(midband_veto_triggered),
            "reason": midband_veto_reason,
        },
        "feature_snapshot": {
            name: float(feature_values.get(name, 0.0))
            for name in feature_names
            if name in feature_values
        },
    }


def upstream_trade_gate_reasons(entry: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    confluence = entry.get("confluence") if isinstance(entry.get("confluence"), Mapping) else {}
    if bool(coherence.get("triggered", False)):
        reasons.append("forecast_coherence_gate")
    if bool(confluence.get("triggered", False)):
        reasons.append("confluence_gate")
    return reasons


def apply_trade_decision_stage(
    summary: SummaryPayload,
    execution_contexts: ExecutionContexts,
    policy: Mapping[str, Any],
    *,
    default_residual_std: float,
    regime_neutral: str,
    default_fee_bps: float,
    default_slippage_bps: float,
    apply_trade_decision_model: Callable[..., Dict[str, Any]],
    upstream_trade_gate_reasons: Callable[[Mapping[str, Any]], list[str]],
    append_gate_trace: Callable[..., None],
) -> SummaryPayload:
    for label, entry in summary.items():
        residual_std = float(execution_contexts.get(label, {}).get("residual_std", default_residual_std))
        regime_state = str(entry.get("regime_state") or regime_neutral)
        payload = apply_trade_decision_model(
            result=entry,
            horizon_label=label,
            regime_state=regime_state,
            residual_std=residual_std,
            policy=policy,
            fee_bps=float(default_fee_bps),
            slippage_bps=float(default_slippage_bps),
        )
        reason = str(payload.get("reason", ""))
        if reason in {"disabled", "missing_model", "bad_model_shape"}:
            entry["trade_decision"] = payload
            continue
        upstream_reasons = upstream_trade_gate_reasons(entry)
        if upstream_reasons:
            payload["pre_upstream_gate_triggered"] = bool(payload.get("triggered", False))
            payload["triggered"] = False
            payload["blocked"] = True
            payload["blocking_reason"] = ",".join(upstream_reasons)
            payload["upstream_gate_reasons"] = list(upstream_reasons)
            append_gate_trace(
                entry,
                stage="trade_decision",
                reason=payload["blocking_reason"],
                triggered=True,
                blocking=True,
            )
        else:
            if bool(policy.get("replace_threshold_rule", True)):
                entry["signal_ensemble"] = int(payload.get("proposed_signal_ensemble", entry.get("signal_ensemble", 0)))
                entry["trade_action"] = str(payload.get("proposed_trade_action", entry.get("trade_action", "hold")))
            if bool(payload.get("triggered", False)):
                append_gate_trace(
                    entry,
                    stage="trade_decision",
                    reason="pass",
                    triggered=True,
                    blocking=False,
                )
            else:
                gate_reason = "threshold_or_envelope_veto"
                midband = payload.get("midband_veto") if isinstance(payload.get("midband_veto"), Mapping) else {}
                weak_band = payload.get("weak_band_veto") if isinstance(payload.get("weak_band_veto"), Mapping) else {}
                if bool(midband.get("triggered", False)):
                    gate_reason = "midband_veto"
                elif bool(weak_band.get("triggered", False)):
                    gate_reason = "weak_band_veto"
                append_gate_trace(
                    entry,
                    stage="trade_decision",
                    reason=gate_reason,
                    triggered=True,
                    blocking=True,
                )
        entry["trade_decision"] = payload
    return summary


def _lookup_oof_expected_net(
    model: Mapping[str, Any],
    prob: float,
    *,
    finite_float_or_none: Callable[[Any], float | None],
    finite_float: Callable[[Any, float], float],
) -> float | None:
    oof_payload = model.get("oof_expected_value") if isinstance(model, Mapping) else None
    if not isinstance(oof_payload, Mapping):
        return None
    bins = oof_payload.get("bins")
    if not isinstance(bins, list):
        return None
    prob_value = finite_float_or_none(prob)
    if prob_value is None:
        return None
    p = float(max(0.0, min(1.0, prob_value)))
    for idx, bucket in enumerate(bins):
        if not isinstance(bucket, Mapping):
            continue
        lo = finite_float(bucket.get("p_min", 0.0), 0.0)
        hi = finite_float(bucket.get("p_max", 1.0), 1.0)
        in_range = (p >= lo and p < hi) if idx < len(bins) - 1 else (p >= lo and p <= hi)
        if in_range:
            return finite_float_or_none(bucket.get("mean_ret_net", 0.0))
    return finite_float_or_none(oof_payload.get("default_expected_net"))


def _lookup_raw_ev_expected_net(
    model: Mapping[str, Any],
    raw_ev: float,
    *,
    finite_float_or_none: Callable[[Any], float | None],
) -> float | None:
    raw_ev_value = finite_float_or_none(raw_ev)
    if raw_ev_value is None:
        return None
    iso_payload = model.get("raw_ev_isotonic") if isinstance(model, Mapping) else None
    if isinstance(iso_payload, Mapping):
        x_vals = iso_payload.get("x_thresholds")
        y_vals = iso_payload.get("y_thresholds")
        if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) >= 2 and len(x_vals) == len(y_vals):
            try:
                x = np.asarray([float(value) for value in x_vals], dtype=float)
                y = np.asarray([float(value) for value in y_vals], dtype=float)
                interpolated = float(np.interp(float(raw_ev_value), x, y, left=y[0], right=y[-1]))
                return finite_float_or_none(interpolated)
            except Exception:
                pass

    payload = model.get("raw_ev_expected_value") if isinstance(model, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    bins = payload.get("bins")
    if not isinstance(bins, list):
        return None
    value = float(raw_ev_value)
    for idx, bucket in enumerate(bins):
        if not isinstance(bucket, Mapping):
            continue
        lo = _finite_float(bucket.get("x_min", float("-inf")), float("-inf"), finite_float_or_none=finite_float_or_none)
        hi = _finite_float(bucket.get("x_max", float("inf")), float("inf"), finite_float_or_none=finite_float_or_none)
        in_range = (value >= lo and value < hi) if idx < len(bins) - 1 else (value >= lo and value <= hi)
        if in_range:
            return finite_float_or_none(bucket.get("mean_ret_net", 0.0))
    return finite_float_or_none(payload.get("default_expected_net"))


def _oof_positive_envelope_status(model: Mapping[str, Any], prob: float, min_samples: int) -> Dict[str, Any]:
    oof_payload = model.get("oof_expected_value") if isinstance(model, Mapping) else None
    if not isinstance(oof_payload, Mapping):
        return {
            "available": False,
            "positive_bin_count": 0,
            "has_positive_bin": False,
            "in_positive_bin": False,
        }

    bins = oof_payload.get("bins")
    if not isinstance(bins, list):
        return {
            "available": False,
            "positive_bin_count": 0,
            "has_positive_bin": False,
            "in_positive_bin": False,
        }

    probability = float(max(0.0, min(1.0, prob)))
    positive_ranges: list[tuple[float, float]] = []
    populated_bin_count = 0
    matched_populated_bin = False
    matched_positive_bin = False
    matched_bin_mean_ret_net: float | None = None
    matched_bin_samples = 0
    best_positive_mean = float("-inf")
    for idx, bucket in enumerate(bins):
        if not isinstance(bucket, Mapping):
            continue
        count = int(bucket.get("samples", 0) or 0)
        mean_ret_net = float(bucket.get("mean_ret_net", 0.0))
        lo = float(bucket.get("p_min", 0.0))
        hi = float(bucket.get("p_max", 1.0))
        in_range = (probability >= lo and probability < hi) if idx < len(bins) - 1 else (probability >= lo and probability <= hi)
        if count >= int(min_samples):
            populated_bin_count += 1
            if in_range:
                matched_populated_bin = True
                matched_positive_bin = mean_ret_net > 0.0
                matched_bin_mean_ret_net = float(mean_ret_net)
                matched_bin_samples = count
            if mean_ret_net > 0.0:
                positive_ranges.append((lo, hi))
                best_positive_mean = max(best_positive_mean, mean_ret_net)

    in_positive = any((probability >= lo and probability <= hi) for lo, hi in positive_ranges)
    return {
        "available": True,
        "positive_bin_count": int(len(positive_ranges)),
        "populated_bin_count": int(populated_bin_count),
        "has_positive_bin": bool(len(positive_ranges) > 0),
        "in_positive_bin": bool(in_positive),
        "matched_populated_bin": bool(matched_populated_bin),
        "matched_positive_bin": bool(matched_positive_bin),
        "matched_bin_mean_ret_net": matched_bin_mean_ret_net,
        "matched_bin_samples": int(matched_bin_samples),
        "best_positive_mean_ret_net": (None if best_positive_mean == float("-inf") else float(best_positive_mean)),
    }


def _finite_float(value: Any, default: float, *, finite_float_or_none: Callable[[Any], float | None]) -> float:
    out = finite_float_or_none(value)
    return float(default) if out is None else float(out)