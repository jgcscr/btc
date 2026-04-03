from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence


def normalize_trend_ignition_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trend_ignition config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key == "model_path":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key == "probability_threshold":
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "cooldown_hours":
            normalized[key] = float(raw_value) if raw_value is not None else None
        else:
            stderr_write(f"Warning: Unknown trend_ignition config key '{raw_key}' ignored.\n")
    return normalized


def normalize_direction_fallback_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("direction_only_fallback config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in {
            "prob_threshold",
            "max_negative_ev",
            "size_factor",
            "stop_take_ratio",
            "cooldown_hours",
            "ignition_ev_extension",
        }:
            normalized[key] = float(raw_value) if raw_value is not None else None
        else:
            stderr_write(f"Warning: Unknown direction_only_fallback config key '{raw_key}' ignored.\n")
    return normalized


def normalize_adaptive_thresholds_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("adaptive_thresholds config must be a mapping.")

    numeric_keys = {
        "breakout_score_threshold",
        "chop_score_threshold",
        "breakout_scale",
        "chop_scale",
        "p_up_min_floor",
        "p_up_min_ceiling",
        "ret_min_floor",
        "ret_min_ceiling",
    }
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in numeric_keys:
            normalized[key] = float(raw_value) if raw_value is not None else None
        else:
            stderr_write(f"Warning: Unknown adaptive_thresholds config key '{raw_key}' ignored.\n")
    return normalized


def normalize_target_range_block(
    value: Mapping[str, Any],
    *,
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("target_range_models config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in {"override_ratio", "confidence_rmse_scale"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "model_dir":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key == "horizons":
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                parts = [part.strip() for part in raw_value.split(",") if part.strip()]
                normalized[key] = [float(part) for part in parts]
            elif isinstance(raw_value, Sequence):
                normalized[key] = [normalize_horizon_value(entry) for entry in raw_value]
            else:
                raise ValueError("horizons in target_range_models must be list/sequence")
        else:
            stderr_write(f"Warning: Unknown target_range_models config key '{raw_key}' ignored.\n")
    return normalized


def normalize_data_quality_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("data_quality config must be a mapping.")

    normalized: Dict[str, Any] = {}
    numeric_keys = {
        "max_staleness_hours",
        "max_missing_ratio",
        "max_zero_volume_ratio",
        "min_rows",
    }
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in numeric_keys:
            if raw_value is None:
                normalized[key] = None
            elif key == "min_rows":
                normalized[key] = int(raw_value)
            else:
                normalized[key] = float(raw_value)
        else:
            stderr_write(f"Warning: Unknown data_quality config key '{raw_key}' ignored.\n")
    return normalized


def normalize_abstention_policy_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("abstention_policy config must be a mapping.")

    numeric_keys = {
        "min_confidence",
        "min_abs_expected_value",
        "min_edge_over_fee",
        "hold_prob_center",
        "hold_prob_band",
    }
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "require_positive_ev"}:
            normalized[key] = bool(raw_value)
        elif key in numeric_keys:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "thresholds_by_horizon_regime":
            if not isinstance(raw_value, Mapping):
                raise ValueError("abstention_policy.thresholds_by_horizon_regime must be a mapping.")
            normalized[key] = dict(raw_value)
        else:
            stderr_write(f"Warning: Unknown abstention_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_regime_model_weights_block(
    value: Mapping[str, Any],
    *,
    regimes: Sequence[str],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("regime_model_weights config must be a mapping.")

    normalized: Dict[str, Any] = {}
    valid_regimes = set(regimes)
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
            continue
        if key in valid_regimes:
            if isinstance(raw_value, Mapping):
                normalized[key] = {str(inner_key): str(inner_value) for inner_key, inner_value in raw_value.items()}
            else:
                normalized[key] = str(raw_value) if raw_value is not None else None
            continue
        stderr_write(f"Warning: Unknown regime_model_weights config key '{raw_key}' ignored.\n")
    return normalized


def normalize_uncertainty_policy_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("uncertainty_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "require_center_cross"}:
            normalized[key] = bool(raw_value)
        elif key in {"alpha", "hold_prob_center", "max_interval_width", "min_component_count"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "thresholds_by_horizon_regime":
            if not isinstance(raw_value, Mapping):
                raise ValueError("thresholds_by_horizon_regime in uncertainty_policy must be a mapping")
            normalized[key] = dict(raw_value)
        else:
            stderr_write(f"Warning: Unknown uncertainty_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_trade_decision_policy_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trade_decision_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {
            "enabled",
            "replace_threshold_rule",
            "require_direction_ret_alignment",
            "use_oof_expected_value",
            "enforce_positive_oof_envelope",
            "block_when_no_positive_oof_bin",
            "allow_raw_ev_fallback_when_no_positive_oof_bin",
        }:
            normalized[key] = bool(raw_value)
        elif key in {
            "threshold",
            "min_expected_net",
            "min_edge_over_fee",
            "positive_oof_min_samples",
            "raw_ev_fallback_quantile",
            "raw_ev_fallback_min_edge_over_fee",
        }:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key in {"oof_expected_value_mode", "positive_oof_envelope_mode"}:
            normalized[key] = str(raw_value).lower() if raw_value is not None else None
        elif key == "model_path":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key == "thresholds_by_horizon_regime":
            if not isinstance(raw_value, Mapping):
                raise ValueError("trade_decision_policy.thresholds_by_horizon_regime must be a mapping.")
            normalized[key] = dict(raw_value)
        elif key == "midband_veto":
            if not isinstance(raw_value, Mapping):
                raise ValueError("trade_decision_policy.midband_veto must be a mapping.")
            normalized[key] = {
                "enabled": bool(raw_value.get("enabled", False)),
                "p_up_low": float(raw_value.get("p_up_low", 0.55)),
                "p_up_high": float(raw_value.get("p_up_high", 0.60)),
                "high_inclusive": bool(raw_value.get("high_inclusive", False)),
                "min_abs_ret_pred": (
                    float(raw_value.get("min_abs_ret_pred")) if raw_value.get("min_abs_ret_pred") is not None else None
                ),
                "max_abs_ret_pred": (
                    float(raw_value.get("max_abs_ret_pred")) if raw_value.get("max_abs_ret_pred") is not None else None
                ),
                "regime_states": [
                    str(item).strip().lower()
                    for item in raw_value.get("regime_states", [])
                    if str(item).strip()
                ],
            }
        elif key == "weak_band_veto":
            if not isinstance(raw_value, Mapping):
                raise ValueError("trade_decision_policy.weak_band_veto must be a mapping.")
            normalized[key] = {
                "enabled": bool(raw_value.get("enabled", False)),
                "p_up_low": float(raw_value.get("p_up_low", 0.55)),
                "p_up_high": float(raw_value.get("p_up_high", 0.60)),
                "high_inclusive": bool(raw_value.get("high_inclusive", False)),
            }
        else:
            stderr_write(f"Warning: Unknown trade_decision_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_regime_model_dirs_block(
    value: Mapping[str, Any],
    *,
    regimes: Sequence[str],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("regime_model_dirs config must be a mapping.")

    normalized: Dict[str, Any] = {}
    valid_regimes = set(regimes)
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
            continue
        if key in valid_regimes:
            if isinstance(raw_value, Mapping):
                normalized[key] = {str(k): str(v) for k, v in raw_value.items() if v is not None}
            else:
                stderr_write(
                    f"Warning: regime_model_dirs.{raw_key} must be a mapping of horizon->path; ignored.\n"
                )
            continue
        stderr_write(f"Warning: Unknown regime_model_dirs config key '{raw_key}' ignored.\n")
    return normalized


def normalize_intrabar_aggregation_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("intrabar_aggregation config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key == "interval":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key in {"hours_multiplier", "max_rows"}:
            normalized[key] = int(raw_value) if raw_value is not None else None
        else:
            stderr_write(f"Warning: Unknown intrabar_aggregation config key '{raw_key}' ignored.\n")
    return normalized


def normalize_feature_coverage_policy_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("feature_coverage_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "block_on_violation"}:
            normalized[key] = bool(raw_value)
        elif key in {"max_imputed_zero_columns", "max_imputed_zero_ratio", "max_source_lag_hours"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "ignored_columns":
            if raw_value is None:
                normalized[key] = []
            elif isinstance(raw_value, str):
                normalized[key] = [item.strip() for item in raw_value.split(",") if item.strip()]
            elif isinstance(raw_value, Sequence):
                normalized[key] = [str(item).strip() for item in raw_value if str(item).strip()]
            else:
                raise ValueError("ignored_columns in feature_coverage_policy must be a list/sequence")
        else:
            stderr_write(f"Warning: Unknown feature_coverage_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_confluence_policy_block(
    value: Mapping[str, Any],
    *,
    parse_targets: Callable[[str], List[float]],
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("confluence_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "require_mid_term_alignment", "require_short_term_alignment"}:
            normalized[key] = bool(raw_value)
        elif key in {"min_support_ratio", "min_mid_term_ratio", "min_short_term_ratio", "dominant_ratio_floor"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "min_aligned_horizons":
            normalized[key] = int(raw_value) if raw_value is not None else None
        elif key in {"min_support_ratio_by_horizon", "min_aligned_horizons_by_horizon"}:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{key} in confluence_policy must be a mapping")
            normalized[key] = dict(raw_value)
        elif key in {"short_horizons", "mid_horizons"}:
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                normalized[key] = parse_targets(raw_value)
            elif isinstance(raw_value, Sequence):
                normalized[key] = [normalize_horizon_value(entry) for entry in raw_value]
            else:
                raise ValueError(f"{key} in confluence_policy must be a list/sequence")
        else:
            stderr_write(f"Warning: Unknown confluence_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_execution_policy_block(
    value: Mapping[str, Any],
    *,
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("execution_policy config must be a mapping.")

    numeric_keys = {
        "min_bias_alignment_ratio",
        "immediate_entry_min_support_ratio",
        "pullback_entry_min_support_ratio",
        "immediate_entry_min_mid_ratio",
        "pullback_entry_min_mid_ratio",
        "high_execution_alignment_ratio",
        "medium_execution_alignment_ratio",
        "entry_zone_atr_mult",
        "max_chase_atr_mult",
        "structure_buffer_atr_mult",
        "short_term_min_mid_ratio",
        "short_term_min_support_ratio",
    }
    integer_keys = {"session_lookback_bars", "swing_lookback_bars"}
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in {"bias_horizons", "execution_horizons", "short_term_strict_horizons"}:
            if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
                raise ValueError(f"{key} in execution_policy must be a list/sequence")
            normalized[key] = [normalize_horizon_value(item) for item in raw_value]
        elif key in numeric_keys:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key in integer_keys:
            normalized[key] = int(raw_value) if raw_value is not None else None
        elif key == "require_bias_alignment":
            normalized[key] = bool(raw_value)
        elif key in {
            "minimum_rr_by_horizon",
            "time_stop_bars_by_horizon",
            "regime_templates",
            "horizon_bias_weights",
            "short_term_min_mid_ratio_by_horizon",
            "short_term_min_support_ratio_by_horizon",
        }:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{key} in execution_policy must be a mapping")
            normalized[key] = dict(raw_value)
        elif key in {
            "partial_take_profit",
            "trailing_stop",
            "analytics",
            "no_trade_guards",
            "adaptive_take_profit",
            "target_range_stop_refinement",
            "pullback_quality",
            "disagreement_severity",
            "coherence_weighting",
            "dynamic_rr_floor",
            "volatility_expansion_stop",
        }:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{key} in execution_policy must be a mapping")
            normalized[key] = dict(raw_value)
        else:
            stderr_write(f"Warning: Unknown execution_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_degradation_monitoring_block(value: Mapping[str, Any], *, stderr_write: Callable[[str], None]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("degradation_monitoring config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in {"lookback_snapshots", "min_snapshots"}:
            normalized[key] = int(raw_value) if raw_value is not None else None
        elif key in {"min_ready_ratio", "max_blocked_ratio", "min_expected_net", "min_confidence"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        else:
            stderr_write(f"Warning: Unknown degradation_monitoring config key '{raw_key}' ignored.\n")
    return normalized


def normalize_forecast_coherence_policy_block(
    value: Mapping[str, Any],
    *,
    parse_targets: Callable[[str], List[float]],
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("forecast_coherence_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {
            "enabled",
            "block_on_direction_ret_mismatch",
            "block_on_direction_projected_price_mismatch",
            "block_on_p_up_ret_mismatch",
            "exclude_blocked_horizons_from_voting",
            "allow_consensus_p_up_ret_relief",
            "consensus_relief_exclude_from_voting",
        }:
            normalized[key] = bool(raw_value)
        elif key in {"p_up_neutral_band", "min_p_up_edge", "min_abs_ret_pred", "consensus_relief_max_p_up_edge"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key in {"horizons", "consensus_relief_horizons"}:
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                normalized[key] = parse_targets(raw_value)
            elif isinstance(raw_value, Sequence):
                normalized[key] = [normalize_horizon_value(item) for item in raw_value]
            else:
                raise ValueError(f"{key} in forecast_coherence_policy must be a list/sequence")
        else:
            stderr_write(f"Warning: Unknown forecast_coherence_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_direction_output_policy_block(
    value: Mapping[str, Any],
    *,
    parse_targets: Callable[[str], List[float]],
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("direction_output_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "use_trade_probability_fallback"}:
            normalized[key] = bool(raw_value)
        elif key == "neutral_band":
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "neutral_band_by_horizon":
            if raw_value is None:
                normalized[key] = {}
            elif isinstance(raw_value, Mapping):
                normalized[key] = dict(raw_value)
            else:
                raise ValueError("neutral_band_by_horizon in direction_output_policy must be a mapping")
        elif key == "calibration_path":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key == "probability_shrinkage":
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, Mapping):
                shrinkage: Dict[str, Any] = {}
                for raw_shrink_key, raw_shrink_value in raw_value.items():
                    shrink_key = str(raw_shrink_key).replace("-", "_")
                    if shrink_key == "enabled":
                        shrinkage[shrink_key] = bool(raw_shrink_value)
                    elif shrink_key in {"default_strength", "bypass_edge"}:
                        shrinkage[shrink_key] = float(raw_shrink_value) if raw_shrink_value is not None else None
                    elif shrink_key == "strength_by_horizon":
                        if raw_shrink_value is None:
                            shrinkage[shrink_key] = {}
                        elif isinstance(raw_shrink_value, Mapping):
                            shrinkage[shrink_key] = dict(raw_shrink_value)
                        else:
                            raise ValueError(
                                "strength_by_horizon in direction_output_policy.probability_shrinkage must be a mapping"
                            )
                    elif shrink_key in {"horizons", "regimes"}:
                        if raw_shrink_value is None:
                            shrinkage[shrink_key] = None
                        elif isinstance(raw_shrink_value, str):
                            if shrink_key == "horizons":
                                shrinkage[shrink_key] = parse_targets(raw_shrink_value)
                            else:
                                shrinkage[shrink_key] = [
                                    segment.strip() for segment in raw_shrink_value.split(",") if segment.strip()
                                ]
                        elif isinstance(raw_shrink_value, Sequence):
                            if shrink_key == "horizons":
                                shrinkage[shrink_key] = [normalize_horizon_value(item) for item in raw_shrink_value]
                            else:
                                shrinkage[shrink_key] = [
                                    str(item).strip() for item in raw_shrink_value if str(item).strip()
                                ]
                        else:
                            raise ValueError(
                                f"{shrink_key} in direction_output_policy.probability_shrinkage must be a list/sequence"
                            )
                    else:
                        stderr_write(
                            f"Warning: Unknown direction_output_policy.probability_shrinkage key '{raw_shrink_key}' ignored.\n"
                        )
                normalized[key] = shrinkage
            else:
                raise ValueError("direction_output_policy.probability_shrinkage must be a mapping")
        elif key == "marginal_rerank":
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, Mapping):
                marginal_rerank: Dict[str, Any] = {}
                for raw_marginal_key, raw_marginal_value in raw_value.items():
                    marginal_key = str(raw_marginal_key).replace("-", "_")
                    if marginal_key in {"enabled", "use_raw_probability_gate"}:
                        marginal_rerank[marginal_key] = bool(raw_marginal_value)
                    elif marginal_key in {"lower", "upper"}:
                        marginal_rerank[marginal_key] = float(raw_marginal_value) if raw_marginal_value is not None else None
                    elif marginal_key == "min_component_count":
                        marginal_rerank[marginal_key] = int(raw_marginal_value) if raw_marginal_value is not None else None
                    elif marginal_key == "horizons":
                        if raw_marginal_value is None:
                            marginal_rerank[marginal_key] = None
                        elif isinstance(raw_marginal_value, str):
                            marginal_rerank[marginal_key] = parse_targets(raw_marginal_value)
                        elif isinstance(raw_marginal_value, Sequence):
                            marginal_rerank[marginal_key] = [normalize_horizon_value(item) for item in raw_marginal_value]
                        else:
                            raise ValueError(
                                "horizons in direction_output_policy.marginal_rerank must be a list/sequence"
                            )
                    elif marginal_key == "weight_specs":
                        if raw_marginal_value is None:
                            marginal_rerank[marginal_key] = {}
                        elif isinstance(raw_marginal_value, Mapping):
                            marginal_rerank[marginal_key] = {
                                str(name): str(spec)
                                for name, spec in raw_marginal_value.items()
                                if spec is not None
                            }
                        else:
                            raise ValueError(
                                "weight_specs in direction_output_policy.marginal_rerank must be a mapping"
                            )
                    else:
                        stderr_write(
                            f"Warning: Unknown direction_output_policy.marginal_rerank key '{raw_marginal_key}' ignored.\n"
                        )
                normalized[key] = marginal_rerank
            else:
                raise ValueError("direction_output_policy.marginal_rerank must be a mapping")
        elif key == "horizons":
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                normalized[key] = parse_targets(raw_value)
            elif isinstance(raw_value, Sequence):
                normalized[key] = [normalize_horizon_value(item) for item in raw_value]
            else:
                raise ValueError("horizons in direction_output_policy must be a list/sequence")
        else:
            stderr_write(f"Warning: Unknown direction_output_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_direction_ensemble_policy_block(
    value: Mapping[str, Any],
    *,
    parse_targets: Callable[[str], List[float]],
    normalize_horizon_value: Callable[[Any], float],
    stderr_write: Callable[[str], None],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("direction_ensemble_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
        elif key in {"lookback_bars", "min_history_points"}:
            normalized[key] = int(raw_value) if raw_value is not None else None
        elif key in {"max_correlation", "min_mean_abs_probability_gap"}:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "horizons":
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                normalized[key] = parse_targets(raw_value)
            elif isinstance(raw_value, Sequence):
                normalized[key] = [normalize_horizon_value(item) for item in raw_value]
            else:
                raise ValueError("horizons in direction_ensemble_policy must be a list/sequence")
        elif key == "model_groups":
            if not isinstance(raw_value, Mapping):
                raise ValueError("model_groups in direction_ensemble_policy must be a mapping")
            normalized[key] = dict(raw_value)
        elif key in {
            "max_active_by_horizon",
            "max_models_per_group_by_horizon",
            "priority_by_horizon",
            "preferred_groups_by_horizon",
        }:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{key} in direction_ensemble_policy must be a mapping")
            normalized[key] = dict(raw_value)
        else:
            stderr_write(f"Warning: Unknown direction_ensemble_policy config key '{raw_key}' ignored.\n")
    return normalized


def normalize_config_value(
    name: str,
    value: Any,
    *,
    default_targets: Sequence[float],
    config_int_fields: Sequence[str],
    config_float_fields: Sequence[str],
    config_bool_fields: Sequence[str],
    config_path_fields: Sequence[str],
    config_allowed_keys: Sequence[str],
    regimes: Sequence[str],
    bool_env: Callable[[str], bool],
    parse_targets: Callable[[str], List[float]],
    normalize_horizon_value: Callable[[Any], float],
    normalize_horizon_float_map: Callable[..., Dict[float, float]],
    normalize_horizon_regime_float_map: Callable[..., Dict[float, Dict[str, float]]],
    stderr_write: Callable[[str], None],
) -> Any:
    if name == "targets":
        if value is None:
            return list(default_targets)
        if isinstance(value, str):
            return parse_targets(value)
        if isinstance(value, Sequence):
            normalized: List[float] = []
            for entry in value:
                normalized.append(normalize_horizon_value(entry))
            if not normalized:
                raise ValueError("Targets list from config cannot be empty.")
            return normalized
        raise ValueError(f"Invalid targets entry in config: {value!r}")
    if name in config_int_fields:
        if value is None:
            return None
        return int(value)
    if name in config_float_fields:
        if value is None:
            return None
        return float(value)
    if name in config_bool_fields:
        if isinstance(value, str):
            return bool_env(value)
        return bool(value)
    if name in config_path_fields:
        if value is None:
            return None
        return str(value)
    if name == "dir_model_weights":
        if value is None:
            return None
        return str(value)
    if name == "disabled_horizons":
        if value is None:
            return []
        if isinstance(value, str):
            return parse_targets(value)
        if isinstance(value, Sequence):
            return [normalize_horizon_value(entry) for entry in value]
        raise ValueError(f"Invalid disabled_horizons entry in config: {value!r}")
    if name == "spot_provider":
        if value is None:
            return None
        return str(value)
    if name in config_allowed_keys:
        if name == "trend_ignition" and value is not None:
            return normalize_trend_ignition_block(value, stderr_write=stderr_write)
        if name == "direction_only_fallback" and value is not None:
            return normalize_direction_fallback_block(value, stderr_write=stderr_write)
        if name == "adaptive_thresholds" and value is not None:
            return normalize_adaptive_thresholds_block(value, stderr_write=stderr_write)
        if name == "target_range_models" and value is not None:
            return normalize_target_range_block(
                value,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "data_quality" and value is not None:
            return normalize_data_quality_block(value, stderr_write=stderr_write)
        if name == "abstention_policy" and value is not None:
            return normalize_abstention_policy_block(value, stderr_write=stderr_write)
        if name == "uncertainty_policy" and value is not None:
            return normalize_uncertainty_policy_block(value, stderr_write=stderr_write)
        if name == "trade_decision_policy" and value is not None:
            return normalize_trade_decision_policy_block(value, stderr_write=stderr_write)
        if name == "regime_model_weights" and value is not None:
            return normalize_regime_model_weights_block(value, regimes=regimes, stderr_write=stderr_write)
        if name == "regime_model_dirs" and value is not None:
            return normalize_regime_model_dirs_block(value, regimes=regimes, stderr_write=stderr_write)
        if name == "intrabar_aggregation" and value is not None:
            return normalize_intrabar_aggregation_block(value, stderr_write=stderr_write)
        if name == "feature_coverage_policy" and value is not None:
            return normalize_feature_coverage_policy_block(value, stderr_write=stderr_write)
        if name == "confluence_policy" and value is not None:
            return normalize_confluence_policy_block(
                value,
                parse_targets=parse_targets,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "execution_policy" and value is not None:
            return normalize_execution_policy_block(
                value,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "forecast_coherence_policy" and value is not None:
            return normalize_forecast_coherence_policy_block(
                value,
                parse_targets=parse_targets,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "direction_output_policy" and value is not None:
            return normalize_direction_output_policy_block(
                value,
                parse_targets=parse_targets,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "direction_ensemble_policy" and value is not None:
            return normalize_direction_ensemble_policy_block(
                value,
                parse_targets=parse_targets,
                normalize_horizon_value=normalize_horizon_value,
                stderr_write=stderr_write,
            )
        if name == "degradation_monitoring" and value is not None:
            return normalize_degradation_monitoring_block(value, stderr_write=stderr_write)
        if name == "position_size_cap_by_horizon" and value is not None:
            return normalize_horizon_float_map(value, minimum=0.0, maximum=1.0)
        if name == "confidence_min_by_horizon_regime" and value is not None:
            return normalize_horizon_regime_float_map(value, minimum=0.0, maximum=1.0)
        return value
    raise ValueError(f"Unsupported config key: {name}")