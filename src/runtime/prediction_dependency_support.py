from __future__ import annotations

import math
import sys
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from src.config_trading import (
    DEFAULT_DIR_MODELS_1H,
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX,
)
from src.runtime.confluence_support import apply_confluence_policy, resolve_confluence_policy
from src.runtime.dataset_profile_support import DatasetCandidate, DatasetProfile
from src.runtime.direction_output_support import (
    apply_probability_calibration,
    build_direction_output as runtime_build_direction_output,
    resolve_direction_output_policy,
    resolve_probability_calibration,
    resolve_trade_probability_for_horizon,
)
from src.runtime.execution_policy_support import apply_execution_policy, resolve_execution_policy
from src.runtime.execution_policy_support import (
    build_entry_zone,
    classify_execution_tier,
    compute_atr_like_price_distance,
    compute_disagreement_severity,
    compute_excursion_priors,
    compute_recent_structure,
    compute_pullback_quality_score,
    execution_alignment_ratio,
    execution_side,
    finite_float,
    lookup_horizon_value as runtime_lookup_horizon_value,
    refine_stop_with_target_range,
    resolve_execution_target_reward,
    resolve_execution_upstream_hold_reason,
    resolve_stop_with_guardrails,
    summarize_bias_context,
)
from src.runtime.forecast_coherence_support import apply_forecast_coherence_policy, resolve_forecast_coherence_policy
from src.runtime.gate_trace_support import append_gate_trace
from src.runtime.horizon_support import coerce_numeric_horizon, format_horizon_label, normalize_horizon_value
from src.runtime.model_resolution_support import (
    direction_configs_for_horizon as runtime_direction_configs_for_horizon,
    model_paths_for_horizon as runtime_model_paths_for_horizon,
    prepare_base_direction_configs as runtime_prepare_base_direction_configs,
)
from src.runtime.policy_support import (
    apply_regime_weight_overrides,
    get_active_regime_weight_override,
    normalize_horizon_float_map,
    normalize_horizon_regime_float_map,
    normalize_threshold_overrides,
    resolve_confidence_min_for_horizon,
    resolve_direction_ensemble_policy,
    resolve_regime_model_dirs_policy,
    resolve_regime_model_weights_policy,
    resolve_regime_specific_dir_path,
    resolve_regression_dir_path,
    resolve_regression_model_dirs_policy,
    resolve_thresholds_for_horizon,
    scope_direction_ensemble_policy,
)
from src.runtime.prediction_defaults import (
    BREAKOUT_RET_NORMALIZER,
    BREAKOUT_VOL_NORMALIZER,
    CONFIDENCE_MIN_DEFAULT,
    DIR_VERSION_OVERRIDES,
    EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS,
    EXECUTION_POLICY_DEFAULT_MIN_SAMPLES,
    EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT,
    EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN,
    EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_HORIZONS,
    EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION,
    MODEL_VERSION_PRIORITY,
    MIN_DIRECTIONAL_RETURN_BUFFER,
    REGIME_CALIBRATION_MIN_PLATT_SLOPE,
    REGIME_CHOP,
    REGIME_NEUTRAL,
    REGIME_TREND,
    TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE,
    TARGET_RANGE_DEFAULT_HORIZONS,
    TARGET_RANGE_DEFAULT_OVERRIDE_RATIO,
)
from src.runtime.prediction_paths import (
    DATASET_15M_PATH,
    DATASET_1H_PATH,
    DATASET_MULTI_PATH,
    DIRECTION_FALLBACK_STATE_PATH,
    MODEL_ROOT,
    TARGET_RANGE_MODEL_DIR,
    TREND_IGNITION_STATE_PATH,
)
from src.runtime.post_prediction_pipeline import apply_post_prediction_policies
from src.runtime.post_trade_support import (
    apply_abstention_policy,
    apply_uncertainty_abstention,
    apply_post_trade_gates,
    resolve_abstention_expected_value,
    resolve_abstention_policy,
    resolve_abstention_policy_for_horizon,
    resolve_uncertainty_settings,
    resolve_uncertainty_policy,
)
from src.runtime.prediction_pipeline import PredictionPipelineDependencies
from src.runtime.prediction_result_support import build_prediction_result
from src.runtime.refresh_support import (
    dataset_profile_for_horizon,
    load_prepared,
    load_prepared_offline,
    project_price,
    select_dataset_candidate,
)
from src.runtime.regime_policy_support import (
    apply_adaptive_thresholds,
    classify_regime_from_score,
    compute_breakout_scores,
    compute_profile_breakout_score,
    load_last_trigger_ts,
    resolve_adaptive_thresholds_policy,
    resolve_direction_fallback_policy,
    resolve_trend_ignition_payload,
    write_last_trigger_ts,
)
from src.runtime.summary_support import build_stub_summary, finite_float_or_none
from src.runtime.target_range_support import (
    apply_target_range_overrides,
    confidence_from_rmse,
    evaluate_direction_only_fallback,
    load_target_range_model,
    load_target_range_models,
    predict_target_range_prices,
    resolve_target_range_policy,
    target_range_label,
)
from src.runtime.trade_decision_support import apply_trade_decision_stage, resolve_trade_decision_policy
from src.runtime.trust_hardening_support import apply_trust_hardening, resolve_trust_hardening_policy
from src.trading.direction_config import (
    apply_path_overrides,
    clone_direction_model_configs,
    direction_configs_to_weight_map,
    log_direction_model_configs,
    resolve_direction_model_configs,
)
from src.trading.ensembles import parse_weight_spec
from src.trading.signals import (
    format_ts_iso,
    load_trend_ignition_classifier,
    MIN_RESIDUAL_STD,
    prepare_data_for_signals,
    prepare_data_for_signals_from_ohlcv,
)
from src.utils.model_artifact_selection import resolve_best_versioned_model_file


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def lookup_horizon_value(mapping: Mapping[float, float] | None, horizon: float, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    numeric_horizon = normalize_horizon_value(horizon)
    if numeric_horizon in mapping:
        return float(mapping[numeric_horizon])
    for key, value in mapping.items():
        try:
            if abs(float(key) - numeric_horizon) <= 1e-6:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def compute_position_size(
    confidence_score: float,
    *,
    confidence_min: float,
    size_floor: float,
    size_cap: float,
) -> float:
    confidence_min = max(0.0, min(1.0, float(confidence_min)))
    size_floor = max(0.0, float(size_floor))
    size_cap = max(size_floor, float(size_cap))
    if confidence_score <= confidence_min:
        return 0.0
    scaled = (confidence_score - confidence_min) / max(1e-8, 1.0 - confidence_min)
    return float(min(size_cap, max(size_floor, scaled * size_cap)))


def coerce_row_value(value: Any) -> float | None:
    return finite_float_or_none(value)


def _direction_from_ret_pred(value: Any) -> str:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close: Any, projected_price: Any) -> str:
    close_value = finite_float_or_none(close)
    projected_value = finite_float_or_none(projected_price)
    if close_value is None or projected_value is None or close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"
    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def _direction_from_probability(value: Any, *, neutral_band: float = 0.0) -> str:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _parse_horizon_label(value: str) -> float:
    lowered = str(value).strip().lower()
    if lowered.endswith("h"):
        return float(lowered[:-1])
    if lowered.endswith("m"):
        return float(lowered[:-1]) / 60.0
    return float(lowered)


def _coerce_result_horizon(value: Any) -> float | None:
    numeric = finite_float_or_none(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _direction_vote(entry: Mapping[str, Any]) -> str:
    return "up" if str(entry.get("direction_next", "down")).lower() == "up" else "down"


def _resolve_target_range_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return resolve_target_range_policy(
        config,
        target_range_model_dir=TARGET_RANGE_MODEL_DIR,
        default_override_ratio=TARGET_RANGE_DEFAULT_OVERRIDE_RATIO,
        default_confidence_scale=TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE,
        default_horizons=TARGET_RANGE_DEFAULT_HORIZONS,
    )


def _load_prepared(dataset_path, *, target_column: str, offline: bool = False):
    return load_prepared(
        dataset_path,
        target_column=target_column,
        offline=offline,
        load_prepared_offline_fn=lambda dataset_path, *, base_horizon: load_prepared_offline(
            dataset_path,
            base_horizon=base_horizon,
            prepare_data_for_signals_from_ohlcv_fn=prepare_data_for_signals_from_ohlcv,
            format_ts_iso_fn=format_ts_iso,
            stderr_write=sys.stderr.write,
        ),
        prepare_data_for_signals_fn=prepare_data_for_signals,
        format_ts_iso_fn=format_ts_iso,
    )


def _resolve_trend_ignition_payload(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return resolve_trend_ignition_payload(
        config,
        load_trend_ignition_classifier=lambda path: load_trend_ignition_classifier(str(path)),
        load_state=lambda: load_last_trigger_ts(TREND_IGNITION_STATE_PATH),
        stderr_write=sys.stderr.write,
    )


def _resolve_direction_fallback_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return resolve_direction_fallback_policy(
        config,
        load_state=lambda: load_last_trigger_ts(DIRECTION_FALLBACK_STATE_PATH),
    )


def _compute_breakout_scores(
    prepared_bundles: Mapping[str, tuple[Any, int, float, str]],
    volatility_snapshots: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    return compute_breakout_scores(
        prepared_bundles,
        volatility_snapshots,
        compute_profile_breakout_score=lambda prepared, index, snapshot: compute_profile_breakout_score(
            prepared,
            index,
            snapshot,
            breakout_vol_normalizer=BREAKOUT_VOL_NORMALIZER,
            breakout_ret_normalizer=BREAKOUT_RET_NORMALIZER,
        ),
    )


def _classify_regime_from_score(score: float, policy: Mapping[str, Any]) -> str:
    return classify_regime_from_score(
        score,
        policy,
        regime_trend=REGIME_TREND,
        regime_neutral=REGIME_NEUTRAL,
        regime_chop=REGIME_CHOP,
    )


def _apply_adaptive_thresholds(
    policy: Mapping[str, Any],
    base_p_up: float,
    base_ret: float,
    regime_state: str,
) -> tuple[float, float, float]:
    return apply_adaptive_thresholds(
        policy,
        base_p_up,
        base_ret,
        regime_state,
        regime_trend=REGIME_TREND,
        regime_chop=REGIME_CHOP,
    )


def _resolve_abstention_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_abstention_policy(
        config,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_uncertainty_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_uncertainty_policy(
        config,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_trade_decision_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_trade_decision_policy(
        config,
        finite_float_or_none=finite_float_or_none,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _resolve_regime_model_weights_policy(config: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    return resolve_regime_model_weights_policy(
        config,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
        parse_weight_spec=parse_weight_spec,
    )


def _resolve_regime_model_dirs_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_regime_model_dirs_policy(
        config,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
    )


def _resolve_confluence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_confluence_policy(
        config,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_execution_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_execution_policy(
        config,
        normalize_horizon_value=normalize_horizon_value,
        coerce_numeric_horizon=coerce_numeric_horizon,
        default_lookback_bars=EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS,
        default_min_samples=EXECUTION_POLICY_DEFAULT_MIN_SAMPLES,
        default_target_range_stop_horizons=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_HORIZONS,
        default_target_range_stop_confidence_min=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN,
        default_target_range_stop_buffer_std_mult=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT,
        default_target_range_stop_min_tighten_fraction=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION,
    )


def _resolve_forecast_coherence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_forecast_coherence_policy(
        config,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_direction_output_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_direction_output_policy(
        config,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_direction_ensemble_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_direction_ensemble_policy(
        config,
        coerce_numeric_horizon=coerce_numeric_horizon,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_trust_hardening_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return resolve_trust_hardening_policy(
        config,
        normalize_horizon_value=normalize_horizon_value,
        coerce_numeric_horizon=coerce_numeric_horizon,
    )


def _prepare_base_direction_configs(**kwargs):
    return runtime_prepare_base_direction_configs(
        **kwargs,
        default_dir_models_1h=DEFAULT_DIR_MODELS_1H,
        resolve_direction_model_configs_fn=resolve_direction_model_configs,
    )


def _dataset_profile_for_horizon(horizon: float):
    return dataset_profile_for_horizon(
        horizon,
        dataset_multi_path=DATASET_MULTI_PATH,
        dataset_1h_path=DATASET_1H_PATH,
        dataset_15m_path=DATASET_15M_PATH,
        dataset_candidate_type=DatasetCandidate,
        dataset_profile_type=DatasetProfile,
    )


def _model_paths_for_horizon(horizon: float):
    return runtime_model_paths_for_horizon(
        horizon,
        format_horizon_label=format_horizon_label,
        normalize_horizon_value=normalize_horizon_value,
        model_root=MODEL_ROOT,
        model_version_priority=MODEL_VERSION_PRIORITY,
        dir_version_overrides=DIR_VERSION_OVERRIDES,
        resolve_best_versioned_model_file_fn=resolve_best_versioned_model_file,
        stderr_write=sys.stderr.write,
    )


def _resolve_regime_specific_dir_path(default_path, *, regime_state: str, horizon_label: str, policy: Mapping[str, Any]):
    return resolve_regime_specific_dir_path(
        default_path,
        regime_state=regime_state,
        horizon_label=horizon_label,
        policy=policy,
        expected_filename=f"xgb_dir{horizon_label}_model.json",
        version_priority=MODEL_VERSION_PRIORITY,
        resolve_best_versioned_model_file=resolve_best_versioned_model_file,
        stderr_write=sys.stderr.write,
    )


def _resolve_regression_dir_path(default_path, *, horizon_label: str, policy: Mapping[str, Any]):
    return resolve_regression_dir_path(
        default_path,
        horizon_label=horizon_label,
        policy=policy,
        expected_filename=f"xgb_ret{horizon_label}_model.json",
        version_priority=MODEL_VERSION_PRIORITY,
        resolve_best_versioned_model_file=resolve_best_versioned_model_file,
        stderr_write=sys.stderr.write,
    )


def _registry_model_exists(model_name: str) -> bool:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.get_registered_model(model_name)
        return True
    except Exception:
        return False


def _direction_configs_for_horizon(base_configs, **kwargs):
    return runtime_direction_configs_for_horizon(
        base_configs,
        **kwargs,
        normalize_horizon_value=normalize_horizon_value,
        default_transformer_model_dir_by_suffix=DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX,
        model_root=MODEL_ROOT,
        model_version_priority=MODEL_VERSION_PRIORITY,
        clone_direction_model_configs_fn=clone_direction_model_configs,
        apply_path_overrides_fn=apply_path_overrides,
        log_direction_model_configs_fn=log_direction_model_configs,
        direction_configs_to_weight_map_fn=direction_configs_to_weight_map,
        registry_model_exists_fn=_registry_model_exists,
    )


def _resolve_thresholds_for_horizon(
    horizon: float,
    default_p_up: float,
    default_ret: float,
    overrides: Mapping[float, Dict[str, float]] | None,
) -> Dict[str, float]:
    return resolve_thresholds_for_horizon(
        horizon,
        default_p_up,
        default_ret,
        overrides,
        normalize_horizon_value=normalize_horizon_value,
    )


def _resolve_direction_threshold_for_horizon(*, direction_threshold: float, auto_direction_threshold: bool, horizon_p_up: float) -> float:
    if not auto_direction_threshold:
        return float(direction_threshold)
    return max(0.5, float(horizon_p_up))


def _compute_directional_stop_take_prices(
    *,
    close: float,
    ret_pred: float,
    residual_std: float,
    direction_signal: int,
) -> tuple[float, float]:
    min_buffer = max(MIN_DIRECTIONAL_RETURN_BUFFER, residual_std * 0.1)
    if int(direction_signal) >= 1:
        stop_return = min(ret_pred - residual_std, -min_buffer)
        take_return = max(ret_pred + residual_std, min_buffer)
    else:
        stop_return = max(ret_pred + residual_std, min_buffer)
        take_return = min(ret_pred - residual_std, -min_buffer)
    return project_price(close, stop_return), project_price(close, take_return)


def _resolve_direction_signal_for_horizon(
    *,
    raw_probability: float,
    calibrated_probability: float,
    threshold: float,
    close: float,
    projected_price: float,
    ret_pred: float,
    calibration_key: str | None,
    calibration_used_regime_key: bool,
) -> int:
    directional_threshold = max(float(threshold), 0.5)
    calibrated_signal = int(float(calibrated_probability) >= directional_threshold)
    raw_signal = int(float(raw_probability) >= directional_threshold)
    raw_side = "up" if raw_signal == 1 else "down"
    calibrated_side = "up" if calibrated_signal == 1 else "down"
    ret_side = _direction_from_ret_pred(ret_pred)
    projected_side = _direction_from_projected_price(close, projected_price)

    if ret_side == projected_side and ret_side in {"up", "down"}:
        return 1 if ret_side == "up" else 0
    if raw_signal == calibrated_signal:
        return calibrated_signal
    if ret_side == raw_side and projected_side == raw_side:
        return raw_signal
    if ret_side == calibrated_side and projected_side == calibrated_side:
        return calibrated_signal
    if calibration_key is None or calibration_used_regime_key:
        return calibrated_signal
    return calibrated_signal


def _derive_probability_alignment_features(
    *,
    close: float,
    projected_price: float,
    ret_pred: float,
    raw_probability: float,
    resolved_probability: float,
    direction: str,
    neutral_band: float,
    probability_guard: Mapping[str, Any] | None,
    calibration_used_regime_key: bool,
) -> Dict[str, float | str]:
    direction_side = str(direction).strip().lower()
    ret_side = _direction_from_ret_pred(ret_pred)
    projected_side = _direction_from_projected_price(close, projected_price)
    raw_side = _direction_from_probability(raw_probability, neutral_band=neutral_band)
    resolved_side = _direction_from_probability(resolved_probability, neutral_band=neutral_band)
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else "neutral"
    raw_gap = float(resolved_probability) - float(raw_probability)
    return {
        "raw_p_up": float(raw_probability),
        "raw_calibrated_probability_gap": float(raw_gap),
        "probability_alignment_gap": float(abs(raw_gap)),
        "raw_p_up_side": raw_side,
        "resolved_p_up_side": resolved_side,
        "ret_pred_side": ret_side,
        "projected_price_side": projected_side,
        "forecast_consensus_side": consensus_side,
        "raw_p_up_ret_mismatch": float(raw_side in {"up", "down"} and ret_side in {"up", "down"} and raw_side != ret_side),
        "p_up_ret_mismatch": float(resolved_side in {"up", "down"} and ret_side in {"up", "down"} and resolved_side != ret_side),
        "raw_p_up_direction_mismatch": float(raw_side in {"up", "down"} and direction_side in {"up", "down"} and raw_side != direction_side),
        "p_up_direction_mismatch": float(resolved_side in {"up", "down"} and direction_side in {"up", "down"} and resolved_side != direction_side),
        "ret_projected_price_consensus": float(consensus_side in {"up", "down"}),
        "probability_calibration_guard_applied": float(bool(isinstance(probability_guard, Mapping) and probability_guard.get("applied"))),
        "probability_calibration_used_regime_key": float(bool(calibration_used_regime_key)),
    }


def _build_direction_output(**kwargs) -> Mapping[str, Any]:
    return runtime_build_direction_output(
        **kwargs,
        apply_probability_calibration=apply_probability_calibration,
        resolve_probability_calibration=lambda calibration_map, label, regime_state: resolve_probability_calibration(
            calibration_map,
            label,
            regime_state,
            regime_calibration_min_platt_slope=REGIME_CALIBRATION_MIN_PLATT_SLOPE,
        ),
        parse_horizon_label=_parse_horizon_label,
        lookup_horizon_value=lookup_horizon_value,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
    )


def _build_prediction_result(**kwargs):
    return build_prediction_result(
        **kwargs,
        project_price=project_price,
        get_active_regime_weight_override=lambda regime_state, horizon, policy: get_active_regime_weight_override(
            regime_state=regime_state,
            horizon=horizon,
            policy=policy,
            normalize_horizon_value=normalize_horizon_value,
        ),
        derive_probability_alignment_features=_derive_probability_alignment_features,
        build_direction_output=_build_direction_output,
        apply_target_range_overrides=apply_target_range_overrides,
        evaluate_direction_only_fallback=evaluate_direction_only_fallback,
        finite_float_or_none=finite_float_or_none,
        coerce_row_value=coerce_row_value,
    )


def _load_target_range_models(policy: Mapping[str, Any] | None, horizons: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    return load_target_range_models(
        policy,
        horizons,
        target_range_model_dir=TARGET_RANGE_MODEL_DIR,
        load_target_range_model_fn=lambda path: load_target_range_model(path, stderr_write=sys.stderr.write),
        stderr_write=sys.stderr.write,
    )


def _apply_forecast_coherence_policy(summary: Dict[str, Dict[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return apply_forecast_coherence_policy(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=finite_float_or_none,
        append_gate_trace=append_gate_trace,
    )


def _apply_trust_hardening(summary: Dict[str, Dict[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return apply_trust_hardening(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=finite_float_or_none,
    )


def _apply_confluence_policy(summary: Dict[str, Dict[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return apply_confluence_policy(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=finite_float_or_none,
        append_gate_trace=append_gate_trace,
    )


def _apply_post_trade_gates(
    summary: Dict[str, Dict[str, Any]],
    *,
    confidence_min: float,
    abstention_policy: Mapping[str, Any],
    uncertainty_policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return apply_post_trade_gates(
        summary,
        confidence_min=confidence_min,
        abstention_policy=abstention_policy,
        uncertainty_policy=uncertainty_policy,
        default_fee_bps=DEFAULT_FEE_BPS,
        default_slippage_bps=DEFAULT_SLIPPAGE_BPS,
        regime_neutral=REGIME_NEUTRAL,
        append_gate_trace=append_gate_trace,
        resolve_abstention_expected_value=resolve_abstention_expected_value,
        resolve_abstention_policy_for_horizon=lambda policy, *, horizon, regime_state: resolve_abstention_policy_for_horizon(
            policy,
            horizon=horizon,
            regime_state=regime_state,
            normalize_horizon_value=normalize_horizon_value,
        ),
        apply_abstention_policy=apply_abstention_policy,
        apply_uncertainty_abstention=lambda **kwargs: apply_uncertainty_abstention(
            **kwargs,
            resolve_uncertainty_settings=lambda policy, *, horizon, regime_state: resolve_uncertainty_settings(
                policy,
                horizon=horizon,
                regime_state=regime_state,
                normalize_horizon_value=normalize_horizon_value,
            ),
        ),
        coerce_result_horizon=_coerce_result_horizon,
    )


def _apply_execution_policy(
    summary: Dict[str, Dict[str, Any]],
    execution_contexts: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return apply_execution_policy(
        summary,
        execution_contexts,
        policy,
        regime_neutral=REGIME_NEUTRAL,
        execution_policy_default_lookback_bars=EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS,
        execution_policy_default_min_samples=EXECUTION_POLICY_DEFAULT_MIN_SAMPLES,
        summarize_bias_context=summarize_bias_context,
        execution_side=execution_side,
        direction_vote=_direction_vote,
        execution_alignment_ratio=execution_alignment_ratio,
        classify_execution_tier=classify_execution_tier,
        compute_atr_like_price_distance=lambda frame, *, index, fallback_close, fallback_return_std: compute_atr_like_price_distance(
            frame,
            index=index,
            fallback_close=fallback_close,
            fallback_return_std=fallback_return_std,
            min_residual_std=MIN_RESIDUAL_STD,
        ),
        compute_recent_structure=compute_recent_structure,
        build_entry_zone=build_entry_zone,
        compute_pullback_quality_score=compute_pullback_quality_score,
        compute_disagreement_severity=compute_disagreement_severity,
        compute_excursion_priors=compute_excursion_priors,
        finite_float_or_none=finite_float_or_none,
        finite_float=finite_float,
        resolve_stop_with_guardrails=resolve_stop_with_guardrails,
        refine_stop_with_target_range=lambda **kwargs: refine_stop_with_target_range(
            **kwargs,
            normalize_horizon_value=normalize_horizon_value,
            default_confidence_min=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN,
            default_buffer_std_mult=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT,
            default_min_tighten_fraction=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION,
        ),
        resolve_execution_target_reward=resolve_execution_target_reward,
        lookup_horizon_value=runtime_lookup_horizon_value,
        resolve_execution_upstream_hold_reason=resolve_execution_upstream_hold_reason,
    )


def _build_stub_summary(
    targets,
    p_up_min,
    ret_min,
    *,
    close: float = 0.0,
    ts_iso: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
):
    return build_stub_summary(
        targets,
        p_up_min,
        ret_min,
        close=close,
        ts_iso=ts_iso,
        thresholds_by_horizon=thresholds_by_horizon,
        normalize_horizon_value=normalize_horizon_value,
        format_horizon_label=format_horizon_label,
        resolve_thresholds_for_horizon=_resolve_thresholds_for_horizon,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        regime_neutral=REGIME_NEUTRAL,
    )


def _write_trend_ignition_state(ts_value: str) -> None:
    write_last_trigger_ts(TREND_IGNITION_STATE_PATH, ts_value)


def _write_direction_fallback_state(ts_value: str) -> None:
    write_last_trigger_ts(DIRECTION_FALLBACK_STATE_PATH, ts_value)


def build_prediction_pipeline_dependencies() -> PredictionPipelineDependencies:
    return PredictionPipelineDependencies(
        normalize_horizon_value=normalize_horizon_value,
        normalize_threshold_overrides=lambda overrides: normalize_threshold_overrides(
            overrides,
            coerce_numeric_horizon=coerce_numeric_horizon,
            normalize_horizon_value=normalize_horizon_value,
        ),
        normalize_horizon_regime_float_map=lambda raw, minimum=0.0, maximum=None: normalize_horizon_regime_float_map(
            raw,
            finite_float_or_none=finite_float_or_none,
            coerce_numeric_horizon=coerce_numeric_horizon,
            normalize_horizon_value=normalize_horizon_value,
            minimum=minimum,
            maximum=maximum,
        ),
        normalize_horizon_float_map=lambda raw, minimum=0.0, maximum=None: normalize_horizon_float_map(
            raw,
            coerce_numeric_horizon=coerce_numeric_horizon,
            normalize_horizon_value=normalize_horizon_value,
            minimum=minimum,
            maximum=maximum,
        ),
        dataset_profile_for_horizon=_dataset_profile_for_horizon,
        select_dataset_candidate=select_dataset_candidate,
        prepare_base_direction_configs=_prepare_base_direction_configs,
        load_prepared=_load_prepared,
        resolve_trend_ignition_payload=_resolve_trend_ignition_payload,
        resolve_direction_fallback_policy=_resolve_direction_fallback_policy,
        resolve_adaptive_thresholds_policy=resolve_adaptive_thresholds_policy,
        resolve_target_range_policy=_resolve_target_range_policy,
        resolve_abstention_policy=_resolve_abstention_policy,
        resolve_uncertainty_policy=_resolve_uncertainty_policy,
        resolve_trade_decision_policy=_resolve_trade_decision_policy,
        resolve_regime_model_weights_policy=_resolve_regime_model_weights_policy,
        resolve_regime_model_dirs_policy=_resolve_regime_model_dirs_policy,
        resolve_regression_model_dirs_policy=resolve_regression_model_dirs_policy,
        resolve_confluence_policy=_resolve_confluence_policy,
        resolve_execution_policy=_resolve_execution_policy,
        resolve_forecast_coherence_policy=_resolve_forecast_coherence_policy,
        resolve_direction_output_policy=_resolve_direction_output_policy,
        resolve_direction_ensemble_policy=_resolve_direction_ensemble_policy,
        resolve_trust_hardening_policy=_resolve_trust_hardening_policy,
        compute_breakout_scores=_compute_breakout_scores,
        load_target_range_models=_load_target_range_models,
        format_horizon_label=format_horizon_label,
        model_paths_for_horizon=_model_paths_for_horizon,
        classify_regime_from_score=_classify_regime_from_score,
        resolve_regime_specific_dir_path=_resolve_regime_specific_dir_path,
        resolve_regression_dir_path=_resolve_regression_dir_path,
        direction_configs_for_horizon=_direction_configs_for_horizon,
        resolve_thresholds_for_horizon=_resolve_thresholds_for_horizon,
        apply_adaptive_thresholds=_apply_adaptive_thresholds,
        apply_regime_weight_overrides=lambda base_weights, *, regime_state, horizon=None, policy=None: apply_regime_weight_overrides(
            base_weights,
            regime_state=regime_state,
            horizon=horizon,
            policy=policy,
            normalize_horizon_value=normalize_horizon_value,
        ),
        scope_direction_ensemble_policy=lambda policy, horizon: scope_direction_ensemble_policy(
            policy,
            horizon,
            normalize_horizon_value=normalize_horizon_value,
        ),
        resolve_direction_threshold_for_horizon=_resolve_direction_threshold_for_horizon,
        project_price=project_price,
        resolve_trade_probability_for_horizon=lambda **kwargs: resolve_trade_probability_for_horizon(
            **kwargs,
            regime_calibration_min_platt_slope=REGIME_CALIBRATION_MIN_PLATT_SLOPE,
            apply_probability_calibration=apply_probability_calibration,
            direction_from_ret_pred=_direction_from_ret_pred,
            direction_from_projected_price=_direction_from_projected_price,
            direction_from_probability=_direction_from_probability,
        ),
        resolve_direction_signal_for_horizon=_resolve_direction_signal_for_horizon,
        compute_directional_stop_take_prices=_compute_directional_stop_take_prices,
        resolve_confidence_min_for_horizon=lambda base_confidence_min, overrides, *, horizon, regime_state: resolve_confidence_min_for_horizon(
            base_confidence_min,
            overrides,
            horizon=horizon,
            regime_state=regime_state,
            normalize_horizon_value=normalize_horizon_value,
            format_horizon_label=format_horizon_label,
        ),
        lookup_horizon_value=lookup_horizon_value,
        compute_position_size=compute_position_size,
        parse_iso_timestamp=parse_iso_timestamp,
        predict_target_range_prices=predict_target_range_prices,
        build_prediction_result=_build_prediction_result,
        get_active_regime_weight_override=lambda regime_state, horizon=None, policy=None: get_active_regime_weight_override(
            regime_state=regime_state,
            horizon=horizon,
            policy=policy,
            normalize_horizon_value=normalize_horizon_value,
        ),
        derive_probability_alignment_features=_derive_probability_alignment_features,
        build_direction_output=_build_direction_output,
        apply_target_range_overrides=apply_target_range_overrides,
        evaluate_direction_only_fallback=evaluate_direction_only_fallback,
        finite_float_or_none=finite_float_or_none,
        coerce_row_value=coerce_row_value,
        write_trend_ignition_state=_write_trend_ignition_state,
        write_direction_fallback_state=_write_direction_fallback_state,
        apply_post_prediction_policies=apply_post_prediction_policies,
        apply_forecast_coherence_policy=_apply_forecast_coherence_policy,
        apply_trust_hardening_stage=_apply_trust_hardening,
        apply_confluence_policy=_apply_confluence_policy,
        apply_trade_decision_stage=apply_trade_decision_stage,
        apply_post_trade_gates=_apply_post_trade_gates,
        apply_execution_policy=_apply_execution_policy,
        build_stub_summary=_build_stub_summary,
        stderr_write=sys.stderr.write,
        regime_neutral=REGIME_NEUTRAL,
        target_range_default_confidence_scale=TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE,
    )