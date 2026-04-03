"""Refresh local Binance US-driven features and emit multi-horizon signals."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from datetime import datetime, timezone
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib

import yaml

import numpy as np
import pandas as pd

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.processed.compute_technical_features import process_technical_features
from src.scripts.build_training_dataset import main as build_1h_dataset
from src.scripts.build_training_dataset_15m import main as build_15m_dataset
from src.scripts.build_training_dataset_multi_horizon import build_multi_horizon_dataset
from src.scripts.build_signal_baseline import (
    DEFAULT_COLUMNS as BASELINE_DEFAULT_COLUMNS,
    _append_detected_meta_columns,
    baseline_to_dataframe,
    compute_baseline,
    load_dataframe,
)
from src.config_trading import (
    DEFAULT_DIR_MODEL_WEIGHTS_1H,
    DEFAULT_DIR_MODELS_1H,
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX,
)
from src.trading.direction_config import (
    DirectionModelConfig,
    apply_path_overrides,
    clone_direction_model_configs,
    direction_configs_to_weight_map,
    log_direction_model_configs,
    resolve_direction_model_configs,
)
from src.trading.ensembles import parse_weight_spec
from src.trading.signals import (
    DEFAULT_RESIDUAL_STD,
    MIN_RESIDUAL_STD,
    PreparedData,
    compute_signal_for_index,
    format_ts_iso,
    load_residual_std_from_dataset,
    load_models,
    load_trend_ignition_classifier,
    populate_sequence_cache_from_prepared,
    prepare_data_for_signals,
    prepare_data_for_signals_from_ohlcv,
)
from src.trading.thresholds import load_calibrated_thresholds
from src.trading.volatility import DEFAULT_REALIZED_WINDOWS, add_volatility_columns, latest_volatility_snapshot
from src.trading.data_quality import DataQualityError, DataQualityPolicy, evaluate_ohlcv_quality
from src.config_trading import DEFAULT_DIR_MODEL_DIR_1H
from src.utils.model_artifact_selection import resolve_best_versioned_model_file
from src.runtime.forecast_coherence_support import (
    apply_forecast_coherence_policy as runtime_apply_forecast_coherence_policy,
    coherence_weight_multiplier as runtime_coherence_weight_multiplier,
    forecast_coherence_excluded as runtime_forecast_coherence_excluded,
    resolve_forecast_coherence_policy as runtime_resolve_forecast_coherence_policy,
)
from src.runtime.trade_decision_support import (
    apply_trade_decision_model as runtime_apply_trade_decision_model,
    apply_trade_decision_stage as runtime_apply_trade_decision_stage,
    lookup_raw_ev_fallback_threshold as runtime_lookup_raw_ev_fallback_threshold,
    resolve_trade_decision_policy as runtime_resolve_trade_decision_policy,
    resolve_trade_decision_threshold as runtime_resolve_trade_decision_threshold,
    upstream_trade_gate_reasons as runtime_upstream_trade_gate_reasons,
)
from src.runtime.post_trade_support import (
    apply_abstention_policy as runtime_apply_abstention_policy,
    apply_post_trade_gates as runtime_apply_post_trade_gates,
    apply_uncertainty_abstention as runtime_apply_uncertainty_abstention,
    resolve_abstention_expected_value as runtime_resolve_abstention_expected_value,
    resolve_abstention_policy as runtime_resolve_abstention_policy,
    resolve_abstention_policy_for_horizon as runtime_resolve_abstention_policy_for_horizon,
    resolve_uncertainty_policy as runtime_resolve_uncertainty_policy,
    resolve_uncertainty_settings as runtime_resolve_uncertainty_settings,
)
from src.runtime.confluence_support import (
    apply_confluence_policy as runtime_apply_confluence_policy,
    resolve_confluence_policy as runtime_resolve_confluence_policy,
)
from src.runtime.execution_policy_support import (
    apply_execution_policy as runtime_apply_execution_policy,
    resolve_execution_policy as runtime_resolve_execution_policy,
)
from src.runtime.output_support import (
    build_trade_ready_monitoring_payload as runtime_build_trade_ready_monitoring_payload,
    refresh_meta_baseline as runtime_refresh_meta_baseline,
    write_monitoring_artifact as runtime_write_monitoring_artifact,
    write_monitoring_payload_file as runtime_write_monitoring_payload_file,
)
from src.runtime.quality_support import (
    evaluate_data_quality as runtime_evaluate_data_quality,
    evaluate_feature_coverage as runtime_evaluate_feature_coverage,
    resolve_data_quality_policy as runtime_resolve_data_quality_policy,
    resolve_feature_coverage_policy as runtime_resolve_feature_coverage_policy,
)
from src.runtime.direction_output_support import (
    apply_probability_calibration as runtime_apply_probability_calibration,
    build_direction_output as runtime_build_direction_output,
    resolve_direction_output_policy as runtime_resolve_direction_output_policy,
    resolve_probability_calibration as runtime_resolve_probability_calibration,
    resolve_trade_probability_for_horizon as runtime_resolve_trade_probability_for_horizon,
)
from src.runtime.local_feature_support import (
    build_ohlcv_frame_from_tidy as runtime_build_ohlcv_frame_from_tidy,
    compute_intrabar_features_from_15m as runtime_compute_intrabar_features_from_15m,
    enrich_local_features_for_model as runtime_enrich_local_features_for_model,
    load_training_feature_names as runtime_load_training_feature_names,
    merge_override_features as runtime_merge_override_features,
    pivot_tidy_spot_ohlcv as runtime_pivot_tidy_spot_ohlcv,
    prepare_local_feature_bundle as runtime_prepare_local_feature_bundle,
    read_timeseries_frame as runtime_read_timeseries_frame,
    summarize_frame as runtime_summarize_frame,
)
from src.runtime.target_range_support import (
    apply_target_range_overrides as runtime_apply_target_range_overrides,
    confidence_from_rmse as runtime_confidence_from_rmse,
    evaluate_direction_only_fallback as runtime_evaluate_direction_only_fallback,
    load_target_range_model as runtime_load_target_range_model,
    load_target_range_models as runtime_load_target_range_models,
    predict_single_target_model as runtime_predict_single_target_model,
    predict_target_range_prices as runtime_predict_target_range_prices,
    resolve_target_range_policy as runtime_resolve_target_range_policy,
    target_range_label as runtime_target_range_label,
)
from src.runtime.regime_policy_support import (
    apply_adaptive_thresholds as runtime_apply_adaptive_thresholds,
    classify_regime_from_score as runtime_classify_regime_from_score,
    compute_breakout_scores as runtime_compute_breakout_scores,
    compute_profile_breakout_score as runtime_compute_profile_breakout_score,
    derive_regime_labels_from_frame as runtime_derive_regime_labels_from_frame,
    inactive_direction_fallback as runtime_inactive_direction_fallback,
    load_last_trigger_ts as runtime_load_last_trigger_ts,
    resolve_adaptive_thresholds_policy as runtime_resolve_adaptive_thresholds_policy,
    resolve_direction_fallback_policy as runtime_resolve_direction_fallback_policy,
    resolve_trend_ignition_payload as runtime_resolve_trend_ignition_payload,
    write_last_trigger_ts as runtime_write_last_trigger_ts,
)
from src.runtime.policy_support import (
    apply_regime_weight_overrides as runtime_apply_regime_weight_overrides,
    get_active_regime_weight_override as runtime_get_active_regime_weight_override,
    normalize_horizon_float_map as runtime_normalize_horizon_float_map,
    normalize_horizon_regime_float_map as runtime_normalize_horizon_regime_float_map,
    normalize_threshold_overrides as runtime_normalize_threshold_overrides,
    resolve_confidence_min_for_horizon as runtime_resolve_confidence_min_for_horizon,
    resolve_direction_ensemble_policy as runtime_resolve_direction_ensemble_policy,
    resolve_regime_model_dirs_policy as runtime_resolve_regime_model_dirs_policy,
    resolve_regime_model_weights_policy as runtime_resolve_regime_model_weights_policy,
    resolve_regime_specific_dir_path as runtime_resolve_regime_specific_dir_path,
    resolve_thresholds_for_horizon as runtime_resolve_thresholds_for_horizon,
    scope_direction_ensemble_policy as runtime_scope_direction_ensemble_policy,
)
from src.runtime.config_normalization_support import (
    normalize_abstention_policy_block as runtime_normalize_abstention_policy_block,
    normalize_adaptive_thresholds_block as runtime_normalize_adaptive_thresholds_block,
    normalize_confluence_policy_block as runtime_normalize_confluence_policy_block,
    normalize_config_value as runtime_normalize_config_value,
    normalize_data_quality_block as runtime_normalize_data_quality_block,
    normalize_degradation_monitoring_block as runtime_normalize_degradation_monitoring_block,
    normalize_direction_ensemble_policy_block as runtime_normalize_direction_ensemble_policy_block,
    normalize_direction_fallback_block as runtime_normalize_direction_fallback_block,
    normalize_direction_output_policy_block as runtime_normalize_direction_output_policy_block,
    normalize_execution_policy_block as runtime_normalize_execution_policy_block,
    normalize_feature_coverage_policy_block as runtime_normalize_feature_coverage_policy_block,
    normalize_forecast_coherence_policy_block as runtime_normalize_forecast_coherence_policy_block,
    normalize_intrabar_aggregation_block as runtime_normalize_intrabar_aggregation_block,
    normalize_regime_model_dirs_block as runtime_normalize_regime_model_dirs_block,
    normalize_regime_model_weights_block as runtime_normalize_regime_model_weights_block,
    normalize_target_range_block as runtime_normalize_target_range_block,
    normalize_trade_decision_policy_block as runtime_normalize_trade_decision_policy_block,
    normalize_trust_hardening_policy_block as runtime_normalize_trust_hardening_policy_block,
    normalize_trend_ignition_block as runtime_normalize_trend_ignition_block,
    normalize_uncertainty_policy_block as runtime_normalize_uncertainty_policy_block,
)
from src.runtime.post_prediction_pipeline import apply_post_prediction_policies as runtime_apply_post_prediction_policies
from src.runtime.trust_hardening_support import (
    apply_trust_hardening as runtime_apply_trust_hardening,
    resolve_trust_hardening_policy as runtime_resolve_trust_hardening_policy,
)
from src.runtime.prediction_result_support import build_prediction_result as runtime_build_prediction_result
from src.runtime.refresh_support import (
    base_horizon_for_target_column as runtime_base_horizon_for_target_column,
    dataset_profile_for_horizon as runtime_dataset_profile_for_horizon,
    load_cli_config as runtime_load_cli_config,
    load_prepared as runtime_load_prepared,
    load_prepared_offline as runtime_load_prepared_offline,
    periods_per_hour_for_base_horizon as runtime_periods_per_hour_for_base_horizon,
    project_price as runtime_project_price,
    select_dataset_candidate as runtime_select_dataset_candidate,
    warn_missing_thresholds as runtime_warn_missing_thresholds,
)
from src.runtime.model_resolution_support import (
    direction_configs_for_horizon as runtime_direction_configs_for_horizon,
    model_paths_for_horizon as runtime_model_paths_for_horizon,
    model_suffix_candidates as runtime_model_suffix_candidates,
    prepare_base_direction_configs as runtime_prepare_base_direction_configs,
)
from src.runtime.refresh_cli_support import parse_refresh_args as runtime_parse_refresh_args
from src.runtime.summary_support import (
    build_stub_summary as runtime_build_stub_summary,
    build_execution_prior_summary as runtime_build_execution_prior_summary,
    build_blocked_trade_analytics as runtime_build_blocked_trade_analytics,
    build_degradation_monitoring as runtime_build_degradation_monitoring,
    build_operator_summary_compact as runtime_build_operator_summary_compact,
    build_prompt_forecast_clause as runtime_build_prompt_forecast_clause,
    build_prompt_ready_summary as runtime_build_prompt_ready_summary,
    confidence_level_from_score as runtime_confidence_level_from_score,
    format_usd_value as runtime_format_usd_value,
    prompt_confluence_rank as runtime_prompt_confluence_rank,
    prompt_direction_label as runtime_prompt_direction_label,
    prompt_effective_direction as runtime_prompt_effective_direction,
    prompt_entry_rank as runtime_prompt_entry_rank,
    prompt_reason_rank as runtime_prompt_reason_rank,
    prompt_status_rank as runtime_prompt_status_rank,
    select_prompt_candidate_entries as runtime_select_prompt_candidate_entries,
    select_prompt_preferred_entry as runtime_select_prompt_preferred_entry,
    write_prediction_summary as runtime_write_prediction_summary,
)
from src.runtime.prediction_pipeline import (
    PredictionPipelineConfig,
    PredictionPipelineDependencies,
    execute_prediction_pipeline as runtime_execute_prediction_pipeline,
)
from src.data.macro_loader import (
    DEFAULT_MACRO_METADATA_PATH,
    DEFAULT_MACRO_OUTPUT_PATH,
    DEFAULT_MACRO_START_DATE,
    MACRO_FEATURE_COLUMNS,
    build_macro_feature_frame,
    build_source_manifest as build_macro_source_manifest,
    load_macro_features,
    resolve_incremental_start_date,
)
from src.data.onchain_loader import (
    DEFAULT_ONCHAIN_METADATA_PATH,
    DEFAULT_ONCHAIN_OUTPUT_PATH,
    DEFAULT_ONCHAIN_START_DATE,
    ONCHAIN_FEATURE_COLUMNS,
    OnchainAPIError,
    build_onchain_feature_frame,
    build_onchain_source_manifest,
    load_onchain_features,
    resolve_incremental_start_timestamp,
    write_onchain_source_manifest,
)
from src.trading.intrabar_features import compute_hourly_intrabar_features

DEFAULT_HOURS = 360
DEFAULT_TARGETS = (0.25, 1, 4, 8, 12)
DEFAULT_P_UP_MIN = 0.45
DEFAULT_RET_MIN = 0.0
MODEL_ROOT = Path("artifacts/models")
MODEL_VERSION_PRIORITY: tuple[str, ...] = ("v2", "v1")
DIR_VERSION_OVERRIDES: dict[str, tuple[str, ...]] = {
    "4h": ("v2", "v1"),
    "8h": ("v2", "v1"),
    "12h": ("v2", "v1"),
}
DATASET_DIR = Path("artifacts/datasets")
LATEST_PREDICTION_PATH = Path("artifacts/predictions/latest.json")
DATASET_1H_PATH = DATASET_DIR / "btc_features_1h_splits.npz"
DATASET_MULTI_PATH = DATASET_DIR / "btc_features_multi_horizon_splits.npz"
DATASET_15M_PATH = DATASET_DIR / "btc_features_15m_splits.npz"
HISTORY_PREDICTION_PATH = Path("artifacts/predictions/history.json")
TRADE_READY_MONITOR_PATH = Path("artifacts/monitoring/trade_ready_summary.json")
MONITORING_LATEST_PATH = Path("artifacts/monitoring/latest.json")
META_BASELINE_JSON_PATH = Path("artifacts/monitoring/meta_baseline.json")
META_BASELINE_PARQUET_PATH = Path("artifacts/monitoring/meta_baseline.parquet")
META_BASELINE_SOURCE_CSV = Path("artifacts/backtests/backtest_signals_meta_ensemble.csv")
TREND_IGNITION_STATE_PATH = Path("artifacts/monitoring/trend_ignition_state.json")
DIRECTION_FALLBACK_STATE_PATH = Path("artifacts/monitoring/direction_fallback_state.json")
DATA_QUALITY_MONITOR_PATH = Path("artifacts/monitoring/data_quality_latest.json")
TARGET_RANGE_MODEL_DIR = Path("artifacts/models/target_ranges")
TARGET_RANGE_DEFAULT_HORIZONS: tuple[float, ...] = (4.0, 8.0, 12.0)
TARGET_RANGE_DEFAULT_OVERRIDE_RATIO = 0.01
TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE = 0.01
REGIME_CALIBRATION_MIN_PLATT_SLOPE = 0.05
CONFIDENCE_MIN_DEFAULT = 0.0
POSITION_SIZE_FLOOR_DEFAULT = 0.0
POSITION_SIZE_CAP_DEFAULT = 1.0
MIN_DIRECTIONAL_RETURN_BUFFER = 0.001
EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS = 240
EXECUTION_POLICY_DEFAULT_MIN_SAMPLES = 40
EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_HORIZONS: tuple[float, ...] = (8.0, 12.0)
EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN = 0.72
EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT = 1.5
EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION = 0.1
DEGRADATION_MONITORING_DEFAULT_LOOKBACK = 30
DEGRADATION_MONITORING_DEFAULT_MIN_SNAPSHOTS = 10

BREAKOUT_VOL_NORMALIZER = 0.05
BREAKOUT_RET_NORMALIZER = 0.002
REGIME_TREND = "trend_ignition"
REGIME_NEUTRAL = "neutral"
REGIME_CHOP = "chop"

LOCAL_FEATURE_OPTIONAL_PATHS: tuple[tuple[str, str], ...] = (
    ("macro_path", "macro"),
    ("onchain_path", "onchain"),
    ("funding_path", "funding"),
    ("intrabar_path", "intrabar"),
)

LOCAL_FEATURE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "macro": tuple(),
    "funding": ("funding_rate", "funding_rate_annualized"),
    "onchain": ("onchain_active_addresses", "onchain_transaction_count"),
}

HORIZON_PRECISION = 6

CONFIG_ALLOWED_KEYS = {
    "hours",
    "targets",
    "p_up_min",
    "ret_min",
    "direction_threshold",
    "auto_direction_threshold",
    "thresholds_json",
    "dry_run",
    "spot_provider",
    "use_local_features",
    "features_path",
    "macro_path",
    "onchain_path",
    "funding_path",
    "intrabar_path",
    "write_artifacts",
    "disable_monitoring_latest",
    "dir_lstm_path",
    "dir_bilstm_path",
    "dir_gru_path",
    "dir_cnn_lstm_path",
    "dir_cnn_bilstm_path",
    "dir_garch_lstm_path",
    "dir_transformer_path",
    "dir_model_config_json",
    "dir_model_weights",
    "trend_ignition",
    "direction_only_fallback",
    "adaptive_thresholds",
    "target_range_models",
    "platt_calibration",
    "data_quality",
    "confidence_min",
    "confidence_min_by_horizon_regime",
    "position_size_floor",
    "position_size_cap",
    "position_size_cap_by_horizon",
    "abstention_policy",
    "uncertainty_policy",
    "trade_decision_policy",
    "regime_model_weights",
    "regime_model_dirs",
    "intrabar_aggregation",
    "feature_coverage_policy",
    "confluence_policy",
    "execution_policy",
    "forecast_coherence_policy",
    "direction_output_policy",
    "direction_ensemble_policy",
    "trust_hardening_policy",
    "degradation_monitoring",
    "disabled_horizons",
}
# boolean config keys; converted with _bool_env
CONFIG_BOOL_FIELDS = {
    "dry_run",
    "use_local_features",
    "write_artifacts",
    "disable_monitoring_latest",
    "auto_direction_threshold",
}
CONFIG_FLOAT_FIELDS = {
    "p_up_min",
    "ret_min",
    "direction_threshold",
    "confidence_min",
    "position_size_floor",
    "position_size_cap",
}
CONFIG_INT_FIELDS = {"hours"}
CONFIG_PATH_FIELDS = {
    "thresholds_json",
    "features_path",
    "macro_path",
    "onchain_path",
    "funding_path",
    "intrabar_path",
    "dir_lstm_path",
    "dir_bilstm_path",
    "dir_gru_path",
    "dir_cnn_lstm_path",
    "dir_cnn_bilstm_path",
    "dir_garch_lstm_path",
    "dir_transformer_path",
    "dir_model_config_json",
}


@dataclass(frozen=True)
class DatasetCandidate:
    path: Path
    target_column: str
    base_horizon: float
    offline_only: bool = False


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    candidates: Tuple[DatasetCandidate, ...]


def _normalize_horizon_value(value: float | int | str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid horizon value: {value}") from exc
    if math.isnan(numeric) or numeric <= 0:
        raise ValueError(f"Horizons must be positive numbers (got {value}).")
    return round(numeric, HORIZON_PRECISION)


def _format_horizon_label(value: float) -> str:
    if value >= 1:
        if float(value).is_integer():
            return f"{int(value)}h"
        return f"{value:g}h"
    minutes = round(value * 60)
    if minutes % 1 == 0:
        return f"{int(minutes)}m"
    return f"{minutes:g}m"


def _normalize_trend_ignition_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_trend_ignition_block(value, stderr_write=sys.stderr.write)


def _normalize_direction_fallback_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_direction_fallback_block(value, stderr_write=sys.stderr.write)


def _normalize_adaptive_thresholds_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_adaptive_thresholds_block(value, stderr_write=sys.stderr.write)


def _normalize_target_range_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_target_range_block(
        value,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_data_quality_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_data_quality_block(value, stderr_write=sys.stderr.write)


def _normalize_abstention_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_abstention_policy_block(value, stderr_write=sys.stderr.write)


def _normalize_regime_model_weights_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_regime_model_weights_block(
        value,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
        stderr_write=sys.stderr.write,
    )


def _normalize_uncertainty_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_uncertainty_policy_block(value, stderr_write=sys.stderr.write)


def _normalize_trade_decision_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_trade_decision_policy_block(value, stderr_write=sys.stderr.write)


def _normalize_regime_model_dirs_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_regime_model_dirs_block(
        value,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
        stderr_write=sys.stderr.write,
    )


def _normalize_intrabar_aggregation_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_intrabar_aggregation_block(value, stderr_write=sys.stderr.write)


def _normalize_feature_coverage_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_feature_coverage_policy_block(value, stderr_write=sys.stderr.write)


def _normalize_confluence_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_confluence_policy_block(
        value,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_execution_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_execution_policy_block(
        value,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_degradation_monitoring_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_degradation_monitoring_block(value, stderr_write=sys.stderr.write)


def _normalize_forecast_coherence_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_forecast_coherence_policy_block(
        value,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_direction_output_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_direction_output_policy_block(
        value,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_direction_ensemble_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_direction_ensemble_policy_block(
        value,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_trust_hardening_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_normalize_trust_hardening_policy_block(
        value,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _normalize_config_value(name: str, value: Any) -> Any:
    return runtime_normalize_config_value(
        name,
        value,
        default_targets=DEFAULT_TARGETS,
        config_int_fields=tuple(CONFIG_INT_FIELDS),
        config_float_fields=tuple(CONFIG_FLOAT_FIELDS),
        config_bool_fields=tuple(CONFIG_BOOL_FIELDS),
        config_path_fields=tuple(CONFIG_PATH_FIELDS),
        config_allowed_keys=tuple(CONFIG_ALLOWED_KEYS),
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
        bool_env=_bool_env,
        parse_targets=parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        normalize_horizon_float_map=lambda raw, minimum=0.0, maximum=None: _normalize_horizon_float_map(
            raw,
            minimum=minimum,
            maximum=maximum,
        ),
        normalize_horizon_regime_float_map=lambda raw, minimum=0.0, maximum=None: _normalize_horizon_regime_float_map(
            raw,
            minimum=minimum,
            maximum=maximum,
        ),
        stderr_write=sys.stderr.write,
    )


def _load_cli_config(path: str | None) -> Dict[str, Any]:
    return runtime_load_cli_config(
        path,
        config_allowed_keys=tuple(CONFIG_ALLOWED_KEYS),
        normalize_config_value=_normalize_config_value,
        yaml_safe_load=yaml.safe_load,
        stderr_write=sys.stderr.write,
    )


def _dataset_profile_for_horizon(horizon: float) -> DatasetProfile:
    return runtime_dataset_profile_for_horizon(
        horizon,
        dataset_multi_path=DATASET_MULTI_PATH,
        dataset_1h_path=DATASET_1H_PATH,
        dataset_15m_path=DATASET_15M_PATH,
        dataset_candidate_type=DatasetCandidate,
        dataset_profile_type=DatasetProfile,
    )


def _select_dataset_candidate(profile: DatasetProfile) -> tuple[DatasetCandidate, bool]:
    return runtime_select_dataset_candidate(profile)


def _horizon_sort_key(label: str) -> float | str:
    label = label.strip()
    if label.endswith("h"):
        body = label[:-1]
        if body.replace(".", "", 1).isdigit():
            return float(body)
    if label.endswith("m"):
        body = label[:-1]
        if body.replace(".", "", 1).isdigit():
            return float(body) / 60.0
    return label


def parse_targets(value: str) -> List[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one horizon must be provided.")
    targets: List[float] = []
    for part in parts:
        try:
            horizon = _normalize_horizon_value(part)
        except ValueError as exc:  # pragma: no cover - CLI validation guard
            raise argparse.ArgumentTypeError(f"Invalid horizon: {part}") from exc
        targets.append(horizon)
    return targets


def _bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _threshold_lookup_keys(horizon: float) -> List[int | float | str]:
    normalized = _normalize_horizon_value(horizon)
    keys: List[int | float | str] = [normalized]
    formatted = format(normalized, "g")
    keys.append(formatted)
    keys.append(f"{formatted}h")
    if normalized < 1.0:
        minute_value = round(normalized * 60)
        keys.append(f"{minute_value}m")
    if float(normalized).is_integer():
        int_key = int(round(normalized))
        keys.append(int_key)
        keys.append(str(int_key))
    return list(dict.fromkeys(keys))


def _coerce_numeric_horizon(value: int | float | str) -> float | None:
    try:
        return _normalize_horizon_value(value)
    except ValueError:
        if isinstance(value, str) and value.endswith("m"):
            body = value[:-1]
            try:
                minutes = float(body)
            except ValueError:
                return None
            if minutes <= 0:
                return None
            return round(minutes / 60.0, HORIZON_PRECISION)
    return None


def _normalize_horizon_float_map(raw: Any, *, minimum: float = 0.0, maximum: float | None = None) -> Dict[float, float]:
    return runtime_normalize_horizon_float_map(
        raw,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        minimum=minimum,
        maximum=maximum,
    )


def _normalize_horizon_regime_float_map(
    raw: Any,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> Dict[float, Dict[str, float]]:
    return runtime_normalize_horizon_regime_float_map(
        raw,
        finite_float_or_none=_finite_float_or_none,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        minimum=minimum,
        maximum=maximum,
    )


def _resolve_confidence_min_for_horizon(
    base_confidence_min: float,
    overrides: Mapping[float, Mapping[str, float]] | None,
    *,
    horizon: float,
    regime_state: str,
) -> tuple[float, str]:
    return runtime_resolve_confidence_min_for_horizon(
        base_confidence_min,
        overrides,
        horizon=horizon,
        regime_state=regime_state,
        normalize_horizon_value=_normalize_horizon_value,
        format_horizon_label=_format_horizon_label,
    )


def _normalize_threshold_overrides(
    overrides: Mapping[int | float | str, Dict[str, float]] | None,
) -> Dict[float, Dict[str, float]]:
    return runtime_normalize_threshold_overrides(
        overrides,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_thresholds_for_horizon(
    horizon: float,
    default_p_up: float,
    default_ret: float,
    overrides: Mapping[float, Dict[str, float]] | None,
) -> Dict[str, float]:
    return runtime_resolve_thresholds_for_horizon(
        horizon,
        default_p_up,
        default_ret,
        overrides,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _warn_missing_thresholds(
    targets: Iterable[float],
    thresholds: Mapping[int | float | str, Dict[str, float]] | None,
    source_path: str | None,
) -> None:
    runtime_warn_missing_thresholds(
        list(targets),
        thresholds,
        source_path,
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        format_horizon_label=_format_horizon_label,
        stderr_write=sys.stderr.write,
    )


def _build_stub_summary(
    targets: Iterable[float],
    p_up_min: float,
    ret_min: float,
    close: float = 0.0,
    ts_iso: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    return runtime_build_stub_summary(
        targets,
        p_up_min,
        ret_min,
        close=close,
        ts_iso=ts_iso,
        thresholds_by_horizon=thresholds_by_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        format_horizon_label=_format_horizon_label,
        resolve_thresholds_for_horizon=_resolve_thresholds_for_horizon,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        regime_neutral=REGIME_NEUTRAL,
    )


def _parse_iso_timestamp(value: str) -> datetime:
    sanitized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(sanitized)


def _load_trend_ignition_state() -> Optional[str]:
    return runtime_load_last_trigger_ts(TREND_IGNITION_STATE_PATH)


def _write_trend_ignition_state(ts_value: str) -> None:
    runtime_write_last_trigger_ts(TREND_IGNITION_STATE_PATH, ts_value)


def _resolve_trend_ignition_payload(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return runtime_resolve_trend_ignition_payload(
        config,
        load_trend_ignition_classifier=lambda path: load_trend_ignition_classifier(str(path)),
        load_state=_load_trend_ignition_state,
        stderr_write=sys.stderr.write,
    )


def _load_direction_fallback_state() -> Optional[str]:
    return runtime_load_last_trigger_ts(DIRECTION_FALLBACK_STATE_PATH)


def _write_direction_fallback_state(ts_value: str) -> None:
    runtime_write_last_trigger_ts(DIRECTION_FALLBACK_STATE_PATH, ts_value)


def _inactive_direction_fallback(
    reason: str,
    *,
    side: Optional[str] = None,
    cooldown_active: bool = False,
    size_factor: float = 0.0,
) -> Dict[str, Any]:
    return runtime_inactive_direction_fallback(
        reason,
        side=side,
        cooldown_active=cooldown_active,
        size_factor=size_factor,
    )


def _resolve_direction_fallback_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return runtime_resolve_direction_fallback_policy(
        config,
        load_state=_load_direction_fallback_state,
    )


def _resolve_adaptive_thresholds_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return runtime_resolve_adaptive_thresholds_policy(config)


def _compute_profile_breakout_score(
    prepared: PreparedData,
    index: int,
    volatility_snapshot: Mapping[str, Any] | None,
) -> float:
    return runtime_compute_profile_breakout_score(
        prepared,
        index,
        volatility_snapshot,
        breakout_vol_normalizer=BREAKOUT_VOL_NORMALIZER,
        breakout_ret_normalizer=BREAKOUT_RET_NORMALIZER,
    )


def _derive_regime_labels_from_frame(
    frame: pd.DataFrame,
    *,
    volatility_col: str,
    breakout_score_threshold: float,
    chop_score_threshold: float,
) -> pd.Series:
    return runtime_derive_regime_labels_from_frame(
        frame,
        volatility_col=volatility_col,
        breakout_score_threshold=breakout_score_threshold,
        chop_score_threshold=chop_score_threshold,
        breakout_vol_normalizer=BREAKOUT_VOL_NORMALIZER,
        breakout_ret_normalizer=BREAKOUT_RET_NORMALIZER,
        regime_trend=REGIME_TREND,
        regime_neutral=REGIME_NEUTRAL,
        regime_chop=REGIME_CHOP,
    )


def _compute_breakout_scores(
    prepared_bundles: Mapping[str, tuple[PreparedData, int, float, str]],
    volatility_snapshots: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    return runtime_compute_breakout_scores(
        prepared_bundles,
        volatility_snapshots,
        compute_profile_breakout_score=_compute_profile_breakout_score,
    )


def _classify_regime_from_score(score: float, policy: Mapping[str, Any]) -> str:
    return runtime_classify_regime_from_score(
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
    return runtime_apply_adaptive_thresholds(
        policy,
        base_p_up,
        base_ret,
        regime_state,
        regime_trend=REGIME_TREND,
        regime_chop=REGIME_CHOP,
    )


def _resolve_target_range_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return runtime_resolve_target_range_policy(
        config,
        target_range_model_dir=TARGET_RANGE_MODEL_DIR,
        default_override_ratio=TARGET_RANGE_DEFAULT_OVERRIDE_RATIO,
        default_confidence_scale=TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE,
        default_horizons=TARGET_RANGE_DEFAULT_HORIZONS,
    )


def _resolve_feature_coverage_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_feature_coverage_policy(config)


def _resolve_confluence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_confluence_policy(
        config,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_execution_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_execution_policy(
        config,
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        default_lookback_bars=EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS,
        default_min_samples=EXECUTION_POLICY_DEFAULT_MIN_SAMPLES,
        default_target_range_stop_horizons=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_HORIZONS,
        default_target_range_stop_confidence_min=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN,
        default_target_range_stop_buffer_std_mult=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT,
        default_target_range_stop_min_tighten_fraction=EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION,
    )


def _resolve_forecast_coherence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_forecast_coherence_policy(
        config,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_direction_output_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_direction_output_policy(
        config,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_trust_hardening_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_trust_hardening_policy(
        config,
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )


def _evaluate_feature_coverage(metadata: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    return runtime_evaluate_feature_coverage(metadata, policy)


def _coerce_result_horizon(value: Any) -> float | None:
    try:
        return _normalize_horizon_value(value)
    except ValueError:
        return None


def _direction_vote(entry: Mapping[str, Any]) -> str:
    return "up" if str(entry.get("direction_next", "down")).lower() == "up" else "down"


def _direction_from_ret_pred(value: Any) -> str:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close: Any, projected_price: Any) -> str:
    close_value = _finite_float_or_none(close)
    projected_value = _finite_float_or_none(projected_price)
    if close_value is None or projected_value is None:
        return "neutral"
    if close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"

    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def _direction_from_probability(value: Any, *, neutral_band: float = 0.0) -> str:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _resolve_direction_threshold_for_horizon(
    *,
    direction_threshold: float,
    auto_direction_threshold: bool,
    horizon_p_up: float,
) -> float:
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
    return _project_price(close, stop_return), _project_price(close, take_return)


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
        forecast_consensus_signal = 1 if ret_side == "up" else 0
        if calibrated_side != ret_side:
            return forecast_consensus_signal

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


def _apply_forecast_coherence_policy(
    summary: Dict[str, Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_forecast_coherence_policy(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
        append_gate_trace=_append_gate_trace,
    )


def _parse_horizon_label(value: str) -> float:
    lowered = str(value).strip().lower()
    if lowered.endswith("h"):
        return float(lowered[:-1])
    if lowered.endswith("m"):
        return float(lowered[:-1]) / 60.0
    return float(lowered)


def _forecast_coherence_excluded(entry: Mapping[str, Any]) -> bool:
    return runtime_forecast_coherence_excluded(entry) or bool(entry.get("excluded_from_voting", False))


def _coherence_weight_multiplier(
    entry: Mapping[str, Any],
    *,
    horizon: float,
    policy: Mapping[str, Any],
) -> float:
    weighting_cfg = policy.get("coherence_weighting") if isinstance(policy.get("coherence_weighting"), Mapping) else {}
    base_multiplier = _lookup_horizon_value(
        weighting_cfg.get("by_horizon", {}) if isinstance(weighting_cfg.get("by_horizon"), Mapping) else {},
        horizon,
        1.0,
    )
    base_multiplier = max(float(base_multiplier), 0.0)
    if not bool(weighting_cfg.get("enabled", False)):
        return base_multiplier

    multiplier = base_multiplier
    min_multiplier = max(min(float(weighting_cfg.get("min_multiplier", 0.1) or 0.1), 1.5), 0.0)
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    low_trust_penalty = max(min(float(weighting_cfg.get("low_trust_penalty", 0.35) or 0.35), 1.0), 0.0)
    blocked_penalty = max(min(float(weighting_cfg.get("blocked_penalty", 1.0) or 1.0), 1.0), 0.0)
    p_up_conflict_penalty = max(min(float(weighting_cfg.get("p_up_conflict_penalty", 0.2) or 0.2), 1.0), 0.0)
    consensus_bonus = max(float(weighting_cfg.get("consensus_bonus", 0.1) or 0.1), 0.0)

    if bool(coherence.get("triggered")):
        multiplier *= max(0.0, 1.0 - blocked_penalty)
    elif bool(coherence.get("low_trust")):
        multiplier *= max(0.0, 1.0 - low_trust_penalty)

    ret_side = str(coherence.get("ret_pred_side") or _direction_from_ret_pred(entry.get("ret_pred")))
    projected_side = str(
        coherence.get("projected_price_side")
        or _direction_from_projected_price(entry.get("close"), entry.get("projected_price"))
    )
    p_up_side = str(
        coherence.get("p_up_side")
        or _direction_from_probability(entry.get("p_up"), neutral_band=float(weighting_cfg.get("neutral_band", 0.02) or 0.02))
    )
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else None
    if consensus_side is not None and p_up_side in {"up", "down"}:
        if p_up_side != consensus_side:
            multiplier *= max(0.0, 1.0 - p_up_conflict_penalty)
        else:
            multiplier *= 1.0 + consensus_bonus

    return max(float(multiplier), min_multiplier)


def _apply_confluence_policy(
    summary: Dict[str, Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_confluence_policy(
        summary,
        policy,
        forecast_coherence_excluded=_forecast_coherence_excluded,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        lookup_horizon_value=_lookup_horizon_value,
        append_gate_trace=_append_gate_trace,
    )


def _apply_trust_hardening(
    summary: Dict[str, Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_trust_hardening(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )


def _lookup_horizon_value(mapping: Mapping[float, float], horizon: float, default: float) -> float:
    numeric_horizon = _normalize_horizon_value(horizon)
    if numeric_horizon in mapping:
        return float(mapping[numeric_horizon])
    for key, value in mapping.items():
        if abs(float(key) - numeric_horizon) <= 1e-6:
            return float(value)
    return float(default)


def _dominant_direction_from_scores(up_score: float, down_score: float) -> tuple[str, float]:
    total = max(float(up_score) + float(down_score), 0.0)
    if total <= 0.0:
        return "neutral", 0.0
    if up_score > down_score:
        return "up", float(up_score / total)
    if down_score > up_score:
        return "down", float(down_score / total)
    return "neutral", 0.5


def _compute_weighted_direction_scores(
    labeled_entries: Sequence[tuple[str, Mapping[str, Any], float]],
    *,
    weights: Mapping[float, float] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_weights = weights or {}
    up_score = 0.0
    down_score = 0.0
    details: List[Dict[str, Any]] = []
    for label, entry, horizon in labeled_entries:
        direction = _direction_vote(entry)
        if direction not in {"up", "down"}:
            continue
        base_weight = max(_lookup_horizon_value(resolved_weights, horizon, 1.0), 0.0)
        confidence = max(float(entry.get("confidence_score") or 0.0), 0.0)
        coherence_multiplier = _coherence_weight_multiplier(entry, horizon=horizon, policy=policy or {})
        trust_weight = max(float(entry.get("voting_weight_after_trust") or 1.0), 0.0)
        weighted_vote = base_weight * coherence_multiplier * trust_weight * (0.5 + 0.5 * min(confidence, 1.0))
        if direction == "up":
            up_score += weighted_vote
        else:
            down_score += weighted_vote
        details.append(
            {
                "label": label,
                "horizon_hours": float(horizon),
                "direction": direction,
                "base_weight": float(base_weight),
                "confidence_score": float(confidence),
                "coherence_multiplier": float(coherence_multiplier),
                "trust_weight": float(trust_weight),
                "weighted_vote": float(weighted_vote),
            }
        )
    dominant_direction, dominant_ratio = _dominant_direction_from_scores(up_score, down_score)
    return {
        "dominant_direction": dominant_direction,
        "dominant_ratio": float(dominant_ratio),
        "up_score": float(up_score),
        "down_score": float(down_score),
        "total_score": float(up_score + down_score),
        "details": details,
    }


def _resolve_execution_upstream_hold_reason(entry: Mapping[str, Any]) -> str:
    trade_decision = entry.get("trade_decision") if isinstance(entry.get("trade_decision"), Mapping) else {}
    if trade_decision.get("confluence_gate_triggered"):
        return "confluence_gate"

    blocking_reason = str(trade_decision.get("blocking_reason") or "").strip()
    if blocking_reason:
        return blocking_reason

    weak_band_veto = trade_decision.get("weak_band_veto") if isinstance(trade_decision.get("weak_band_veto"), Mapping) else {}
    if weak_band_veto.get("triggered"):
        return str(weak_band_veto.get("reason") or "weak_band_veto")

    midband_veto = trade_decision.get("midband_veto") if isinstance(trade_decision.get("midband_veto"), Mapping) else {}
    if midband_veto.get("triggered"):
        return str(midband_veto.get("reason") or "midband_veto")

    abstention = entry.get("abstention") if isinstance(entry.get("abstention"), Mapping) else {}
    if abstention.get("triggered"):
        return str(abstention.get("reason") or "abstention_gate")

    return "upstream_model_hold"


def _execution_side(entry: Mapping[str, Any]) -> str:
    return "long" if _direction_vote(entry) == "up" else "short"


def _compute_atr_like_price_distance(
    frame: pd.DataFrame,
    *,
    index: int,
    fallback_close: float,
    fallback_return_std: float,
    window: int = 14,
) -> float:
    start = max(0, index - max(window, 2) + 1)
    history = frame.iloc[start : index + 1].copy()
    if {"high", "low", "close"}.issubset(history.columns):
        high = pd.to_numeric(history["high"], errors="coerce")
        low = pd.to_numeric(history["low"], errors="coerce")
        close = pd.to_numeric(history["close"], errors="coerce")
        valid_close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_close.empty:
            anchor = float(valid_close.tail(window).median())
            if anchor > 0.0 and fallback_close > 0.0:
                deviation = abs(anchor / fallback_close - 1.0)
                if deviation > 0.5:
                    return max(float(fallback_close) * max(abs(float(fallback_return_std)), MIN_RESIDUAL_STD), 1e-8)
            elif anchor <= 0.0 and fallback_close > 0.0:
                return max(float(fallback_close) * max(abs(float(fallback_return_std)), MIN_RESIDUAL_STD), 1e-8)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        atr = pd.to_numeric(true_range, errors="coerce").tail(window).mean()
        if pd.notna(atr) and float(atr) > 0.0:
            return float(atr)
    return max(float(fallback_close) * max(abs(float(fallback_return_std)), MIN_RESIDUAL_STD), 1e-8)


def _compute_recent_structure(
    frame: pd.DataFrame,
    *,
    index: int,
    session_lookback_bars: int,
    swing_lookback_bars: int,
    atr_distance: float,
    fallback_price: float,
) -> Dict[str, float]:
    start_session = max(0, index - max(session_lookback_bars, 2) + 1)
    start_swing = max(0, index - max(swing_lookback_bars, 2) + 1)
    session_frame = frame.iloc[start_session : index + 1].copy()
    swing_frame = frame.iloc[start_swing : index + 1].copy()

    def _safe_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column not in df.columns:
            return pd.Series([default], dtype=float)
        series = pd.to_numeric(df[column], errors="coerce")
        series = series.dropna()
        if series.empty:
            return pd.Series([default], dtype=float)
        return series.astype(float)

    high_session = float(_safe_series(session_frame, "high", fallback_price).max())
    low_session = float(_safe_series(session_frame, "low", fallback_price).min())
    swing_high = float(_safe_series(swing_frame, "high", fallback_price).max())
    swing_low = float(_safe_series(swing_frame, "low", fallback_price).min())
    close_series = _safe_series(session_frame, "close", fallback_price)
    volume_series = _safe_series(session_frame, "volume", 0.0)
    if float(volume_series.sum()) > 0.0 and len(close_series) == len(volume_series):
        vwap = float((close_series * volume_series).sum() / volume_series.sum())
    else:
        vwap = float(close_series.iloc[-1]) if not close_series.empty else float(fallback_price)

    if fallback_price > 0.0:
        structure_values = (high_session, low_session, swing_high, swing_low, vwap)
        invalid_structure = any(value <= 0.0 for value in structure_values)
        if not invalid_structure:
            invalid_structure = any(abs(value / fallback_price - 1.0) > 0.5 for value in structure_values)
        if invalid_structure:
            high_session = float(fallback_price + atr_distance)
            low_session = float(max(fallback_price - atr_distance, 1e-8))
            swing_high = float(fallback_price + atr_distance * 1.5)
            swing_low = float(max(fallback_price - atr_distance * 1.5, 1e-8))
            vwap = float(fallback_price)

    return {
        "session_high": high_session,
        "session_low": low_session,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "vwap": vwap,
        "atr_distance": float(max(atr_distance, 1e-8)),
    }


def _compute_excursion_priors(
    frame: pd.DataFrame,
    *,
    index: int,
    horizon_steps: int,
    side: str,
    lookback_bars: int,
    min_samples: int,
    mae_quantile: float,
    mfe_quantile: float,
    current_regime: str | None = None,
    current_volatility: float | None = None,
    bucket_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "sample_count": 0,
        "mae_distance": None,
        "mfe_distance": None,
        "peak_step_p50": None,
        "adverse_step_p50": None,
        "source": "global",
        "matched_regime": None,
        "volatility_bucket": None,
        "bucket_threshold": None,
    }
    if horizon_steps <= 0 or index <= horizon_steps:
        return result
    if not {"high", "low", "close"}.issubset(frame.columns):
        return result

    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    end = max(index - horizon_steps, 0)
    start = max(0, end - max(lookback_bars, min_samples))
    selected_indices = list(range(start, end))

    normalized_regime = str(current_regime or "").strip().lower() or None
    if bucket_policy and bool(bucket_policy.get("enabled", False)) and start < end:
        regime_col = str(bucket_policy.get("regime_col") or "regime_state")
        volatility_col = str(bucket_policy.get("volatility_col") or "volatility_realized_24h")
        min_bucket_samples = max(int(bucket_policy.get("min_bucket_samples") or min_samples), 1)
        low_vol_quantile = max(min(float(bucket_policy.get("low_vol_quantile") or 0.5), 0.95), 0.05)
        breakout_score_threshold = float(bucket_policy.get("breakout_score_threshold") or 0.8)
        chop_score_threshold = float(bucket_policy.get("chop_score_threshold") or 0.3)

        regime_matches: List[int] = []
        bucket_matches: List[int] = []
        regime_bucket_matches: List[int] = []
        regime_match_used = False

        if regime_col in frame.columns:
            regime_series = frame[regime_col].iloc[start:end].fillna("").astype(str).str.strip().str.lower()
            if normalized_regime is not None:
                regime_matches = [start + offset for offset, value in enumerate(regime_series) if value == normalized_regime]
                regime_match_used = bool(regime_matches)
        elif normalized_regime is not None:
            derived_regimes = _derive_regime_labels_from_frame(
                frame.iloc[start:end].copy(),
                volatility_col=volatility_col,
                breakout_score_threshold=breakout_score_threshold,
                chop_score_threshold=chop_score_threshold,
            )
            regime_matches = [start + offset for offset, value in enumerate(derived_regimes.astype(str).str.lower()) if value == normalized_regime]
            regime_match_used = bool(regime_matches)

        if volatility_col in frame.columns and current_volatility is not None and math.isfinite(float(current_volatility)):
            volatility_history = pd.to_numeric(frame[volatility_col].iloc[start:end], errors="coerce")
            valid_history = volatility_history.dropna()
            if not valid_history.empty:
                bucket_threshold = float(valid_history.quantile(low_vol_quantile))
                current_bucket = "low_vol" if float(current_volatility) <= bucket_threshold else "high_vol"
                bucket_matches = [
                    start + offset
                    for offset, value in enumerate(volatility_history)
                    if pd.notna(value)
                    and ((current_bucket == "low_vol" and float(value) <= bucket_threshold) or (current_bucket == "high_vol" and float(value) > bucket_threshold))
                ]
                result["volatility_bucket"] = current_bucket
                result["bucket_threshold"] = bucket_threshold

        if regime_matches and bucket_matches:
            regime_bucket_matches = sorted(set(regime_matches).intersection(bucket_matches))

        if len(regime_bucket_matches) >= min_bucket_samples:
            selected_indices = regime_bucket_matches
            result["source"] = "regime_volatility_bucket"
        elif len(regime_matches) >= min_bucket_samples:
            selected_indices = regime_matches
            result["source"] = "regime_bucket"
        elif len(bucket_matches) >= min_bucket_samples:
            selected_indices = bucket_matches
            result["source"] = "volatility_bucket"

        if result["source"] in {"regime_bucket", "regime_volatility_bucket"} and regime_match_used:
            result["matched_regime"] = normalized_regime

    maes: List[float] = []
    mfes: List[float] = []
    peak_steps: List[int] = []
    adverse_steps: List[int] = []
    for cursor in selected_indices:
        entry = close[cursor]
        if not math.isfinite(entry) or entry <= 0.0:
            continue
        future_high = high[cursor + 1 : cursor + 1 + horizon_steps]
        future_low = low[cursor + 1 : cursor + 1 + horizon_steps]
        if future_high.size == 0 or future_low.size == 0:
            continue
        if side == "long":
            favorable_idx = int(np.nanargmax(future_high))
            adverse_idx = int(np.nanargmin(future_low))
            favorable = max(float(future_high[favorable_idx]) / entry - 1.0, 0.0)
            adverse = max(1.0 - float(future_low[adverse_idx]) / entry, 0.0)
        else:
            favorable_idx = int(np.nanargmin(future_low))
            adverse_idx = int(np.nanargmax(future_high))
            favorable = max(entry / float(future_low[favorable_idx]) - 1.0, 0.0)
            adverse = max(float(future_high[adverse_idx]) / entry - 1.0, 0.0)
        if not math.isfinite(favorable) or not math.isfinite(adverse):
            continue
        mfes.append(favorable)
        maes.append(adverse)
        peak_steps.append(favorable_idx + 1)
        adverse_steps.append(adverse_idx + 1)

    if len(maes) < min_samples or len(mfes) < min_samples:
        result["sample_count"] = len(maes)
        return result

    result.update(
        {
            "available": True,
            "sample_count": len(maes),
            "mae_distance": float(np.quantile(np.asarray(maes, dtype=float), mae_quantile)),
            "mfe_distance": float(np.quantile(np.asarray(mfes, dtype=float), mfe_quantile)),
            "peak_step_p50": int(round(float(np.quantile(np.asarray(peak_steps, dtype=float), 0.5)))),
            "adverse_step_p50": int(round(float(np.quantile(np.asarray(adverse_steps, dtype=float), 0.5)))),
        }
    )
    return result


def _summarize_bias_context(
    summary: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    bias_horizons = set(policy.get("bias_horizons", []))
    execution_horizons = set(policy.get("execution_horizons", []))
    short_term_horizons = set(policy.get("short_term_strict_horizons", []))
    weights = policy.get("horizon_bias_weights") if isinstance(policy.get("horizon_bias_weights"), Mapping) else {}
    bias_entries: List[tuple[str, Mapping[str, Any], float]] = []
    execution_entries: List[tuple[str, Mapping[str, Any], float]] = []
    short_entries: List[tuple[str, Mapping[str, Any], float]] = []
    mid_entries: List[tuple[str, Mapping[str, Any], float]] = []
    for label, entry in summary.items():
        if _forecast_coherence_excluded(entry):
            continue
        horizon = _coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        if horizon in bias_horizons:
            bias_entries.append((label, entry, horizon))
            mid_entries.append((label, entry, horizon))
        if horizon in execution_horizons:
            execution_entries.append((label, entry, horizon))
        if horizon in short_term_horizons:
            short_entries.append((label, entry, horizon))

    bias_scores = _compute_weighted_direction_scores(bias_entries, weights=weights, policy=policy)
    execution_scores = _compute_weighted_direction_scores(execution_entries, weights=weights, policy=policy)
    short_term_scores = _compute_weighted_direction_scores(short_entries, weights=weights, policy=policy)
    mid_term_scores = _compute_weighted_direction_scores(mid_entries, weights=weights, policy=policy)

    bias_direction = str(bias_scores.get("dominant_direction", "neutral"))
    bias_alignment_ratio = float(bias_scores.get("dominant_ratio", 0.0))
    min_bias_alignment_ratio = max(min(float(policy.get("min_bias_alignment_ratio", 0.0) or 0.0), 1.0), 0.0)
    if bias_direction != "neutral" and bias_alignment_ratio < min_bias_alignment_ratio:
        bias_direction = "neutral"

    direction_support_horizons: Dict[str, List[str]] = {"up": [], "down": []}
    for label, entry, _horizon in bias_entries:
        direction = _direction_vote(entry)
        if direction in direction_support_horizons:
            direction_support_horizons[direction].append(label)

    return {
        "bias_direction": bias_direction,
        "bias_direction_pre_threshold": str(bias_scores.get("dominant_direction", "neutral")),
        "bias_alignment_ratio": bias_alignment_ratio,
        "bias_scores": bias_scores,
        "execution_scores": execution_scores,
        "short_term_scores": short_term_scores,
        "mid_term_scores": mid_term_scores,
        "short_term_direction": str(short_term_scores.get("dominant_direction", "neutral")),
        "short_term_alignment_ratio": float(short_term_scores.get("dominant_ratio", 0.0)),
        "mid_term_direction": str(mid_term_scores.get("dominant_direction", "neutral")),
        "mid_term_alignment_ratio": float(mid_term_scores.get("dominant_ratio", 0.0)),
        "min_bias_alignment_ratio": float(min_bias_alignment_ratio),
        "direction_support_horizons": direction_support_horizons,
        "execution_entries": execution_entries,
    }


def _execution_alignment_ratio(
    execution_entries: Sequence[tuple[str, Mapping[str, Any], float]],
    *,
    direction: str,
    weights: Mapping[float, float] | None = None,
) -> float:
    if not execution_entries:
        return 0.0
    score_payload = _compute_weighted_direction_scores(execution_entries, weights=weights)
    total = float(score_payload.get("total_score", 0.0) or 0.0)
    if total <= 0.0:
        return 0.0
    if direction == "up":
        return float(score_payload.get("up_score", 0.0) or 0.0) / total
    if direction == "down":
        return float(score_payload.get("down_score", 0.0) or 0.0) / total
    return 0.0


def _classify_execution_tier(
    entry: Mapping[str, Any],
    *,
    bias_direction: str,
    execution_alignment_ratio: float,
    policy: Mapping[str, Any],
) -> str:
    direction = _direction_vote(entry)
    horizon = _coerce_result_horizon(entry.get("horizon_hours")) or 0.0
    support_ratio = float(entry.get("confluence_support_ratio") or 0.0)
    mid_ratio = float(entry.get("confluence_mid_term_ratio") or 0.0)
    if bias_direction != "neutral" and direction != bias_direction:
        return "low"
    if horizon in set(policy.get("short_term_strict_horizons", [])):
        strict_mid_ratio = _lookup_horizon_value(
            policy.get("short_term_min_mid_ratio_by_horizon", {}),
            horizon,
            float(policy.get("short_term_min_mid_ratio", 0.67)),
        )
        strict_support_ratio = _lookup_horizon_value(
            policy.get("short_term_min_support_ratio_by_horizon", {}),
            horizon,
            float(policy.get("short_term_min_support_ratio", 0.75)),
        )
        if support_ratio < strict_support_ratio or mid_ratio < strict_mid_ratio:
            return "low"
    if (
        support_ratio >= float(policy.get("immediate_entry_min_support_ratio", 0.8))
        and mid_ratio >= float(policy.get("immediate_entry_min_mid_ratio", 0.67))
        and execution_alignment_ratio >= float(policy.get("high_execution_alignment_ratio", 1.0))
    ):
        return "high"
    if (
        support_ratio >= float(policy.get("pullback_entry_min_support_ratio", 0.6))
        and mid_ratio >= float(policy.get("pullback_entry_min_mid_ratio", 0.5))
        and execution_alignment_ratio >= float(policy.get("medium_execution_alignment_ratio", 0.5))
    ):
        return "medium"
    return "low"


def _build_entry_zone(
    *,
    market_price: float,
    side: str,
    structure: Mapping[str, float],
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any] | None = None,
) -> Dict[str, float | bool | str]:
    atr_distance = float(structure.get("atr_distance", 0.0))
    template_zone_mult = float((regime_template or {}).get("entry_zone_atr_mult") or 0.0)
    entry_zone_width = atr_distance * (
        template_zone_mult if template_zone_mult > 0.0 else float(policy.get("entry_zone_atr_mult", 0.25))
    )
    session_high = float(structure.get("session_high", market_price))
    session_low = float(structure.get("session_low", market_price))
    range_size = max(session_high - session_low, atr_distance)
    vwap = float(structure.get("vwap", market_price))
    if side == "long":
        preferred = min(market_price, max(vwap, session_low + range_size * 0.382))
    else:
        preferred = max(market_price, min(vwap, session_high - range_size * 0.382))
    zone_low = preferred - entry_zone_width
    zone_high = preferred + entry_zone_width
    in_zone = zone_low <= market_price <= zone_high
    return {
        "preferred_entry_price": float(preferred),
        "entry_zone_low": float(zone_low),
        "entry_zone_high": float(zone_high),
        "entry_ready": bool(in_zone),
        "vwap_reference": vwap,
    }


def _resolve_uncertainty_settings(
    policy: Mapping[str, Any],
    *,
    horizon: float | None,
    regime_state: str,
) -> Dict[str, Any]:
    return runtime_resolve_uncertainty_settings(
        policy,
        horizon=horizon,
        regime_state=regime_state,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _compute_recent_candle_expansion(
    frame: pd.DataFrame,
    *,
    index: int,
    window: int,
) -> float:
    if frame.empty:
        return 1.0
    start = max(0, index - max(window, 2) + 1)
    history = frame.iloc[start : index + 1].copy()
    if history.empty:
        return 1.0
    if {"high", "low"}.issubset(history.columns):
        ranges = (pd.to_numeric(history["high"], errors="coerce") - pd.to_numeric(history["low"], errors="coerce")).abs()
    else:
        closes = pd.to_numeric(history.get("close"), errors="coerce")
        ranges = closes.diff().abs()
    clean = ranges.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 1.0
    latest = float(clean.iloc[-1])
    baseline = float(clean.iloc[:-1].median()) if clean.size > 1 else float(clean.median())
    if baseline <= 0.0:
        return 1.0
    return float(latest / baseline)


def _compute_pullback_quality_score(
    *,
    entry: Mapping[str, Any],
    frame: pd.DataFrame,
    index: int,
    market_price: float,
    side: str,
    structure: Mapping[str, float],
    atr_distance: float,
    horizon: float,
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any],
) -> Dict[str, Any]:
    pullback_cfg = policy.get("pullback_quality") if isinstance(policy.get("pullback_quality"), Mapping) else {}
    if not pullback_cfg.get("enabled"):
        return {
            "enabled": False,
            "score": 1.0,
            "min_score": 0.0,
            "triggered": False,
            "vwap_deviation_atr": 0.0,
            "range_expansion_1h": _finite_float(entry.get("range_expansion_1h"), 0.0),
            "candle_expansion_ratio": 1.0,
        }

    vwap = float(structure.get("vwap", market_price))
    safe_atr = max(float(atr_distance), 1e-8)
    vwap_deviation_atr = abs(market_price - vwap) / safe_atr
    max_vwap_deviation = float(pullback_cfg.get("max_vwap_deviation_atr", 1.5))
    vwap_score = max(0.0, 1.0 - (vwap_deviation_atr / max(max_vwap_deviation, 1e-8)))

    range_expansion = abs(_finite_float(entry.get("range_expansion_1h"), 0.0))
    range_threshold = float(pullback_cfg.get("range_expansion_penalty_threshold", 1.25))
    if range_expansion <= range_threshold:
        range_score = 1.0
    else:
        range_score = max(0.0, 1.0 - min(range_expansion - range_threshold, 1.0))

    candle_expansion_ratio = _compute_recent_candle_expansion(
        frame,
        index=index,
        window=int(pullback_cfg.get("candle_expansion_window", 8)),
    )
    max_candle_expansion = float(pullback_cfg.get("max_candle_expansion_ratio", 2.0))
    if candle_expansion_ratio <= 1.0:
        candle_score = 1.0
    else:
        candle_score = max(
            0.0,
            1.0 - ((candle_expansion_ratio - 1.0) / max(max_candle_expansion - 1.0, 1e-8)),
        )

    momentum_penalty = 0.0
    if side == "long" and _finite_float(entry.get("momentum_slope_2h"), 0.0) < 0.0:
        momentum_penalty = min(abs(_finite_float(entry.get("momentum_slope_2h"), 0.0)) * 10.0, 0.15)
    if side == "short" and _finite_float(entry.get("momentum_slope_2h"), 0.0) > 0.0:
        momentum_penalty = min(abs(_finite_float(entry.get("momentum_slope_2h"), 0.0)) * 10.0, 0.15)

    score = max(0.0, min(1.0, 0.45 * vwap_score + 0.30 * range_score + 0.25 * candle_score - momentum_penalty))
    min_score = _lookup_horizon_value(
        pullback_cfg.get("min_score_by_horizon", {}),
        horizon,
        max(float(regime_template.get("pullback_quality_floor", 0.0) or 0.0), 0.0),
    )
    min_score = max(min_score, float(regime_template.get("pullback_quality_floor", 0.0) or 0.0))
    return {
        "enabled": True,
        "score": float(score),
        "min_score": float(min_score),
        "triggered": bool(score < min_score),
        "vwap_deviation_atr": float(vwap_deviation_atr),
        "range_expansion_1h": float(range_expansion),
        "candle_expansion_ratio": float(candle_expansion_ratio),
    }


def _compute_disagreement_severity(
    entry: Mapping[str, Any],
    *,
    bias_context: Mapping[str, Any],
    policy: Mapping[str, Any],
    atr_distance: float,
    structure: Mapping[str, float],
) -> Dict[str, Any]:
    disagreement_cfg = policy.get("disagreement_severity") if isinstance(policy.get("disagreement_severity"), Mapping) else {}
    if not disagreement_cfg.get("enabled", True):
        return {
            "enabled": False,
            "score": 0.0,
            "triggered": False,
            "pullback_only": False,
            "reasons": [],
        }

    direction = _direction_vote(entry)
    short_direction = str(bias_context.get("short_term_direction", "neutral"))
    mid_direction = str(bias_context.get("mid_term_direction", "neutral"))
    short_ratio = float(bias_context.get("short_term_alignment_ratio", 0.0) or 0.0)
    mid_ratio = float(bias_context.get("mid_term_alignment_ratio", 0.0) or 0.0)
    score = 0.0
    reasons: List[str] = []

    if short_direction in {"up", "down"} and mid_direction in {"up", "down"} and short_direction != mid_direction:
        score += 0.5
        reasons.append("short_mid_direction_conflict")
    if mid_direction in {"up", "down"} and direction == mid_direction and short_direction not in {"neutral", mid_direction}:
        score += 0.15
        reasons.append("short_term_countertrend")
    alignment_gap = abs(mid_ratio - short_ratio)
    if alignment_gap > 0.1:
        score += min(alignment_gap, 0.2)
        reasons.append("alignment_gap")

    vwap = float(structure.get("vwap", _finite_float(entry.get("close"), 0.0)))
    if atr_distance > 0.0:
        vwap_deviation_atr = abs(_finite_float(entry.get("close"), 0.0) - vwap) / max(atr_distance, 1e-8)
        if vwap_deviation_atr >= float(disagreement_cfg.get("vwap_extension_penalty_atr", 0.75)):
            score += 0.1
            reasons.append("vwap_extension")

    range_expansion = abs(_finite_float(entry.get("range_expansion_1h"), 0.0))
    if range_expansion >= float(disagreement_cfg.get("range_expansion_penalty_threshold", 1.0)):
        score += 0.1
        reasons.append("range_expansion")

    score = max(0.0, min(1.0, score))
    block_threshold = float(disagreement_cfg.get("block_threshold", 0.7))
    pullback_threshold = float(disagreement_cfg.get("pullback_threshold", 0.45))
    return {
        "enabled": True,
        "score": float(score),
        "triggered": bool(score >= block_threshold),
        "pullback_only": bool(score >= pullback_threshold and score < block_threshold),
        "reasons": reasons,
        "short_term_direction": short_direction,
        "mid_term_direction": mid_direction,
        "short_term_alignment_ratio": float(short_ratio),
        "mid_term_alignment_ratio": float(mid_ratio),
    }


def _resolve_stop_with_guardrails(
    *,
    side: str,
    planned_entry: float,
    existing_stop: float,
    structure_stop: float,
    analytic_stop: float | None,
    atr_distance: float,
    guards_cfg: Mapping[str, Any],
    analytic_stop_preferred: bool = False,
) -> Dict[str, Any]:
    def _valid_stop(stop_value: float | None) -> bool:
        if stop_value is None or not math.isfinite(float(stop_value)):
            return False
        numeric_stop = float(stop_value)
        if side == "long":
            return numeric_stop < planned_entry
        return numeric_stop > planned_entry

    def _distance(stop_value: float) -> float:
        return planned_entry - stop_value if side == "long" else stop_value - planned_entry

    candidates: List[Dict[str, Any]] = []
    for source_name, stop_value in (
        ("existing", existing_stop),
        ("structure", structure_stop),
        ("analytics", analytic_stop),
    ):
        if not _valid_stop(stop_value):
            continue
        numeric_stop = float(stop_value)
        candidates.append(
            {
                "source": source_name,
                "stop_loss": numeric_stop,
                "risk_unit": _distance(numeric_stop),
            }
        )

    if not candidates:
        fallback_risk = max(atr_distance * 0.5, 1e-8)
        fallback_stop = planned_entry - fallback_risk if side == "long" else planned_entry + fallback_risk
        return {
            "stop_loss": fallback_stop,
            "risk_unit": fallback_risk,
            "source": "atr_fallback",
            "adjustment": {
                "applied": True,
                "type": "atr_fallback",
                "reason": "no_valid_stop_candidates",
                "risk_unit_before": None,
                "risk_unit_after": fallback_risk,
            },
        }

    if analytic_stop_preferred:
        priority = {"analytics": 0, "structure": 1, "existing": 2}
        selected = min(
            candidates,
            key=lambda item: (priority.get(str(item.get("source")), 99), abs(float(item["risk_unit"]) - atr_distance)),
        )
    else:
        selected = max(candidates, key=lambda item: float(item["risk_unit"]))
    selected_stop = float(selected["stop_loss"])
    risk_unit = float(selected["risk_unit"])
    adjustment: Dict[str, Any] | None = None

    if guards_cfg.get("enabled"):
        min_stop = float(guards_cfg.get("min_stop_distance_atr_mult", 0.35)) * atr_distance
        max_stop = float(guards_cfg.get("max_stop_distance_atr_mult", 3.0)) * atr_distance
        if min_stop > 0.0 and risk_unit < min_stop:
            adjusted_stop = planned_entry - min_stop if side == "long" else planned_entry + min_stop
            adjustment = {
                "applied": True,
                "type": "expanded_to_min_stop_distance",
                "reason": "stop_too_tight_near_invalidation",
                "from_source": str(selected["source"]),
                "risk_unit_before": risk_unit,
                "risk_unit_after": min_stop,
            }
            selected_stop = float(adjusted_stop)
            risk_unit = float(min_stop)
        elif max_stop > 0.0 and risk_unit > max_stop:
            within_band = [item for item in candidates if float(item["risk_unit"]) <= max_stop]
            if within_band:
                replacement = max(within_band, key=lambda item: float(item["risk_unit"]))
                adjustment = {
                    "applied": True,
                    "type": "replaced_with_guardrail_candidate",
                    "reason": "stop_too_wide",
                    "from_source": str(selected["source"]),
                    "to_source": str(replacement["source"]),
                    "risk_unit_before": risk_unit,
                    "risk_unit_after": float(replacement["risk_unit"]),
                }
                selected = replacement
                selected_stop = float(replacement["stop_loss"])
                risk_unit = float(replacement["risk_unit"])
            else:
                adjusted_stop = planned_entry - max_stop if side == "long" else planned_entry + max_stop
                adjustment = {
                    "applied": True,
                    "type": "capped_to_max_stop_distance",
                    "reason": "stop_too_wide",
                    "from_source": str(selected["source"]),
                    "risk_unit_before": risk_unit,
                    "risk_unit_after": max_stop,
                }
                selected_stop = float(adjusted_stop)
                risk_unit = float(max_stop)

    return {
        "stop_loss": float(selected_stop),
        "risk_unit": float(max(risk_unit, 1e-8)),
        "source": str(selected.get("source", "unknown")),
        "adjustment": adjustment,
    }


def _refine_stop_with_target_range(
    *,
    side: str,
    planned_entry: float,
    selected_stop: float,
    risk_unit: float,
    atr_distance: float,
    horizon: float,
    projected_high: float | None,
    projected_low: float | None,
    projected_high_confidence: float | None,
    projected_low_confidence: float | None,
    projected_high_residual_std: float | None,
    projected_low_residual_std: float | None,
    policy: Mapping[str, Any],
    guards_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    refinement_cfg = (
        policy.get("target_range_stop_refinement")
        if isinstance(policy.get("target_range_stop_refinement"), Mapping)
        else {}
    )
    if not refinement_cfg.get("enabled"):
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    scoped_horizons = set(refinement_cfg.get("horizons", []))
    if scoped_horizons and _normalize_horizon_value(horizon) not in scoped_horizons:
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    if side == "long":
        projected_adverse = _finite_float_or_none(projected_low)
        confidence = _finite_float_or_none(projected_low_confidence)
        residual_std = _finite_float_or_none(projected_low_residual_std)
        projection_field = "projected_low"
        tighten_only = projected_adverse is not None and projected_adverse > selected_stop and projected_adverse < planned_entry
    else:
        projected_adverse = _finite_float_or_none(projected_high)
        confidence = _finite_float_or_none(projected_high_confidence)
        residual_std = _finite_float_or_none(projected_high_residual_std)
        projection_field = "projected_high"
        tighten_only = projected_adverse is not None and projected_adverse < selected_stop and projected_adverse > planned_entry

    if not tighten_only or confidence is None:
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    confidence_min = float(refinement_cfg.get("confidence_min", EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN))
    if confidence < confidence_min:
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    residual_std_value = max(float(residual_std or 0.0), 0.0)
    uncertainty_buffer = max(
        planned_entry * residual_std_value * float(refinement_cfg.get("buffer_std_mult", EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT)),
        atr_distance * 0.1,
        1e-8,
    )
    if side == "long":
        candidate_stop = min(float(projected_adverse) - uncertainty_buffer, planned_entry - 1e-8)
        candidate_risk = planned_entry - candidate_stop
    else:
        candidate_stop = max(float(projected_adverse) + uncertainty_buffer, planned_entry + 1e-8)
        candidate_risk = candidate_stop - planned_entry

    min_stop = float(guards_cfg.get("min_stop_distance_atr_mult", 0.35)) * atr_distance if guards_cfg.get("enabled") else 0.0
    candidate_risk = max(candidate_risk, min_stop, 1e-8)
    candidate_stop = planned_entry - candidate_risk if side == "long" else planned_entry + candidate_risk

    if candidate_risk >= risk_unit:
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    tighten_fraction = (float(risk_unit) - float(candidate_risk)) / max(float(risk_unit), 1e-8)
    min_tighten_fraction = float(
        refinement_cfg.get("min_tighten_fraction", EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION)
    )
    if tighten_fraction < min_tighten_fraction:
        return {
            "applied": False,
            "stop_loss": float(selected_stop),
            "risk_unit": float(risk_unit),
            "details": None,
        }

    return {
        "applied": True,
        "stop_loss": float(candidate_stop),
        "risk_unit": float(candidate_risk),
        "details": {
            "applied": True,
            "type": "target_range_stop_tightened",
            "projection_field": projection_field,
            "projected_level": float(projected_adverse),
            "confidence": float(confidence),
            "confidence_min": float(confidence_min),
            "uncertainty_buffer": float(uncertainty_buffer),
            "risk_unit_before": float(risk_unit),
            "risk_unit_after": float(candidate_risk),
            "tighten_fraction": float(tighten_fraction),
        },
    }


def _apply_execution_policy(
    summary: Dict[str, Dict[str, Any]],
    contexts: Mapping[str, Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_execution_policy(
        summary,
        contexts,
        policy,
        regime_neutral=REGIME_NEUTRAL,
        execution_policy_default_lookback_bars=EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS,
        execution_policy_default_min_samples=EXECUTION_POLICY_DEFAULT_MIN_SAMPLES,
        summarize_bias_context=_summarize_bias_context,
        execution_side=_execution_side,
        direction_vote=_direction_vote,
        execution_alignment_ratio=lambda execution_entries, direction, weights: _execution_alignment_ratio(
            execution_entries,
            direction=direction,
            weights=weights,
        ),
        classify_execution_tier=lambda entry, bias_direction, execution_alignment_ratio, policy: _classify_execution_tier(
            entry,
            bias_direction=bias_direction,
            execution_alignment_ratio=execution_alignment_ratio,
            policy=policy,
        ),
        compute_atr_like_price_distance=_compute_atr_like_price_distance,
        compute_recent_structure=_compute_recent_structure,
        build_entry_zone=_build_entry_zone,
        compute_pullback_quality_score=_compute_pullback_quality_score,
        compute_disagreement_severity=_compute_disagreement_severity,
        compute_excursion_priors=_compute_excursion_priors,
        finite_float_or_none=_finite_float_or_none,
        finite_float=_finite_float,
        resolve_stop_with_guardrails=_resolve_stop_with_guardrails,
        refine_stop_with_target_range=_refine_stop_with_target_range,
        resolve_execution_target_reward=_resolve_execution_target_reward,
        lookup_horizon_value=_lookup_horizon_value,
        resolve_execution_upstream_hold_reason=_resolve_execution_upstream_hold_reason,
    )


def _build_execution_prior_summary(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return runtime_build_execution_prior_summary(summary)


def _resolve_execution_target_reward(
    *,
    side: str,
    planned_entry: float,
    existing_take: float,
    projected_high: float | None,
    projected_low: float | None,
    analytics_payload: Mapping[str, Any],
    risk_unit: float,
    horizon: float,
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any],
    regime_state: str = REGIME_NEUTRAL,
) -> Dict[str, Any]:
    rr_floor = _lookup_horizon_value(policy.get("minimum_rr_by_horizon", {}), horizon, 1.0)
    rr_floor *= float(regime_template.get("tp_multiplier", 1.0) or 1.0)
    dynamic_rr_cfg = policy.get("dynamic_rr_floor") if isinstance(policy.get("dynamic_rr_floor"), Mapping) else {}
    dynamic_rr_applied = False
    dynamic_rr_ratio = None
    if bool(dynamic_rr_cfg.get("enabled", False)) and bool(analytics_payload.get("available")):
        sample_count = int(analytics_payload.get("sample_count") or 0)
        min_samples = max(int(dynamic_rr_cfg.get("min_samples", 40) or 40), 1)
        mae_distance = _finite_float_or_none(analytics_payload.get("mae_distance"))
        mfe_distance = _finite_float_or_none(analytics_payload.get("mfe_distance"))
        if sample_count >= min_samples and mae_distance is not None and mfe_distance is not None and mae_distance > 0.0:
            realized_ratio = max(mfe_distance / max(mae_distance, 1e-8), 0.0)
            regime_multiplier = 1.0
            regime_map = dynamic_rr_cfg.get("regime_multiplier") if isinstance(dynamic_rr_cfg.get("regime_multiplier"), Mapping) else {}
            if regime_map:
                regime_multiplier = max(float(regime_map.get(str(regime_state).strip().lower(), 1.0) or 1.0), 0.0)
            scaled_ratio = realized_ratio * max(float(dynamic_rr_cfg.get("mfe_mae_scale", 0.9) or 0.9), 0.0) * regime_multiplier
            max_adjustment = max(min(float(dynamic_rr_cfg.get("max_adjustment", 0.35) or 0.35), 1.0), 0.0)
            floor_reduction = rr_floor * max_adjustment
            bounded_floor = max(rr_floor - floor_reduction, scaled_ratio)
            min_floor = _lookup_horizon_value(
                dynamic_rr_cfg.get("min_floor_by_horizon", {}),
                horizon,
                float(dynamic_rr_cfg.get("default_floor", 0.0) or 0.0),
            )
            max_floor = _lookup_horizon_value(dynamic_rr_cfg.get("max_floor_by_horizon", {}), horizon, rr_floor)
            rr_floor = max(min(bounded_floor, max_floor), min_floor)
            dynamic_rr_applied = True
            dynamic_rr_ratio = realized_ratio
    effective_rr_floor = rr_floor
    existing_reward = abs(existing_take - planned_entry)
    projection_reward = 0.0
    if side == "long" and projected_high is not None:
        projection_reward = max(projected_high - planned_entry, 0.0)
    elif side == "short" and projected_low is not None:
        projection_reward = max(planned_entry - projected_low, 0.0)

    analytics_available = bool(analytics_payload.get("available"))
    analytic_mfe_reward = planned_entry * float(analytics_payload.get("mfe_distance") or 0.0)
    if analytics_available and analytic_mfe_reward > 0.0:
        projection_cap_ratio = float(
            ((policy.get("analytics", {}).get("regime_volatility_buckets") or {}).get("max_projection_mfe_ratio") or 1.25)
        )
        projection_reward = min(projection_reward, analytic_mfe_reward * projection_cap_ratio) if projection_reward > 0.0 else 0.0

    min_reward = rr_floor * risk_unit
    adapted = False
    status = "pass"
    reason = "pass"

    if analytics_available and analytic_mfe_reward > 0.0:
        feasible_reward = max(analytic_mfe_reward, projection_reward)
        if feasible_reward < min_reward:
            adaptive_cfg = policy.get("adaptive_take_profit", {}) if isinstance(policy.get("adaptive_take_profit"), Mapping) else {}
            adaptive_rr_floor = rr_floor * float(adaptive_cfg.get("min_rr_fraction_of_floor", 1.0) or 1.0)
            feasible_rr = feasible_reward / max(risk_unit, 1e-8)
            if bool(adaptive_cfg.get("enabled", False)) and feasible_rr >= adaptive_rr_floor:
                adapted = True
                effective_rr_floor = adaptive_rr_floor
                final_reward = max(feasible_reward, adaptive_rr_floor * risk_unit)
            else:
                status = "rejected"
                reason = "insufficient_mfe_headroom"
                final_reward = max(feasible_reward, 0.0)
        else:
            final_reward = max(min_reward, analytic_mfe_reward, projection_reward)
    else:
        final_reward = max(existing_reward, projection_reward, min_reward)

    risk_reward_ratio = final_reward / max(risk_unit, 1e-8)
    if status == "pass" and risk_reward_ratio < effective_rr_floor:
        status = "rejected"
        reason = "risk_reward_below_floor"

    selected_take = planned_entry + final_reward if side == "long" else planned_entry - final_reward
    target_source = "existing_or_projection"
    if analytics_available:
        target_source = "analytics_mfe_adaptive" if adapted else "analytics_mfe"

    return {
        "status": status,
        "reason": reason,
        "selected_take": float(selected_take),
        "risk_reward_ratio": float(risk_reward_ratio),
        "target_management": {
            "source": target_source,
            "adapted_to_mfe_headroom": adapted,
            "analytics_available": analytics_available,
            "original_rr_floor": float(rr_floor),
            "effective_rr_floor": float(effective_rr_floor),
            "analytic_mfe_reward": float(analytic_mfe_reward) if analytic_mfe_reward > 0.0 else None,
            "projection_reward": float(projection_reward) if projection_reward > 0.0 else None,
            "selected_reward": float(final_reward),
            "dynamic_rr_floor_applied": bool(dynamic_rr_applied),
            "dynamic_realized_rr_ratio": float(dynamic_rr_ratio) if dynamic_rr_ratio is not None else None,
        },
    }


def _confidence_level_from_score(value: Any) -> str:
    return runtime_confidence_level_from_score(value, finite_float_or_none=_finite_float_or_none)


def _prompt_direction_label(direction: str) -> str:
    return runtime_prompt_direction_label(direction)


def _format_usd_value(value: Any) -> str | None:
    return runtime_format_usd_value(value, finite_float_or_none=_finite_float_or_none)


def _prompt_effective_direction(entry: Mapping[str, Any]) -> str:
    return runtime_prompt_effective_direction(entry)


def _build_prompt_forecast_clause(label: str, entry: Mapping[str, Any]) -> str:
    return runtime_build_prompt_forecast_clause(
        label,
        entry,
        finite_float_or_none=_finite_float_or_none,
    )


def _prompt_status_rank(status: str) -> int:
    return runtime_prompt_status_rank(status)


def _prompt_reason_rank(reason: str | None) -> int:
    return runtime_prompt_reason_rank(reason)


def _prompt_confluence_rank(tier: str | None) -> int:
    return runtime_prompt_confluence_rank(tier)


def _prompt_entry_rank(label: str, entry: Mapping[str, Any]) -> tuple[int, int, int, int, float, float, float, float]:
    return runtime_prompt_entry_rank(
        label,
        entry,
        coerce_result_horizon=_coerce_result_horizon,
        finite_float_or_none=_finite_float_or_none,
    )


def _select_prompt_candidate_entries(
    summary: Mapping[str, Mapping[str, Any]],
) -> List[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]]:
    return runtime_select_prompt_candidate_entries(
        summary,
        coerce_result_horizon=_coerce_result_horizon,
        finite_float_or_none=_finite_float_or_none,
    )


def _select_prompt_preferred_entry(
    summary: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, Mapping[str, Any] | None, Dict[str, Any] | None]:
    return runtime_select_prompt_preferred_entry(
        summary,
        coerce_result_horizon=_coerce_result_horizon,
        finite_float_or_none=_finite_float_or_none,
    )


def _build_prompt_ready_summary(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return runtime_build_prompt_ready_summary(
        summary,
        select_prompt_preferred_entry=_select_prompt_preferred_entry,
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )


def _build_operator_summary_compact(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    preferred_label: str | None,
    preferred_entry: Mapping[str, Any] | None,
    market_direction: str,
    execution_state: str,
    blocking_factors: Sequence[str],
) -> Dict[str, Any]:
    return runtime_build_operator_summary_compact(
        summary,
        preferred_label=preferred_label,
        preferred_entry=preferred_entry,
        market_direction=market_direction,
        execution_state=execution_state,
        blocking_factors=blocking_factors,
        prompt_effective_direction=_prompt_effective_direction,
    )


def _resolve_data_quality_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_data_quality_policy(config)


def _resolve_abstention_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_abstention_policy(
        config,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_abstention_policy_for_horizon(
    policy: Mapping[str, Any],
    *,
    horizon: float | None,
    regime_state: str,
) -> Dict[str, Any]:
    return runtime_resolve_abstention_policy_for_horizon(
        policy,
        horizon=horizon,
        regime_state=regime_state,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_regime_model_weights_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    return runtime_resolve_regime_model_weights_policy(
        config,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        parse_weight_spec=parse_weight_spec,
    )


def _resolve_direction_ensemble_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_direction_ensemble_policy(
        config,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _scope_direction_ensemble_policy(policy: Mapping[str, Any] | None, horizon: float) -> Dict[str, Any]:
    return runtime_scope_direction_ensemble_policy(
        policy,
        horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_uncertainty_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_uncertainty_policy(
        config,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _resolve_degradation_monitoring_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lookback_snapshots": max(int(cfg.get("lookback_snapshots") or DEGRADATION_MONITORING_DEFAULT_LOOKBACK), 3),
        "min_snapshots": max(int(cfg.get("min_snapshots") or DEGRADATION_MONITORING_DEFAULT_MIN_SNAPSHOTS), 1),
        "min_ready_ratio": max(min(float(cfg.get("min_ready_ratio") or 0.1), 1.0), 0.0),
        "max_blocked_ratio": max(min(float(cfg.get("max_blocked_ratio") or 0.85), 1.0), 0.0),
        "min_expected_net": float(cfg.get("min_expected_net") or 0.0),
        "min_confidence": max(min(float(cfg.get("min_confidence") or 0.0), 1.0), 0.0),
    }


def _resolve_trade_decision_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_trade_decision_policy(
        config,
        finite_float_or_none=_finite_float_or_none,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        stderr_write=sys.stderr.write,
    )


def _resolve_trade_decision_threshold(
    policy: Mapping[str, Any],
    *,
    horizon_label: str | None,
    regime_state: str,
) -> tuple[float, str]:
    return runtime_resolve_trade_decision_threshold(
        policy,
        horizon_label=horizon_label,
        regime_state=regime_state,
        normalize_horizon_value=_normalize_horizon_value,
        parse_horizon_label=_parse_horizon_label,
        format_horizon_label=_format_horizon_label,
    )


def _sigmoid(value: float) -> float:
    clipped = max(min(float(value), 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_float(value: Any, default: float = 0.0) -> float:
    out = _finite_float_or_none(value)
    return float(default) if out is None else float(out)


def _lookup_raw_ev_fallback_threshold(model: Mapping[str, Any], quantile: float) -> float | None:
    return runtime_lookup_raw_ev_fallback_threshold(
        model,
        quantile,
        finite_float_or_none=_finite_float_or_none,
    )


def _apply_trade_decision_model(
    *,
    result: Dict[str, Any],
    horizon_label: str | None = None,
    regime_state: str,
    residual_std: float,
    policy: Mapping[str, Any],
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    return runtime_apply_trade_decision_model(
        result=result,
        horizon_label=horizon_label,
        regime_state=regime_state,
        residual_std=residual_std,
        policy=policy,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        regime_trend=REGIME_TREND,
        regime_neutral=REGIME_NEUTRAL,
        regime_chop=REGIME_CHOP,
        resolve_trade_decision_threshold=_resolve_trade_decision_threshold,
        sigmoid=_sigmoid,
        finite_float_or_none=_finite_float_or_none,
        finite_float=_finite_float,
    )


def _append_gate_trace(
    entry: Dict[str, Any],
    *,
    stage: str,
    reason: str,
    triggered: bool,
    blocking: bool,
) -> None:
    trace = entry.get("gate_trace")
    if not isinstance(trace, list):
        trace = []
        entry["gate_trace"] = trace
    trace.append(
        {
            "stage": stage,
            "reason": reason,
            "triggered": bool(triggered),
            "blocking": bool(blocking),
        }
    )


def _upstream_trade_gate_reasons(entry: Mapping[str, Any]) -> List[str]:
    return runtime_upstream_trade_gate_reasons(entry)


def _apply_trade_decision_stage(
    summary: Dict[str, Dict[str, Any]],
    execution_contexts: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_trade_decision_stage(
        summary,
        execution_contexts,
        policy,
        default_residual_std=DEFAULT_RESIDUAL_STD,
        regime_neutral=REGIME_NEUTRAL,
        default_fee_bps=DEFAULT_FEE_BPS,
        default_slippage_bps=DEFAULT_SLIPPAGE_BPS,
        apply_trade_decision_model=_apply_trade_decision_model,
        upstream_trade_gate_reasons=_upstream_trade_gate_reasons,
        append_gate_trace=_append_gate_trace,
    )


def _apply_post_trade_gates(
    summary: Dict[str, Dict[str, Any]],
    *,
    confidence_min: float,
    abstention_policy: Mapping[str, Any],
    uncertainty_policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return runtime_apply_post_trade_gates(
        summary,
        confidence_min=confidence_min,
        abstention_policy=abstention_policy,
        uncertainty_policy=uncertainty_policy,
        default_fee_bps=DEFAULT_FEE_BPS,
        default_slippage_bps=DEFAULT_SLIPPAGE_BPS,
        regime_neutral=REGIME_NEUTRAL,
        append_gate_trace=_append_gate_trace,
        resolve_abstention_expected_value=_resolve_abstention_expected_value,
        resolve_abstention_policy_for_horizon=_resolve_abstention_policy_for_horizon,
        apply_abstention_policy=_apply_abstention_policy,
        apply_uncertainty_abstention=_apply_uncertainty_abstention,
        coerce_result_horizon=_coerce_result_horizon,
    )


def _resolve_regime_model_dirs_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    return runtime_resolve_regime_model_dirs_policy(
        config,
        regimes=(REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP),
    )


def _resolve_regime_specific_dir_path(
    default_path: Path,
    *,
    regime_state: str,
    horizon_label: str,
    policy: Mapping[str, Any],
) -> Path:
    return runtime_resolve_regime_specific_dir_path(
        default_path,
        regime_state=regime_state,
        horizon_label=horizon_label,
        policy=policy,
        expected_filename=f"xgb_dir{horizon_label}_model.json",
        version_priority=MODEL_VERSION_PRIORITY,
        resolve_best_versioned_model_file=resolve_best_versioned_model_file,
        stderr_write=sys.stderr.write,
    )


def _apply_regime_weight_overrides(
    base_weights: Mapping[str, float],
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    return runtime_apply_regime_weight_overrides(
        base_weights,
        regime_state=regime_state,
        horizon=horizon,
        policy=policy,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _get_active_regime_weight_override(
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, float]]:
    return runtime_get_active_regime_weight_override(
        regime_state=regime_state,
        horizon=horizon,
        policy=policy,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _apply_abstention_policy(
    *,
    trade_action: str,
    p_up: float,
    confidence_score: float,
    expected_value: float,
    fee_bps: float,
    slippage_bps: float,
    policy: Mapping[str, Any],
) -> tuple[bool, str]:
    return runtime_apply_abstention_policy(
        trade_action=trade_action,
        p_up=p_up,
        confidence_score=confidence_score,
        expected_value=expected_value,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        policy=policy,
    )


def _resolve_abstention_expected_value(
    expected_value: float,
    trade_decision: Mapping[str, Any] | None,
) -> tuple[float, str]:
    return runtime_resolve_abstention_expected_value(expected_value, trade_decision)


def _apply_uncertainty_abstention(
    *,
    trade_action: str,
    p_up_components: Mapping[str, Any],
    horizon: float | None,
    regime_state: str,
    policy: Mapping[str, Any],
) -> tuple[bool, str, Dict[str, Any]]:
    return runtime_apply_uncertainty_abstention(
        trade_action=trade_action,
        p_up_components=p_up_components,
        horizon=horizon,
        regime_state=regime_state,
        policy=policy,
        resolve_uncertainty_settings=_resolve_uncertainty_settings,
    )


def _compute_confidence_score(p_up: float, expected_value: float, residual_std: float) -> float:
    # Blend directional conviction with risk-adjusted edge into a bounded confidence score.
    directional = min(1.0, abs(p_up - 0.5) * 2.0)
    denom = max(abs(residual_std), 1e-8)
    edge = max(-1.0, min(1.0, expected_value / denom))
    edge_component = (edge + 1.0) * 0.5
    return float(max(0.0, min(1.0, 0.6 * directional + 0.4 * edge_component)))


def _compute_position_size(
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
    scaled = (confidence_score - confidence_min) / max(1e-8, (1.0 - confidence_min))
    return float(min(size_cap, max(size_floor, scaled * size_cap)))


def _target_range_label(horizon: float) -> str:
    return runtime_target_range_label(horizon)


def _load_target_range_model(path: Path) -> Optional[Dict[str, Any]]:
    return runtime_load_target_range_model(path, stderr_write=sys.stderr.write)


def _load_target_range_models(
    policy: Mapping[str, Any] | None,
    horizons: Sequence[float],
) -> Dict[float, Dict[str, Any]]:
    return runtime_load_target_range_models(
        policy,
        horizons,
        target_range_model_dir=TARGET_RANGE_MODEL_DIR,
        load_target_range_model_fn=_load_target_range_model,
        stderr_write=sys.stderr.write,
    )


def _predict_single_target_model(payload: Mapping[str, Any], row: pd.Series) -> float:
    return runtime_predict_single_target_model(payload, row)


def _confidence_from_rmse(rmse: float | None, scale: float) -> float:
    return runtime_confidence_from_rmse(rmse, scale)


def _predict_target_range_prices(
    bundle: Mapping[str, Any],
    row: pd.Series,
    *,
    close: float,
    confidence_scale: float,
) -> Dict[str, float]:
    return runtime_predict_target_range_prices(
        bundle,
        row,
        close=close,
        confidence_scale=confidence_scale,
        finite_float_or_none=_finite_float_or_none,
    )


def _apply_target_range_overrides(
    stop_loss: float,
    take_profit: float,
    projection: Mapping[str, float],
    override_ratio: float,
    direction: int,
) -> tuple[Dict[str, Dict[str, float] | None], float, float]:
    return runtime_apply_target_range_overrides(
        stop_loss,
        take_profit,
        projection,
        override_ratio,
        direction,
    )


def _evaluate_direction_only_fallback(
    policy: Optional[Dict[str, Any]],
    *,
    p_up: float,
    signal_dir_only: int,
    expected_value: float,
    projected_price: float,
    signal_ts: str,
    trend_prob: float,
    trend_threshold: Optional[float],
) -> tuple[Dict[str, Any], bool]:
    return runtime_evaluate_direction_only_fallback(
        policy,
        p_up=p_up,
        signal_dir_only=signal_dir_only,
        expected_value=expected_value,
        projected_price=projected_price,
        signal_ts=signal_ts,
        trend_prob=trend_prob,
        trend_threshold=trend_threshold,
        inactive_direction_fallback=_inactive_direction_fallback,
        parse_iso_timestamp=_parse_iso_timestamp,
    )


def run_ingestion(
    hours: int,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    provider: str = "binanceus",
) -> Path:
    if provider != "binanceus":
        raise ValueError(f"Unsupported provider '{provider}'. Binance-only mode requires --spot-provider binanceus.")

    limit = max(hours, 1)
    print(f"Fetching {limit} {interval} klines from Binance US for {symbol}...")
    output_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=limit)
    print(f"Saved spot tidy parquet to {output_path}")
    return output_path


def _pivot_tidy_spot_ohlcv(path: Path) -> pd.DataFrame:
    return runtime_pivot_tidy_spot_ohlcv(path)


def _compute_intrabar_features_from_15m(path_15m_tidy: Path) -> pd.DataFrame:
    return runtime_compute_intrabar_features_from_15m(path_15m_tidy)


def _build_ohlcv_frame_from_tidy(df: pd.DataFrame) -> pd.DataFrame:
    return runtime_build_ohlcv_frame_from_tidy(df)


def _write_data_quality_payload(payload: Mapping[str, Any]) -> None:
    DATA_QUALITY_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_MONITOR_PATH.write_text(json.dumps(payload, indent=2))


def _evaluate_data_quality(
    frame: pd.DataFrame,
    policy_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return runtime_evaluate_data_quality(
        frame,
        policy_config,
        data_quality_policy_type=DataQualityPolicy,
        evaluate_ohlcv_quality=evaluate_ohlcv_quality,
        data_quality_error_type=DataQualityError,
        write_data_quality_payload=_write_data_quality_payload,
    )


def run_feature_builders(price_source: Path | None = None) -> Dict[str, str]:
    results: Dict[str, str] = {}
    print("Recomputing technical indicator features...")
    technical_path = process_technical_features(price_source=price_source, include_history=True)
    results["technical"] = str(technical_path)

    try:
        existing_macro = load_macro_features(DEFAULT_MACRO_OUTPUT_PATH) if DEFAULT_MACRO_OUTPUT_PATH.exists() else None
        macro_start = resolve_incremental_start_date(
            existing_macro,
            default_start_date=DEFAULT_MACRO_START_DATE,
        )
        macro_frame = build_macro_feature_frame(start_date=macro_start, existing=existing_macro)
        DEFAULT_MACRO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        macro_frame.to_parquet(DEFAULT_MACRO_OUTPUT_PATH, index=False)
        macro_manifest = build_macro_source_manifest()
        macro_manifest["row_count"] = int(len(macro_frame))
        macro_manifest["ts_start"] = macro_frame["ts"].min().isoformat() if not macro_frame.empty else None
        macro_manifest["ts_end"] = macro_frame["ts"].max().isoformat() if not macro_frame.empty else None
        macro_manifest["refresh"] = {
            "requested_start_date": macro_start,
            "output_path": str(DEFAULT_MACRO_OUTPUT_PATH),
        }
        DEFAULT_MACRO_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_MACRO_METADATA_PATH.write_text(json.dumps(macro_manifest, indent=2), encoding="utf-8")
        results["macro"] = str(DEFAULT_MACRO_OUTPUT_PATH)
        print(f"Refreshed macro features at {DEFAULT_MACRO_OUTPUT_PATH}")
    except Exception as exc:
        print(f"Warning: macro feature refresh failed: {exc}", file=sys.stderr)

    try:
        existing_onchain = load_onchain_features(DEFAULT_ONCHAIN_OUTPUT_PATH) if DEFAULT_ONCHAIN_OUTPUT_PATH.exists() else None
        onchain_start = resolve_incremental_start_timestamp(
            existing_onchain,
            default_start=DEFAULT_ONCHAIN_START_DATE,
        )
        onchain_frame = build_onchain_feature_frame(start_ts=onchain_start, existing=existing_onchain)
        DEFAULT_ONCHAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        onchain_frame.to_parquet(DEFAULT_ONCHAIN_OUTPUT_PATH, index=False)
        onchain_manifest = build_onchain_source_manifest()
        onchain_manifest["row_count"] = int(len(onchain_frame))
        onchain_manifest["ts_start"] = onchain_frame["ts"].min().isoformat() if not onchain_frame.empty else None
        onchain_manifest["ts_end"] = onchain_frame["ts"].max().isoformat() if not onchain_frame.empty else None
        onchain_manifest["refresh"] = {
            "requested_start_ts": onchain_start,
            "output_path": str(DEFAULT_ONCHAIN_OUTPUT_PATH),
        }
        write_onchain_source_manifest(DEFAULT_ONCHAIN_METADATA_PATH, onchain_manifest)
        results["onchain"] = str(DEFAULT_ONCHAIN_OUTPUT_PATH)
        print(f"Refreshed on-chain features at {DEFAULT_ONCHAIN_OUTPUT_PATH}")
    except OnchainAPIError as exc:
        print(f"Warning: on-chain feature refresh skipped: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: on-chain feature refresh failed: {exc}", file=sys.stderr)

    return results


def rebuild_datasets(horizons: Sequence[float]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 1h dataset splits...")
    build_1h_dataset(str(DATASET_DIR))

    hourly_targets = {int(round(h)) for h in horizons if h >= 1.0}
    expanded_horizons = sorted(hourly_targets | {1, 4})
    print(f"Building multi-horizon dataset for horizons {expanded_horizons}...")
    build_multi_horizon_dataset(
        output_dir=str(DATASET_DIR),
        horizons=expanded_horizons,
        train_frac=0.7,
        val_frac=0.15,
    )

    if any(h < 1.0 for h in horizons):
        print("Detected sub-hourly targets; refreshing 15m dataset splits...")
        build_15m_dataset(str(DATASET_DIR))


def _read_timeseries_frame(path: str, label: str) -> pd.DataFrame:
    return runtime_read_timeseries_frame(path, label)


def _summarize_frame(df: pd.DataFrame, label: str, path: str) -> Dict[str, Any]:
    return runtime_summarize_frame(df, label, path)


def _merge_override_features(base: pd.DataFrame, extra: pd.DataFrame, label: str) -> tuple[pd.DataFrame, List[str]]:
    return runtime_merge_override_features(base, extra, label)


def _load_training_feature_names() -> List[str] | None:
    return runtime_load_training_feature_names(
        DATASET_MULTI_PATH,
        DATASET_1H_PATH,
        stderr_write=sys.stderr.write,
    )


def _enrich_local_features_for_model(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    return runtime_enrich_local_features_for_model(
        frame,
        required_columns=required_columns,
    )


def _prepare_local_feature_bundle(
    *,
    features_path: str,
    hours: int,
    optional_sources: Mapping[str, str] | None = None,
) -> tuple[tuple[PreparedData, int, float, str], Dict[str, Any]]:
    return runtime_prepare_local_feature_bundle(
        features_path=features_path,
        hours=hours,
        optional_sources=optional_sources,
        dataset_multi_path=DATASET_MULTI_PATH,
        dataset_1h_path=DATASET_1H_PATH,
        local_feature_required_columns=LOCAL_FEATURE_REQUIRED_COLUMNS,
        stderr_write=sys.stderr.write,
    )


def _model_suffix_candidates(horizon: float) -> List[str]:
    return runtime_model_suffix_candidates(horizon, normalize_horizon_value=_normalize_horizon_value)


def _model_paths_for_horizon(horizon: float) -> tuple[Path, Path]:
    return runtime_model_paths_for_horizon(
        horizon,
        format_horizon_label=_format_horizon_label,
        normalize_horizon_value=_normalize_horizon_value,
        model_root=MODEL_ROOT,
        model_version_priority=MODEL_VERSION_PRIORITY,
        dir_version_overrides=DIR_VERSION_OVERRIDES,
        resolve_best_versioned_model_file_fn=resolve_best_versioned_model_file,
        stderr_write=sys.stderr.write,
    )


def _prepare_base_direction_configs(
    *,
    config_json_path: str | None,
    weight_spec: str | None,
    dir_lstm_path: str | None,
    dir_bilstm_path: str | None,
    dir_gru_path: str | None,
    dir_cnn_lstm_path: str | None,
    dir_cnn_bilstm_path: str | None,
    dir_garch_lstm_path: str | None,
    dir_transformer_path: str | None,
) -> List[DirectionModelConfig]:
    return runtime_prepare_base_direction_configs(
        config_json_path=config_json_path,
        weight_spec=weight_spec,
        dir_lstm_path=dir_lstm_path,
        dir_bilstm_path=dir_bilstm_path,
        dir_gru_path=dir_gru_path,
        dir_cnn_lstm_path=dir_cnn_lstm_path,
        dir_cnn_bilstm_path=dir_cnn_bilstm_path,
        dir_garch_lstm_path=dir_garch_lstm_path,
        dir_transformer_path=dir_transformer_path,
        default_dir_models_1h=DEFAULT_DIR_MODELS_1H,
        resolve_direction_model_configs_fn=resolve_direction_model_configs,
    )


def _direction_configs_for_horizon(
    base_configs: Sequence[DirectionModelConfig],
    *,
    dir_model_path: str,
    horizon: float,
    horizon_label: str,
) -> tuple[List[DirectionModelConfig], Dict[str, float]]:
    def _registry_model_exists(model_name: str) -> bool:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.get_registered_model(model_name)
            return True
        except Exception:
            return False

    return runtime_direction_configs_for_horizon(
        base_configs,
        dir_model_path=dir_model_path,
        horizon=horizon,
        horizon_label=horizon_label,
        normalize_horizon_value=_normalize_horizon_value,
        default_transformer_model_dir_by_suffix=DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX,
        model_root=MODEL_ROOT,
        model_version_priority=MODEL_VERSION_PRIORITY,
        clone_direction_model_configs_fn=clone_direction_model_configs,
        apply_path_overrides_fn=apply_path_overrides,
        log_direction_model_configs_fn=log_direction_model_configs,
        direction_configs_to_weight_map_fn=direction_configs_to_weight_map,
        registry_model_exists_fn=_registry_model_exists,
    )


def _load_platt_calibration(path: str | None) -> Dict[str, Dict[str, Any]]:
    return runtime_load_probability_calibration(path, stderr_write=sys.stderr.write)


def _apply_probability_calibration(p: float, params: Mapping[str, Any]) -> float:
    return runtime_apply_probability_calibration(p, params)


def _resolve_probability_calibration(
    platt_calibration: Mapping[str, Mapping[str, Any]] | None,
    label: str,
    regime_state: str,
) -> tuple[str | None, Mapping[str, Any] | None, bool]:
    return runtime_resolve_probability_calibration(
        platt_calibration,
        label,
        regime_state,
        regime_calibration_min_platt_slope=REGIME_CALIBRATION_MIN_PLATT_SLOPE,
    )


def _resolve_trade_probability_for_horizon(
    *,
    platt_calibration: Mapping[str, Mapping[str, Any]] | None,
    label: str,
    regime_state: str,
    raw_probability: float,
    close: float,
    projected_price: float,
    ret_pred: float,
    neutral_band: float = 0.02,
) -> tuple[float, str | None, bool, Dict[str, Any] | None]:
    return runtime_resolve_trade_probability_for_horizon(
        platt_calibration=platt_calibration,
        label=label,
        regime_state=regime_state,
        raw_probability=raw_probability,
        close=close,
        projected_price=projected_price,
        ret_pred=ret_pred,
        neutral_band=neutral_band,
        regime_calibration_min_platt_slope=REGIME_CALIBRATION_MIN_PLATT_SLOPE,
        apply_probability_calibration=_apply_probability_calibration,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
    )


def _build_direction_output(
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
) -> Dict[str, Any]:
    return runtime_build_direction_output(
        enabled=enabled,
        scoped=scoped,
        label=label,
        regime_state=regime_state,
        signal_dir_only=signal_dir_only,
        raw_probability=raw_probability,
        trade_probability=trade_probability,
        ret_pred=ret_pred,
        close=close,
        projected_price=projected_price,
        p_up_components=p_up_components,
        policy=policy,
        apply_probability_calibration=_apply_probability_calibration,
        resolve_probability_calibration=_resolve_probability_calibration,
        parse_horizon_label=_parse_horizon_label,
        lookup_horizon_value=_lookup_horizon_value,
        direction_from_ret_pred=_direction_from_ret_pred,
        direction_from_projected_price=_direction_from_projected_price,
        direction_from_probability=_direction_from_probability,
    )


def _load_prepared(dataset_path: Path, *, target_column: str, offline: bool = False) -> tuple:
    return runtime_load_prepared(
        dataset_path,
        target_column=target_column,
        offline=offline,
        load_prepared_offline_fn=_load_prepared_offline,
        prepare_data_for_signals_fn=prepare_data_for_signals,
        format_ts_iso_fn=format_ts_iso,
    )


def _base_horizon_for_target_column(target_column: str) -> float:
    return runtime_base_horizon_for_target_column(target_column)


def _periods_per_hour_for_base_horizon(base_horizon: float) -> int:
    return runtime_periods_per_hour_for_base_horizon(base_horizon)


def _load_prepared_offline(dataset_path: Path, *, base_horizon: float) -> tuple[PreparedData, int, float, str]:
    return runtime_load_prepared_offline(
        dataset_path,
        base_horizon=base_horizon,
        prepare_data_for_signals_from_ohlcv_fn=prepare_data_for_signals_from_ohlcv,
        format_ts_iso_fn=format_ts_iso,
        stderr_write=sys.stderr.write,
    )


def _project_price(close: float, log_return: float) -> float:
    return runtime_project_price(close, log_return)


def run_predictions(
    targets: Iterable[float],
    p_up_min: float,
    ret_min: float,
    direction_threshold: float = 0.5,
    auto_direction_threshold: bool = False,
    offline: bool = False,
    dir_lstm_path: str | None = None,
    dir_bilstm_path: str | None = None,
    dir_gru_path: str | None = None,
    dir_cnn_lstm_path: str | None = None,
    dir_cnn_bilstm_path: str | None = None,
    dir_garch_lstm_path: str | None = None,
    dir_transformer_path: str | None = None,
    dir_model_config_json: str | None = None,
    dir_model_weights: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
    prepared_override: tuple[PreparedData, int, float, str] | None = None,
    trend_ignition: Mapping[str, Any] | None = None,
    direction_only_fallback: Mapping[str, Any] | None = None,
    adaptive_thresholds: Mapping[str, Any] | None = None,
    target_range_models: Mapping[str, Any] | None = None,
    platt_calibration: Mapping[str, Mapping[str, Any]] | None = None,
    abstention_policy: Mapping[str, Any] | None = None,
    uncertainty_policy: Mapping[str, Any] | None = None,
    trade_decision_policy: Mapping[str, Any] | None = None,
    regime_model_weights: Mapping[str, Any] | None = None,
    regime_model_dirs: Mapping[str, Any] | None = None,
    confluence_policy: Mapping[str, Any] | None = None,
    execution_policy: Mapping[str, Any] | None = None,
    forecast_coherence_policy: Mapping[str, Any] | None = None,
    direction_output_policy: Mapping[str, Any] | None = None,
    direction_ensemble_policy: Mapping[str, Any] | None = None,
    trust_hardening_policy: Mapping[str, Any] | None = None,
    latest_close: float | None = None,
    confidence_min: float = CONFIDENCE_MIN_DEFAULT,
    confidence_min_by_horizon_regime: Mapping[float | int | str, Mapping[str, float]] | None = None,
    position_size_floor: float = POSITION_SIZE_FLOOR_DEFAULT,
    position_size_cap: float = POSITION_SIZE_CAP_DEFAULT,
    position_size_cap_by_horizon: Mapping[float | int | str, float] | None = None,
    disabled_horizons: Sequence[float] | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    config = PredictionPipelineConfig(
        targets=targets,
        p_up_min=p_up_min,
        ret_min=ret_min,
        direction_threshold=direction_threshold,
        auto_direction_threshold=auto_direction_threshold,
        offline=offline,
        dir_lstm_path=dir_lstm_path,
        dir_bilstm_path=dir_bilstm_path,
        dir_gru_path=dir_gru_path,
        dir_cnn_lstm_path=dir_cnn_lstm_path,
        dir_cnn_bilstm_path=dir_cnn_bilstm_path,
        dir_garch_lstm_path=dir_garch_lstm_path,
        dir_transformer_path=dir_transformer_path,
        dir_model_config_json=dir_model_config_json,
        dir_model_weights=dir_model_weights,
        thresholds_by_horizon=thresholds_by_horizon,
        prepared_override=prepared_override,
        trend_ignition=trend_ignition,
        direction_only_fallback=direction_only_fallback,
        adaptive_thresholds=adaptive_thresholds,
        target_range_models=target_range_models,
        platt_calibration=platt_calibration,
        abstention_policy=abstention_policy,
        uncertainty_policy=uncertainty_policy,
        trade_decision_policy=trade_decision_policy,
        regime_model_weights=regime_model_weights,
        regime_model_dirs=regime_model_dirs,
        confluence_policy=confluence_policy,
        execution_policy=execution_policy,
        forecast_coherence_policy=forecast_coherence_policy,
        direction_output_policy=direction_output_policy,
        direction_ensemble_policy=direction_ensemble_policy,
        trust_hardening_policy=trust_hardening_policy,
        latest_close=latest_close,
        confidence_min=confidence_min,
        confidence_min_by_horizon_regime=confidence_min_by_horizon_regime,
        position_size_floor=position_size_floor,
        position_size_cap=position_size_cap,
        position_size_cap_by_horizon=position_size_cap_by_horizon,
        disabled_horizons=disabled_horizons,
    )
    deps = PredictionPipelineDependencies(
        normalize_horizon_value=_normalize_horizon_value,
        normalize_threshold_overrides=_normalize_threshold_overrides,
        normalize_horizon_regime_float_map=_normalize_horizon_regime_float_map,
        normalize_horizon_float_map=_normalize_horizon_float_map,
        dataset_profile_for_horizon=_dataset_profile_for_horizon,
        select_dataset_candidate=_select_dataset_candidate,
        prepare_base_direction_configs=_prepare_base_direction_configs,
        load_prepared=_load_prepared,
        resolve_trend_ignition_payload=_resolve_trend_ignition_payload,
        resolve_direction_fallback_policy=_resolve_direction_fallback_policy,
        resolve_adaptive_thresholds_policy=_resolve_adaptive_thresholds_policy,
        resolve_target_range_policy=_resolve_target_range_policy,
        resolve_abstention_policy=_resolve_abstention_policy,
        resolve_uncertainty_policy=_resolve_uncertainty_policy,
        resolve_trade_decision_policy=_resolve_trade_decision_policy,
        resolve_regime_model_weights_policy=_resolve_regime_model_weights_policy,
        resolve_regime_model_dirs_policy=_resolve_regime_model_dirs_policy,
        resolve_confluence_policy=_resolve_confluence_policy,
        resolve_execution_policy=_resolve_execution_policy,
        resolve_forecast_coherence_policy=_resolve_forecast_coherence_policy,
        resolve_direction_output_policy=_resolve_direction_output_policy,
        resolve_direction_ensemble_policy=_resolve_direction_ensemble_policy,
        resolve_trust_hardening_policy=_resolve_trust_hardening_policy,
        compute_breakout_scores=_compute_breakout_scores,
        load_target_range_models=_load_target_range_models,
        format_horizon_label=_format_horizon_label,
        model_paths_for_horizon=_model_paths_for_horizon,
        classify_regime_from_score=_classify_regime_from_score,
        resolve_regime_specific_dir_path=_resolve_regime_specific_dir_path,
        direction_configs_for_horizon=_direction_configs_for_horizon,
        resolve_thresholds_for_horizon=_resolve_thresholds_for_horizon,
        apply_adaptive_thresholds=_apply_adaptive_thresholds,
        apply_regime_weight_overrides=_apply_regime_weight_overrides,
        scope_direction_ensemble_policy=_scope_direction_ensemble_policy,
        resolve_direction_threshold_for_horizon=_resolve_direction_threshold_for_horizon,
        project_price=_project_price,
        resolve_trade_probability_for_horizon=_resolve_trade_probability_for_horizon,
        resolve_direction_signal_for_horizon=_resolve_direction_signal_for_horizon,
        compute_directional_stop_take_prices=_compute_directional_stop_take_prices,
        resolve_confidence_min_for_horizon=_resolve_confidence_min_for_horizon,
        lookup_horizon_value=_lookup_horizon_value,
        compute_position_size=_compute_position_size,
        parse_iso_timestamp=_parse_iso_timestamp,
        predict_target_range_prices=_predict_target_range_prices,
        build_prediction_result=runtime_build_prediction_result,
        get_active_regime_weight_override=_get_active_regime_weight_override,
        derive_probability_alignment_features=_derive_probability_alignment_features,
        build_direction_output=_build_direction_output,
        apply_target_range_overrides=_apply_target_range_overrides,
        evaluate_direction_only_fallback=_evaluate_direction_only_fallback,
        finite_float_or_none=_finite_float_or_none,
        coerce_row_value=lambda value: None if pd.isna(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]) else float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]),
        write_trend_ignition_state=_write_trend_ignition_state,
        write_direction_fallback_state=_write_direction_fallback_state,
        apply_post_prediction_policies=runtime_apply_post_prediction_policies,
        apply_forecast_coherence_policy=_apply_forecast_coherence_policy,
        apply_trust_hardening_stage=_apply_trust_hardening,
        apply_confluence_policy=_apply_confluence_policy,
        apply_trade_decision_stage=_apply_trade_decision_stage,
        apply_post_trade_gates=_apply_post_trade_gates,
        apply_execution_policy=_apply_execution_policy,
        build_stub_summary=_build_stub_summary,
        stderr_write=sys.stderr.write,
        regime_neutral=REGIME_NEUTRAL,
        target_range_default_confidence_scale=TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE,
    )
    return runtime_execute_prediction_pipeline(config, deps)


def write_summary(
    summary: Dict[str, Dict[str, Any]],
    *,
    degradation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return runtime_write_prediction_summary(
        summary,
        degradation_policy=degradation_policy,
        latest_prediction_path=LATEST_PREDICTION_PATH,
        history_prediction_path=HISTORY_PREDICTION_PATH,
        build_prompt_ready_summary_fn=_build_prompt_ready_summary,
        build_blocked_trade_analytics_fn=_build_blocked_trade_analytics,
        build_degradation_monitoring_fn=lambda history, policy: _build_degradation_monitoring(history, policy=policy),
        print_fn=print,
    )


def _build_blocked_trade_analytics(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return runtime_build_blocked_trade_analytics(summary)


def _build_degradation_monitoring(
    history: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return runtime_build_degradation_monitoring(
        history,
        policy=policy,
        resolve_degradation_monitoring_policy=_resolve_degradation_monitoring_policy,
        horizon_sort_key=_horizon_sort_key,
        finite_float_or_none=_finite_float_or_none,
    )


def _build_trade_ready_monitoring_payload(predictions_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return runtime_build_trade_ready_monitoring_payload(
        predictions_payload,
        args,
        horizon_sort_key=_horizon_sort_key,
        format_horizon_label=_format_horizon_label,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        position_size_floor_default=POSITION_SIZE_FLOOR_DEFAULT,
        position_size_cap_default=POSITION_SIZE_CAP_DEFAULT,
    )


def _write_monitoring_payload_file(payload: dict[str, Any], path: Path) -> None:
    runtime_write_monitoring_payload_file(payload, path)


def _write_monitoring_latest(
    predictions_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return runtime_write_monitoring_artifact(
        predictions_payload,
        args,
        output_path=MONITORING_LATEST_PATH,
        horizon_sort_key=_horizon_sort_key,
        format_horizon_label=_format_horizon_label,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        position_size_floor_default=POSITION_SIZE_FLOOR_DEFAULT,
        position_size_cap_default=POSITION_SIZE_CAP_DEFAULT,
        payload=payload,
    )


def _write_trade_ready_monitoring(
    predictions_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return runtime_write_monitoring_artifact(
        predictions_payload,
        args,
        output_path=TRADE_READY_MONITOR_PATH,
        horizon_sort_key=_horizon_sort_key,
        format_horizon_label=_format_horizon_label,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        position_size_floor_default=POSITION_SIZE_FLOOR_DEFAULT,
        position_size_cap_default=POSITION_SIZE_CAP_DEFAULT,
        payload=payload,
    )


def _refresh_meta_baseline() -> None:
    runtime_refresh_meta_baseline(
        source_csv=META_BASELINE_SOURCE_CSV,
        json_path=META_BASELINE_JSON_PATH,
        parquet_path=META_BASELINE_PARQUET_PATH,
        load_dataframe=load_dataframe,
        compute_baseline=compute_baseline,
        baseline_to_dataframe=baseline_to_dataframe,
        append_detected_meta_columns=_append_detected_meta_columns,
        default_columns=list(BASELINE_DEFAULT_COLUMNS),
        stderr_write=sys.stderr.write,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return runtime_parse_refresh_args(
        argv,
        load_cli_config=_load_cli_config,
        parse_targets=parse_targets,
        default_hours=DEFAULT_HOURS,
        default_targets=DEFAULT_TARGETS,
        default_p_up_min=DEFAULT_P_UP_MIN,
        default_ret_min=DEFAULT_RET_MIN,
        confidence_min_default=CONFIDENCE_MIN_DEFAULT,
        position_size_floor_default=POSITION_SIZE_FLOOR_DEFAULT,
        position_size_cap_default=POSITION_SIZE_CAP_DEFAULT,
        default_dir_model_weights_1h=DEFAULT_DIR_MODEL_WEIGHTS_1H,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if getattr(args, "config", None):
        print(f"Loaded CLI defaults from config: {args.config}")
    from src.runtime.models import RuntimeMode
    from src.runtime.refresh_pipeline import execute_refresh_pipeline

    execute_refresh_pipeline(args, mode=RuntimeMode.RESEARCH)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
