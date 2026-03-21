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
    "onchain": ("onchain_large_transfer_count", "onchain_whale_transfer_count"),
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
    "degradation_monitoring",
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
            print(
                f"Warning: Unknown trend_ignition config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_direction_fallback_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown direction_only_fallback config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_adaptive_thresholds_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(f"Warning: Unknown adaptive_thresholds config key '{raw_key}' ignored.", file=sys.stderr)
    return normalized


def _normalize_target_range_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
                normalized[key] = [float(entry) for entry in raw_value]
            else:
                raise ValueError("horizons in target_range_models must be list/sequence")
        else:
            print(f"Warning: Unknown target_range_models config key '{raw_key}' ignored.", file=sys.stderr)
    return normalized


def _normalize_data_quality_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown data_quality config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_abstention_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
        else:
            print(
                f"Warning: Unknown abstention_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_regime_model_weights_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("regime_model_weights config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
            continue
        if key in {REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP}:
            if isinstance(raw_value, Mapping):
                normalized[key] = {str(inner_key): str(inner_value) for inner_key, inner_value in raw_value.items()}
            else:
                normalized[key] = str(raw_value) if raw_value is not None else None
            continue
        print(
            f"Warning: Unknown regime_model_weights config key '{raw_key}' ignored.",
            file=sys.stderr,
        )
    return normalized


def _normalize_uncertainty_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown uncertainty_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_trade_decision_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
        elif key == "oof_expected_value_mode":
            normalized[key] = str(raw_value).lower() if raw_value is not None else None
        elif key == "positive_oof_envelope_mode":
            normalized[key] = str(raw_value).lower() if raw_value is not None else None
        elif key == "model_path":
            normalized[key] = str(raw_value) if raw_value is not None else None
        elif key == "midband_veto":
            if not isinstance(raw_value, Mapping):
                raise ValueError("trade_decision_policy.midband_veto must be a mapping.")
            normalized[key] = {
                "enabled": bool(raw_value.get("enabled", False)),
                "p_up_low": float(raw_value.get("p_up_low", 0.55)),
                "p_up_high": float(raw_value.get("p_up_high", 0.60)),
                "high_inclusive": bool(raw_value.get("high_inclusive", False)),
                "min_abs_ret_pred": (
                    float(raw_value.get("min_abs_ret_pred"))
                    if raw_value.get("min_abs_ret_pred") is not None
                    else None
                ),
                "max_abs_ret_pred": (
                    float(raw_value.get("max_abs_ret_pred"))
                    if raw_value.get("max_abs_ret_pred") is not None
                    else None
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
            print(
                f"Warning: Unknown trade_decision_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_regime_model_dirs_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("regime_model_dirs config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key == "enabled":
            normalized[key] = bool(raw_value)
            continue
        if key in {REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP}:
            if isinstance(raw_value, Mapping):
                normalized[key] = {str(k): str(v) for k, v in raw_value.items() if v is not None}
            else:
                print(
                    f"Warning: regime_model_dirs.{raw_key} must be a mapping of horizon->path; ignored.",
                    file=sys.stderr,
                )
            continue
        print(
            f"Warning: Unknown regime_model_dirs config key '{raw_key}' ignored.",
            file=sys.stderr,
        )
    return normalized


def _normalize_intrabar_aggregation_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown intrabar_aggregation config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_feature_coverage_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown feature_coverage_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_confluence_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
                normalized[key] = [_normalize_horizon_value(entry) for entry in raw_value]
            else:
                raise ValueError(f"{key} in confluence_policy must be a list/sequence")
        else:
            print(
                f"Warning: Unknown confluence_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_execution_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("execution_policy config must be a mapping.")

    numeric_keys = {
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
            normalized[key] = [_normalize_horizon_value(item) for item in raw_value]
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
        }:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{key} in execution_policy must be a mapping")
            normalized[key] = dict(raw_value)
        else:
            print(f"Warning: Unknown execution_policy config key '{raw_key}' ignored.", file=sys.stderr)
    return normalized


def _normalize_degradation_monitoring_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
            print(
                f"Warning: Unknown degradation_monitoring config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_forecast_coherence_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
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
        elif key in {
            "p_up_neutral_band",
            "min_p_up_edge",
            "min_abs_ret_pred",
            "consensus_relief_max_p_up_edge",
        }:
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key in {"horizons", "consensus_relief_horizons"}:
            if raw_value is None:
                normalized[key] = None
            elif isinstance(raw_value, str):
                normalized[key] = parse_targets(raw_value)
            elif isinstance(raw_value, Sequence):
                normalized[key] = [_normalize_horizon_value(item) for item in raw_value]
            else:
                raise ValueError("horizons in forecast_coherence_policy must be a list/sequence")
        else:
            print(
                f"Warning: Unknown forecast_coherence_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_direction_output_policy_block(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("direction_output_policy config must be a mapping.")

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).replace("-", "_")
        if key in {"enabled", "use_trade_probability_fallback"}:
            normalized[key] = bool(raw_value)
        elif key == "neutral_band":
            normalized[key] = float(raw_value) if raw_value is not None else None
        elif key == "calibration_path":
            normalized[key] = str(raw_value) if raw_value is not None else None
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
                            marginal_rerank[marginal_key] = [_normalize_horizon_value(item) for item in raw_marginal_value]
                        else:
                            raise ValueError("horizons in direction_output_policy.marginal_rerank must be a list/sequence")
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
                            raise ValueError("weight_specs in direction_output_policy.marginal_rerank must be a mapping")
                    else:
                        print(
                            f"Warning: Unknown direction_output_policy.marginal_rerank key '{raw_marginal_key}' ignored.",
                            file=sys.stderr,
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
                normalized[key] = [_normalize_horizon_value(item) for item in raw_value]
            else:
                raise ValueError("horizons in direction_output_policy must be a list/sequence")
        else:
            print(
                f"Warning: Unknown direction_output_policy config key '{raw_key}' ignored.",
                file=sys.stderr,
            )
    return normalized


def _normalize_config_value(name: str, value: Any) -> Any:
    if name == "targets":
        if value is None:
            return list(DEFAULT_TARGETS)
        if isinstance(value, str):
            return parse_targets(value)
        if isinstance(value, Sequence):
            normalized: List[float] = []
            for entry in value:
                normalized.append(_normalize_horizon_value(entry))
            if not normalized:
                raise ValueError("Targets list from config cannot be empty.")
            return normalized
        raise ValueError(f"Invalid targets entry in config: {value!r}")
    if name in CONFIG_INT_FIELDS:
        if value is None:
            return None
        return int(value)
    if name in CONFIG_FLOAT_FIELDS:
        if value is None:
            return None
        return float(value)
    if name in CONFIG_BOOL_FIELDS:
        if isinstance(value, str):
            return _bool_env(value)
        return bool(value)
    if name in CONFIG_PATH_FIELDS:
        if value is None:
            return None
        return str(value)
    if name == "dir_model_weights":
        if value is None:
            return None
        return str(value)
    if name == "spot_provider":
        if value is None:
            return None
        return str(value)
    if name in CONFIG_ALLOWED_KEYS:
        if name == "trend_ignition" and value is not None:
            return _normalize_trend_ignition_block(value)
        if name == "direction_only_fallback" and value is not None:
            return _normalize_direction_fallback_block(value)
        if name == "adaptive_thresholds" and value is not None:
            return _normalize_adaptive_thresholds_block(value)
        if name == "target_range_models" and value is not None:
            return _normalize_target_range_block(value)
        if name == "data_quality" and value is not None:
            return _normalize_data_quality_block(value)
        if name == "abstention_policy" and value is not None:
            return _normalize_abstention_policy_block(value)
        if name == "uncertainty_policy" and value is not None:
            return _normalize_uncertainty_policy_block(value)
        if name == "trade_decision_policy" and value is not None:
            return _normalize_trade_decision_policy_block(value)
        if name == "regime_model_weights" and value is not None:
            return _normalize_regime_model_weights_block(value)
        if name == "regime_model_dirs" and value is not None:
            return _normalize_regime_model_dirs_block(value)
        if name == "intrabar_aggregation" and value is not None:
            return _normalize_intrabar_aggregation_block(value)
        if name == "feature_coverage_policy" and value is not None:
            return _normalize_feature_coverage_policy_block(value)
        if name == "confluence_policy" and value is not None:
            return _normalize_confluence_policy_block(value)
        if name == "execution_policy" and value is not None:
            return _normalize_execution_policy_block(value)
        if name == "forecast_coherence_policy" and value is not None:
            return _normalize_forecast_coherence_policy_block(value)
        if name == "direction_output_policy" and value is not None:
            return _normalize_direction_output_policy_block(value)
        if name == "degradation_monitoring" and value is not None:
            return _normalize_degradation_monitoring_block(value)
        if name == "position_size_cap_by_horizon" and value is not None:
            return _normalize_horizon_float_map(value, minimum=0.0, maximum=1.0)
        return value
    raise ValueError(f"Unsupported config key: {name}")


def _load_cli_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    try:
        raw_data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse config file {resolved}: {exc}") from exc
    if raw_data is None:
        return {"config": str(resolved)}
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Config file must contain a mapping/dict (got {type(raw_data).__name__}).")
    normalized: Dict[str, Any] = {}
    for raw_key, value in raw_data.items():
        if not isinstance(raw_key, str):
            print(f"Ignoring non-string config key: {raw_key}", file=sys.stderr)
            continue
        key = raw_key.replace("-", "_")
        if key not in CONFIG_ALLOWED_KEYS:
            print(f"Warning: Unknown config key '{raw_key}' ignored.", file=sys.stderr)
            continue
        normalized[key] = _normalize_config_value(key, value)
    normalized["config"] = str(resolved)
    return normalized


def _dataset_profile_for_horizon(horizon: float) -> DatasetProfile:
    hourly_candidates = (
        DatasetCandidate(DATASET_MULTI_PATH, "ret_1h", 1.0, offline_only=False),
        DatasetCandidate(DATASET_1H_PATH, "ret_1h", 1.0, offline_only=False),
    )
    if horizon < 1.0:
        sub_candidates = (
            DatasetCandidate(DATASET_15M_PATH, "ret_15m", 0.25, offline_only=True),
            *hourly_candidates,
        )
        return DatasetProfile(key="15m", candidates=sub_candidates)
    return DatasetProfile(key="hourly", candidates=hourly_candidates)


def _select_dataset_candidate(profile: DatasetProfile) -> tuple[DatasetCandidate, bool]:
    if not profile.candidates:
        raise RuntimeError(f"Dataset profile {profile.key} does not define any candidates.")
    for idx, candidate in enumerate(profile.candidates):
        if candidate.path.exists():
            return candidate, idx > 0
    return profile.candidates[-1], True


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
    if not isinstance(raw, Mapping):
        return {}
    resolved: Dict[float, float] = {}
    for key, value in raw.items():
        horizon = _coerce_numeric_horizon(key)
        if horizon is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        numeric_value = max(numeric_value, minimum)
        if maximum is not None:
            numeric_value = min(numeric_value, maximum)
        resolved[horizon] = numeric_value
    return resolved


def _resolve_thresholds_for_horizon(
    horizon: float,
    default_p_up: float,
    default_ret: float,
    overrides: Mapping[int | float | str, Dict[str, float]] | None,
) -> Dict[str, float]:
    entry: Dict[str, float] | None = None
    if overrides:
        for key in _threshold_lookup_keys(horizon):
            entry = overrides.get(key)  # type: ignore[arg-type]
            if entry is not None:
                break

    p_up_value = float((entry or {}).get("p_up_min", default_p_up))
    ret_value = float((entry or {}).get("ret_min", default_ret))
    resolved: Dict[str, float] = {
        "p_up_min": p_up_value,
        "ret_min": ret_value,
    }
    if entry:
        optional_float_keys = ("max_drawdown", "volatility_ceiling", "volatility_mult", "expected_value_multiplier")
        for key in optional_float_keys:
            if key in entry:
                try:
                    resolved[key] = float(entry[key])
                except (TypeError, ValueError):
                    continue
        metric_key = entry.get("volatility_metric")
        if isinstance(metric_key, str) and metric_key.strip():
            resolved["volatility_metric"] = metric_key.strip()
    return resolved


def _warn_missing_thresholds(
    targets: Iterable[float],
    thresholds: Mapping[int | float | str, Dict[str, float]] | None,
    source_path: str | None,
) -> None:
    if not thresholds:
        return
    requested = {_normalize_horizon_value(h) for h in targets}
    available: set[float] = set()
    for key in thresholds.keys():
        numeric = _coerce_numeric_horizon(key)
        if numeric is not None:
            available.add(numeric)
    missing = sorted(requested - available)
    if missing:
        label = ", ".join(_format_horizon_label(h) for h in missing)
        source = source_path or "provided thresholds JSON"
        print(
            f"Warning: {source} is missing calibrated entries for horizons {label}; "
            "falling back to CLI defaults.",
            file=sys.stderr,
        )


def _build_stub_summary(
    targets: Iterable[float],
    p_up_min: float,
    ret_min: float,
    close: float = 0.0,
    ts_iso: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    generated_ts = ts_iso or datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Dict[str, float | str | int]] = {}
    normalized_targets = sorted({_normalize_horizon_value(h) for h in targets})
    for horizon in normalized_targets:
        label = _format_horizon_label(horizon)
        horizon_thresholds = _resolve_thresholds_for_horizon(
            horizon,
            p_up_min,
            ret_min,
            thresholds_by_horizon,
        )
        horizon_p_up = horizon_thresholds["p_up_min"]
        horizon_ret = horizon_thresholds["ret_min"]
        summary[label] = {
            "timestamp": generated_ts,
            "horizon_hours": horizon,
            "close": close,
            "p_up": 0.5,
            "p_trend_ignition": 0.0,
            "ignition_state": 0,
            "ignition_cooldown_active": False,
            "ret_pred": 0.0,
            "projected_price": close,
            "signal_ensemble": 0,
            "signal_dir_only": 0,
            "confidence_score": 0.0,
            "position_size": 0.0,
            "confidence_min": CONFIDENCE_MIN_DEFAULT,
            "confidence_filter_triggered": False,
            "p_up_components": {},
            "stop_loss": close,
            "take_profit": close,
            "expected_value": 0.0,
            "thresholds": horizon_thresholds,
            "regime_state": REGIME_NEUTRAL,
            "regime_score": 0.0,
            "projected_high": close,
            "projected_low": close,
            "projected_high_confidence": 0.0,
            "projected_low_confidence": 0.0,
            "volatility": {
                "snapshot": {},
                "ceiling": horizon_thresholds.get("volatility_ceiling"),
                "triggered": False,
            },
            "volatility_flag": False,
            "target_range_overrides": {
                "stop_loss": None,
                "take_profit": None,
            },
            "execution_plan": {
                "enabled": False,
                "status": "dry_run",
                "reason": "dry_run",
            },
            "direction_only_fallback": {
                "active": False,
                "side": None,
                "size_factor": 0.0,
                "stop_loss_fallback": None,
                "take_profit_fallback": None,
                "reason": "dry_run",
                "cooldown_active": False,
            },
        }
        summary[label]["thresholds"]["p_up_min_effective"] = horizon_p_up
        summary[label]["thresholds"]["ret_min_effective"] = horizon_ret
        summary[label]["thresholds"]["adaptive_scale"] = 1.0
    return summary


def _parse_iso_timestamp(value: str) -> datetime:
    sanitized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(sanitized)


def _load_trend_ignition_state() -> Optional[str]:
    if not TREND_IGNITION_STATE_PATH.exists():
        return None
    try:
        payload = json.loads(TREND_IGNITION_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    ts_value = payload.get("last_trigger_ts")
    if isinstance(ts_value, str) and ts_value.strip():
        return ts_value
    return None


def _write_trend_ignition_state(ts_value: str) -> None:
    TREND_IGNITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TREND_IGNITION_STATE_PATH.write_text(json.dumps({"last_trigger_ts": ts_value}, indent=2))


def _resolve_trend_ignition_payload(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    enabled = config.get("enabled")
    if enabled is False:
        return None
    model_path = config.get("model_path")
    if not model_path:
        return None

    try:
        payload = load_trend_ignition_classifier(str(model_path))
    except FileNotFoundError as exc:
        print(f"Warning: {exc}; trend ignition support disabled.", file=sys.stderr)
        return None

    threshold = config.get("probability_threshold")
    cooldown = config.get("cooldown_hours")
    payload["threshold"] = float(threshold) if threshold is not None else 0.6
    payload["cooldown_hours"] = max(float(cooldown) if cooldown is not None else 0.0, 0.0)
    payload["last_trigger_ts"] = _load_trend_ignition_state()
    return payload


def _load_direction_fallback_state() -> Optional[str]:
    if not DIRECTION_FALLBACK_STATE_PATH.exists():
        return None
    try:
        payload = json.loads(DIRECTION_FALLBACK_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    ts_value = payload.get("last_trigger_ts")
    if isinstance(ts_value, str) and ts_value.strip():
        return ts_value
    return None


def _write_direction_fallback_state(ts_value: str) -> None:
    DIRECTION_FALLBACK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIRECTION_FALLBACK_STATE_PATH.write_text(json.dumps({"last_trigger_ts": ts_value}, indent=2))


def _inactive_direction_fallback(
    reason: str,
    *,
    side: Optional[str] = None,
    cooldown_active: bool = False,
    size_factor: float = 0.0,
) -> Dict[str, Any]:
    return {
        "active": False,
        "side": side,
        "size_factor": size_factor,
        "stop_loss_fallback": None,
        "take_profit_fallback": None,
        "reason": reason,
        "cooldown_active": cooldown_active,
    }


def _resolve_direction_fallback_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    enabled = config.get("enabled")
    if enabled is False:
        policy: Dict[str, Any] = {
            "enabled": False,
            "prob_threshold": float(config.get("prob_threshold") or 0.5),
            "max_negative_ev": float(config.get("max_negative_ev") or 0.0),
            "size_factor": float(config.get("size_factor") or 0.0),
            "stop_take_ratio": float(config.get("stop_take_ratio") or 0.0),
            "cooldown_hours": float(config.get("cooldown_hours") or 0.0),
            "ignition_ev_extension": float(config.get("ignition_ev_extension") or 0.0),
            "last_trigger_ts": _load_direction_fallback_state(),
        }
        return policy

    policy = {
        "enabled": True,
        "prob_threshold": float(config.get("prob_threshold") or 0.6),
        "max_negative_ev": max(float(config.get("max_negative_ev") or 0.0), 0.0),
        "size_factor": max(float(config.get("size_factor") or 1.0), 0.0),
        "stop_take_ratio": max(float(config.get("stop_take_ratio") or 0.0), 0.0),
        "cooldown_hours": max(float(config.get("cooldown_hours") or 0.0), 0.0),
        "ignition_ev_extension": max(float(config.get("ignition_ev_extension") or 0.0), 0.0),
        "last_trigger_ts": _load_direction_fallback_state(),
    }
    return policy


def _resolve_adaptive_thresholds_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None

    policy: Dict[str, Any] = {}
    policy["enabled"] = bool(config.get("enabled", False))
    policy["breakout_score_threshold"] = float(config.get("breakout_score_threshold") or 0.8)
    policy["chop_score_threshold"] = float(config.get("chop_score_threshold") or 0.3)
    policy["breakout_scale"] = max(float(config.get("breakout_scale") or 0.9), 0.0)
    policy["chop_scale"] = max(float(config.get("chop_scale") or 1.1), 0.0)
    for clamp_key in ("p_up_min_floor", "p_up_min_ceiling", "ret_min_floor", "ret_min_ceiling"):
        clamp_value = config.get(clamp_key)
        policy[clamp_key] = float(clamp_value) if clamp_value is not None else None
    return policy


def _compute_profile_breakout_score(
    prepared: PreparedData,
    index: int,
    volatility_snapshot: Mapping[str, Any] | None,
) -> float:
    snapshot = volatility_snapshot or {}
    vol_component = 0.0
    for value in snapshot.values():
        try:
            vol_component = max(vol_component, float(value))
        except (TypeError, ValueError):
            continue

    ret_component = 0.0
    if index > 0 and "close" in prepared.df_all.columns:
        try:
            current_close = float(prepared.df_all["close"].iloc[index])
            prev_close = float(prepared.df_all["close"].iloc[index - 1])
            if current_close > 0 and prev_close > 0:
                ret_component = abs(math.log(current_close / prev_close))
        except (ValueError, ZeroDivisionError, IndexError):
            ret_component = 0.0

    norm_vol = min(vol_component / BREAKOUT_VOL_NORMALIZER, 2.0) if BREAKOUT_VOL_NORMALIZER else 0.0
    norm_ret = min(ret_component / BREAKOUT_RET_NORMALIZER, 2.0) if BREAKOUT_RET_NORMALIZER else 0.0
    score = (norm_vol + norm_ret) / 2.0
    return round(score, 6)


def _derive_regime_labels_from_frame(
    frame: pd.DataFrame,
    *,
    volatility_col: str,
    breakout_score_threshold: float,
    chop_score_threshold: float,
) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame.columns else pd.Series(np.nan, index=frame.index)
    volatility = pd.to_numeric(frame.get(volatility_col), errors="coerce") if volatility_col in frame.columns else pd.Series(0.0, index=frame.index)
    ret_component = pd.Series(0.0, index=frame.index, dtype=float)
    valid_close = close > 0.0
    if valid_close.any():
        ret_component = np.log(close.where(valid_close)).diff().abs().fillna(0.0)
    vol_component = volatility.fillna(0.0).abs()
    norm_vol = (vol_component / BREAKOUT_VOL_NORMALIZER).clip(lower=0.0, upper=2.0) if BREAKOUT_VOL_NORMALIZER else vol_component * 0.0
    norm_ret = (ret_component / BREAKOUT_RET_NORMALIZER).clip(lower=0.0, upper=2.0) if BREAKOUT_RET_NORMALIZER else ret_component * 0.0
    score = ((norm_vol + norm_ret) / 2.0).fillna(0.0)

    labels = pd.Series(REGIME_NEUTRAL, index=frame.index, dtype=object)
    labels.loc[score >= breakout_score_threshold] = REGIME_TREND
    labels.loc[score <= chop_score_threshold] = REGIME_CHOP
    return labels


def _compute_breakout_scores(
    prepared_bundles: Mapping[str, tuple[PreparedData, int, float, str]],
    volatility_snapshots: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for key, bundle in prepared_bundles.items():
        prepared, index, _close, _ts = bundle
        snapshot = volatility_snapshots.get(key, {})
        scores[key] = _compute_profile_breakout_score(prepared, index, snapshot)
    return scores


def _classify_regime_from_score(score: float, policy: Mapping[str, Any]) -> str:
    breakout_threshold = float(policy.get("breakout_score_threshold", 1.0))
    chop_threshold = float(policy.get("chop_score_threshold", 0.0))
    if score >= breakout_threshold:
        return REGIME_TREND
    if score <= chop_threshold:
        return REGIME_CHOP
    return REGIME_NEUTRAL


def _apply_adaptive_thresholds(
    policy: Mapping[str, Any],
    base_p_up: float,
    base_ret: float,
    regime_state: str,
) -> tuple[float, float, float]:
    if not policy.get("enabled"):
        return base_p_up, base_ret, 1.0

    if regime_state == REGIME_TREND:
        scale = float(policy.get("breakout_scale", 1.0))
    elif regime_state == REGIME_CHOP:
        scale = float(policy.get("chop_scale", 1.0))
    else:
        scale = 1.0

    scaled_p = base_p_up * scale
    scaled_ret = base_ret * scale

    floor = policy.get("p_up_min_floor")
    if floor is not None:
        scaled_p = max(scaled_p, float(floor))
    ceiling = policy.get("p_up_min_ceiling")
    if ceiling is not None:
        scaled_p = min(scaled_p, float(ceiling))

    ret_floor = policy.get("ret_min_floor")
    if ret_floor is not None:
        scaled_ret = max(scaled_ret, float(ret_floor))
    ret_ceiling = policy.get("ret_min_ceiling")
    if ret_ceiling is not None:
        scaled_ret = min(scaled_ret, float(ret_ceiling))

    return scaled_p, scaled_ret, scale


def _resolve_target_range_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None

    policy = {
        "enabled": bool(config.get("enabled", False)),
        "model_dir": Path(config.get("model_dir") or TARGET_RANGE_MODEL_DIR).expanduser(),
        "override_ratio": max(float(config.get("override_ratio") or TARGET_RANGE_DEFAULT_OVERRIDE_RATIO), 0.0),
        "confidence_rmse_scale": max(
            float(config.get("confidence_rmse_scale") or TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE),
            1e-6,
        ),
    }

    horizons = config.get("horizons")
    if horizons is None:
        policy["horizons"] = list(TARGET_RANGE_DEFAULT_HORIZONS)
    else:
        policy["horizons"] = sorted({float(h) for h in horizons if float(h) > 0})
    return policy


def _resolve_feature_coverage_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "max_imputed_zero_columns": max(float(cfg.get("max_imputed_zero_columns") or 1e9), 0.0),
        "max_imputed_zero_ratio": max(float(cfg.get("max_imputed_zero_ratio") or 1.0), 0.0),
        "max_source_lag_hours": max(float(cfg.get("max_source_lag_hours") or 1e9), 0.0),
        "block_on_violation": bool(cfg.get("block_on_violation", True)),
        "ignored_columns": sorted({str(column).strip() for column in (cfg.get("ignored_columns") or []) if str(column).strip()}),
    }


def _resolve_confluence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    short_horizons = cfg.get("short_horizons") or [0.25, 1.0]
    mid_horizons = cfg.get("mid_horizons") or [4.0, 8.0, 12.0]

    def _normalize_horizon_map(raw: Any, *, minimum: float = 0.0, maximum: float | None = None) -> Dict[float, float]:
        if not isinstance(raw, Mapping):
            return {}
        resolved: Dict[float, float] = {}
        for key, value in raw.items():
            horizon = _coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            numeric_value = max(numeric_value, minimum)
            if maximum is not None:
                numeric_value = min(numeric_value, maximum)
            resolved[horizon] = numeric_value
        return resolved

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "short_horizons": sorted({_normalize_horizon_value(v) for v in short_horizons}),
        "mid_horizons": sorted({_normalize_horizon_value(v) for v in mid_horizons}),
        "min_support_ratio": max(min(float(cfg.get("min_support_ratio") or 0.6), 1.0), 0.0),
        "min_support_ratio_by_horizon": _normalize_horizon_map(
            cfg.get("min_support_ratio_by_horizon"),
            minimum=0.0,
            maximum=1.0,
        ),
        "min_mid_term_ratio": max(min(float(cfg.get("min_mid_term_ratio") or 0.5), 1.0), 0.0),
        "min_short_term_ratio": max(min(float(cfg.get("min_short_term_ratio") or 0.5), 1.0), 0.0),
        "dominant_ratio_floor": max(min(float(cfg.get("dominant_ratio_floor") or 0.55), 1.0), 0.0),
        "min_aligned_horizons": max(int(cfg.get("min_aligned_horizons") or 2), 1),
        "min_aligned_horizons_by_horizon": _normalize_horizon_map(
            cfg.get("min_aligned_horizons_by_horizon"),
            minimum=1.0,
        ),
        "require_mid_term_alignment": bool(cfg.get("require_mid_term_alignment", True)),
        "require_short_term_alignment": bool(cfg.get("require_short_term_alignment", False)),
    }


def _resolve_execution_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}

    def _normalize_float_map(raw: Any, *, minimum: float = 0.0) -> Dict[float, float]:
        if not isinstance(raw, Mapping):
            return {}
        resolved: Dict[float, float] = {}
        for key, value in raw.items():
            horizon = _coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                resolved[horizon] = max(float(value), minimum)
            except (TypeError, ValueError):
                continue
        return resolved

    partial_cfg = cfg.get("partial_take_profit") if isinstance(cfg.get("partial_take_profit"), Mapping) else {}
    trailing_cfg = cfg.get("trailing_stop") if isinstance(cfg.get("trailing_stop"), Mapping) else {}
    analytics_cfg = cfg.get("analytics") if isinstance(cfg.get("analytics"), Mapping) else {}
    analytics_bucket_cfg = (
        analytics_cfg.get("regime_volatility_buckets")
        if isinstance(analytics_cfg.get("regime_volatility_buckets"), Mapping)
        else {}
    )
    guards_cfg = cfg.get("no_trade_guards") if isinstance(cfg.get("no_trade_guards"), Mapping) else {}
    adaptive_tp_cfg = cfg.get("adaptive_take_profit") if isinstance(cfg.get("adaptive_take_profit"), Mapping) else {}
    target_range_stop_cfg = (
        cfg.get("target_range_stop_refinement")
        if isinstance(cfg.get("target_range_stop_refinement"), Mapping)
        else {}
    )
    raw_regime_templates = cfg.get("regime_templates") if isinstance(cfg.get("regime_templates"), Mapping) else {}
    regime_templates: Dict[str, Dict[str, Any]] = {}
    for regime_name, raw_template in raw_regime_templates.items():
        if not isinstance(raw_template, Mapping):
            continue
        entry_mode_by_tier = raw_template.get("entry_mode_by_tier") if isinstance(raw_template.get("entry_mode_by_tier"), Mapping) else {}
        regime_templates[str(regime_name)] = {
            "tp_multiplier": max(float(raw_template.get("tp_multiplier", 1.0) or 1.0), 0.1),
            "time_stop_multiplier": max(float(raw_template.get("time_stop_multiplier", 1.0) or 1.0), 0.1),
            "size_multiplier": max(float(raw_template.get("size_multiplier", 1.0) or 1.0), 0.0),
            "entry_zone_atr_mult": max(float(raw_template.get("entry_zone_atr_mult", 0.0) or 0.0), 0.0),
            "max_chase_atr_mult": max(float(raw_template.get("max_chase_atr_mult", 0.0) or 0.0), 0.0),
            "pullback_quality_floor": max(float(raw_template.get("pullback_quality_floor", 0.0) or 0.0), 0.0),
            "entry_mode_by_tier": {
                str(tier).strip().lower(): str(mode).strip().lower()
                for tier, mode in entry_mode_by_tier.items()
                if str(tier).strip() and str(mode).strip()
            },
        }

    pullback_quality_cfg = cfg.get("pullback_quality") if isinstance(cfg.get("pullback_quality"), Mapping) else {}
    disagreement_cfg = cfg.get("disagreement_severity") if isinstance(cfg.get("disagreement_severity"), Mapping) else {}
    coherence_weighting_cfg = cfg.get("coherence_weighting") if isinstance(cfg.get("coherence_weighting"), Mapping) else {}

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "bias_horizons": sorted({_normalize_horizon_value(v) for v in (cfg.get("bias_horizons") or [4.0, 8.0, 12.0])}),
        "execution_horizons": sorted({_normalize_horizon_value(v) for v in (cfg.get("execution_horizons") or [0.25, 1.0])}),
        "horizon_bias_weights": _normalize_float_map(cfg.get("horizon_bias_weights"), minimum=0.0),
        "short_term_strict_horizons": sorted(
            {_normalize_horizon_value(v) for v in (cfg.get("short_term_strict_horizons") or [1.0])}
        ),
        "short_term_min_mid_ratio": max(min(float(cfg.get("short_term_min_mid_ratio") or 0.67), 1.0), 0.0),
        "short_term_min_support_ratio": max(min(float(cfg.get("short_term_min_support_ratio") or 0.75), 1.0), 0.0),
        "short_term_min_mid_ratio_by_horizon": _normalize_float_map(
            cfg.get("short_term_min_mid_ratio_by_horizon"),
            minimum=0.0,
        ),
        "min_bias_alignment_ratio": max(min(float(cfg.get("min_bias_alignment_ratio") or 0.0), 1.0), 0.0),
        "short_term_min_support_ratio_by_horizon": _normalize_float_map(
            cfg.get("short_term_min_support_ratio_by_horizon"),
            minimum=0.0,
        ),
        "require_bias_alignment": bool(cfg.get("require_bias_alignment", True)),
        "immediate_entry_min_support_ratio": max(min(float(cfg.get("immediate_entry_min_support_ratio") or 0.8), 1.0), 0.0),
        "pullback_entry_min_support_ratio": max(min(float(cfg.get("pullback_entry_min_support_ratio") or 0.6), 1.0), 0.0),
        "immediate_entry_min_mid_ratio": max(min(float(cfg.get("immediate_entry_min_mid_ratio") or 0.67), 1.0), 0.0),
        "pullback_entry_min_mid_ratio": max(min(float(cfg.get("pullback_entry_min_mid_ratio") or 0.5), 1.0), 0.0),
        "high_execution_alignment_ratio": max(min(float(cfg.get("high_execution_alignment_ratio") or 1.0), 1.0), 0.0),
        "medium_execution_alignment_ratio": max(min(float(cfg.get("medium_execution_alignment_ratio") or 0.5), 1.0), 0.0),
        "entry_zone_atr_mult": max(float(cfg.get("entry_zone_atr_mult") or 0.25), 0.01),
        "max_chase_atr_mult": max(float(cfg.get("max_chase_atr_mult") or 0.35), 0.0),
        "session_lookback_bars": max(int(cfg.get("session_lookback_bars") or 8), 2),
        "swing_lookback_bars": max(int(cfg.get("swing_lookback_bars") or 6), 2),
        "structure_buffer_atr_mult": max(float(cfg.get("structure_buffer_atr_mult") or 0.2), 0.0),
        "minimum_rr_by_horizon": _normalize_float_map(cfg.get("minimum_rr_by_horizon"), minimum=0.0),
        "time_stop_bars_by_horizon": {
            horizon: max(int(round(value)), 1)
            for horizon, value in _normalize_float_map(cfg.get("time_stop_bars_by_horizon"), minimum=1.0).items()
        },
        "partial_take_profit": {
            "enabled": bool(partial_cfg.get("enabled", False)),
            "tp1_r_multiple": max(float(partial_cfg.get("tp1_r_multiple") or 1.0), 0.1),
            "tp1_size_fraction": max(min(float(partial_cfg.get("tp1_size_fraction") or 0.5), 1.0), 0.0),
            "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
        },
        "trailing_stop": {
            "enabled": bool(trailing_cfg.get("enabled", False)),
            "activation_r_multiple": max(float(trailing_cfg.get("activation_r_multiple") or 1.0), 0.1),
            "trail_buffer_atr_mult": max(float(trailing_cfg.get("trail_buffer_atr_mult") or 0.75), 0.0),
        },
        "analytics": {
            "enabled": bool(analytics_cfg.get("enabled", False)),
            "lookback_bars": max(int(analytics_cfg.get("lookback_bars") or EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS), 10),
            "mae_quantile": max(min(float(analytics_cfg.get("mae_quantile") or 0.75), 0.99), 0.5),
            "mfe_quantile": max(min(float(analytics_cfg.get("mfe_quantile") or 0.6), 0.99), 0.5),
            "min_samples": max(int(analytics_cfg.get("min_samples") or EXECUTION_POLICY_DEFAULT_MIN_SAMPLES), 10),
            "regime_volatility_buckets": {
                "enabled": bool(analytics_bucket_cfg.get("enabled", False)),
                "regime_col": str(analytics_bucket_cfg.get("regime_col") or "regime_state"),
                "volatility_col": str(analytics_bucket_cfg.get("volatility_col") or "volatility_realized_24h"),
                "min_bucket_samples": max(int(analytics_bucket_cfg.get("min_bucket_samples") or 12), 1),
                "low_vol_quantile": max(min(float(analytics_bucket_cfg.get("low_vol_quantile") or 0.5), 0.95), 0.05),
                "max_projection_mfe_ratio": max(float(analytics_bucket_cfg.get("max_projection_mfe_ratio") or 1.25), 0.5),
                "breakout_score_threshold": float(analytics_bucket_cfg.get("breakout_score_threshold") or 0.8),
                "chop_score_threshold": float(analytics_bucket_cfg.get("chop_score_threshold") or 0.3),
            },
        },
        "no_trade_guards": {
            "enabled": bool(guards_cfg.get("enabled", False)),
            "min_stop_distance_atr_mult": max(float(guards_cfg.get("min_stop_distance_atr_mult") or 0.35), 0.0),
            "max_stop_distance_atr_mult": max(float(guards_cfg.get("max_stop_distance_atr_mult") or 3.0), 0.0),
            "max_entry_deviation_atr_mult": max(float(guards_cfg.get("max_entry_deviation_atr_mult") or 1.25), 0.0),
            "require_favorable_entry_zone": bool(guards_cfg.get("require_favorable_entry_zone", True)),
        },
        "adaptive_take_profit": {
            "enabled": bool(adaptive_tp_cfg.get("enabled", True)),
            "min_rr_fraction_of_floor": max(
                min(float(adaptive_tp_cfg.get("min_rr_fraction_of_floor") or 0.85), 1.0),
                0.0,
            ),
        },
        "target_range_stop_refinement": {
            "enabled": bool(target_range_stop_cfg.get("enabled", False)),
            "horizons": sorted(
                {
                    _normalize_horizon_value(v)
                    for v in (
                        target_range_stop_cfg.get("horizons")
                        or EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_HORIZONS
                    )
                }
            ),
            "confidence_min": max(
                min(
                    float(
                        target_range_stop_cfg.get("confidence_min")
                        or EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_CONFIDENCE_MIN
                    ),
                    1.0,
                ),
                0.0,
            ),
            "buffer_std_mult": max(
                float(
                    target_range_stop_cfg.get("buffer_std_mult")
                    or EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_BUFFER_STD_MULT
                ),
                0.0,
            ),
            "min_tighten_fraction": max(
                min(
                    float(
                        target_range_stop_cfg.get("min_tighten_fraction")
                        or EXECUTION_POLICY_DEFAULT_TARGET_RANGE_STOP_MIN_TIGHTEN_FRACTION
                    ),
                    1.0,
                ),
                0.0,
            ),
        },
        "pullback_quality": {
            "enabled": bool(pullback_quality_cfg.get("enabled", False)),
            "min_score_by_horizon": _normalize_float_map(pullback_quality_cfg.get("min_score_by_horizon"), minimum=0.0),
            "max_vwap_deviation_atr": max(float(pullback_quality_cfg.get("max_vwap_deviation_atr") or 1.5), 0.1),
            "max_candle_expansion_ratio": max(float(pullback_quality_cfg.get("max_candle_expansion_ratio") or 2.0), 0.1),
            "candle_expansion_window": max(int(pullback_quality_cfg.get("candle_expansion_window") or 8), 2),
            "range_expansion_penalty_threshold": max(float(pullback_quality_cfg.get("range_expansion_penalty_threshold") or 1.25), 0.0),
        },
        "disagreement_severity": {
            "enabled": bool(disagreement_cfg.get("enabled", True)),
            "block_threshold": max(min(float(disagreement_cfg.get("block_threshold") or 0.7), 1.0), 0.0),
            "pullback_threshold": max(min(float(disagreement_cfg.get("pullback_threshold") or 0.45), 1.0), 0.0),
            "vwap_extension_penalty_atr": max(float(disagreement_cfg.get("vwap_extension_penalty_atr") or 0.75), 0.0),
            "range_expansion_penalty_threshold": max(float(disagreement_cfg.get("range_expansion_penalty_threshold") or 1.0), 0.0),
        },
        "coherence_weighting": {
            "enabled": bool(coherence_weighting_cfg.get("enabled", False)),
            "low_trust_penalty": max(min(float(coherence_weighting_cfg.get("low_trust_penalty") or 0.35), 1.0), 0.0),
            "blocked_penalty": max(min(float(coherence_weighting_cfg.get("blocked_penalty") or 1.0), 1.0), 0.0),
            "p_up_conflict_penalty": max(min(float(coherence_weighting_cfg.get("p_up_conflict_penalty") or 0.2), 1.0), 0.0),
            "consensus_bonus": max(float(coherence_weighting_cfg.get("consensus_bonus") or 0.1), 0.0),
            "neutral_band": max(float(coherence_weighting_cfg.get("neutral_band") or 0.02), 0.0),
            "min_multiplier": max(min(float(coherence_weighting_cfg.get("min_multiplier") or 0.1), 1.0), 0.0),
            "by_horizon": _normalize_float_map(coherence_weighting_cfg.get("by_horizon"), minimum=0.0),
        },
        "regime_templates": regime_templates,
    }


def _resolve_forecast_coherence_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    horizons = cfg.get("horizons") or [1.0, 4.0, 8.0, 12.0]
    consensus_relief_horizons = cfg.get("consensus_relief_horizons") or [1.0, 4.0]
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "horizons": sorted({_normalize_horizon_value(v) for v in horizons}),
        "block_on_direction_ret_mismatch": bool(cfg.get("block_on_direction_ret_mismatch", True)),
        "block_on_direction_projected_price_mismatch": bool(cfg.get("block_on_direction_projected_price_mismatch", True)),
        "block_on_p_up_ret_mismatch": bool(cfg.get("block_on_p_up_ret_mismatch", True)),
        "p_up_neutral_band": max(float(cfg.get("p_up_neutral_band") or 0.02), 0.0),
        "min_p_up_edge": max(float(cfg.get("min_p_up_edge") or 0.05), 0.0),
        "min_abs_ret_pred": max(float(cfg.get("min_abs_ret_pred") or 0.0), 0.0),
        "allow_consensus_p_up_ret_relief": bool(cfg.get("allow_consensus_p_up_ret_relief", False)),
        "consensus_relief_horizons": sorted({_normalize_horizon_value(v) for v in consensus_relief_horizons}),
        "consensus_relief_max_p_up_edge": max(float(cfg.get("consensus_relief_max_p_up_edge") or 0.12), 0.0),
        "consensus_relief_exclude_from_voting": bool(cfg.get("consensus_relief_exclude_from_voting", False)),
        "exclude_blocked_horizons_from_voting": bool(cfg.get("exclude_blocked_horizons_from_voting", True)),
    }


def _resolve_direction_output_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
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
        "horizons": sorted({_normalize_horizon_value(v) for v in horizons}),
        "neutral_band": max(float(cfg.get("neutral_band") or 0.0), 0.0),
        "use_trade_probability_fallback": bool(cfg.get("use_trade_probability_fallback", True)),
        "calibration_path": str(cfg.get("calibration_path") or "") or None,
        "calibration_map": calibration_map,
        "marginal_rerank": {
            "enabled": bool(marginal_rerank_cfg.get("enabled", False)) and bool(marginal_weight_specs),
            "horizons": sorted({_normalize_horizon_value(v) for v in marginal_horizons}),
            "lower": lower,
            "upper": upper,
            "min_component_count": max(int(marginal_rerank_cfg.get("min_component_count") or 2), 1),
            "use_raw_probability_gate": bool(marginal_rerank_cfg.get("use_raw_probability_gate", True)),
            "weight_specs": marginal_weight_specs,
        },
    }


def _evaluate_feature_coverage(metadata: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    feature_alignment = metadata.get("feature_alignment", {}) if isinstance(metadata, Mapping) else {}
    source_freshness = metadata.get("source_freshness", {}) if isinstance(metadata, Mapping) else {}
    imputed_zero_columns = feature_alignment.get("imputed_zero_columns", []) if isinstance(feature_alignment, Mapping) else []
    required_columns = int(feature_alignment.get("required_columns", 0) or 0) if isinstance(feature_alignment, Mapping) else 0
    ignored_columns = set(policy.get("ignored_columns", [])) if isinstance(policy, Mapping) else set()
    ignored_imputed_zero_columns = []
    effective_imputed_zero_columns = []
    if isinstance(imputed_zero_columns, list):
        ignored_imputed_zero_columns = [column for column in imputed_zero_columns if column in ignored_columns]
        effective_imputed_zero_columns = [column for column in imputed_zero_columns if column not in ignored_columns]
    effective_required_columns = max(required_columns - len(ignored_imputed_zero_columns), 0)
    imputed_zero_count = len(effective_imputed_zero_columns)
    imputed_zero_ratio = (imputed_zero_count / effective_required_columns) if effective_required_columns > 0 else 0.0
    max_lag_hours = 0.0
    stale_sources: List[str] = []
    if isinstance(source_freshness, Mapping):
        for source_name, payload in source_freshness.items():
            if not isinstance(payload, Mapping):
                continue
            lag_hours = float(payload.get("lag_hours") or 0.0)
            max_lag_hours = max(max_lag_hours, lag_hours)
            if lag_hours > float(policy.get("max_source_lag_hours", 1e9)):
                stale_sources.append(str(source_name))

    failed_checks: List[str] = []
    if imputed_zero_count > float(policy.get("max_imputed_zero_columns", 1e9)):
        failed_checks.append("imputed_zero_columns")
    if imputed_zero_ratio > float(policy.get("max_imputed_zero_ratio", 1.0)):
        failed_checks.append("imputed_zero_ratio")
    if stale_sources:
        failed_checks.append("stale_sources")

    return {
        "enabled": bool(policy.get("enabled", False)),
        "ok": not failed_checks,
        "imputed_zero_count": int(imputed_zero_count),
        "imputed_zero_ratio": float(imputed_zero_ratio),
        "effective_required_columns": int(effective_required_columns),
        "ignored_columns": sorted(ignored_columns),
        "ignored_imputed_zero_columns": ignored_imputed_zero_columns,
        "effective_imputed_zero_columns": effective_imputed_zero_columns,
        "max_source_lag_hours": float(max_lag_hours),
        "stale_sources": stale_sources,
        "failed_checks": failed_checks,
        "block_on_violation": bool(policy.get("block_on_violation", True)),
    }


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

    # When both forecast views agree, let that consensus break classifier ties.
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


def _parse_horizon_label(value: str) -> float:
    lowered = str(value).strip().lower()
    if lowered.endswith("h"):
        return float(lowered[:-1])
    if lowered.endswith("m"):
        return float(lowered[:-1]) / 60.0
    return float(lowered)


def _forecast_coherence_excluded(entry: Mapping[str, Any]) -> bool:
    payload = entry.get("forecast_coherence")
    return bool(isinstance(payload, Mapping) and payload.get("exclude_from_voting"))


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
    if not summary:
        return summary

    enabled = bool(policy.get("enabled", False))
    scoped_horizons = set(policy.get("horizons", []))
    neutral_band = float(policy.get("p_up_neutral_band", 0.02) or 0.0)
    min_p_up_edge = float(policy.get("min_p_up_edge", 0.05) or 0.0)
    min_abs_ret_pred = float(policy.get("min_abs_ret_pred", 0.0) or 0.0)
    exclude_from_voting = bool(policy.get("exclude_blocked_horizons_from_voting", True))
    allow_consensus_relief = bool(policy.get("allow_consensus_p_up_ret_relief", False))
    consensus_relief_horizons = set(policy.get("consensus_relief_horizons", []))
    consensus_relief_max_p_up_edge = float(policy.get("consensus_relief_max_p_up_edge", 0.12) or 0.0)
    consensus_relief_exclude_from_voting = bool(policy.get("consensus_relief_exclude_from_voting", False))

    for entry in summary.values():
        horizon = _coerce_result_horizon(entry.get("horizon_hours"))
        direction = _direction_vote(entry)
        ret_side = _direction_from_ret_pred(entry.get("ret_pred"))
        projected_side = _direction_from_projected_price(entry.get("close"), entry.get("projected_price"))
        p_up_side = _direction_from_probability(entry.get("p_up"), neutral_band=neutral_band)
        p_up_value = _finite_float_or_none(entry.get("p_up"))
        ret_pred_value = abs(float(entry.get("ret_pred", 0.0)))
        p_up_edge = abs(p_up_value - 0.5) if p_up_value is not None else None
        consensus_relief_applied = False

        payload = {
            "enabled": enabled,
            "evaluated": bool(enabled and horizon in scoped_horizons),
            "exclude_from_voting": False,
            "direction_side": direction,
            "ret_pred_side": ret_side,
            "projected_price_side": projected_side,
            "p_up_side": p_up_side,
            "triggered": False,
            "reasons": [],
            "advisory_reasons": [],
            "low_trust": False,
            "consensus_relief_applied": False,
        }

        if not enabled or horizon not in scoped_horizons:
            entry["forecast_coherence"] = payload
            continue

        reasons: List[str] = []
        if bool(policy.get("block_on_direction_ret_mismatch", True)) and ret_side != "neutral" and direction != ret_side:
            reasons.append("direction_ret_mismatch")
        if (
            bool(policy.get("block_on_direction_projected_price_mismatch", True))
            and projected_side != "neutral"
            and direction != projected_side
        ):
            reasons.append("direction_projected_price_mismatch")
        if (
            bool(policy.get("block_on_p_up_ret_mismatch", True))
            and p_up_side != "neutral"
            and ret_side != "neutral"
            and p_up_side != ret_side
            and p_up_edge is not None
            and p_up_edge >= min_p_up_edge
            and ret_pred_value >= min_abs_ret_pred
        ):
            consensus_relief_applied = bool(
                allow_consensus_relief
                and horizon in consensus_relief_horizons
                and direction in {"up", "down"}
                and direction == ret_side == projected_side
                and p_up_edge <= consensus_relief_max_p_up_edge
            )
            if not consensus_relief_applied:
                reasons.append("p_up_ret_mismatch")

        advisory_reasons: List[str] = []
        consensus_side = direction if direction == ret_side == projected_side and direction in {"up", "down"} else None
        if consensus_relief_applied:
            advisory_reasons.append("consensus_p_up_ret_mismatch_relief")
        if (
            bool(policy.get("block_on_p_up_ret_mismatch", True))
            and consensus_side is not None
            and p_up_side != "neutral"
            and p_up_side != consensus_side
            and p_up_edge is not None
            and p_up_edge < min_p_up_edge
            and ret_pred_value >= min_abs_ret_pred
            and not reasons
            and not consensus_relief_applied
            and (str(entry.get("trade_action", "hold")) == "hold" or not bool(entry.get("signal_ensemble", 0)))
        ):
            advisory_reasons.append("low_edge_p_up_ret_mismatch")

        payload["reasons"] = reasons
        payload["advisory_reasons"] = advisory_reasons
        payload["triggered"] = bool(reasons)
        payload["low_trust"] = bool(advisory_reasons)
        payload["consensus_relief_applied"] = bool(consensus_relief_applied)
        payload["exclude_from_voting"] = bool(reasons) and exclude_from_voting
        if advisory_reasons and exclude_from_voting:
            payload["exclude_from_voting"] = bool(
                not consensus_relief_applied or consensus_relief_exclude_from_voting
            )
        entry["forecast_coherence"] = payload
        if reasons:
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["direction_next_display"] = "neutral"
            direction_output = entry.get("direction_output")
            if isinstance(direction_output, dict):
                direction_output["coherence_override"] = {
                    "applied": True,
                    "reason": "forecast_coherence_gate",
                    "raw_direction": direction_output.get("direction"),
                }
                direction_output["direction"] = "neutral"
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["pre_forecast_coherence_triggered"] = bool(trade_decision.get("triggered", False))
                trade_decision["triggered"] = False
                trade_decision["blocked"] = True
                trade_decision["blocking_reason"] = "forecast_coherence_gate"
                trade_decision["forecast_coherence_gate_triggered"] = True
                trade_decision["forecast_coherence_gate_reasons"] = reasons
        elif advisory_reasons:
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["forecast_coherence_low_trust"] = True
                trade_decision["forecast_coherence_low_trust_reasons"] = advisory_reasons
    return summary


def _apply_confluence_policy(
    summary: Dict[str, Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if not summary:
        return summary

    labeled_entries: List[tuple[str, Dict[str, Any], float]] = []
    for label, entry in summary.items():
        if _forecast_coherence_excluded(entry):
            continue
        horizon = _coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        labeled_entries.append((label, entry, horizon))

    if not labeled_entries:
        return summary

    short_horizons = set(policy.get("short_horizons", []))
    mid_horizons = set(policy.get("mid_horizons", []))
    up_count = sum(1 for _label, entry, _h in labeled_entries if _direction_vote(entry) == "up")
    down_count = len(labeled_entries) - up_count
    dominant_direction = "neutral"
    dominant_ratio = 0.5
    if up_count > down_count:
        dominant_direction = "up"
        dominant_ratio = up_count / len(labeled_entries)
    elif down_count > up_count:
        dominant_direction = "down"
        dominant_ratio = down_count / len(labeled_entries)

    for label, entry, horizon in labeled_entries:
        current_direction = _direction_vote(entry)
        aligned = [(other_label, other_entry, other_h) for other_label, other_entry, other_h in labeled_entries if _direction_vote(other_entry) == current_direction]
        aligned_count = len(aligned)
        support_ratio = aligned_count / len(labeled_entries)
        min_aligned_horizons = int(
            round(
                _lookup_horizon_value(
                    policy.get("min_aligned_horizons_by_horizon", {}) if isinstance(policy.get("min_aligned_horizons_by_horizon"), Mapping) else {},
                    horizon,
                    float(policy.get("min_aligned_horizons", 2)),
                )
            )
        )
        min_support_ratio = _lookup_horizon_value(
            policy.get("min_support_ratio_by_horizon", {}) if isinstance(policy.get("min_support_ratio_by_horizon"), Mapping) else {},
            horizon,
            float(policy.get("min_support_ratio", 0.6)),
        )

        short_entries = [item for item in labeled_entries if item[2] in short_horizons]
        mid_entries = [item for item in labeled_entries if item[2] in mid_horizons]
        short_ratio = (
            sum(1 for _other_label, other_entry, _other_h in short_entries if _direction_vote(other_entry) == current_direction) / len(short_entries)
            if short_entries else None
        )
        mid_ratio = (
            sum(1 for _other_label, other_entry, _other_h in mid_entries if _direction_vote(other_entry) == current_direction) / len(mid_entries)
            if mid_entries else None
        )

        confluence_triggered = False
        reasons: List[str] = []
        if str(entry.get("trade_action", "hold")) != "hold":
            if aligned_count < min_aligned_horizons:
                confluence_triggered = True
                reasons.append("aligned_horizons_below_min")
            if support_ratio < min_support_ratio:
                confluence_triggered = True
                reasons.append("support_ratio_below_min")
            if (
                bool(policy.get("require_mid_term_alignment", True))
                and mid_ratio is not None
                and mid_ratio < float(policy.get("min_mid_term_ratio", 0.5))
            ):
                confluence_triggered = True
                reasons.append("mid_term_ratio_below_min")
            if (
                bool(policy.get("require_short_term_alignment", False))
                and short_ratio is not None
                and short_ratio < float(policy.get("min_short_term_ratio", 0.5))
            ):
                confluence_triggered = True
                reasons.append("short_term_ratio_below_min")
            if dominant_direction != "neutral" and dominant_direction != current_direction and dominant_ratio >= float(policy.get("dominant_ratio_floor", 0.55)):
                confluence_triggered = True
                reasons.append("dominant_direction_conflict")

        entry["confluence"] = {
            "enabled": bool(policy.get("enabled", False)),
            "dominant_direction": dominant_direction,
            "dominant_ratio": float(dominant_ratio),
            "aligned_horizons": int(aligned_count),
            "total_horizons": int(len(labeled_entries)),
            "support_ratio": float(support_ratio),
            "short_term_ratio": None if short_ratio is None else float(short_ratio),
            "mid_term_ratio": None if mid_ratio is None else float(mid_ratio),
            "triggered": bool(confluence_triggered),
            "reasons": reasons,
        }
        entry["confluence_support_ratio"] = float(support_ratio)
        entry["confluence_short_term_ratio"] = None if short_ratio is None else float(short_ratio)
        entry["confluence_mid_term_ratio"] = None if mid_ratio is None else float(mid_ratio)
        entry["confluence_direction_matches_dominant"] = (
            0.0 if dominant_direction == "neutral" else float(dominant_direction == current_direction)
        )
        if confluence_triggered:
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["confluence_gate_triggered"] = True
                trade_decision["confluence_gate_reasons"] = reasons
    return summary


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
        weighted_vote = base_weight * coherence_multiplier * (0.5 + 0.5 * min(confidence, 1.0))
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
    horizon_overrides = raw_overrides.get(_normalize_horizon_value(horizon))
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
    if not summary:
        return summary

    bias_context = _summarize_bias_context(summary, policy)
    bias_direction = str(bias_context.get("bias_direction", "neutral"))
    bias_alignment_ratio = float(bias_context.get("bias_alignment_ratio", 0.0))
    execution_entries = bias_context.get("execution_entries", [])
    weights = policy.get("horizon_bias_weights") if isinstance(policy.get("horizon_bias_weights"), Mapping) else {}

    for label, entry in summary.items():
        market_price = float(entry.get("close", entry.get("entry_price", 0.0)) or 0.0)
        entry["market_price"] = market_price
        entry["execution_prior_provenance"] = {
            "analytics_source": "unavailable",
            "matched_regime": None,
            "volatility_bucket": None,
            "bucket_threshold": None,
            "sample_count": 0,
            "stop_source": None,
            "stop_adjustment_type": None,
            "target_source": "existing_or_projection",
        }
        side = _execution_side(entry)
        direction = _direction_vote(entry)
        upstream_hold = str(entry.get("trade_action", "hold")) == "hold"
        execution_alignment_ratio = _execution_alignment_ratio(execution_entries, direction=direction, weights=weights)
        tier = _classify_execution_tier(
            entry,
            bias_direction=bias_direction,
            execution_alignment_ratio=execution_alignment_ratio,
            policy=policy,
        )
        bias_scores = bias_context.get("bias_scores") if isinstance(bias_context.get("bias_scores"), Mapping) else {}
        execution_scores = bias_context.get("execution_scores") if isinstance(bias_context.get("execution_scores"), Mapping) else {}
        bias_score_value = float((bias_scores.get("up_score") if direction == "up" else bias_scores.get("down_score")) or 0.0)
        execution_score_value = float((execution_scores.get("up_score") if direction == "up" else execution_scores.get("down_score")) or 0.0)
        support_horizons = list((bias_context.get("direction_support_horizons") or {}).get(direction, []))
        entry["bias_score"] = bias_score_value
        entry["execution_score"] = execution_score_value
        entry["bias_support_horizons"] = support_horizons
        entry["bias_support_is_8h_standalone"] = support_horizons == ["8h"]
        plan: Dict[str, Any] = {
            "enabled": bool(policy.get("enabled", False)),
            "bias_direction": bias_direction,
            "bias_alignment_ratio": bias_alignment_ratio,
            "execution_alignment_ratio": float(execution_alignment_ratio),
            "bias_score": float(bias_score_value),
            "execution_score": float(execution_score_value),
            "confluence_tier": tier,
            "status": "ready",
            "reason": "pass",
            "side": side,
            "entry_mode": "disabled",
            "pending_trade_action": side,
            "partial_take_profit": None,
            "time_stop": None,
            "trailing_stop": None,
            "analytics": {"available": False},
            "structure": None,
            "stop_management": None,
        }
        if not bool(policy.get("enabled", False)):
            entry["execution_plan"] = plan
            continue

        forecast_coherence = entry.get("forecast_coherence")
        if isinstance(forecast_coherence, Mapping) and forecast_coherence.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "forecast_coherence_gate"
            entry["execution_plan"] = plan
            continue

        if bool(policy.get("require_bias_alignment", True)) and bias_direction != "neutral" and direction != bias_direction:
            plan["status"] = "rejected"
            plan["reason"] = "bias_direction_conflict"
            entry["execution_plan"] = plan
            continue

        context = contexts.get(label)
        if not context:
            plan["status"] = "rejected"
            plan["reason"] = "missing_execution_context"
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["execution_plan"] = plan
            continue

        prepared = context["prepared"]
        index = int(context["index"])
        horizon = float(context["horizon"])
        residual_std = float(context["residual_std"])
        regime_state = str(entry.get("regime_state", REGIME_NEUTRAL))
        regime_template = (policy.get("regime_templates") or {}).get(regime_state, {})
        horizon_steps = max(int(round(horizon)), 1)
        atr_distance = _compute_atr_like_price_distance(
            prepared.df_all,
            index=index,
            fallback_close=market_price,
            fallback_return_std=residual_std,
        )
        structure = _compute_recent_structure(
            prepared.df_all,
            index=index,
            session_lookback_bars=int(policy.get("session_lookback_bars", 8)),
            swing_lookback_bars=int(policy.get("swing_lookback_bars", 6)),
            atr_distance=atr_distance,
            fallback_price=market_price,
        )
        plan["structure"] = structure
        entry_zone = _build_entry_zone(
            market_price=market_price,
            side=side,
            structure=structure,
            policy=policy,
            regime_template=regime_template,
        )
        preferred_entry = float(entry_zone["preferred_entry_price"])
        plan.update(entry_zone)

        pullback_quality = _compute_pullback_quality_score(
            entry=entry,
            frame=prepared.df_all,
            index=index,
            market_price=market_price,
            side=side,
            structure=structure,
            atr_distance=atr_distance,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
        )
        disagreement_severity = _compute_disagreement_severity(
            entry,
            bias_context=bias_context,
            policy=policy,
            atr_distance=atr_distance,
            structure=structure,
        )
        plan["pullback_quality"] = pullback_quality
        plan["disagreement_severity"] = disagreement_severity
        entry["disagreement_severity"] = disagreement_severity

        template_max_chase = float(regime_template.get("max_chase_atr_mult", 0.0) or 0.0)
        max_chase = (template_max_chase if template_max_chase > 0.0 else float(policy.get("max_chase_atr_mult", 0.35))) * atr_distance
        market_deviation = abs(market_price - preferred_entry)
        if tier == "high" and (bool(entry_zone["entry_ready"]) or market_deviation <= max_chase):
            entry_mode = "immediate"
            planned_entry = market_price
        elif tier in {"high", "medium"}:
            entry_mode = "pullback"
            planned_entry = preferred_entry
        else:
            entry_mode = "blocked"
            planned_entry = preferred_entry

        template_entry_modes = regime_template.get("entry_mode_by_tier") if isinstance(regime_template.get("entry_mode_by_tier"), Mapping) else {}
        template_entry_mode = str(template_entry_modes.get(tier) or "").strip().lower()
        if template_entry_mode in {"immediate", "pullback", "blocked"}:
            if template_entry_mode == "blocked":
                entry_mode = "blocked"
            elif template_entry_mode == "pullback" and entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif template_entry_mode == "immediate" and entry_mode == "pullback" and bool(entry_zone["entry_ready"]):
                entry_mode = "immediate"
                planned_entry = market_price

        if disagreement_severity.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "short_term_disagreement"
        elif disagreement_severity.get("pullback_only") and entry_mode == "immediate":
            entry_mode = "pullback"
            planned_entry = preferred_entry

        if pullback_quality.get("triggered"):
            if entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif entry_mode == "pullback":
                plan["status"] = "rejected"
                plan["reason"] = "pullback_quality_insufficient"
        plan["entry_mode"] = entry_mode

        analytics_cfg = policy.get("analytics", {}) if isinstance(policy.get("analytics"), Mapping) else {}
        analytics_payload = {"available": False}
        if analytics_cfg.get("enabled"):
            analytics_payload = _compute_excursion_priors(
                prepared.df_all,
                index=index,
                horizon_steps=horizon_steps,
                side=side,
                lookback_bars=int(analytics_cfg.get("lookback_bars", EXECUTION_POLICY_DEFAULT_LOOKBACK_BARS)),
                min_samples=int(analytics_cfg.get("min_samples", EXECUTION_POLICY_DEFAULT_MIN_SAMPLES)),
                mae_quantile=float(analytics_cfg.get("mae_quantile", 0.75)),
                mfe_quantile=float(analytics_cfg.get("mfe_quantile", 0.6)),
                current_regime=regime_state,
                current_volatility=_finite_float_or_none((entry.get("volatility") or {}).get("current")),
                bucket_policy=analytics_cfg.get("regime_volatility_buckets"),
            )
        plan["analytics"] = analytics_payload

        existing_stop = float(entry.get("stop_loss", planned_entry))
        existing_take = float(entry.get("take_profit", planned_entry))
        structure_buffer = atr_distance * float(policy.get("structure_buffer_atr_mult", 0.2))
        if side == "long":
            structure_stop = min(float(structure["session_low"]), float(structure["swing_low"])) - structure_buffer
            analytic_stop = planned_entry * (1.0 - float(analytics_payload.get("mae_distance") or 0.0))
        else:
            structure_stop = max(float(structure["session_high"]), float(structure["swing_high"])) + structure_buffer
            analytic_stop = planned_entry * (1.0 + float(analytics_payload.get("mae_distance") or 0.0))
        analytic_stop_value = analytic_stop if analytics_payload.get("available") else None

        guards_cfg = policy.get("no_trade_guards", {}) if isinstance(policy.get("no_trade_guards"), Mapping) else {}
        stop_resolution = _resolve_stop_with_guardrails(
            side=side,
            planned_entry=planned_entry,
            existing_stop=existing_stop,
            structure_stop=structure_stop,
            analytic_stop=analytic_stop_value,
            atr_distance=atr_distance,
            guards_cfg=guards_cfg,
            analytic_stop_preferred=bool(analytics_payload.get("available")) and str(analytics_payload.get("source")) != "global",
        )
        selected_stop = float(stop_resolution["stop_loss"])
        risk_unit = float(stop_resolution["risk_unit"])
        stop_refinement = _refine_stop_with_target_range(
            side=side,
            planned_entry=planned_entry,
            selected_stop=selected_stop,
            risk_unit=risk_unit,
            atr_distance=atr_distance,
            horizon=horizon,
            projected_high=_finite_float_or_none(entry.get("projected_high")),
            projected_low=_finite_float_or_none(entry.get("projected_low")),
            projected_high_confidence=_finite_float_or_none(entry.get("projected_high_confidence")),
            projected_low_confidence=_finite_float_or_none(entry.get("projected_low_confidence")),
            projected_high_residual_std=_finite_float_or_none(entry.get("projected_high_residual_std")),
            projected_low_residual_std=_finite_float_or_none(entry.get("projected_low_residual_std")),
            policy=policy,
            guards_cfg=guards_cfg,
        )
        if stop_refinement.get("applied"):
            selected_stop = float(stop_refinement["stop_loss"])
            risk_unit = float(stop_refinement["risk_unit"])
        plan["stop_management"] = {
            "source": stop_resolution.get("source"),
            "adjustment": stop_resolution.get("adjustment"),
            "target_range_refinement": stop_refinement.get("details"),
        }

        if guards_cfg.get("enabled"):
            max_entry_dev = float(guards_cfg.get("max_entry_deviation_atr_mult", 1.25)) * atr_distance
            if bool(guards_cfg.get("require_favorable_entry_zone", True)) and market_deviation > max_entry_dev and entry_mode == "immediate":
                plan["status"] = "rejected"
                plan["reason"] = "entry_too_extended"

        target_resolution = _resolve_execution_target_reward(
            side=side,
            planned_entry=planned_entry,
            existing_take=existing_take,
            projected_high=_finite_float_or_none(entry.get("projected_high")),
            projected_low=_finite_float_or_none(entry.get("projected_low")),
            analytics_payload=analytics_payload,
            risk_unit=risk_unit,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
        )
        selected_take = float(target_resolution["selected_take"])
        risk_reward_ratio = float(target_resolution["risk_reward_ratio"])
        plan["target_management"] = dict(target_resolution["target_management"])
        if target_resolution["status"] != "pass":
            plan["status"] = "rejected"
            plan["reason"] = str(target_resolution["reason"])

        partial_cfg = policy.get("partial_take_profit", {}) if isinstance(policy.get("partial_take_profit"), Mapping) else {}
        partial_take_profit = None
        if partial_cfg.get("enabled"):
            tp1_distance = risk_unit * float(partial_cfg.get("tp1_r_multiple", 1.0))
            tp1_price = planned_entry + tp1_distance if side == "long" else planned_entry - tp1_distance
            partial_take_profit = {
                "enabled": True,
                "tp1_price": tp1_price,
                "tp1_size_fraction": float(partial_cfg.get("tp1_size_fraction", 0.5)),
                "tp2_price": selected_take,
                "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
            }

        trailing_cfg = policy.get("trailing_stop", {}) if isinstance(policy.get("trailing_stop"), Mapping) else {}
        trailing_stop = None
        if trailing_cfg.get("enabled"):
            activation_distance = risk_unit * float(trailing_cfg.get("activation_r_multiple", 1.0))
            trailing_stop = {
                "enabled": True,
                "activation_price": planned_entry + activation_distance if side == "long" else planned_entry - activation_distance,
                "trail_buffer": atr_distance * float(trailing_cfg.get("trail_buffer_atr_mult", 0.75)),
            }

        time_stop_map = policy.get("time_stop_bars_by_horizon", {}) if isinstance(policy.get("time_stop_bars_by_horizon"), Mapping) else {}
        base_time_stop = max(int(round(_lookup_horizon_value(time_stop_map, horizon, max(horizon_steps, 1)))), 1)
        time_stop_mult = float(regime_template.get("time_stop_multiplier", 1.0) or 1.0)
        recommended_time_stop = max(int(round(base_time_stop * time_stop_mult)), 1)
        if analytics_payload.get("available") and analytics_payload.get("peak_step_p50"):
            recommended_time_stop = min(recommended_time_stop, max(int(analytics_payload["peak_step_p50"] * 1.25), 1))
        time_stop_payload = {
            "enabled": True,
            "bars": recommended_time_stop,
            "reason": "stagnation_exit",
        }

        if plan["status"] == "ready" and entry_mode == "pullback":
            if bool(entry_zone["entry_ready"]):
                plan["status"] = "ready"
                plan["reason"] = "pass"
            else:
                plan["status"] = "waiting_pullback"
                plan["reason"] = "await_pullback_entry_zone"
        elif plan["status"] == "ready" and entry_mode == "blocked":
            plan["status"] = "rejected"
            plan["reason"] = "low_execution_confluence"

        position_size = float(entry.get("position_size", 0.0))
        position_size *= float(regime_template.get("size_multiplier", 1.0) or 1.0)
        if tier == "medium":
            position_size *= 0.85
        elif tier == "low":
            position_size = 0.0

        plan["partial_take_profit"] = partial_take_profit
        plan["time_stop"] = time_stop_payload
        plan["trailing_stop"] = trailing_stop

        entry["entry_price"] = float(planned_entry)
        entry["stop_loss"] = float(selected_stop)
        entry["take_profit"] = float(selected_take)
        entry["risk_reward_ratio"] = float(risk_reward_ratio)
        entry["position_size"] = float(max(position_size, 0.0))
        analytics_payload_final = plan.get("analytics") if isinstance(plan.get("analytics"), Mapping) else {}
        stop_management = plan.get("stop_management") if isinstance(plan.get("stop_management"), Mapping) else {}
        entry["execution_prior_provenance"] = {
            "analytics_source": analytics_payload_final.get("source", "unavailable") if analytics_payload_final else "unavailable",
            "matched_regime": analytics_payload_final.get("matched_regime"),
            "volatility_bucket": analytics_payload_final.get("volatility_bucket"),
            "bucket_threshold": analytics_payload_final.get("bucket_threshold"),
            "sample_count": analytics_payload_final.get("sample_count"),
            "stop_source": stop_management.get("source"),
            "stop_adjustment_type": (stop_management.get("adjustment") or {}).get("type") if stop_management else None,
            "target_source": str((plan.get("target_management") or {}).get("source") or "existing_or_projection"),
        }
        entry["execution_plan"] = plan

        if upstream_hold and plan["status"] == "ready":
            plan["status"] = "bias_only_ready"
            plan["reason"] = _resolve_execution_upstream_hold_reason(entry)
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        elif plan["status"] != "ready":
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        else:
            entry["trade_action"] = side
    return summary


def _build_execution_prior_summary(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    analytics_source_counts: Dict[str, int] = {}
    stop_source_counts: Dict[str, int] = {}
    target_source_counts: Dict[str, int] = {}
    for entry in summary.values():
        provenance = entry.get("execution_prior_provenance") if isinstance(entry, Mapping) else None
        if not isinstance(provenance, Mapping):
            continue
        analytics_source = str(provenance.get("analytics_source") or "unavailable")
        stop_source = str(provenance.get("stop_source") or "unknown")
        target_source = str(provenance.get("target_source") or "unknown")
        analytics_source_counts[analytics_source] = analytics_source_counts.get(analytics_source, 0) + 1
        stop_source_counts[stop_source] = stop_source_counts.get(stop_source, 0) + 1
        target_source_counts[target_source] = target_source_counts.get(target_source, 0) + 1
    return {
        "analytics_source_counts": analytics_source_counts,
        "stop_source_counts": stop_source_counts,
        "target_source_counts": target_source_counts,
    }


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
) -> Dict[str, Any]:
    rr_floor = _lookup_horizon_value(policy.get("minimum_rr_by_horizon", {}), horizon, 1.0)
    rr_floor *= float(regime_template.get("tp_multiplier", 1.0) or 1.0)
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
        },
    }


def _confidence_level_from_score(value: Any) -> str:
    score = _finite_float_or_none(value)
    if score is None:
        return "Low"
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def _prompt_direction_label(direction: str) -> str:
    normalized = str(direction).strip().lower()
    if normalized == "up":
        return "Long"
    if normalized == "down":
        return "Short"
    return "Neutral"


def _format_usd_value(value: Any) -> str | None:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return None
    return f"${numeric:,.2f}"


def _prompt_effective_direction(entry: Mapping[str, Any]) -> str:
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    direction_display = str(entry.get("direction_next_display") or "neutral").lower()
    if bool(coherence.get("triggered")) and direction_display == "neutral":
        internal_direction = str(entry.get("direction_next") or "neutral").lower()
        if internal_direction in {"up", "down"}:
            return internal_direction
    return direction_display


def _build_prompt_forecast_clause(label: str, entry: Mapping[str, Any]) -> str:
    direction_display = _prompt_effective_direction(entry)
    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    projected_high = _format_usd_value(entry.get("projected_high"))
    projected_low = _format_usd_value(entry.get("projected_low"))
    projected_price = _format_usd_value(entry.get("projected_price"))

    clause = f"{label}: {direction_display}"
    if projected_high and projected_low:
        clause += f", projected range {projected_low} to {projected_high}"
    elif projected_price:
        clause += f", projected price {projected_price}"

    if coherence.get("triggered"):
        clause += " (coherence blocked)"
    elif plan.get("reason") not in {None, "pass", "upstream_model_hold", "confluence_gate", "await_pullback_entry_zone"}:
        clause += f" ({plan.get('reason')})"
    elif plan.get("status") == "bias_only_ready":
        hold_reason = "confluence gate" if plan.get("reason") == "confluence_gate" else "upstream hold"
        clause += f" (bias ready, {hold_reason})"
    return clause


def _prompt_status_rank(status: str) -> int:
    return {
        "ready": 0,
        "waiting_pullback": 1,
        "bias_only_ready": 2,
        "analysis_only": 3,
        "rejected": 4,
        "no_trade": 5,
    }.get(str(status or "rejected"), 6)


def _prompt_reason_rank(reason: str | None) -> int:
    return {
        "pass": 0,
        "await_pullback_entry_zone": 1,
        "upstream_model_hold": 2,
        "confluence_gate": 2,
        "low_execution_confluence": 3,
        "insufficient_mfe_headroom": 4,
        "bias_direction_conflict": 5,
        "forecast_coherence_gate": 6,
    }.get(str(reason or "pass"), 7)


def _prompt_confluence_rank(tier: str | None) -> int:
    return {
        "high": 0,
        "medium": 1,
        "low": 2,
    }.get(str(tier or "low"), 3)


def _prompt_entry_rank(label: str, entry: Mapping[str, Any]) -> tuple[int, int, int, int, float, float, float, float]:
    horizon = _coerce_result_horizon(entry.get("horizon_hours"))
    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
    status_rank = _prompt_status_rank(str(plan.get("status") or "rejected"))
    reason_rank = _prompt_reason_rank(plan.get("reason"))
    confluence_rank = _prompt_confluence_rank(plan.get("confluence_tier"))
    execution_alignment = float(plan.get("execution_alignment_ratio") or 0.0)
    bias_alignment = float(plan.get("bias_alignment_ratio") or 0.0)
    execution_score = float(plan.get("execution_score") or 0.0)
    bias_score = float(plan.get("bias_score") or 0.0)
    confidence_score = float(_finite_float_or_none(entry.get("confidence_score")) or 0.0)
    horizon_preference = {4.0: 0, 12.0: 1, 8.0: 2, 1.0: 3, 0.25: 4}
    preference_rank = float(horizon_preference.get(horizon, 9.0))
    if bool(entry.get("bias_support_is_8h_standalone")) and horizon == 8.0:
        preference_rank += 2.0
    return (
        status_rank,
        reason_rank,
        confluence_rank,
        -execution_alignment,
        -bias_alignment,
        -execution_score,
        -bias_score,
        -confidence_score,
        preference_rank,
        -(float(horizon) if horizon is not None else 0.0),
    )


def _select_prompt_candidate_entries(
    summary: Mapping[str, Mapping[str, Any]],
) -> List[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]]:
    ranked_entries: List[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]] = []
    directional_hourly_or_higher_present = False
    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        horizon = _coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        direction_display = _prompt_effective_direction(entry)
        rank = _prompt_entry_rank(label, entry)
        ranked_entries.append((rank, label, entry))
        if direction_display in {"up", "down"} and horizon >= 1.0:
            directional_hourly_or_higher_present = True

    if directional_hourly_or_higher_present:
        filtered_entries = []
        for rank, label, entry in ranked_entries:
            horizon = _coerce_result_horizon(entry.get("horizon_hours"))
            direction_display = _prompt_effective_direction(entry)
            if direction_display in {"up", "down"} and horizon is not None and horizon < 1.0:
                continue
            filtered_entries.append((rank, label, entry))
        return filtered_entries
    return ranked_entries


def _select_prompt_preferred_entry(
    summary: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, Mapping[str, Any] | None, Dict[str, Any] | None]:
    ranked_entries = _select_prompt_candidate_entries(summary)
    side_entries: Dict[str, List[tuple[tuple[int, int, int, float, float, float, float, float], str, Mapping[str, Any]]]] = {
        "up": [],
        "down": [],
    }
    for rank, label, entry in ranked_entries:
        direction_display = _prompt_effective_direction(entry)
        if direction_display in side_entries:
            side_entries[direction_display].append((rank, label, entry))

    side_profiles: List[
        tuple[
            tuple[int, int, int, int, int, int, float, float, float, float],
            str,
            tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]],
            Dict[str, Any],
        ]
    ] = []
    for side, entries in side_entries.items():
        if not entries:
            continue
        ordered_entries = sorted(entries, key=lambda item: item[0])
        ready_like_count = 0
        high_timeframe_count = 0
        avg_execution_alignment = 0.0
        avg_bias_alignment = 0.0
        support_horizons: List[str] = []
        for _rank, label, entry in ordered_entries:
            plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
            status = str(plan.get("status") or "rejected")
            if status in {"ready", "waiting_pullback", "bias_only_ready"}:
                ready_like_count += 1
            horizon = _coerce_result_horizon(entry.get("horizon_hours"))
            if horizon is not None and horizon >= 8.0:
                high_timeframe_count += 1
            avg_execution_alignment += float(plan.get("execution_alignment_ratio") or 0.0)
            avg_bias_alignment += float(plan.get("bias_alignment_ratio") or 0.0)
            support_horizons.append(label)
        avg_execution_alignment /= max(len(ordered_entries), 1)
        avg_bias_alignment /= max(len(ordered_entries), 1)
        best_rank, best_label, best_entry = ordered_entries[0]
        side_rank = (
            best_rank[0],
            -ready_like_count,
            -high_timeframe_count,
            -len(ordered_entries),
            best_rank[1],
            best_rank[2],
            -avg_execution_alignment,
            -avg_bias_alignment,
            best_rank[5],
            best_rank[6],
        )
        side_profiles.append(
            (
                side_rank,
                side,
                (best_rank, best_label, best_entry),
                {
                    "side": side,
                    "support_horizons": support_horizons,
                    "support_count": len(ordered_entries),
                    "high_timeframe_count": high_timeframe_count,
                    "ready_like_count": ready_like_count,
                    "avg_execution_alignment": float(avg_execution_alignment),
                    "avg_bias_alignment": float(avg_bias_alignment),
                    "conflict_present": sum(1 for entries in side_entries.values() if entries) > 1,
                },
            )
        )

    if side_profiles:
        side_profiles.sort(key=lambda item: item[0])
        _side_rank, _side, best_entry_tuple, side_profile = side_profiles[0]
        _best_rank, best_label, best_entry = best_entry_tuple
        return best_label, best_entry, side_profile

    ranked_entries.sort(key=lambda item: item[0])
    if ranked_entries:
        _rank, preferred_label, preferred_entry = ranked_entries[0]
        return preferred_label, preferred_entry, None
    return None, None, None


def _build_prompt_ready_summary(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    preferred_label, preferred_entry, side_profile = _select_prompt_preferred_entry(summary)

    trend_parts: List[str] = []
    blocking_factors: List[str] = []
    for label in sorted(summary.keys(), key=_horizon_sort_key):
        entry = summary[label]
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
        clause = _build_prompt_forecast_clause(label, entry)
        if coherence.get("triggered"):
            blocking_factors.extend(str(reason) for reason in coherence.get("reasons", []))
        elif plan.get("reason") not in {None, "pass", "upstream_model_hold", "confluence_gate", "await_pullback_entry_zone"}:
            blocking_factors.append(str(plan.get("reason")))
        trend_parts.append(clause)

    selected_direction = "Neutral"
    preferred_horizon = None
    confidence_level = "Low"
    tradeable = False
    execution_state = "no_trade"
    pending_trade_action = None
    entry_point = None
    stop_loss = None
    take_profit = None
    risk_reward_ratio = None
    rationale = "No horizon produced a coherent executable trade setup."

    if preferred_entry is not None and preferred_label is not None:
        direction_display = _prompt_effective_direction(preferred_entry)
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        target_management = plan.get("target_management") if isinstance(plan.get("target_management"), Mapping) else {}
        selected_direction = _prompt_direction_label(direction_display)
        preferred_horizon = preferred_label
        confidence_level = _confidence_level_from_score(preferred_entry.get("confidence_score"))
        execution_state = str(plan.get("status") or "no_trade")
        pending_trade_action = str(plan.get("pending_trade_action") or "").lower() or None
        tradeable = execution_state in {"ready", "waiting_pullback"} and selected_direction != "Neutral"
        if execution_state in {"ready", "waiting_pullback", "bias_only_ready"} and selected_direction != "Neutral":
            entry_point = _finite_float_or_none(preferred_entry.get("entry_price"))
            stop_loss = _finite_float_or_none(preferred_entry.get("stop_loss"))
            take_profit = _finite_float_or_none(preferred_entry.get("take_profit"))
            risk_reward_ratio = _finite_float_or_none(preferred_entry.get("risk_reward_ratio"))
        rationale_parts = [f"Preferred horizon {preferred_label} carries the strongest post-policy bias."]
        if side_profile and side_profile.get("conflict_present"):
            support_horizons = side_profile.get("support_horizons") or []
            support_text = ", ".join(str(value) for value in support_horizons)
            rationale_parts[0] = (
                f"Preferred horizon {preferred_label} wins side arbitration for {selected_direction.lower()} bias "
                f"across {support_text}."
            )
        if plan.get("reason") == "forecast_coherence_gate":
            rationale_parts.append("The forecast remains directional, but forecast coherence blocks it from execution.")
        elif plan.get("status") == "bias_only_ready":
            rationale_parts.append("The horizon is structurally aligned, but the upstream action remains hold.")
        elif plan.get("status") == "waiting_pullback":
            rationale_parts.append("Bias is valid, but price is outside the preferred entry zone.")
        if target_management.get("adapted_to_mfe_headroom"):
            rationale_parts.append("Take-profit was resized to empirical MFE headroom instead of rejecting the setup.")
        elif plan.get("reason") not in {None, "pass"}:
            rationale_parts.append(f"Current blocker: {plan.get('reason')}.")
        rationale = " ".join(rationale_parts)

    formatted_response = "\n".join(
        [
            "Market Outlook & Strategy",
            f"Selected Direction: {selected_direction}",
            f"Preferred Horizon: {preferred_horizon or 'None'}",
            f"Confidence Level: {confidence_level}",
            f"Pending Trade Action: {(pending_trade_action or 'hold').title() if selected_direction != 'Neutral' else 'Hold'}",
            "",
            "Trade Execution Plan (USD)",
            f"Entry Point: {_format_usd_value(entry_point) or 'No trade'}",
            f"Stop Loss: {_format_usd_value(stop_loss) or 'No trade'}",
            f"Take Profit: {_format_usd_value(take_profit) or 'No trade'}",
            f"Risk/Reward Ratio: {f'{risk_reward_ratio:.2f}' if risk_reward_ratio is not None else 'Not applicable'}",
            "",
            "Analysis Summary",
            f"Trend Forecast: {'; '.join(trend_parts)}",
            f"Rationale: {rationale}",
        ]
    )

    blocking_factors = sorted({factor for factor in blocking_factors if factor})
    operator_compact = _build_operator_summary_compact(
        summary,
        preferred_label=preferred_label,
        preferred_entry=preferred_entry,
        market_direction=selected_direction,
        execution_state=execution_state,
        blocking_factors=blocking_factors,
    )
    return {
        "market_outlook_strategy": {
            "selected_direction": selected_direction,
            "preferred_horizon": preferred_horizon,
            "confidence_level": confidence_level,
            "pending_trade_action": pending_trade_action,
            "tradeable": tradeable,
            "execution_state": execution_state,
        },
        "trade_execution_plan_usd": {
            "entry_point": entry_point,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
        },
        "analysis_summary": {
            "trend_forecast": trend_parts,
            "rationale": rationale,
            "blocking_factors": blocking_factors,
        },
        "operator_summary_compact": operator_compact,
        "formatted_response": formatted_response,
    }


def _build_operator_summary_compact(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    preferred_label: str | None,
    preferred_entry: Mapping[str, Any] | None,
    market_direction: str,
    execution_state: str,
    blocking_factors: Sequence[str],
) -> Dict[str, Any]:
    normalized_market_direction = str(market_direction).strip().lower()
    if normalized_market_direction == "long":
        normalized_market_direction = "up"
    elif normalized_market_direction == "short":
        normalized_market_direction = "down"
    elif normalized_market_direction not in {"up", "down"}:
        normalized_market_direction = "neutral"

    primary_blocker = None
    if blocking_factors:
        primary_blocker = str(blocking_factors[0])
    elif preferred_entry is not None:
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        if plan.get("reason") not in {None, "pass"}:
            primary_blocker = str(plan.get("reason"))

    action = "stand_aside"
    if execution_state == "ready":
        action = "enter_now"
    elif execution_state == "waiting_pullback":
        action = "wait_for_pullback"
    elif execution_state == "bias_only_ready":
        action = "bias_only"

    support_horizons = []
    max_disagreement_score = 0.0
    caution_flags: List[str] = []
    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        if _prompt_effective_direction(entry) == normalized_market_direction:
            support_horizons.append(label)
        disagreement = plan.get("disagreement_severity") if isinstance(plan.get("disagreement_severity"), Mapping) else {}
        max_disagreement_score = max(max_disagreement_score, float(disagreement.get("score") or 0.0))
        if bool(entry.get("bias_support_is_8h_standalone")):
            caution_flags.append("8h_standalone_bias")
        if disagreement.get("triggered"):
            caution_flags.append("short_term_disagreement")

    if preferred_entry is not None:
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        pullback_quality = plan.get("pullback_quality") if isinstance(plan.get("pullback_quality"), Mapping) else {}
        if pullback_quality.get("triggered"):
            caution_flags.append("pullback_quality_insufficient")

    return {
        "market_bias": str(market_direction),
        "preferred_horizon": preferred_label,
        "recommended_operator_action": action,
        "primary_blocker": primary_blocker,
        "support_horizons": support_horizons,
        "max_disagreement_score": float(max_disagreement_score),
        "caution_flags": sorted(set(caution_flags)),
    }


def _resolve_data_quality_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "max_staleness_hours": float(cfg.get("max_staleness_hours") or 2.0),
        "max_missing_ratio": float(cfg.get("max_missing_ratio") or 0.01),
        "max_zero_volume_ratio": float(cfg.get("max_zero_volume_ratio") or 0.2),
        "min_rows": int(cfg.get("min_rows") or 120),
    }


def _resolve_abstention_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "min_confidence": max(0.0, min(1.0, float(cfg.get("min_confidence") or 0.0))),
        "min_abs_expected_value": max(float(cfg.get("min_abs_expected_value") or 0.0), 0.0),
        "min_edge_over_fee": max(float(cfg.get("min_edge_over_fee") or 0.0), 0.0),
        "require_positive_ev": bool(cfg.get("require_positive_ev", False)),
        "hold_prob_center": max(0.0, min(1.0, float(cfg.get("hold_prob_center") or 0.5))),
        "hold_prob_band": max(0.0, min(0.5, float(cfg.get("hold_prob_band") or 0.0))),
    }


def _resolve_regime_model_weights_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    if not bool(config.get("enabled", False)):
        return {"enabled": False, "weights_by_regime": {}, "weights_by_regime_horizon": {}}

    weights_by_regime: Dict[str, Dict[str, float]] = {}
    weights_by_regime_horizon: Dict[str, Dict[float, Dict[str, float]]] = {}
    for regime in (REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP):
        raw = config.get(regime)
        if not raw:
            continue
        if isinstance(raw, Mapping):
            per_horizon: Dict[float, Dict[str, float]] = {}
            for raw_horizon, raw_weights in raw.items():
                horizon = _coerce_numeric_horizon(raw_horizon)
                if horizon is None:
                    continue
                parsed = parse_weight_spec(str(raw_weights))
                if parsed:
                    per_horizon[_normalize_horizon_value(horizon)] = {str(k): float(v) for k, v in parsed.items()}
            if per_horizon:
                weights_by_regime_horizon[regime] = per_horizon
            continue

        parsed = parse_weight_spec(str(raw))
        if parsed:
            weights_by_regime[regime] = {str(k): float(v) for k, v in parsed.items()}
    return {
        "enabled": True,
        "weights_by_regime": weights_by_regime,
        "weights_by_regime_horizon": weights_by_regime_horizon,
    }


def _resolve_uncertainty_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    alpha = float(cfg.get("alpha") or 0.2)
    alpha = max(0.01, min(0.49, alpha))
    thresholds_by_horizon_regime: Dict[float, Dict[str, Dict[str, Any]]] = {}
    raw_thresholds = cfg.get("thresholds_by_horizon_regime") if isinstance(cfg.get("thresholds_by_horizon_regime"), Mapping) else {}
    for raw_horizon, raw_regimes in raw_thresholds.items():
        horizon = _coerce_numeric_horizon(raw_horizon)
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
            thresholds_by_horizon_regime[_normalize_horizon_value(horizon)] = resolved_regimes
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "alpha": alpha,
        "hold_prob_center": max(0.0, min(1.0, float(cfg.get("hold_prob_center") or 0.5))),
        "max_interval_width": max(float(cfg.get("max_interval_width") or 1.0), 0.0),
        "require_center_cross": bool(cfg.get("require_center_cross", True)),
        "min_component_count": max(int(float(cfg.get("min_component_count") or 3)), 1),
        "thresholds_by_horizon_regime": thresholds_by_horizon_regime,
    }


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
                print(f"Warning: failed to parse trade decision model {path}: {exc}", file=sys.stderr)
        else:
            print(f"Warning: trade decision model not found at {path}; policy disabled.", file=sys.stderr)
    enabled = bool(cfg.get("enabled", False) and model_payload is not None)
    midband_veto_cfg = cfg.get("midband_veto") if isinstance(cfg.get("midband_veto"), Mapping) else {}
    weak_band_veto_cfg = cfg.get("weak_band_veto") if isinstance(cfg.get("weak_band_veto"), Mapping) else {}
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
                for value in (midband_veto_cfg.get("regime_states", []) if isinstance(midband_veto_cfg.get("regime_states", []), list) else [])
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


def _lookup_oof_expected_net(model: Mapping[str, Any], prob: float) -> float | None:
    oof_payload = model.get("oof_expected_value") if isinstance(model, Mapping) else None
    if not isinstance(oof_payload, Mapping):
        return None
    bins = oof_payload.get("bins")
    if not isinstance(bins, list):
        return None
    prob_value = _finite_float_or_none(prob)
    if prob_value is None:
        return None
    p = float(max(0.0, min(1.0, prob_value)))
    for idx, bucket in enumerate(bins):
        if not isinstance(bucket, Mapping):
            continue
        lo = _finite_float(bucket.get("p_min", 0.0), 0.0)
        hi = _finite_float(bucket.get("p_max", 1.0), 1.0)
        in_range = (p >= lo and p < hi) if idx < len(bins) - 1 else (p >= lo and p <= hi)
        if in_range:
            return _finite_float_or_none(bucket.get("mean_ret_net", 0.0))
    default_value = oof_payload.get("default_expected_net")
    return _finite_float_or_none(default_value)


def _lookup_raw_ev_expected_net(model: Mapping[str, Any], raw_ev: float) -> float | None:
    raw_ev_value = _finite_float_or_none(raw_ev)
    if raw_ev_value is None:
        return None
    iso_payload = model.get("raw_ev_isotonic") if isinstance(model, Mapping) else None
    if isinstance(iso_payload, Mapping):
        x_vals = iso_payload.get("x_thresholds")
        y_vals = iso_payload.get("y_thresholds")
        if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) >= 2 and len(x_vals) == len(y_vals):
            try:
                x = np.asarray([float(v) for v in x_vals], dtype=float)
                y = np.asarray([float(v) for v in y_vals], dtype=float)
                interpolated = float(np.interp(float(raw_ev_value), x, y, left=y[0], right=y[-1]))
                return _finite_float_or_none(interpolated)
            except Exception:
                pass

    payload = model.get("raw_ev_expected_value") if isinstance(model, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    bins = payload.get("bins")
    if not isinstance(bins, list):
        return None
    x = float(raw_ev_value)
    for idx, bucket in enumerate(bins):
        if not isinstance(bucket, Mapping):
            continue
        lo = _finite_float(bucket.get("x_min", float("-inf")), float("-inf"))
        hi = _finite_float(bucket.get("x_max", float("inf")), float("inf"))
        in_range = (x >= lo and x < hi) if idx < len(bins) - 1 else (x >= lo and x <= hi)
        if in_range:
            return _finite_float_or_none(bucket.get("mean_ret_net", 0.0))
    default_value = payload.get("default_expected_net")
    return _finite_float_or_none(default_value)


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

    p = float(max(0.0, min(1.0, prob)))
    positive_ranges: List[tuple[float, float]] = []
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
        in_range = (p >= lo and p < hi) if idx < len(bins) - 1 else (p >= lo and p <= hi)
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

    in_positive = any((p >= lo and p <= hi) for lo, hi in positive_ranges)
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


def _lookup_raw_ev_fallback_threshold(model: Mapping[str, Any], quantile: float) -> float | None:
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
        return _finite_float_or_none(direct)

    # Fallback to nearest available quantile key if exact one is missing.
    best_dist = float("inf")
    best_value: float | None = None
    for k, v in quantiles.items():
        if not isinstance(k, str) or not k.startswith("q"):
            continue
        try:
            kq = float(k[1:]) / 100.0
            dist = abs(kq - q)
            if dist < best_dist:
                best_dist = dist
                best_value = _finite_float_or_none(v)
        except Exception:
            continue
    return best_value


def _apply_trade_decision_model(
    *,
    result: Dict[str, Any],
    regime_state: str,
    residual_std: float,
    policy: Mapping[str, Any],
    fee_bps: float,
    slippage_bps: float,
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

    feature_names = [str(v) for v in model.get("feature_columns", [])] if isinstance(model.get("feature_columns"), list) else []
    coefficients = [float(v) for v in model.get("coefficients", [])] if isinstance(model.get("coefficients"), list) else []
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
        "regime_is_trend": 1.0 if regime_state == REGIME_TREND else 0.0,
        "regime_is_neutral": 1.0 if regime_state == REGIME_NEUTRAL else 0.0,
        "regime_is_chop": 1.0 if regime_state == REGIME_CHOP else 0.0,
    }

    logit = intercept
    for name, coef in zip(feature_names, coefficients):
        logit += coef * float(feature_values.get(name, 0.0))
    trade_prob = _sigmoid(logit)

    threshold = max(0.0, min(1.0, float(policy.get("threshold", 0.55))))
    expected_net_raw = _finite_float(result.get("expected_value", 0.0), 0.0)
    expected_net_oof = _lookup_oof_expected_net(model, trade_prob)
    expected_net_raw_calibrated = _lookup_raw_ev_expected_net(model, expected_net_raw)
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
    ret_pred = _finite_float(result.get("ret_pred", 0.0), 0.0)
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
    if bool(policy.get("enforce_positive_oof_envelope", False)) and envelope.get("available", False):
        has_positive = bool(envelope.get("has_positive_bin", False))
        in_positive = bool(envelope.get("in_positive_bin", False))
        matched_populated_bin = bool(envelope.get("matched_populated_bin", False))
        matched_positive_bin = bool(envelope.get("matched_positive_bin", False))
        raw_ev_fallback_threshold: float | None = None
        raw_ev_fallback_pass = False
        if envelope_mode == "populated_bin_sign":
            if matched_populated_bin and not matched_positive_bin:
                trade_ok = False
        elif has_positive and not in_positive:
            trade_ok = False
        if (not has_positive) and bool(policy.get("block_when_no_positive_oof_bin", True)):
            allow_raw_fallback = bool(policy.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False))
            if allow_raw_fallback:
                fallback_threshold = _lookup_raw_ev_fallback_threshold(
                    model,
                    quantile=float(policy.get("raw_ev_fallback_quantile", 0.9)),
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
    else:
        raw_ev_fallback_threshold = None
        raw_ev_fallback_pass = False

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
                for value in (midband_veto_cfg.get("regime_states", []) if isinstance(midband_veto_cfg.get("regime_states", []), list) else [])
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

        result["signal_ensemble"] = int(trade_ok)
        result["trade_action"] = (
            "long" if int(trade_ok) == 1 and signal_dir_only == 1 else
            "short" if int(trade_ok) == 1 and signal_dir_only == 0 else
            "hold"
        )
    else:
        midband_veto_triggered = False
        midband_veto_reason = "replace_threshold_rule_disabled"

    return {
        "enabled": True,
        "triggered": bool(trade_ok),
        "trade_probability": float(trade_prob),
        "threshold": float(threshold),
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


def _resolve_regime_model_dirs_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not config or not bool(config.get("enabled", False)):
        return {"enabled": False, "paths": {}}

    paths: Dict[str, Dict[str, str]] = {}
    for regime in (REGIME_TREND, REGIME_NEUTRAL, REGIME_CHOP):
        raw = config.get(regime)
        if isinstance(raw, Mapping):
            paths[regime] = {str(k): str(v) for k, v in raw.items() if v is not None}
    return {"enabled": True, "paths": paths}


def _resolve_regime_specific_dir_path(
    default_path: Path,
    *,
    regime_state: str,
    horizon_label: str,
    policy: Mapping[str, Any],
) -> Path:
    if not policy or not bool(policy.get("enabled", False)):
        return default_path
    path_map = policy.get("paths", {})
    regime_map = path_map.get(regime_state) if isinstance(path_map, Mapping) else None
    if not isinstance(regime_map, Mapping):
        return default_path
    override = regime_map.get(horizon_label)
    if not override:
        return default_path
    override_path = Path(str(override)).expanduser()
    override_path = resolve_best_versioned_model_file(
        override_path,
        expected_filename=f"xgb_dir{horizon_label}_model.json",
        version_priority=MODEL_VERSION_PRIORITY,
    )
    if not override_path.exists():
        print(
            f"Warning: regime model dir override not found for {horizon_label}@{regime_state}: {override_path}",
            file=sys.stderr,
        )
        return default_path
    return override_path


def _apply_regime_weight_overrides(
    base_weights: Mapping[str, float],
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    resolved = {str(k): float(v) for k, v in base_weights.items()}
    if not policy or not bool(policy.get("enabled")):
        return resolved
    normalized_horizon = _normalize_horizon_value(horizon) if horizon is not None else None
    weights_by_regime_horizon = policy.get("weights_by_regime_horizon") or {}
    if normalized_horizon is not None:
        horizon_overrides = weights_by_regime_horizon.get(regime_state)
        if isinstance(horizon_overrides, Mapping):
            override = horizon_overrides.get(normalized_horizon)
            if isinstance(override, Mapping):
                for key, value in override.items():
                    resolved[str(key)] = float(value)
                return resolved
    weights_by_regime = policy.get("weights_by_regime") or {}
    override = weights_by_regime.get(regime_state)
    if not isinstance(override, Mapping):
        return resolved
    for key, value in override.items():
        key_str = str(key)
        # Accept both model names and type aliases in the weight map.
        resolved[key_str] = float(value)
    return resolved


def _get_active_regime_weight_override(
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, float]]:
    if not policy or not bool(policy.get("enabled")):
        return None
    normalized_horizon = _normalize_horizon_value(horizon) if horizon is not None else None
    weights_by_regime_horizon = policy.get("weights_by_regime_horizon") or {}
    if normalized_horizon is not None:
        horizon_overrides = weights_by_regime_horizon.get(regime_state)
        if isinstance(horizon_overrides, Mapping):
            override = horizon_overrides.get(normalized_horizon)
            if isinstance(override, Mapping):
                return {str(k): float(v) for k, v in override.items()}
    weights_by_regime = policy.get("weights_by_regime") or {}
    override = weights_by_regime.get(regime_state)
    if isinstance(override, Mapping):
        return {str(k): float(v) for k, v in override.items()}
    return None


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


def _resolve_abstention_expected_value(
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


def _apply_uncertainty_abstention(
    *,
    trade_action: str,
    p_up_components: Mapping[str, Any],
    horizon: float | None,
    regime_state: str,
    policy: Mapping[str, Any],
) -> tuple[bool, str, Dict[str, Any]]:
    if trade_action == "hold":
        return False, "already_hold", {"available": False}
    if not bool(policy.get("enabled", False)):
        return False, "disabled", {"available": False}

    vals: List[float] = []
    for value in p_up_components.values():
        try:
            vals.append(float(value))
        except Exception:
            continue
    if len(vals) < int(policy.get("min_component_count", 3)):
        return False, "insufficient_components", {"available": False, "component_count": len(vals)}

    settings = _resolve_uncertainty_settings(policy, horizon=horizon, regime_state=regime_state)
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
    if float(horizon).is_integer():
        return f"{int(round(horizon))}h"
    return f"{horizon:g}h"


def _load_target_range_model(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:  # pragma: no cover - corrupted artifact guard
        print(f"Warning: failed to load target-range model at {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, Mapping) or "model" not in payload:
        print(f"Warning: malformed target-range payload at {path}; skipping.", file=sys.stderr)
        return None
    feature_names = payload.get("feature_names") or []
    payload = dict(payload)
    payload["feature_names"] = [str(name) for name in feature_names]
    metrics = payload.get("metrics") or {}
    payload["metrics"] = {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    return payload


def _load_target_range_models(
    policy: Mapping[str, Any] | None,
    horizons: Sequence[float],
) -> Dict[float, Dict[str, Any]]:
    if not policy or not policy.get("enabled"):
        return {}

    model_dir: Path = policy.get("model_dir")  # type: ignore[assignment]
    model_dir = model_dir or TARGET_RANGE_MODEL_DIR
    target_horizons = policy.get("horizons") or horizons

    bundles: Dict[float, Dict[str, Any]] = {}
    for horizon in horizons:
        if float(horizon) not in {float(h) for h in target_horizons}:
            continue
        label = _target_range_label(horizon)
        high_path = model_dir / f"{label}_high.joblib"
        low_path = model_dir / f"{label}_low.joblib"
        high_payload = _load_target_range_model(high_path)
        low_payload = _load_target_range_model(low_path)
        if not high_payload or not low_payload:
            missing = []
            if not high_payload:
                missing.append(high_path.name)
            if not low_payload:
                missing.append(low_path.name)
            print(
                f"Warning: skipping target-range models for {label} horizon (missing {', '.join(missing)}).",
                file=sys.stderr,
            )
            continue
        bundles[horizon] = {
            "high": high_payload,
            "low": low_payload,
        }
    return bundles


def _predict_single_target_model(payload: Mapping[str, Any], row: pd.Series) -> float:
    model = payload.get("model")
    feature_names: Sequence[str] = payload.get("feature_names") or []
    if not feature_names:
        raise RuntimeError("Target-range model payload missing feature_names for inference")
    values = [float(row.get(name, 0.0)) for name in feature_names]
    vector = np.asarray(values, dtype=float).reshape(1, -1)
    prediction = model.predict(vector)
    return float(prediction[0])


def _confidence_from_rmse(rmse: float | None, scale: float) -> float:
    if rmse is None:
        return 0.0
    return max(0.0, min(1.0, math.exp(-rmse / max(scale, 1e-6))))


def _predict_target_range_prices(
    bundle: Mapping[str, Any],
    row: pd.Series,
    *,
    close: float,
    confidence_scale: float,
) -> Dict[str, float]:
    high_payload = bundle.get("high")
    low_payload = bundle.get("low")
    if not high_payload or not low_payload:
        raise RuntimeError("Incomplete target-range bundle supplied for inference")

    high_ret = _predict_single_target_model(high_payload, row)
    low_ret = _predict_single_target_model(low_payload, row)
    projected_high = close * math.exp(high_ret)
    projected_low = close * math.exp(low_ret)

    rmse_high = high_payload.get("metrics", {}).get("val_rmse")
    rmse_low = low_payload.get("metrics", {}).get("val_rmse")
    return {
        "projected_high": projected_high,
        "projected_low": projected_low,
        "projected_high_confidence": _confidence_from_rmse(rmse_high, confidence_scale),
        "projected_low_confidence": _confidence_from_rmse(rmse_low, confidence_scale),
        "projected_high_rmse": _finite_float_or_none(rmse_high),
        "projected_low_rmse": _finite_float_or_none(rmse_low),
        "projected_high_residual_std": _finite_float_or_none(high_payload.get("metrics", {}).get("val_residual_std")),
        "projected_low_residual_std": _finite_float_or_none(low_payload.get("metrics", {}).get("val_residual_std")),
    }


def _apply_target_range_overrides(
    stop_loss: float,
    take_profit: float,
    projection: Mapping[str, float],
    override_ratio: float,
    direction: int,
) -> tuple[Dict[str, Dict[str, float] | None], float, float]:
    overrides = {
        "stop_loss": None,
        "take_profit": None,
    }
    updated_stop = stop_loss
    updated_take = take_profit

    projected_high = projection.get("projected_high")
    projected_low = projection.get("projected_low")

    if projected_high is not None and direction >= 1:
        if projected_high >= take_profit * (1.0 + override_ratio):
            overrides["take_profit"] = {
                "previous": take_profit,
                "updated": projected_high,
                "reason": "target_range_high",
            }
            updated_take = projected_high
    elif projected_low is not None and direction <= 0:
        if projected_low <= take_profit * (1.0 - override_ratio):
            overrides["take_profit"] = {
                "previous": take_profit,
                "updated": projected_low,
                "reason": "target_range_low",
            }
            updated_take = projected_low

    if projected_low is not None and direction >= 1:
        if projected_low <= stop_loss * (1.0 - override_ratio):
            overrides["stop_loss"] = {
                "previous": stop_loss,
                "updated": projected_low,
                "reason": "target_range_low",
            }
            updated_stop = projected_low
    elif projected_high is not None and direction <= 0:
        if projected_high >= stop_loss * (1.0 + override_ratio):
            overrides["stop_loss"] = {
                "previous": stop_loss,
                "updated": projected_high,
                "reason": "target_range_high",
            }
            updated_stop = projected_high

    return overrides, updated_stop, updated_take


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
    if policy is None:
        return _inactive_direction_fallback("not_configured"), False
    size_factor = float(policy.get("size_factor", 0.0))
    if not policy.get("enabled", True):
        return _inactive_direction_fallback("disabled", size_factor=size_factor), False

    side = "long" if int(signal_dir_only or 0) == 1 else "short"
    side_prob = p_up if side == "long" else 1.0 - p_up
    threshold = float(policy.get("prob_threshold", 0.5))
    if side_prob < threshold:
        return _inactive_direction_fallback("insufficient_probability", side=side, size_factor=size_factor), False

    if expected_value >= 0.0:
        return _inactive_direction_fallback("non_negative_ev", side=side, size_factor=size_factor), False

    limit = float(policy.get("max_negative_ev", 0.0))
    allowed_negative = limit
    ignition_extension_reason = False
    ignition_extension = float(policy.get("ignition_ev_extension", 0.0))
    if ignition_extension and trend_threshold is not None:
        if trend_prob >= trend_threshold:
            allowed_negative += ignition_extension
            ignition_extension_reason = True

    if expected_value < -allowed_negative:
        reason = "ev_below_band_ignition_extension" if ignition_extension_reason else "ev_below_band"
        return _inactive_direction_fallback(reason, side=side, size_factor=size_factor), False

    cooldown_hours = float(policy.get("cooldown_hours", 0.0))
    last_ts = policy.get("last_trigger_ts")
    cooldown_active = False
    if cooldown_hours > 0 and isinstance(last_ts, str) and last_ts.strip():
        try:
            elapsed = (
                _parse_iso_timestamp(signal_ts) - _parse_iso_timestamp(last_ts)
            ).total_seconds() / 3600.0
            if elapsed < cooldown_hours:
                cooldown_active = True
        except ValueError:
            cooldown_active = False
    if cooldown_active:
        return _inactive_direction_fallback(
            "cooldown_active",
            side=side,
            cooldown_active=True,
            size_factor=size_factor,
        ), False

    ratio = max(float(policy.get("stop_take_ratio", 0.0)), 0.0)
    projected = float(projected_price)
    if ratio == 0.0:
        stop_loss = projected
        take_profit = projected
    else:
        if side == "long":
            stop_loss = projected * (1.0 - ratio)
            take_profit = projected * (1.0 + ratio)
        else:
            stop_loss = projected * (1.0 + ratio)
            take_profit = projected * (1.0 - ratio)

    reason = "ev_within_band_ignition_extension" if ignition_extension_reason else "ev_within_band"
    payload = {
        "active": True,
        "side": side,
        "size_factor": size_factor,
        "stop_loss_fallback": stop_loss,
        "take_profit_fallback": take_profit,
        "reason": reason,
        "cooldown_active": False,
    }
    policy["last_trigger_ts"] = signal_ts
    return payload, True


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
    tidy = pd.read_parquet(path)
    wide = tidy.pivot(index="ts", columns="metric", values="value").reset_index()
    rename_map = {
        "spot_open": "open",
        "spot_high": "high",
        "spot_low": "low",
        "spot_close": "close",
        "spot_volume": "volume",
        "spot_quote_volume": "quote_volume",
        "spot_num_trades": "num_trades",
        "spot_taker_buy_base_volume": "taker_buy_base_volume",
        "spot_taker_buy_quote_volume": "taker_buy_quote_volume",
    }
    wide = wide.rename(columns=rename_map)
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True, errors="coerce")
    wide = wide.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return wide


def _compute_intrabar_features_from_15m(path_15m_tidy: Path) -> pd.DataFrame:
    frame = _pivot_tidy_spot_ohlcv(path_15m_tidy)
    required = {"open", "high", "low", "close", "volume"}
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise RuntimeError(f"15m frame missing columns required for intrabar aggregation: {missing}")

    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for optional_col in ("num_trades", "taker_buy_base_volume"):
        if optional_col not in frame.columns:
            frame[optional_col] = 0.0
        frame[optional_col] = pd.to_numeric(frame[optional_col], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).copy()

    frame["hour_ts"] = frame["ts"].dt.ceil("h")
    frame["bar_ret"] = frame["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    frame["is_up_bar"] = (frame["close"] > frame["open"]).astype(float)
    frame["range"] = (frame["high"] - frame["low"]).abs()
    frame["body"] = (frame["close"] - frame["open"]).abs()
    wick = (frame["range"] - frame["body"]).clip(lower=0.0)
    frame["wick_ratio"] = wick / frame["range"].replace(0.0, np.nan)

    grouped = frame.groupby("hour_ts", as_index=False).agg(
        intrabar_ret_std_15m_1h=("bar_ret", "std"),
        intrabar_up_bar_ratio_15m_1h=("is_up_bar", "mean"),
        intrabar_wick_ratio_15m_1h=("wick_ratio", "mean"),
        intrabar_sum_range_15m_1h=("range", "sum"),
        intrabar_sum_volume_15m_1h=("volume", "sum"),
        intrabar_mean_trade_count_15m_1h=("num_trades", "mean"),
        intrabar_taker_buy_base_sum_15m_1h=("taker_buy_base_volume", "sum"),
        intrabar_open_first_15m_1h=("open", "first"),
        intrabar_close_last_15m_1h=("close", "last"),
        intrabar_high_max_15m_1h=("high", "max"),
        intrabar_low_min_15m_1h=("low", "min"),
    )

    grouped["intrabar_trend_strength_15m_1h"] = (
        (grouped["intrabar_close_last_15m_1h"] - grouped["intrabar_open_first_15m_1h"]).abs()
        /
        (grouped["intrabar_high_max_15m_1h"] - grouped["intrabar_low_min_15m_1h"]).replace(0.0, np.nan)
    )
    grouped["intrabar_range_ratio_15m_1h"] = (
        grouped["intrabar_sum_range_15m_1h"]
        /
        grouped["intrabar_close_last_15m_1h"].replace(0.0, np.nan)
    )
    grouped["intrabar_taker_buy_ratio_15m_1h"] = (
        grouped["intrabar_taker_buy_base_sum_15m_1h"]
        /
        grouped["intrabar_sum_volume_15m_1h"].replace(0.0, np.nan)
    )

    grouped = grouped.rename(columns={"hour_ts": "ts"})
    grouped = grouped.drop(
        columns=[
            "intrabar_open_first_15m_1h",
            "intrabar_close_last_15m_1h",
            "intrabar_high_max_15m_1h",
            "intrabar_low_min_15m_1h",
        ],
        errors="ignore",
    )
    return grouped.sort_values("ts").reset_index(drop=True)


def _build_ohlcv_frame_from_tidy(df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "spot_open": "open",
        "spot_high": "high",
        "spot_low": "low",
        "spot_close": "close",
        "spot_volume": "volume",
    }
    subset = df[df["metric"].isin(metric_map.keys())].copy()
    if subset.empty:
        raise DataQualityError("No OHLCV metrics found in ingestion output")
    subset["metric"] = subset["metric"].map(metric_map)
    ohlcv = subset.pivot(index="ts", columns="metric", values="value").reset_index()
    return ohlcv


def _write_data_quality_payload(payload: Mapping[str, Any]) -> None:
    DATA_QUALITY_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_MONITOR_PATH.write_text(json.dumps(payload, indent=2))


def _evaluate_data_quality(
    frame: pd.DataFrame,
    policy_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    policy_values = _resolve_data_quality_policy(policy_config)
    policy = DataQualityPolicy(
        max_staleness_hours=float(policy_values["max_staleness_hours"]),
        max_missing_ratio=float(policy_values["max_missing_ratio"]),
        max_zero_volume_ratio=float(policy_values["max_zero_volume_ratio"]),
        min_rows=int(policy_values["min_rows"]),
    )
    payload: Dict[str, Any] = {
        "ok": True,
        "policy": {
            "enabled": bool(policy_values["enabled"]),
            "max_staleness_hours": policy.max_staleness_hours,
            "max_missing_ratio": policy.max_missing_ratio,
            "max_zero_volume_ratio": policy.max_zero_volume_ratio,
            "min_rows": policy.min_rows,
        },
    }
    try:
        payload.update(evaluate_ohlcv_quality(frame, policy))
    except DataQualityError as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["row_count"] = int(len(frame))
    _write_data_quality_payload(payload)
    return payload


def run_feature_builders(price_source: Path | None = None) -> Dict[str, str]:
    results: Dict[str, str] = {}
    print("Recomputing technical indicator features...")
    technical_path = process_technical_features(price_source=price_source, include_history=True)
    results["technical"] = str(technical_path)
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
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} override not found at {resolved}")

    ext = resolved.suffix.lower()
    if ext in {".csv", ".tsv"}:
        df = pd.read_csv(resolved)
    else:
        try:
            df = pd.read_parquet(resolved)
        except Exception as exc:  # pragma: no cover - pyarrow missing or invalid parquet
            raise RuntimeError(f"Failed to read {label} override at {resolved}: {exc}") from exc

    if "ts" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError(f"{label} override at {resolved} must include a 'ts' column")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].dt.floor("h")
    df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return df


def _summarize_frame(df: pd.DataFrame, label: str, path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": str(Path(path).expanduser()),
        "row_count": int(len(df)),
        "columns": int(len(df.columns)),
        "label": label,
    }
    if not df.empty and "ts" in df.columns:
        summary["ts_start"] = str(df["ts"].min().isoformat())
        summary["ts_end"] = str(df["ts"].max().isoformat())
    return summary


def _merge_override_features(base: pd.DataFrame, extra: pd.DataFrame, label: str) -> tuple[pd.DataFrame, List[str]]:
    if extra.empty:
        print(f"Override '{label}' is empty; skipping merge.")
        return base, []

    columns_before = set(base.columns)
    merged = pd.merge_asof(
        base.sort_values("ts"),
        extra.sort_values("ts"),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("ts").reset_index(drop=True)
    new_columns = [col for col in merged.columns if col not in columns_before]
    if new_columns:
        preview = ", ".join(new_columns[:5])
        suffix = "..." if len(new_columns) > 5 else ""
        print(f"Merged {len(new_columns)} '{label}' columns: {preview}{suffix}")
    else:
        print(f"Override '{label}' did not contribute new columns; check schema overlap.")
    return merged, new_columns


def _load_training_feature_names() -> List[str] | None:
    dataset_path = DATASET_MULTI_PATH if DATASET_MULTI_PATH.exists() else DATASET_1H_PATH
    if not dataset_path.exists():
        print(
            "Warning: dataset NPZ missing; falling back to local feature column order.",
            file=sys.stderr,
        )
        return None

    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        if "feature_names" not in dataset_npz.files:
            print(
                f"Warning: {dataset_path} missing feature_names; using local column order.",
                file=sys.stderr,
            )
            return None
        data = dataset_npz["feature_names"].tolist()

    feature_names = [str(name) for name in data]
    return feature_names


def _enrich_local_features_for_model(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    required = set(required_columns)
    if not required:
        return frame, []

    enriched = frame.copy()
    added: List[str] = []

    def _record_added(column: str) -> None:
        if column in required and column not in added:
            added.append(column)

    def _add_numeric_column(column: str, values: pd.Series) -> None:
        if column not in required or column in enriched.columns:
            return
        series = pd.to_numeric(values, errors="coerce")
        enriched[column] = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        _record_added(column)

    close = pd.to_numeric(enriched["close"], errors="coerce") if "close" in enriched.columns else None
    volume = pd.to_numeric(enriched["volume"], errors="coerce") if "volume" in enriched.columns else None

    if close is not None:
        ma_7 = close.rolling(window=7, min_periods=3).mean()
        ma_24 = close.rolling(window=24, min_periods=6).mean()
        _add_numeric_column("ma_close_7h", ma_7)
        _add_numeric_column("ma_close_24h", ma_24)
        if "ma_ratio_7_24" in required and "ma_ratio_7_24" not in enriched.columns:
            denom = ma_24.replace(0.0, np.nan)
            ratio = (ma_7 / denom).replace([np.inf, -np.inf], np.nan)
            _add_numeric_column("ma_ratio_7_24", ratio)

        if "vol_24h" in required and "vol_24h" not in enriched.columns:
            vol_24h = close.rolling(window=24, min_periods=6).std(ddof=0)
            _add_numeric_column("vol_24h", vol_24h)

        _add_numeric_column("close_delta_1h", close.diff())
        _add_numeric_column("close_pct_change_1h", close.pct_change())

        if "close_zscore_7h" in required and "close_zscore_7h" not in enriched.columns:
            std_7 = close.rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
            if "ma_close_7h" in enriched.columns:
                z_7 = (close - pd.to_numeric(enriched["ma_close_7h"], errors="coerce")) / std_7
                _add_numeric_column("close_zscore_7h", z_7)
        if "close_zscore_24h" in required and "close_zscore_24h" not in enriched.columns:
            std_24 = close.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
            if "ma_close_24h" in enriched.columns:
                z_24 = (close - pd.to_numeric(enriched["ma_close_24h"], errors="coerce")) / std_24
                _add_numeric_column("close_zscore_24h", z_24)

        ret_columns = [
            column
            for column in required
            if column.startswith("ret_")
            and column.endswith("h")
            and column not in enriched.columns
            and column not in {"ret_max_4h", "ret_min_4h", "ret_max_8h", "ret_min_8h", "ret_max_12h", "ret_min_12h"}
        ]
        for column in sorted(ret_columns):
            horizon_raw = column[4:-1]
            try:
                periods = int(round(float(horizon_raw)))
            except ValueError:
                continue
            if periods <= 0:
                continue
            _add_numeric_column(column, close.pct_change(periods=periods))

        required_volatility = {
            "volatility_realized_24h",
            "volatility_realized_72h",
            "volatility_ewm_24h",
            "volatility_ewm_72h",
            "volatility_garch_like",
        }
        if any(column in required and column not in enriched.columns for column in required_volatility):
            enriched, computed_volatility = add_volatility_columns(
                enriched,
                realized_windows=DEFAULT_REALIZED_WINDOWS,
            )
            for column in computed_volatility:
                _record_added(column)

    if volume is not None:
        _add_numeric_column("volume_delta_1h", volume.diff())
        _add_numeric_column("volume_pct_change_1h", volume.pct_change())

    taker_buy_base = None
    taker_buy_quote = None
    if "taker_buy_base_volume" in enriched.columns:
        taker_buy_base = pd.to_numeric(enriched["taker_buy_base_volume"], errors="coerce")
    elif volume is not None:
        taker_buy_base = volume * 0.5
    if taker_buy_base is not None:
        _add_numeric_column("taker_buy_base_volume", taker_buy_base)

    if "taker_buy_quote_volume" in enriched.columns:
        taker_buy_quote = pd.to_numeric(enriched["taker_buy_quote_volume"], errors="coerce")
    elif taker_buy_base is not None and close is not None:
        taker_buy_quote = taker_buy_base * close
    if taker_buy_quote is not None:
        _add_numeric_column("taker_buy_quote_volume", taker_buy_quote)

    if taker_buy_base is not None and volume is not None:
        taker_sell = (volume - taker_buy_base).clip(lower=0.0)
        cvd_raw = taker_buy_base - taker_sell
        cvd_window = cvd_raw.rolling(window=6, min_periods=2).sum()
        vol_window = volume.rolling(window=6, min_periods=2).sum().replace(0.0, np.nan)
        cvd_ratio = (cvd_window / vol_window).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0)
        _add_numeric_column("cvd_ratio_6h", cvd_ratio)
        cvd_mean = cvd_window.rolling(window=24, min_periods=6).mean()
        cvd_std = cvd_window.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
        cvd_zscore = ((cvd_window - cvd_mean) / cvd_std).replace([np.inf, -np.inf], np.nan).clip(lower=-10.0, upper=10.0)
        _add_numeric_column("cvd_zscore_6h", cvd_zscore)

    if "funding_rate_zscore_24h" in required and "funding_rate_zscore_24h" not in enriched.columns:
        funding_rate = pd.to_numeric(enriched["funding_rate"], errors="coerce") if "funding_rate" in enriched.columns else None
        if funding_rate is None:
            funding_zscore = pd.Series(0.0, index=enriched.index)
        else:
            mean_24 = funding_rate.rolling(window=24, min_periods=6).mean()
            std_24 = funding_rate.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
            funding_zscore = (funding_rate - mean_24) / std_24
        _add_numeric_column("funding_rate_zscore_24h", funding_zscore)

    if "trend_ignition_6h" in required and "trend_ignition_6h" not in enriched.columns and close is not None:
        momentum = close.pct_change().fillna(0.0)
        ignition_score = (momentum > 0.0).astype(float).rolling(window=6, min_periods=1).mean()
        _add_numeric_column("trend_ignition_6h", ignition_score)

    if {"high", "low", "close"}.issubset(enriched.columns):
        high = pd.to_numeric(enriched["high"], errors="coerce")
        low = pd.to_numeric(enriched["low"], errors="coerce")
        close_for_liquidity = pd.to_numeric(enriched["close"], errors="coerce")
        prev_close = close_for_liquidity.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        atr_6h = true_range.rolling(window=6, min_periods=2).mean().replace(0.0, np.nan)
        range_span = (high - low).abs()
        liquidity_ratio = (range_span / atr_6h).replace([np.inf, -np.inf], np.nan)
        _add_numeric_column("liquidity_range_ratio_6h", liquidity_ratio.clip(lower=0.0, upper=10.0))

        mid_price = (high + low) / 2.0
        half_range = (high - low).replace(0.0, np.nan) / 2.0
        close_position = ((close_for_liquidity - mid_price) / half_range).replace([np.inf, -np.inf], np.nan)
        _add_numeric_column("liquidity_close_position_ratio", close_position.clip(lower=-1.0, upper=1.0))

        atr_24h = true_range.rolling(window=24, min_periods=6).mean().replace(0.0, np.nan)
        _add_numeric_column("range_expansion_1h", (true_range / atr_24h).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=10.0))

        session_high = high.rolling(window=8, min_periods=2).max().replace(0.0, np.nan)
        session_low = low.rolling(window=8, min_periods=2).min().replace(0.0, np.nan)
        _add_numeric_column(
            "distance_from_session_high_8h",
            ((close_for_liquidity / session_high) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
        )
        _add_numeric_column(
            "distance_from_session_low_8h",
            ((close_for_liquidity / session_low) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
        )

        if volume is not None:
            typical_price = (high + low + close_for_liquidity) / 3.0
            rolling_notional = (typical_price * volume).rolling(window=8, min_periods=2).sum()
            rolling_volume = volume.rolling(window=8, min_periods=2).sum().replace(0.0, np.nan)
            rolling_vwap = (rolling_notional / rolling_volume).replace([np.inf, -np.inf], np.nan)
            _add_numeric_column(
                "vwap_deviation_8h",
                ((close_for_liquidity / rolling_vwap) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
            )

        _add_numeric_column("momentum_slope_2h", close_for_liquidity.pct_change(periods=2))
        _add_numeric_column("momentum_slope_4h", close_for_liquidity.pct_change(periods=4))

    return enriched, added


def _prepare_local_feature_bundle(
    *,
    features_path: str,
    hours: int,
    optional_sources: Mapping[str, str] | None = None,
) -> tuple[tuple[PreparedData, int, float, str], Dict[str, Any]]:
    base_df = _read_timeseries_frame(features_path, "features")
    metadata: Dict[str, Any] = {
        "features": _summarize_frame(base_df, "features", features_path),
    }

    if optional_sources:
        for label, path in optional_sources.items():
            try:
                frame = _read_timeseries_frame(path, label)
            except Exception as exc:
                print(f"Warning: failed to load local override '{label}' at {path}: {exc}", file=sys.stderr)
                continue
            base_df, added_columns = _merge_override_features(base_df, frame, label)
            summary = _summarize_frame(frame, label, path)
            summary["added_columns"] = added_columns
            required = LOCAL_FEATURE_REQUIRED_COLUMNS.get(label, tuple())
            if required:
                missing = [col for col in required if col not in base_df.columns]
                summary["required_columns"] = list(required)
                summary["missing_required_columns"] = missing
                if missing:
                    print(
                        f"Warning: override '{label}' missing columns {missing}; breakout features may stay zeroed.",
                        file=sys.stderr,
                    )
            metadata[label] = summary

    feature_ts_end = None
    if not base_df.empty and "ts" in base_df.columns:
        feature_ts_end = pd.to_datetime(base_df["ts"], utc=True, errors="coerce").max()
    source_freshness: Dict[str, Any] = {}
    if feature_ts_end is not None:
        for label, payload in metadata.items():
            if not isinstance(payload, Mapping):
                continue
            ts_end_raw = payload.get("ts_end")
            if not isinstance(ts_end_raw, str) or not ts_end_raw.strip():
                continue
            try:
                source_ts_end = pd.to_datetime(ts_end_raw, utc=True)
            except Exception:
                continue
            lag_hours = max((feature_ts_end - source_ts_end).total_seconds() / 3600.0, 0.0)
            source_freshness[str(label)] = {
                "ts_end": source_ts_end.isoformat(),
                "lag_hours": float(lag_hours),
            }

    if hours > 0 and len(base_df) > hours:
        base_df = base_df.iloc[-hours:].reset_index(drop=True)

    feature_names = _load_training_feature_names()
    if feature_names:
        supplemental_feature_names = [
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "cvd_ratio_6h",
            "cvd_zscore_6h",
            "funding_rate_zscore_24h",
            "trend_ignition_6h",
            "range_expansion_1h",
            "distance_from_session_high_8h",
            "distance_from_session_low_8h",
            "vwap_deviation_8h",
            "momentum_slope_2h",
            "momentum_slope_4h",
        ]
        for column in supplemental_feature_names:
            if column not in feature_names:
                feature_names.append(column)

        base_df, synthesized_columns = _enrich_local_features_for_model(
            base_df,
            required_columns=feature_names,
        )
        missing = [col for col in feature_names if col not in base_df.columns]
        if missing:
            unresolved_futures = [
                col
                for col in missing
                if col.startswith("fut_") or col in {"funding_rate", "funding_rate_annualized", "open_interest"}
            ]
            print(
                "Warning: local feature alignment still missing "
                f"{len(missing)} model columns after synthesizing {len(synthesized_columns)} columns; "
                f"imputing zeros ({len(unresolved_futures)} futures/funding/open-interest columns).",
                file=sys.stderr,
            )
            for column in missing:
                base_df[column] = 0.0
        elif synthesized_columns:
            print(
                f"Info: synthesized {len(synthesized_columns)} local model columns from OHLCV context.",
            )

        # Stabilize sparse merged features before signal preparation.
        numeric_features = base_df[feature_names].apply(pd.to_numeric, errors="coerce")
        base_df[feature_names] = numeric_features.ffill().bfill().fillna(0.0)

        metadata["feature_alignment"] = {
            "required_columns": len(feature_names),
            "synthesized_columns": synthesized_columns,
            "imputed_zero_columns": missing,
        }
    else:
        feature_names = [col for col in base_df.columns if col != "ts"]

    metadata["source_freshness"] = source_freshness

    prepared = prepare_data_for_signals_from_ohlcv(
        base_df,
        feature_names=feature_names,
        train_frac=0.7,
    )

    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Local feature overrides produced an empty dataframe.")

    if "close" not in prepared.df_all.columns:
        raise RuntimeError("Local feature overrides must include a 'close' column for predictions.")

    close = float(prepared.df_all["close"].iloc[index])
    ts_value = prepared.df_all["ts"].iloc[index]
    ts_iso = format_ts_iso(ts_value)

    return (prepared, index, close, ts_iso), metadata


def _model_suffix_candidates(horizon: float) -> List[str]:
    normalized = _normalize_horizon_value(horizon)
    candidates: List[str] = []
    if normalized < 1.0:
        minutes = int(round(normalized * 60))
        candidates.append(f"{minutes}m")
        candidates.append(f"{normalized:g}h")
    else:
        if float(normalized).is_integer():
            candidates.append(f"{int(normalized)}h")
        else:
            candidates.append(f"{normalized:g}h")

    if normalized < 1.0 and "1h" not in candidates:
        candidates.append("1h")

    unique: List[str] = []
    for suffix in candidates:
        if suffix not in unique:
            unique.append(suffix)
    return unique


def _model_paths_for_horizon(horizon: float) -> tuple[Path, Path]:
    suffixes = _model_suffix_candidates(horizon)
    label = _format_horizon_label(horizon)
    fallback: tuple[Path, Path] | None = None

    for suffix_idx, suffix in enumerate(suffixes):
        reg_path = resolve_best_versioned_model_file(
            MODEL_ROOT / f"xgb_ret{suffix}_v1",
            expected_filename=f"xgb_ret{suffix}_model.json",
            version_priority=MODEL_VERSION_PRIORITY,
        )
        dir_path = resolve_best_versioned_model_file(
            MODEL_ROOT / f"xgb_dir{suffix}_v1",
            expected_filename=f"xgb_dir{suffix}_model.json",
            version_priority=DIR_VERSION_OVERRIDES.get(suffix, MODEL_VERSION_PRIORITY),
        )

        if fallback is None:
            fallback = (reg_path, dir_path)

        if reg_path.exists() and dir_path.exists():
            if suffix_idx > 0:
                print(
                    f"Info: using {suffix} model artifacts for {label} horizon fallback.",
                    file=sys.stderr,
                )
            return reg_path, dir_path

    if fallback is not None and len(suffixes) > 1:
        print(
            f"Warning: dedicated model artifacts for {label} horizon are missing; using {suffixes[-1]} fallback paths.",
            file=sys.stderr,
        )
        return fallback

    if fallback is None:
        raise RuntimeError(f"Unable to resolve model paths for {label} horizon.")
    return fallback


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
    overrides = {
        "lstm": dir_lstm_path,
        "bilstm": dir_bilstm_path,
        "gru": dir_gru_path,
        "cnn_lstm": dir_cnn_lstm_path,
        "cnn_bilstm": dir_cnn_bilstm_path,
        "garch_lstm": dir_garch_lstm_path,
        "transformer": dir_transformer_path,
    }
    return resolve_direction_model_configs(
        DEFAULT_DIR_MODELS_1H,
        config_json_path=config_json_path,
        weight_spec=weight_spec,
        path_overrides=overrides,
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

    def _sequence_model_overrides() -> Dict[str, str]:
        def _explicit_transformer_path(suffix: str) -> Optional[str]:
            path = DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX.get(suffix)
            if not path:
                return None
            if path.startswith("models:/"):
                parts = path.split("/")
                model_name = parts[1] if len(parts) > 1 else ""
                if model_name and _registry_model_exists(model_name):
                    return path
                return None
            path_obj = Path(path).expanduser()
            return str(path_obj) if path_obj.exists() else None

        overrides: Dict[str, str] = {}
        suffixes = _model_suffix_candidates(horizon)
        seq_types = (
            "lstm",
            "bilstm",
            "gru",
            "cnn_lstm",
            "cnn_bilstm",
            "garch_lstm",
            "transformer",
            "transformer_large",
        )
        for model_type in seq_types:
            for suffix in suffixes:
                if model_type == "transformer":
                    explicit_path = _explicit_transformer_path(suffix)
                    if explicit_path:
                        overrides[model_type] = explicit_path
                        break
                prefix = f"{model_type}_dir{suffix}"
                if model_type == "transformer_large":
                    prefix = f"transformer_dir{suffix}_large"
                for version in MODEL_VERSION_PRIORITY:
                    candidate = MODEL_ROOT / f"{prefix}_{version}"
                    if candidate.exists():
                        overrides[model_type] = str(candidate)
                        break
                if model_type in overrides:
                    break
                if model_type == "transformer" and horizon >= 1.0 and suffix.endswith("h"):
                    use_registry = os.getenv("USE_MLFLOW_REGISTRY", "").lower() in {"1", "true", "yes"}
                    if use_registry:
                        model_name = f"transformer_dir{suffix}"
                        if _registry_model_exists(model_name):
                            overrides[model_type] = f"models:/{model_name}/latest"
                            break
        return overrides

    def _lgbm_model_path() -> Optional[str]:
        suffixes = _model_suffix_candidates(horizon)
        for suffix in suffixes:
            for version in MODEL_VERSION_PRIORITY:
                model_dir = MODEL_ROOT / f"lgbm_dir{suffix}_{version}"
                model_path = model_dir / f"lgbm_dir{suffix}_model.joblib"
                if model_path.exists():
                    return str(model_path)
        return None

    configs = clone_direction_model_configs(base_configs)
    overrides = {"xgb": dir_model_path}
    overrides.update(_sequence_model_overrides())
    apply_path_overrides(configs, overrides)
    lgbm_path = _lgbm_model_path()
    if lgbm_path and not any(entry.get("type") == "lgbm" for entry in configs):
        configs.append(
            {
                "name": "lgbm",
                "type": "lgbm",
                "path": lgbm_path,
                "weight": 1.0,
            }
        )
    log_direction_model_configs(configs, label=f"[run_refresh_and_predict] direction models ({horizon_label})")
    weight_map = direction_configs_to_weight_map(configs)
    return configs, weight_map


def _load_platt_calibration(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        print(f"Warning: Platt calibration file not found at {path_obj}; skipping.", file=sys.stderr)
        return {}
    payload = json.loads(path_obj.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Platt calibration file must contain a JSON object keyed by horizon.")
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        method = str(value.get("method", "platt")).lower()
        if method == "platt" and "a" in value and "b" in value:
            result[str(key)] = {"method": "platt", "a": float(value["a"]), "b": float(value["b"])}
            continue
        if method == "beta" and all(k in value for k in ("a", "b", "c")):
            result[str(key)] = {
                "method": "beta",
                "a": float(value["a"]),
                "b": float(value["b"]),
                "c": float(value["c"]),
            }
            continue
        if method == "isotonic" and all(k in value for k in ("x", "y")):
            x = [float(v) for v in value.get("x", [])]
            y = [float(v) for v in value.get("y", [])]
            if x and y and len(x) == len(y):
                result[str(key)] = {"method": "isotonic", "x": x, "y": y}
                continue
        # Backward-compatible platt payloads without explicit method.
        if "a" in value and "b" in value:
            result[str(key)] = {"method": "platt", "a": float(value["a"]), "b": float(value["b"])}
    return result


def _apply_probability_calibration(p: float, params: Mapping[str, Any]) -> float:
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


def _resolve_probability_calibration(
    platt_calibration: Mapping[str, Mapping[str, Any]] | None,
    label: str,
    regime_state: str,
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
        if method == "platt" and (abs(slope) < REGIME_CALIBRATION_MIN_PLATT_SLOPE or slope <= 0.0):
            base_params = platt_calibration.get(label)
            if isinstance(base_params, Mapping):
                return label, base_params, False
        return regime_key, regime_params, True
    base_params = platt_calibration.get(label)
    if isinstance(base_params, Mapping):
        return label, base_params, False
    return None, None, False


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
    calibration_key, params, calibration_used_regime_key = _resolve_probability_calibration(
        platt_calibration,
        label,
        regime_state,
    )
    probability = float(raw_probability)
    if isinstance(params, Mapping):
        probability = _apply_probability_calibration(float(raw_probability), params)

    ret_side = _direction_from_ret_pred(ret_pred)
    projected_side = _direction_from_projected_price(close, projected_price)
    raw_side = _direction_from_probability(raw_probability, neutral_band=neutral_band)
    calibrated_side = _direction_from_probability(probability, neutral_band=neutral_band)
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else None
    guard_payload: Dict[str, Any] | None = None

    if (
        consensus_side is not None
        and calibration_used_regime_key
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
                base_probability = _apply_probability_calibration(float(raw_probability), base_params)
                base_side = _direction_from_probability(base_probability, neutral_band=neutral_band)
                if base_side == consensus_side:
                    resolved_probability = float(base_probability)
                    resolved_key = label
                    fallback_source = "base_horizon_calibration"

        guard_payload = {
            "applied": True,
            "reason": "regime_calibration_conflicts_with_forecast_consensus",
            "forecast_side": consensus_side,
            "raw_side": raw_side,
            "regime_calibrated_side": calibrated_side,
            "original_applied_key": calibration_key,
            "fallback_source": fallback_source,
            "raw_probability": float(raw_probability),
            "regime_calibrated_probability": float(probability),
            "base_probability": None if base_probability is None else float(base_probability),
            "base_side": base_side,
            "resolved_probability": float(resolved_probability),
        }
        return resolved_probability, resolved_key, resolved_used_regime_key, guard_payload

    return probability, calibration_key, calibration_used_regime_key, guard_payload


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

    neutral_band = float(policy.get("neutral_band", 0.0) or 0.0)
    internal_direction = base_direction
    ret_side = _direction_from_ret_pred(ret_pred)
    projected_side = _direction_from_projected_price(close, projected_price)
    probability = float(trade_probability)
    source = "trade_probability"
    fallback_to_trade_probability = True
    calibration_key = None
    calibration_used_regime_key = False
    calibration_skipped_due_to_marginal_rerank = False
    calibration_map = policy.get("calibration_map") if isinstance(policy.get("calibration_map"), Mapping) else None
    if calibration_map:
        calibration_key, params, calibration_used_regime_key = _resolve_probability_calibration(
            calibration_map,
            label,
            regime_state,
        )
        if isinstance(params, Mapping):
            probability = _apply_probability_calibration(float(raw_probability), params)
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
    if bool(marginal_rerank_policy.get("enabled", False)) and _parse_horizon_label(label) in marginal_horizons:
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

    payload.update(
        {
            "direction": _direction_from_probability(probability, neutral_band=neutral_band),
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


def _load_prepared(dataset_path: Path, *, target_column: str, offline: bool = False) -> tuple:
    if offline:
        return _load_prepared_offline(dataset_path)

    prepared = prepare_data_for_signals(str(dataset_path), target_column=target_column)
    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Prepared dataset has no rows.")
    ts_value = prepared.df_all["ts"].iloc[index]
    close = float(prepared.df_all["close"].iloc[index])
    ts_iso = format_ts_iso(ts_value)
    return prepared, index, close, ts_iso


def _load_prepared_offline(dataset_path: Path) -> tuple[PreparedData, int, float, str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found for offline preparation: {dataset_path}")

    close_snapshot: np.ndarray | None = None
    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        if "feature_names" not in dataset_npz.files:
            raise KeyError("Dataset NPZ missing feature_names for offline preparation.")
        feature_names = dataset_npz["feature_names"].tolist()
        arrays = [dataset_npz[key] for key in ("X_train", "X_val", "X_test") if key in dataset_npz.files]
        if "close_all" in dataset_npz.files:
            close_snapshot = np.asarray(dataset_npz["close_all"], dtype=float)

    if not arrays:
        raise RuntimeError("Dataset NPZ does not contain any feature splits for offline preparation.")

    X_all = np.concatenate(arrays, axis=0)
    if X_all.size == 0:
        raise RuntimeError("Dataset NPZ is empty after concatenation; cannot build offline prepared data.")

    df_features = pd.DataFrame(X_all, columns=feature_names)
    if "close" not in df_features.columns:
        raise RuntimeError("Offline dataset must include a 'close' feature column.")

    periods = len(df_features)
    ts_index = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq="H")
    df_features.insert(0, "ts", ts_index)

    prepared = prepare_data_for_signals_from_ohlcv(
        df_features,
        feature_names=feature_names,
        train_frac=0.7,
    )

    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Offline prepared dataset has no rows.")

    ts_value = prepared.df_all["ts"].iloc[index]
    close = float(prepared.df_all["close"].iloc[index])
    if close_snapshot is not None:
        if len(close_snapshot) == len(df_features):
            close = float(close_snapshot[index])
        else:
            print(
                "Warning: close_all array length mismatch in offline dataset; "
                "falling back to scaled close values.",
                file=sys.stderr,
            )
    ts_iso = format_ts_iso(ts_value)
    return prepared, index, close, ts_iso


def _project_price(close: float, log_return: float) -> float:
    return close * math.exp(log_return)


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
    latest_close: float | None = None,
    confidence_min: float = CONFIDENCE_MIN_DEFAULT,
    position_size_floor: float = POSITION_SIZE_FLOOR_DEFAULT,
    position_size_cap: float = POSITION_SIZE_CAP_DEFAULT,
    position_size_cap_by_horizon: Mapping[float | int | str, float] | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    normalized_targets = sorted({_normalize_horizon_value(h) for h in targets})
    if not normalized_targets:
        return {}

    trend_payload = _resolve_trend_ignition_payload(trend_ignition)
    direction_fallback_policy = _resolve_direction_fallback_policy(direction_only_fallback)
    adaptive_policy = _resolve_adaptive_thresholds_policy(adaptive_thresholds)
    target_range_policy = _resolve_target_range_policy(target_range_models)
    abstention_policy_resolved = _resolve_abstention_policy(abstention_policy)
    uncertainty_policy_resolved = _resolve_uncertainty_policy(uncertainty_policy)
    trade_decision_policy_resolved = _resolve_trade_decision_policy(trade_decision_policy)
    regime_weight_policy = _resolve_regime_model_weights_policy(regime_model_weights)
    regime_model_dirs_policy = _resolve_regime_model_dirs_policy(regime_model_dirs)
    confluence_policy_resolved = _resolve_confluence_policy(confluence_policy)
    execution_policy_resolved = _resolve_execution_policy(execution_policy)
    forecast_coherence_policy_resolved = _resolve_forecast_coherence_policy(forecast_coherence_policy)
    direction_output_policy_resolved = _resolve_direction_output_policy(direction_output_policy)
    confidence_min = max(0.0, min(1.0, float(confidence_min)))
    position_size_floor = max(0.0, float(position_size_floor))
    position_size_cap = max(position_size_floor, float(position_size_cap))
    position_size_cap_by_horizon_resolved = _normalize_horizon_float_map(
        position_size_cap_by_horizon,
        minimum=position_size_floor,
        maximum=position_size_cap,
    )

    profiles: Dict[str, DatasetProfile] = {}
    target_profiles: Dict[float, str] = {}
    horizons_by_profile: Dict[str, List[float]] = defaultdict(list)
    for horizon in normalized_targets:
        profile = _dataset_profile_for_horizon(horizon)
        profiles.setdefault(profile.key, profile)
        target_profiles[horizon] = profile.key
        horizons_by_profile[profile.key].append(horizon)

    resolved_profiles: Dict[str, DatasetCandidate] = {}
    for key, profile in profiles.items():
        candidate, used_fallback = _select_dataset_candidate(profile)
        resolved_profiles[key] = candidate
        if used_fallback:
            print(
                f"Info: using {candidate.path.name} for {key} horizon group (fallback dataset).",
                file=sys.stderr,
            )

    base_direction_configs = _prepare_base_direction_configs(
        config_json_path=dir_model_config_json,
        weight_spec=dir_model_weights,
        dir_lstm_path=dir_lstm_path,
        dir_bilstm_path=dir_bilstm_path,
        dir_gru_path=dir_gru_path,
        dir_cnn_lstm_path=dir_cnn_lstm_path,
        dir_cnn_bilstm_path=dir_cnn_bilstm_path,
        dir_garch_lstm_path=dir_garch_lstm_path,
        dir_transformer_path=dir_transformer_path,
    )

    prepared_bundles: Dict[str, tuple[PreparedData, int, float, str]] = {}
    volatility_snapshots: Dict[str, Dict[str, Any]] = {}
    stub_close = 0.0
    stub_ts = datetime.now(timezone.utc).isoformat()
    for key, candidate in resolved_profiles.items():
        if prepared_override is not None and not candidate.offline_only:
            bundle = prepared_override
        else:
            dataset_path = candidate.path
            if not dataset_path.exists():
                if offline:
                    print(
                        f"Dry run: dataset not found for {key} group (expected {dataset_path}).",
                        file=sys.stderr,
                    )
                    continue
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            bundle = _load_prepared(
                dataset_path,
                target_column=candidate.target_column,
                offline=offline or candidate.offline_only,
            )
        prepared_bundles[key] = bundle
        prepared, index, close_snapshot, ts_snapshot = bundle
        stub_close = close_snapshot
        stub_ts = ts_snapshot
        volatility_snapshots[key] = latest_volatility_snapshot(
            prepared.df_all,
            prepared.volatility_columns or [],
            index=index,
        )

    breakout_scores: Dict[str, float] = {}
    if adaptive_policy and adaptive_policy.get("enabled"):
        breakout_scores = _compute_breakout_scores(prepared_bundles, volatility_snapshots)

    target_range_bundles: Dict[float, Dict[str, Any]] = {}
    if target_range_policy and target_range_policy.get("enabled"):
        target_range_bundles = _load_target_range_models(target_range_policy, normalized_targets)

    residual_std_by_horizon: Dict[float, float] = {}
    for key, candidate in resolved_profiles.items():
        horizons = horizons_by_profile.get(key, [])
        if not horizons:
            continue
        dataset_path = candidate.path
        if not dataset_path.exists():
            continue
        try:
            residuals = load_residual_std_from_dataset(
                str(dataset_path),
                horizons,
                base_horizon=candidate.base_horizon,
            )
            residual_std_by_horizon.update(residuals)
        except FileNotFoundError:
            print(
                f"Warning: residual std dataset missing at {dataset_path}; using default {DEFAULT_RESIDUAL_STD:.4f}.",
                file=sys.stderr,
            )
            for horizon in horizons:
                residual_std_by_horizon[horizon] = DEFAULT_RESIDUAL_STD

    summary: Dict[str, Dict[str, float | str | int]] = {}
    execution_contexts: Dict[str, Dict[str, Any]] = {}
    pending_trend_ts: Optional[str] = None
    pending_direction_fallback_ts: Optional[str] = None
    for horizon in normalized_targets:
        profile_key = target_profiles.get(horizon)
        if profile_key is None:
            continue
        if profile_key not in prepared_bundles:
            print(
                f"Warning: skipping {_format_horizon_label(horizon)} horizon because prepared data is missing.",
                file=sys.stderr,
            )
            continue
        candidate = resolved_profiles[profile_key]
        prepared, index, close, ts_iso = prepared_bundles[profile_key]
        volatility_snapshot = volatility_snapshots.get(profile_key, {})
        row_features = prepared.df_all.iloc[index]
        label = _format_horizon_label(horizon)
        reg_path, dir_path_default = _model_paths_for_horizon(horizon)
        if not reg_path.exists() or not dir_path_default.exists():
            print(
                f"Warning: skipping {label} horizon because model files are missing",
                file=sys.stderr,
            )
            continue

        regime_state = REGIME_NEUTRAL
        regime_score = None
        adaptive_scale = 1.0
        if adaptive_policy and adaptive_policy.get("enabled"):
            profile_score = breakout_scores.get(profile_key)
            if profile_score is not None:
                regime_score = profile_score
                regime_state = _classify_regime_from_score(profile_score, adaptive_policy)

        dir_path = _resolve_regime_specific_dir_path(
            dir_path_default,
            regime_state=regime_state,
            horizon_label=label,
            policy=regime_model_dirs_policy,
        )

        direction_configs, base_dir_weight_map = _direction_configs_for_horizon(
            base_direction_configs,
            dir_model_path=str(dir_path),
            horizon=horizon,
            horizon_label=label,
        )
        models = load_models(
            str(reg_path),
            direction_model_configs=direction_configs,
        )
        if trend_payload:
            models["trend_ignition"] = trend_payload
        populate_sequence_cache_from_prepared(prepared, models)
        horizon_thresholds = _resolve_thresholds_for_horizon(
            horizon,
            p_up_min,
            ret_min,
            thresholds_by_horizon,
        )
        horizon_p_up = horizon_thresholds["p_up_min"]
        horizon_ret = horizon_thresholds["ret_min"]
        if adaptive_policy and adaptive_policy.get("enabled"):
            profile_score = breakout_scores.get(profile_key)
            if profile_score is not None:
                horizon_p_up, horizon_ret, adaptive_scale = _apply_adaptive_thresholds(
                    adaptive_policy,
                    horizon_p_up,
                    horizon_ret,
                    regime_state,
                )
        dir_weight_map = _apply_regime_weight_overrides(
            base_dir_weight_map,
            regime_state=regime_state,
            horizon=horizon,
            policy=regime_weight_policy,
        )

        signal = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=horizon_p_up,
            ret_min=horizon_ret,
            horizon=horizon,
            dir_model_weights=dir_weight_map,
            volatility_snapshot=volatility_snapshot,
            volatility_policy=horizon_thresholds,
            p_up_calibration=None,
        )
        # override direction-only signal using configurable threshold
        try:
            p_val = float(signal.get("p_up", 0.0))
        except Exception:
            p_val = 0.0
        thresh = _resolve_direction_threshold_for_horizon(
            direction_threshold=float(locals().get("direction_threshold", 0.5)),
            auto_direction_threshold=bool(auto_direction_threshold),
            horizon_p_up=float(horizon_p_up),
        )
        signal["signal_dir_only"] = int(p_val >= thresh)
        if latest_close is not None:
            signal['close'] = latest_close
            close = latest_close

        ret_pred = float(signal.get("ret_pred", 0.0))
        raw_p_up = float(signal.get("p_up", 0.0))
        p_up = float(raw_p_up)
        signal_ts = str(signal.get("ts", ts_iso))
        signal_dir_only = int(signal.get("signal_dir_only", 0))
        signal_ensemble = int(signal.get("signal_ensemble", 0))
        residual_std = float(residual_std_by_horizon.get(horizon, DEFAULT_RESIDUAL_STD))
        expected_value = p_up * ret_pred - (1 - p_up) * residual_std
        ev_multiplier = float(horizon_thresholds.get("expected_value_multiplier", 1.0))
        expected_value *= ev_multiplier
        confidence_score = _compute_confidence_score(p_up, expected_value, residual_std)
        calibration_key = None
        calibration_used_regime_key = False
        probability_guard = None

        # Apply optional horizon/regime calibration with a forecast-alignment guard for regime-specific flips.
        if platt_calibration:
            p_up, calibration_key, calibration_used_regime_key, probability_guard = _resolve_trade_probability_for_horizon(
                platt_calibration=platt_calibration,
                label=label,
                regime_state=regime_state,
                raw_probability=raw_p_up,
                close=close,
                projected_price=_project_price(close, ret_pred),
                ret_pred=ret_pred,
            )
            signal_dir_only = _resolve_direction_signal_for_horizon(
                raw_probability=raw_p_up,
                calibrated_probability=p_up,
                threshold=thresh,
                close=close,
                projected_price=_project_price(close, ret_pred),
                ret_pred=ret_pred,
                calibration_key=calibration_key,
                calibration_used_regime_key=calibration_used_regime_key,
            )
            signal_ensemble = int((p_up >= horizon_p_up) and (ret_pred >= horizon_ret) and (not bool(signal.get("volatility_flag"))))
            expected_value = p_up * ret_pred - (1 - p_up) * residual_std
            expected_value *= ev_multiplier
            confidence_score = _compute_confidence_score(p_up, expected_value, residual_std)

        stop_loss_price, take_profit_price = _compute_directional_stop_take_prices(
            close=close,
            ret_pred=ret_pred,
            residual_std=residual_std,
            direction_signal=signal_dir_only,
        )
        effective_position_size_cap = _lookup_horizon_value(
            position_size_cap_by_horizon_resolved,
            horizon,
            position_size_cap,
        )
        position_size = _compute_position_size(
            confidence_score,
            confidence_min=confidence_min,
            size_floor=position_size_floor,
            size_cap=effective_position_size_cap,
        )
        trend_prob = float(signal.get("p_trend_ignition", 0.0))
        ignition_state = 0
        cooldown_active = False
        if trend_payload:
            threshold_value = float(trend_payload.get("threshold", 0.6))
            cooldown_hours = float(trend_payload.get("cooldown_hours", 0.0))
            last_trigger_ts = trend_payload.get("last_trigger_ts")
            if cooldown_hours > 0 and isinstance(last_trigger_ts, str) and last_trigger_ts.strip():
                try:
                    elapsed_hours = (
                        _parse_iso_timestamp(signal_ts) - _parse_iso_timestamp(last_trigger_ts)
                    ).total_seconds() / 3600.0
                    cooldown_active = elapsed_hours < cooldown_hours
                except ValueError:
                    cooldown_active = False
            if trend_prob >= threshold_value and not cooldown_active:
                ignition_state = 1
                if pending_trend_ts is None:
                    pending_trend_ts = signal_ts
        target_projection: Dict[str, float] | None = None
        if target_range_policy and target_range_policy.get("enabled"):
            bundle = target_range_bundles.get(horizon)
            if bundle:
                try:
                    target_projection = _predict_target_range_prices(
                        bundle,
                        row_features,
                        close=close,
                        confidence_scale=float(
                            target_range_policy.get("confidence_rmse_scale", TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE)
                        ),
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    print(
                        f"Warning: failed to compute target-range projection for {label}: {exc}",
                        file=sys.stderr,
                    )

        result = {
            "timestamp": signal_ts,
            "horizon_hours": horizon,
            "close": close,
            "entry_price": close,
            "p_up": p_up,
            "p_trend_ignition": trend_prob,
            "ignition_state": ignition_state,
            "ignition_cooldown_active": cooldown_active if trend_payload else False,
            "ret_pred": ret_pred,
            "projected_price": _project_price(close, ret_pred),
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
            "position_size_cap": effective_position_size_cap,
            "p_up_components": signal.get("p_up_components", {}),
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "expected_value": expected_value,
            "thresholds": horizon_thresholds,
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
            "regime_weight_overrides": _get_active_regime_weight_override(
                regime_state=regime_state,
                horizon=horizon,
                policy=regime_weight_policy,
            ),
            "projected_high": target_projection.get("projected_high") if target_projection else None,
            "projected_low": target_projection.get("projected_low") if target_projection else None,
            "projected_high_confidence": target_projection.get("projected_high_confidence", 0.0)
            if target_projection
            else 0.0,
            "projected_low_confidence": target_projection.get("projected_low_confidence", 0.0)
            if target_projection
            else 0.0,
            "projected_high_rmse": target_projection.get("projected_high_rmse") if target_projection else None,
            "projected_low_rmse": target_projection.get("projected_low_rmse") if target_projection else None,
            "projected_high_residual_std": target_projection.get("projected_high_residual_std") if target_projection else None,
            "projected_low_residual_std": target_projection.get("projected_low_residual_std") if target_projection else None,
            "volatility": signal.get("volatility", {
                "snapshot": volatility_snapshot,
                "metric": None,
                "ceiling": horizon_thresholds.get("volatility_ceiling"),
                "triggered": False,
            }),
            "volatility_flag": bool(signal.get("volatility_flag")),
        }
        probability_alignment_features = _derive_probability_alignment_features(
            close=close,
            projected_price=float(result["projected_price"]),
            ret_pred=ret_pred,
            raw_probability=float(raw_p_up),
            resolved_probability=float(p_up),
            direction=str(result["direction_next"]),
            neutral_band=float(forecast_coherence_policy_resolved.get("p_up_neutral_band", 0.02) or 0.02),
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
        direction_output_scoped = horizon in set(direction_output_policy_resolved.get("horizons", []))
        direction_output = _build_direction_output(
            enabled=bool(direction_output_policy_resolved.get("enabled", False)),
            scoped=direction_output_scoped,
            label=label,
            regime_state=regime_state,
            signal_dir_only=signal_dir_only,
            raw_probability=raw_p_up,
            trade_probability=p_up,
            ret_pred=ret_pred,
            close=close,
            projected_price=_project_price(close, ret_pred),
            p_up_components=signal.get("p_up_components", {}),
            policy=direction_output_policy_resolved,
        )
        result["direction_output"] = direction_output
        result["direction_next_display"] = direction_output.get("direction", result["direction_next"])
        for field in (
            "range_expansion_1h",
            "distance_from_session_high_8h",
            "distance_from_session_low_8h",
            "vwap_deviation_8h",
            "momentum_slope_2h",
            "momentum_slope_4h",
        ):
            if field in row_features.index:
                value = pd.to_numeric(pd.Series([row_features.get(field)]), errors="coerce").iloc[0]
                result[field] = None if pd.isna(value) else float(value)
        if regime_score is not None:
            result["regime_score"] = regime_score
        result["thresholds"]["p_up_min_effective"] = horizon_p_up
        result["thresholds"]["ret_min_effective"] = horizon_ret
        result["thresholds"]["adaptive_scale"] = adaptive_scale

        overrides_payload = {"stop_loss": None, "take_profit": None}
        if target_projection and target_range_policy and target_range_policy.get("enabled"):
            overrides_payload, updated_stop, updated_take = _apply_target_range_overrides(
                result["stop_loss"],
                result["take_profit"],
                target_projection,
                override_ratio=float(target_range_policy.get("override_ratio", TARGET_RANGE_DEFAULT_OVERRIDE_RATIO)),
                direction=int(result["signal_dir_only"]),
            )
            result["stop_loss"] = updated_stop
            result["take_profit"] = updated_take
        result["target_range_overrides"] = overrides_payload
        trade_decision_payload = _apply_trade_decision_model(
            result=result,
            regime_state=regime_state,
            residual_std=residual_std,
            policy=trade_decision_policy_resolved,
            fee_bps=float(DEFAULT_FEE_BPS),
            slippage_bps=float(DEFAULT_SLIPPAGE_BPS),
        )
        result["trade_decision"] = trade_decision_payload
        entry_price = float(result["entry_price"])
        stop_loss = float(result["stop_loss"])
        take_profit = float(result["take_profit"])
        downside = abs(entry_price - stop_loss)
        upside = abs(take_profit - entry_price)
        result["risk_reward_ratio"] = (upside / downside) if downside > 0 else None
        fallback_info, fallback_triggered = _evaluate_direction_only_fallback(
            direction_fallback_policy,
            p_up=p_up,
            signal_dir_only=int(signal.get("signal_dir_only", 0)),
            expected_value=expected_value,
            projected_price=result["projected_price"],
            signal_ts=signal_ts,
            trend_prob=trend_prob,
            trend_threshold=float(trend_payload.get("threshold")) if trend_payload else None,
        )
        result["direction_only_fallback"] = fallback_info
        if result["trade_action"] != "hold" and confidence_score < confidence_min:
            result["trade_action"] = "hold"
            result["confidence_filter_triggered"] = True
        else:
            result["confidence_filter_triggered"] = False
        if fallback_triggered:
            pending_direction_fallback_ts = signal_ts

        abstention_expected_value, abstention_expected_value_source = _resolve_abstention_expected_value(
            expected_value,
            result.get("trade_decision") if isinstance(result.get("trade_decision"), Mapping) else None,
        )

        abstain, abstain_reason = _apply_abstention_policy(
            trade_action=str(result["trade_action"]),
            p_up=p_up,
            confidence_score=confidence_score,
            expected_value=abstention_expected_value,
            fee_bps=float(DEFAULT_FEE_BPS),
            slippage_bps=float(DEFAULT_SLIPPAGE_BPS),
            policy=abstention_policy_resolved,
        )
        result["abstention"] = {
            "enabled": bool(abstention_policy_resolved.get("enabled", False)),
            "triggered": bool(abstain),
            "reason": abstain_reason,
            "expected_value_used": float(abstention_expected_value),
            "expected_value_source": abstention_expected_value_source,
        }
        if abstain:
            result["trade_action"] = "hold"
            result["signal_ensemble"] = 0

        uncertainty_abstain, uncertainty_reason, uncertainty_payload = _apply_uncertainty_abstention(
            trade_action=str(result["trade_action"]),
            p_up_components=result.get("p_up_components", {}),
            horizon=horizon,
            regime_state=regime_state,
            policy=uncertainty_policy_resolved,
        )
        result["uncertainty"] = uncertainty_payload
        if uncertainty_abstain:
            result["trade_action"] = "hold"
            result["signal_ensemble"] = 0
            result["abstention"] = {
                "enabled": True,
                "triggered": True,
                "reason": uncertainty_reason,
            }

        summary[label] = result
        execution_contexts[label] = {
            "prepared": prepared,
            "index": index,
            "horizon": horizon,
            "residual_std": residual_std,
        }
    if trend_payload and pending_trend_ts:
        _write_trend_ignition_state(pending_trend_ts)
    if direction_fallback_policy and pending_direction_fallback_ts:
        _write_direction_fallback_state(pending_direction_fallback_ts)

    if forecast_coherence_policy_resolved.get("enabled"):
        summary = _apply_forecast_coherence_policy(summary, forecast_coherence_policy_resolved)

    if confluence_policy_resolved.get("enabled"):
        summary = _apply_confluence_policy(summary, confluence_policy_resolved)

    if execution_policy_resolved.get("enabled"):
        summary = _apply_execution_policy(summary, execution_contexts, execution_policy_resolved)

    if not summary:
        if offline:
            print("Dry run: model artifacts missing, emitting stub predictions.")
            return _build_stub_summary(
                targets,
                p_up_min,
                ret_min,
                close=stub_close,
                ts_iso=stub_ts,
                thresholds_by_horizon=thresholds_by_horizon,
            )
        raise RuntimeError("No predictions were produced; ensure model artifacts exist.")
    return summary


def write_summary(
    summary: Dict[str, Dict[str, Any]],
    *,
    degradation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    LATEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    execution_prior_summary = _build_execution_prior_summary(summary)
    prompt_ready_summary = _build_prompt_ready_summary(summary)
    blocked_trade_analytics = _build_blocked_trade_analytics(summary)
    json_payload = {
        "generated_at": generated_at,
        "predictions": summary,
        "execution_prior_summary": execution_prior_summary,
        "blocked_trade_analytics": blocked_trade_analytics,
        "prompt_ready_summary": prompt_ready_summary,
    }
    LATEST_PREDICTION_PATH.write_text(json.dumps(json_payload, indent=2))
    print(json.dumps(json_payload, indent=2))

    history_entry = {
        "generated_at": generated_at,
        "predictions": summary,
        "execution_prior_summary": execution_prior_summary,
        "blocked_trade_analytics": blocked_trade_analytics,
        "prompt_ready_summary": prompt_ready_summary,
    }
    history: List[Dict[str, object]] = []
    if HISTORY_PREDICTION_PATH.exists():
        try:
            history = json.loads(HISTORY_PREDICTION_PATH.read_text())
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    history.append(history_entry)
    HISTORY_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PREDICTION_PATH.write_text(json.dumps(history, indent=2))
    json_payload["degradation_monitoring"] = _build_degradation_monitoring(
        history,
        policy=degradation_policy,
    )
    history[-1]["degradation_monitoring"] = json_payload["degradation_monitoring"]
    HISTORY_PREDICTION_PATH.write_text(json.dumps(history, indent=2))
    LATEST_PREDICTION_PATH.write_text(json.dumps(json_payload, indent=2))
    return json_payload


def _build_blocked_trade_analytics(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    by_horizon: Dict[str, Dict[str, Any]] = {}
    blocked_total = 0
    ready_total = 0
    waiting_total = 0
    bias_only_total = 0

    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        status = str(plan.get("status") or "unknown")
        reason = str(plan.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if status == "ready":
            ready_total += 1
        elif status == "waiting_pullback":
            waiting_total += 1
        elif status == "bias_only_ready":
            bias_only_total += 1
            blocked_total += 1
        elif status == "rejected":
            blocked_total += 1

        horizon_payload = by_horizon.setdefault(
            label,
            {"status_counts": {}, "reason_counts": {}, "trade_action": str(entry.get("trade_action") or "hold")},
        )
        horizon_payload["status_counts"][status] = horizon_payload["status_counts"].get(status, 0) + 1
        horizon_payload["reason_counts"][reason] = horizon_payload["reason_counts"].get(reason, 0) + 1

    return {
        "total_horizons": len(summary),
        "ready_total": ready_total,
        "waiting_pullback_total": waiting_total,
        "bias_only_total": bias_only_total,
        "blocked_total": blocked_total,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "by_horizon": by_horizon,
    }


def _build_degradation_monitoring(
    history: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    resolved_policy = _resolve_degradation_monitoring_policy(policy)
    if not resolved_policy.get("enabled", False):
        return {"enabled": False, "basis": "proxy_history"}

    lookback = int(resolved_policy.get("lookback_snapshots", DEGRADATION_MONITORING_DEFAULT_LOOKBACK))
    min_snapshots = int(resolved_policy.get("min_snapshots", DEGRADATION_MONITORING_DEFAULT_MIN_SNAPSHOTS))
    recent_history = list(history[-lookback:])
    horizon_labels: set[str] = set()
    for item in recent_history:
        predictions = item.get("predictions") if isinstance(item, Mapping) else None
        if isinstance(predictions, Mapping):
            horizon_labels.update(str(label) for label in predictions.keys())

    by_horizon: Dict[str, Any] = {}
    alarms: List[Dict[str, Any]] = []
    for horizon_label in sorted(horizon_labels, key=_horizon_sort_key):
        rows: List[Mapping[str, Any]] = []
        for item in recent_history:
            predictions = item.get("predictions") if isinstance(item, Mapping) else None
            entry = predictions.get(horizon_label) if isinstance(predictions, Mapping) else None
            if isinstance(entry, Mapping):
                rows.append(entry)
        if len(rows) < min_snapshots:
            by_horizon[horizon_label] = {
                "samples": len(rows),
                "alarm": False,
                "reasons": ["insufficient_history"],
            }
            continue

        ready_like = 0
        blocked = 0
        confidence_values: List[float] = []
        expected_net_values: List[float] = []
        for entry in rows:
            plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
            status = str(plan.get("status") or "unknown")
            if status in {"ready", "waiting_pullback", "bias_only_ready"}:
                ready_like += 1
            if status in {"rejected", "bias_only_ready"}:
                blocked += 1
            confidence = _finite_float_or_none(entry.get("confidence_score"))
            if confidence is not None:
                confidence_values.append(confidence)
            trade_decision = entry.get("trade_decision") if isinstance(entry.get("trade_decision"), Mapping) else {}
            expected_net = _finite_float_or_none(trade_decision.get("expected_net"))
            if expected_net is not None:
                expected_net_values.append(expected_net)

        sample_count = len(rows)
        ready_ratio = ready_like / max(sample_count, 1)
        blocked_ratio = blocked / max(sample_count, 1)
        avg_confidence = float(sum(confidence_values) / max(len(confidence_values), 1)) if confidence_values else None
        avg_expected_net = float(sum(expected_net_values) / max(len(expected_net_values), 1)) if expected_net_values else None
        reasons: List[str] = []
        if ready_ratio < float(resolved_policy.get("min_ready_ratio", 0.1)):
            reasons.append("ready_ratio_below_floor")
        if blocked_ratio > float(resolved_policy.get("max_blocked_ratio", 0.85)):
            reasons.append("blocked_ratio_above_ceiling")
        if avg_confidence is not None and avg_confidence < float(resolved_policy.get("min_confidence", 0.0)):
            reasons.append("confidence_below_floor")
        if avg_expected_net is not None and avg_expected_net < float(resolved_policy.get("min_expected_net", 0.0)):
            reasons.append("expected_net_below_floor")

        alarm = bool(reasons)
        by_horizon[horizon_label] = {
            "samples": sample_count,
            "ready_ratio": float(ready_ratio),
            "blocked_ratio": float(blocked_ratio),
            "avg_confidence": avg_confidence,
            "avg_expected_net": avg_expected_net,
            "alarm": alarm,
            "reasons": reasons,
        }
        if alarm:
            alarms.append({"horizon": horizon_label, "reasons": reasons})

    return {
        "enabled": True,
        "basis": "proxy_history",
        "lookback_snapshots": lookback,
        "min_snapshots": min_snapshots,
        "alarms": alarms,
        "by_horizon": by_horizon,
    }


def _build_trade_ready_monitoring_payload(predictions_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    predictions = predictions_payload.get("predictions", {})
    horizons: list[dict[str, Any]] = []
    for horizon_key in sorted(predictions.keys(), key=_horizon_sort_key):
        entry = predictions[horizon_key]
        if isinstance(entry, dict):
            horizons.append(entry)
    request = {
        "targets": args.targets,
        "spot_provider": args.spot_provider,
        "hours": args.hours,
        "dry_run": bool(args.dry_run),
        "confidence_min": float(getattr(args, "confidence_min", CONFIDENCE_MIN_DEFAULT)),
        "position_size_floor": float(getattr(args, "position_size_floor", POSITION_SIZE_FLOOR_DEFAULT)),
        "position_size_cap": float(getattr(args, "position_size_cap", POSITION_SIZE_CAP_DEFAULT)),
    }
    position_size_cap_by_horizon = getattr(args, "position_size_cap_by_horizon", None)
    if isinstance(position_size_cap_by_horizon, Mapping) and position_size_cap_by_horizon:
        request["position_size_cap_by_horizon"] = {
            _format_horizon_label(float(key)): float(value)
            for key, value in sorted(position_size_cap_by_horizon.items(), key=lambda item: float(item[0]))
        }
    data_quality_cfg = getattr(args, "data_quality", None)
    if isinstance(data_quality_cfg, Mapping):
        request["data_quality"] = dict(data_quality_cfg)
    metadata = getattr(args, "local_feature_metadata", None)
    if metadata:
        request["local_feature_overrides"] = metadata
    payload = {
        "generated_at": predictions_payload.get("generated_at"),
        "source": "run_refresh_and_predict",
        "request": request,
        "horizons": horizons,
    }
    if isinstance(predictions_payload.get("blocked_trade_analytics"), Mapping):
        payload["blocked_trade_analytics"] = predictions_payload.get("blocked_trade_analytics")
    if isinstance(predictions_payload.get("degradation_monitoring"), Mapping):
        payload["degradation_monitoring"] = predictions_payload.get("degradation_monitoring")
    if isinstance(predictions_payload.get("prompt_ready_summary"), Mapping):
        payload["prompt_ready_summary"] = predictions_payload.get("prompt_ready_summary")
    return payload


def _write_monitoring_payload_file(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_monitoring_latest(
    predictions_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monitoring_payload = payload or _build_trade_ready_monitoring_payload(predictions_payload, args)
    _write_monitoring_payload_file(monitoring_payload, MONITORING_LATEST_PATH)
    return monitoring_payload


def _write_trade_ready_monitoring(
    predictions_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monitoring_payload = payload or _build_trade_ready_monitoring_payload(predictions_payload, args)
    _write_monitoring_payload_file(monitoring_payload, TRADE_READY_MONITOR_PATH)
    return monitoring_payload


def _refresh_meta_baseline() -> None:
    if not META_BASELINE_SOURCE_CSV.exists():
        print(
            f"Meta baseline CSV not found at {META_BASELINE_SOURCE_CSV.as_posix()}; skipping baseline refresh.",
            file=sys.stderr,
        )
        return
    df = load_dataframe(META_BASELINE_SOURCE_CSV, limit=0)
    if df.empty:
        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": 0,
            "columns": {},
            "column_order": list(BASELINE_DEFAULT_COLUMNS),
        }
    else:
        columns = _append_detected_meta_columns(df, BASELINE_DEFAULT_COLUMNS)
        baseline = compute_baseline(df, columns)
    META_BASELINE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_BASELINE_JSON_PATH.write_text(json.dumps(baseline, indent=2))
    META_BASELINE_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_df = baseline_to_dataframe(baseline)
    baseline_df.to_parquet(META_BASELINE_PARQUET_PATH, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML/JSON file that overrides the CLI defaults (hours, targets, thresholds, etc.)."
            " CLI flags still take precedence over config entries."
        ),
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = _load_cli_config(config_args.config)
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Binance US spot data, rebuild local features/datasets, and emit multi-horizon predictions."
        ),
        parents=[config_parser],
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Number of hourly candles to fetch from Binance US (default: 360).",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=list(DEFAULT_TARGETS),
        help="Comma-separated prediction horizons in hours (default: 0.25,1,4,8,12).",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Probability threshold for ensemble activation (default: 0.45).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Return threshold for ensemble activation (default: 0.0).",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability cutoff used for the direction-only signal. "
            "Values above 0.5 make the model more sensitive to downtrends. "
            "Default 0.5 produces the original behaviour."
        ),
    )
    parser.add_argument(
        "--auto-direction-threshold",
        action="store_true",
        help=(
            "Enable automatic computation of the direction-only threshold based "
            "on calibrated p_up_min values from thresholds_json. "
            "If set, --direction-threshold is ignored."
        ),
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=str(Path("artifacts/models/calibrated_thresholds_merged.json")),
        help="Optional JSON file containing per-horizon thresholds; set to an empty string to disable.",
    )
    parser.add_argument(
        "--platt-calibration",
        type=str,
        default=str(Path("artifacts/models/platt_calibration.json")),
        help="Optional JSON file containing Platt scaling coefficients per horizon.",
    )
    parser.add_argument(
        "--data-quality-enabled",
        action="store_true",
        help="Enable hard OHLCV data-quality checks after ingestion/local feature loading.",
    )
    parser.add_argument(
        "--max-staleness-hours",
        type=float,
        default=2.0,
        help="Maximum allowed OHLCV staleness in hours when data quality checks are enabled.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.01,
        help="Maximum allowed ratio of missing hourly timestamps when quality checks are enabled.",
    )
    parser.add_argument(
        "--max-zero-volume-ratio",
        type=float,
        default=0.2,
        help="Maximum allowed ratio of zero-volume rows when quality checks are enabled.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=120,
        help="Minimum required OHLCV rows when quality checks are enabled.",
    )
    parser.add_argument(
        "--confidence-min",
        type=float,
        default=CONFIDENCE_MIN_DEFAULT,
        help="Minimum confidence score required to keep a non-hold trade action (default: 0.0).",
    )
    parser.add_argument(
        "--position-size-floor",
        type=float,
        default=POSITION_SIZE_FLOOR_DEFAULT,
        help="Floor for confidence-scaled position size (default: 0.0).",
    )
    parser.add_argument(
        "--position-size-cap",
        type=float,
        default=POSITION_SIZE_CAP_DEFAULT,
        help="Cap for confidence-scaled position size (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip network-dependent steps and reuse cached datasets/models for smoke testing.",
    )
    parser.add_argument(
        "--replay-offset-bars",
        type=int,
        default=0,
        help=(
            "Replay cached hourly predictions from N bars back using the prepared dataset instead of the latest bar. "
            "Intended for hourly horizons such as 1,4,8,12."
        ),
    )
    parser.add_argument(
        "--spot-provider",
        choices=("binanceus",),
        default="binanceus",
        help="Spot ingestion provider for hourly candles (Binance-only; default: binanceus).",
    )
    parser.add_argument(
        "--use-local-features",
        action="store_true",
        help=(
            "Skip ingestion + feature rebuilds and load hourly features directly from the supplied parquet paths."
            " Requires --features-path."
        ),
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to the merged hourly features parquet/CSV used when --use-local-features is set.",
    )
    parser.add_argument(
        "--macro-path",
        type=str,
        default=None,
        help="Optional macro parquet/CSV used only for metadata when --use-local-features is enabled.",
    )
    parser.add_argument(
        "--onchain-path",
        type=str,
        default=None,
        help="Optional on-chain parquet/CSV used only for metadata when --use-local-features is enabled.",
    )
    parser.add_argument(
        "--funding-path",
        type=str,
        default=None,
        help="Optional funding parquet/CSV used only for metadata when --use-local-features is enabled.",
    )
    parser.add_argument(
        "--intrabar-path",
        type=str,
        default=None,
        help="Optional intrabar (15m->1h aggregated) parquet/CSV merged when --use-local-features is enabled.",
    )
    parser.add_argument(
        "--intrabar-enabled",
        action="store_true",
        help="Fetch 15m Binance candles and aggregate intrabar features into the live inference bundle.",
    )
    parser.add_argument(
        "--intrabar-interval",
        type=str,
        default="15m",
        help="Binance interval used for intrabar aggregation (default: 15m).",
    )
    parser.add_argument(
        "--intrabar-hours-multiplier",
        type=int,
        default=4,
        help="Multiplier for intrabar fetch size relative to --hours (default: 4 for 15m).",
    )
    parser.add_argument(
        "--intrabar-max-rows",
        type=int,
        default=4000,
        help="Upper bound on intrabar rows fetched from Binance for aggregation.",
    )
    parser.add_argument(
        "--trade-decision-enabled",
        action="store_true",
        help="Enable trade decision model override for final trade/no-trade action.",
    )
    parser.add_argument(
        "--trade-decision-disabled",
        action="store_true",
        help="Disable trade decision model policy even if enabled in config.",
    )
    parser.add_argument(
        "--trade-decision-model",
        type=str,
        default=None,
        help="Path to JSON trade decision model artifact.",
    )
    parser.add_argument(
        "--trade-decision-threshold",
        type=float,
        default=None,
        help="Optional probability threshold for trade decision model.",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Update monitoring artifacts (trade_ready_summary + meta baseline) after predictions complete.",
    )
    parser.add_argument(
        "--disable-monitoring-latest",
        action="store_true",
        help="Skip writing artifacts/monitoring/latest.json snapshot (default: enabled).",
    )
    parser.add_argument(
        "--dir-lstm-path",
        type=str,
        default=None,
        help="Optional directory containing the LSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-bilstm-path",
        type=str,
        default=None,
        help="Optional directory containing the BiLSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-gru-path",
        type=str,
        default=None,
        help="Optional directory containing the GRU direction model ensemble.",
    )
    parser.add_argument(
        "--dir-cnn-lstm-path",
        type=str,
        default=None,
        help="Optional directory containing the CNN-LSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-cnn-bilstm-path",
        type=str,
        default=None,
        help="Optional directory containing the CNN-BiLSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-garch-lstm-path",
        type=str,
        default=None,
        help="Optional directory containing the GARCH-LSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-transformer-path",
        type=str,
        default=None,
        help="Optional directory containing the transformer direction model ensemble.",
    )
    parser.add_argument(
        "--dir-model-config-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file describing direction-model entries (list of {type,path,weight}); "
            "overrides the built-in DEFAULT_DIR_MODELS_1H registry."
        ),
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=DEFAULT_DIR_MODEL_WEIGHTS_1H,
        help=(
            "Legacy comma-separated weights for direction models (e.g. transformer:2,lstm:1,xgb:1). "
            "Applied on top of the resolved structured config."
        ),
    )
    if config_defaults:
        config_defaults = {k: v for k, v in config_defaults.items() if k != "config"}
        if config_defaults:
            parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(getattr(args, "replay_offset_bars", 0) or 0) < 0:
        print("Error: --replay-offset-bars must be >= 0.", file=sys.stderr)
        sys.exit(2)
    if not hasattr(args, "trend_ignition"):
        args.trend_ignition = None
    if not hasattr(args, "direction_only_fallback"):
        args.direction_only_fallback = None
    if not hasattr(args, "adaptive_thresholds"):
        args.adaptive_thresholds = None
    if not hasattr(args, "target_range_models"):
        args.target_range_models = None
    if not hasattr(args, "data_quality"):
        args.data_quality = None
    if not hasattr(args, "abstention_policy"):
        args.abstention_policy = None
    if not hasattr(args, "uncertainty_policy"):
        args.uncertainty_policy = None
    if not hasattr(args, "regime_model_weights"):
        args.regime_model_weights = None
    if not hasattr(args, "regime_model_dirs"):
        args.regime_model_dirs = None
    if not hasattr(args, "trade_decision_policy"):
        args.trade_decision_policy = None
    if not hasattr(args, "intrabar_aggregation"):
        args.intrabar_aggregation = None
    if not hasattr(args, "feature_coverage_policy"):
        args.feature_coverage_policy = None
    if not hasattr(args, "confluence_policy"):
        args.confluence_policy = None
    if not hasattr(args, "execution_policy"):
        args.execution_policy = None
    if not hasattr(args, "forecast_coherence_policy"):
        args.forecast_coherence_policy = None
    if not hasattr(args, "direction_output_policy"):
        args.direction_output_policy = None
    if not hasattr(args, "position_size_cap_by_horizon"):
        args.position_size_cap_by_horizon = None
    if not hasattr(args, "degradation_monitoring"):
        args.degradation_monitoring = None
    if args.data_quality is None:
        args.data_quality = {}
    if args.data_quality_enabled:
        args.data_quality["enabled"] = True
    # CLI quality flags always override config values.
    args.data_quality["max_staleness_hours"] = args.max_staleness_hours
    args.data_quality["max_missing_ratio"] = args.max_missing_ratio
    args.data_quality["max_zero_volume_ratio"] = args.max_zero_volume_ratio
    args.data_quality["min_rows"] = args.min_rows
    prepared_override: tuple[PreparedData, int, float, str] | None = None
    args.local_feature_metadata = None

    intrabar_cfg = dict(getattr(args, "intrabar_aggregation", {}) or {})
    if args.intrabar_enabled:
        intrabar_cfg["enabled"] = True
    if args.intrabar_interval:
        intrabar_cfg.setdefault("interval", args.intrabar_interval)
    intrabar_cfg.setdefault("hours_multiplier", args.intrabar_hours_multiplier)
    intrabar_cfg.setdefault("max_rows", args.intrabar_max_rows)
    intrabar_enabled = bool(intrabar_cfg.get("enabled", False))

    trade_decision_cfg = dict(getattr(args, "trade_decision_policy", {}) or {})
    if args.trade_decision_disabled:
        trade_decision_cfg["enabled"] = False
    if args.trade_decision_enabled:
        trade_decision_cfg["enabled"] = True
    if args.trade_decision_model:
        trade_decision_cfg["model_path"] = str(args.trade_decision_model)
    if args.trade_decision_threshold is not None:
        trade_decision_cfg["threshold"] = float(args.trade_decision_threshold)
    args.trade_decision_policy = trade_decision_cfg

    if getattr(args, "config", None):
        print(f"Loaded CLI defaults from config: {args.config}")

    replay_offset_bars = int(getattr(args, "replay_offset_bars", 0) or 0)
    if replay_offset_bars > 0:
        if any(float(target) < 1.0 for target in getattr(args, "targets", []) or []):
            print(
                "Error: --replay-offset-bars currently supports hourly horizons only; remove sub-hour targets such as 0.25.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.use_local_features:
            print("Error: --replay-offset-bars cannot be combined with --use-local-features.", file=sys.stderr)
            sys.exit(2)
        if not args.dry_run:
            print("Info: enabling --dry-run automatically for replay mode.")
            args.dry_run = True

    if args.use_local_features and args.dry_run:
        print("Error: --use-local-features cannot be combined with --dry-run.", file=sys.stderr)
        sys.exit(2)
    if args.use_local_features and not args.features_path:
        print("Error: --features-path is required when --use-local-features is enabled.", file=sys.stderr)
        sys.exit(2)

    latest_close: float | None = None
    latest_spot_features_path: str | None = None
    intrabar_features_path: str | None = None
    if args.use_local_features:
        try:
            optional_sources = {
                label: getattr(args, attr)
                for attr, label in LOCAL_FEATURE_OPTIONAL_PATHS
                if getattr(args, attr, None)
            }
            prepared_override, metadata = _prepare_local_feature_bundle(
                features_path=args.features_path,
                hours=args.hours,
                optional_sources=optional_sources,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Failed to load local feature overrides: {exc}", file=sys.stderr)
            sys.exit(1)
        args.local_feature_metadata = metadata
        labels = ", ".join(sorted(metadata.keys()))
        print(f"Loaded local feature overrides: {labels}")
        coverage_policy = _resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
        coverage_payload = _evaluate_feature_coverage(metadata, coverage_policy)
        args.local_feature_metadata["feature_coverage"] = coverage_payload
        if coverage_policy.get("enabled") and not coverage_payload.get("ok", False) and coverage_payload.get("block_on_violation", True):
            print(
                "Feature coverage gate blocked prediction run: "
                f"{', '.join(coverage_payload.get('failed_checks', []))}",
                file=sys.stderr,
            )
            sys.exit(1)
        quality_policy = _resolve_data_quality_policy(getattr(args, "data_quality", None))
        if quality_policy.get("enabled"):
            try:
                quality_frame = _read_timeseries_frame(args.features_path, "features")
                quality_payload = _evaluate_data_quality(quality_frame, quality_policy)
            except Exception as exc:
                print(f"Data quality check failed: {exc}", file=sys.stderr)
                sys.exit(1)
            if not quality_payload.get("ok", False):
                print(
                    f"Data quality gate blocked prediction run: {quality_payload.get('error', 'unknown data quality failure')}",
                    file=sys.stderr,
                )
                sys.exit(1)
    elif args.dry_run:
        print("Dry run enabled: using cached datasets and skipping ingestion, feature rebuild, and dataset regeneration.")
    else:
        try:
            output_path = run_ingestion(hours=args.hours, provider=args.spot_provider)

            if intrabar_enabled:
                intrabar_interval = str(intrabar_cfg.get("interval") or "15m")
                hours_mult = max(int(intrabar_cfg.get("hours_multiplier") or 4), 1)
                max_rows = max(int(intrabar_cfg.get("max_rows") or 4000), 1)
                intrabar_limit = min(max_rows, max(args.hours * hours_mult, args.hours))
                intrabar_tidy_path = run_ingestion(
                    hours=intrabar_limit,
                    interval=intrabar_interval,
                    provider=args.spot_provider,
                )
                intrabar_df = _compute_intrabar_features_from_15m(intrabar_tidy_path)
                intrabar_output = Path("data/processed/technical") / "intrabar_features_15m_to_1h.parquet"
                intrabar_output.parent.mkdir(parents=True, exist_ok=True)
                intrabar_df.to_parquet(intrabar_output, index=False)
                intrabar_features_path = str(intrabar_output)
                print(
                    "Saved intrabar aggregated features to "
                    f"{intrabar_output} (rows={len(intrabar_df)}, interval={intrabar_interval}).",
                )

            # Save latest price data to spot_klines for dataset building
            if output_path and output_path.exists():
                df = pd.read_parquet(output_path)
                quality_policy = _resolve_data_quality_policy(getattr(args, "data_quality", None))
                if quality_policy.get("enabled"):
                    quality_frame = _build_ohlcv_frame_from_tidy(df)
                    quality_payload = _evaluate_data_quality(quality_frame, quality_policy)
                    if not quality_payload.get("ok", False):
                        raise RuntimeError(
                            "Data quality gate blocked prediction run: "
                            f"{quality_payload.get('error', 'unknown data quality failure')}"
                        )
                # Pivot to wide
                wide_df = df.pivot(index='ts', columns='metric', values='value').reset_index()
                # Rename columns
                rename_map = {
                    'spot_open': 'open',
                    'spot_high': 'high',
                    'spot_low': 'low',
                    'spot_close': 'close',
                    'spot_volume': 'volume',
                    'spot_quote_volume': 'quote_volume',
                    'spot_num_trades': 'num_trades',
                    'spot_taker_buy_base_volume': 'taker_buy_base_volume',
                    'spot_taker_buy_quote_volume': 'taker_buy_quote_volume',
                }
                wide_df = wide_df.rename(columns=rename_map)
                wide_df['interval'] = '1h'
                # Save to spot_klines
                from datetime import datetime
                today = datetime.now().strftime('%Y-%m-%d')
                spot_path = Path('data/spot_klines') / f'btcusdt_spot_1h_{today}.parquet'
                wide_df.to_parquet(spot_path, index=False)
                latest_spot_features_path = str(spot_path)
                print(f"Saved latest price data to {spot_path}")
            latest_close = None
            if output_path and output_path.exists():
                df = pd.read_parquet(output_path)
                close_df = df[df.metric == 'spot_close']
                if not close_df.empty:
                    latest_close = close_df.value.iloc[-1]
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            sys.exit(1)

        feature_build_results: Dict[str, str] = {}
        try:
            feature_build_results = run_feature_builders(price_source=output_path)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Feature rebuild failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            rebuild_datasets(args.targets)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Dataset build failed: {exc}", file=sys.stderr)
            sys.exit(1)

        # Prefer freshly rebuilt local features for inference so timestamps track the newest ingested candle.
        # Dataset NPZ tails can lag when upstream curated features are stale.
        technical_features_path = feature_build_results.get("technical")
        if latest_spot_features_path:
            try:
                prepared_override, metadata = _prepare_local_feature_bundle(
                    features_path=latest_spot_features_path,
                    hours=args.hours,
                    optional_sources={
                        key: value
                        for key, value in {
                            "technical": technical_features_path,
                            "intrabar": intrabar_features_path,
                        }.items()
                        if value
                    }
                    or None,
                )
                args.local_feature_metadata = metadata
                coverage_policy = _resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
                coverage_payload = _evaluate_feature_coverage(metadata, coverage_policy)
                args.local_feature_metadata["feature_coverage"] = coverage_payload
                if coverage_policy.get("enabled") and not coverage_payload.get("ok", False) and coverage_payload.get("block_on_violation", True):
                    raise RuntimeError(
                        "feature coverage gate failed: " + ", ".join(coverage_payload.get("failed_checks", []))
                    )
                print(
                    "Using freshly rebuilt local feature bundle for live inference "
                    f"({latest_spot_features_path}).",
                )
            except Exception as exc:
                coverage_policy = _resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
                if coverage_policy.get("enabled") and coverage_policy.get("block_on_violation", True):
                    print(
                        "Fresh local inference bundle preparation failed and coverage blocking is enabled: "
                        f"{exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(
                    "Warning: failed to prepare fresh local inference bundle; "
                    f"falling back to dataset-based inference ({exc}).",
                    file=sys.stderr,
                )
        elif technical_features_path:
            print(
                "Warning: fresh spot feature file unavailable; local inference override disabled.",
                file=sys.stderr,
            )

    if replay_offset_bars > 0:
        try:
            replay_profile = _dataset_profile_for_horizon(1.0)
            replay_candidate, used_fallback = _select_dataset_candidate(replay_profile)
            prepared, replay_latest_index, _close_snapshot, _ts_snapshot = _load_prepared(
                replay_candidate.path,
                target_column=replay_candidate.target_column,
                offline=True,
            )
            replay_index = replay_latest_index - replay_offset_bars
            if replay_index < 0:
                raise ValueError(
                    f"Replay offset {replay_offset_bars} exceeds prepared dataset length {replay_latest_index + 1}."
                )
            replay_close = float(prepared.df_all["close"].iloc[replay_index])
            replay_ts = format_ts_iso(prepared.df_all["ts"].iloc[replay_index])
            prepared_override = (prepared, replay_index, replay_close, replay_ts)
            latest_close = replay_close
            fallback_msg = " (fallback dataset)" if used_fallback else ""
            print(
                "Replay mode enabled: using hourly cached dataset "
                f"{replay_candidate.path.name}{fallback_msg} at index offset {replay_offset_bars} "
                f"(timestamp={replay_ts})."
            )
        except Exception as exc:
            print(f"Replay preparation failed: {exc}", file=sys.stderr)
            sys.exit(1)

    env_dir_lstm = os.getenv("DIR_LSTM_PATH") or args.dir_lstm_path
    env_dir_bilstm = os.getenv("DIR_BILSTM_PATH") or args.dir_bilstm_path
    env_dir_gru = os.getenv("DIR_GRU_PATH") or args.dir_gru_path
    env_dir_cnn_lstm = os.getenv("DIR_CNN_LSTM_PATH") or args.dir_cnn_lstm_path
    env_dir_cnn_bilstm = os.getenv("DIR_CNN_BILSTM_PATH") or args.dir_cnn_bilstm_path
    env_dir_garch_lstm = os.getenv("DIR_GARCH_LSTM_PATH") or args.dir_garch_lstm_path
    env_dir_transformer = os.getenv("DIR_TRANSFORMER_PATH") or args.dir_transformer_path
    if any([
        env_dir_lstm,
        env_dir_bilstm,
        env_dir_gru,
        env_dir_cnn_lstm,
        env_dir_cnn_bilstm,
        env_dir_garch_lstm,
        env_dir_transformer,
    ]):
        print(
            "Sequence ensemble directories:"
            f" LSTM={env_dir_lstm or 'None'}"
            f", BiLSTM={env_dir_bilstm or 'None'}"
            f", GRU={env_dir_gru or 'None'}"
            f", CNN-LSTM={env_dir_cnn_lstm or 'None'}"
            f", CNN-BiLSTM={env_dir_cnn_bilstm or 'None'}"
            f", GARCH-LSTM={env_dir_garch_lstm or 'None'}"
            f", transformer={env_dir_transformer or 'None'}",
        )

    thresholds_path = args.thresholds_json or None
    platt_calibration = _load_platt_calibration(getattr(args, "platt_calibration", None))
    direction_output_cfg = dict(getattr(args, "direction_output_policy", {}) or {})
    direction_output_calibration_path = direction_output_cfg.get("calibration_path")
    direction_output_cfg["calibration_map"] = _load_platt_calibration(direction_output_calibration_path)
    if args.target_range_models is None:
        target_range_meta = TARGET_RANGE_MODEL_DIR / "metadata.json"
        if target_range_meta.exists():
            args.target_range_models = {
                "enabled": True,
                "model_dir": str(TARGET_RANGE_MODEL_DIR),
            }
    thresholds_by_horizon = load_calibrated_thresholds(thresholds_path)
    if thresholds_by_horizon:
        print(
            "Loaded calibrated thresholds for horizons"
            f" {sorted(thresholds_by_horizon.keys())}"
            f" from {thresholds_path}.",
        )
    _warn_missing_thresholds(args.targets, thresholds_by_horizon, thresholds_path)

    try:
        summary = run_predictions(
            args.targets,
            args.p_up_min,
            args.ret_min,
            direction_threshold=args.direction_threshold,
            auto_direction_threshold=args.auto_direction_threshold,
            offline=args.dry_run,
            dir_lstm_path=env_dir_lstm,
            dir_bilstm_path=env_dir_bilstm,
            dir_gru_path=env_dir_gru,
            dir_cnn_lstm_path=env_dir_cnn_lstm,
            dir_cnn_bilstm_path=env_dir_cnn_bilstm,
            dir_garch_lstm_path=env_dir_garch_lstm,
            dir_transformer_path=env_dir_transformer,
            dir_model_config_json=args.dir_model_config_json or None,
            dir_model_weights=args.dir_model_weights,
            thresholds_by_horizon=thresholds_by_horizon,
            prepared_override=prepared_override,
            trend_ignition=getattr(args, "trend_ignition", None),
            direction_only_fallback=getattr(args, "direction_only_fallback", None),
            adaptive_thresholds=getattr(args, "adaptive_thresholds", None),
            target_range_models=getattr(args, "target_range_models", None),
            platt_calibration=platt_calibration,
            abstention_policy=getattr(args, "abstention_policy", None),
            uncertainty_policy=getattr(args, "uncertainty_policy", None),
            trade_decision_policy=getattr(args, "trade_decision_policy", None),
            regime_model_weights=getattr(args, "regime_model_weights", None),
            regime_model_dirs=getattr(args, "regime_model_dirs", None),
            confluence_policy=getattr(args, "confluence_policy", None),
            execution_policy=getattr(args, "execution_policy", None),
            forecast_coherence_policy=getattr(args, "forecast_coherence_policy", None),
            direction_output_policy=direction_output_cfg,
            latest_close=latest_close,
            confidence_min=float(getattr(args, "confidence_min", CONFIDENCE_MIN_DEFAULT)),
            position_size_floor=float(getattr(args, "position_size_floor", POSITION_SIZE_FLOOR_DEFAULT)),
            position_size_cap=float(getattr(args, "position_size_cap", POSITION_SIZE_CAP_DEFAULT)),
            position_size_cap_by_horizon=getattr(args, "position_size_cap_by_horizon", None),
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Prediction step failed: {exc}", file=sys.stderr)
        sys.exit(1)

    predictions_payload = write_summary(
        summary,
        degradation_policy=getattr(args, "degradation_monitoring", None),
    )

    monitoring_payload: dict[str, Any] | None = None
    if not args.disable_monitoring_latest:
        monitoring_payload = _write_monitoring_latest(predictions_payload, args)

    if args.write_artifacts:
        _write_trade_ready_monitoring(
            predictions_payload,
            args,
            payload=monitoring_payload,
        )
        _refresh_meta_baseline()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
