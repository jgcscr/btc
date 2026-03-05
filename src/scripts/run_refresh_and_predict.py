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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib

import yaml

import numpy as np
import pandas as pd

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.ingestors.tiingo_spot import ingest_tiingo_spot
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
from src.config_trading import DEFAULT_DIR_MODEL_WEIGHTS_1H, DEFAULT_DIR_MODELS_1H
from src.trading.direction_config import (
    DirectionModelConfig,
    apply_path_overrides,
    clone_direction_model_configs,
    direction_configs_to_weight_map,
    log_direction_model_configs,
    resolve_direction_model_configs,
)
from src.trading.signals import (
    DEFAULT_RESIDUAL_STD,
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
from src.trading.volatility import latest_volatility_snapshot

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
TARGET_RANGE_MODEL_DIR = Path("artifacts/models/target_ranges")
TARGET_RANGE_DEFAULT_HORIZONS: tuple[float, ...] = (4.0, 8.0, 12.0)
TARGET_RANGE_DEFAULT_OVERRIDE_RATIO = 0.01
TARGET_RANGE_DEFAULT_CONFIDENCE_SCALE = 0.01

BREAKOUT_VOL_NORMALIZER = 0.05
BREAKOUT_RET_NORMALIZER = 0.002
REGIME_TREND = "trend_ignition"
REGIME_NEUTRAL = "neutral"
REGIME_CHOP = "chop"

LOCAL_FEATURE_OPTIONAL_PATHS: tuple[tuple[str, str], ...] = (
    ("macro_path", "macro"),
    ("onchain_path", "onchain"),
    ("cryptoquant_path", "cryptoquant"),
    ("funding_path", "funding"),
)

LOCAL_FEATURE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "macro": tuple(),
    "cryptoquant": tuple(),
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
    "cryptoquant_path",
    "funding_path",
    "write_artifacts",
    "disable_monitoring_latest",
    "dir_lstm_path",
    "dir_bilstm_path",
    "dir_gru_path",
    "dir_cnn_lstm_path",
    "dir_transformer_path",
    "dir_model_config_json",
    "dir_model_weights",
    "trend_ignition",
    "direction_only_fallback",
    "adaptive_thresholds",
    "target_range_models",
}
# boolean config keys; converted with _bool_env
CONFIG_BOOL_FIELDS = {
    "dry_run",
    "use_local_features",
    "write_artifacts",
    "disable_monitoring_latest",
    "auto_direction_threshold",
}
CONFIG_FLOAT_FIELDS = {"p_up_min", "ret_min", "direction_threshold"}
CONFIG_INT_FIELDS = {"hours"}
CONFIG_PATH_FIELDS = {
    "thresholds_json",
    "features_path",
    "macro_path",
    "onchain_path",
    "cryptoquant_path",
    "funding_path",
    "dir_lstm_path",
    "dir_bilstm_path",
    "dir_gru_path",
    "dir_cnn_lstm_path",
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
    if provider == "tiingo":
        lookback_days = max(math.ceil(hours / 24), 1)
        print(f"Fetching {lookback_days} day(s) of {interval} candles from Tiingo for BTCUSD...")
        output_path = ingest_tiingo_spot(lookback_days=lookback_days)
        print(f"Saved Tiingo spot tidy parquet to {output_path}")
        return output_path

    limit = max(hours, 1)
    print(f"Fetching {limit} {interval} klines from Binance US for {symbol}...")
    output_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=limit)
    print(f"Saved spot tidy parquet to {output_path}")
    return output_path


def run_feature_builders(price_source: Path | None = None) -> Dict[str, str]:
    results: Dict[str, str] = {}
    print("Recomputing technical indicator features...")
    technical_path = process_technical_features(price_source=price_source)
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

    if hours > 0 and len(base_df) > hours:
        base_df = base_df.iloc[-hours:].reset_index(drop=True)

    feature_names = _load_training_feature_names()
    if feature_names:
        missing = [col for col in feature_names if col not in base_df.columns]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            print(
                f"Warning: local features missing {len(missing)} model columns {preview}{suffix}; filling with zeros.",
                file=sys.stderr,
            )
            for column in missing:
                base_df[column] = 0.0
    else:
        feature_names = [col for col in base_df.columns if col != "ts"]

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
    base_suffix = suffixes[0]
    base_reg_version = MODEL_VERSION_PRIORITY[0]
    base_dir_version = DIR_VERSION_OVERRIDES.get(base_suffix, MODEL_VERSION_PRIORITY)[0]

    for suffix_idx, suffix in enumerate(suffixes):
        dir_versions = DIR_VERSION_OVERRIDES.get(suffix, MODEL_VERSION_PRIORITY)
        for reg_version_idx, reg_version in enumerate(MODEL_VERSION_PRIORITY):
            for dir_version_idx, dir_version in enumerate(dir_versions):
                reg_dir = MODEL_ROOT / f"xgb_ret{suffix}_{reg_version}"
                dir_dir = MODEL_ROOT / f"xgb_dir{suffix}_{dir_version}"
                reg_path = reg_dir / f"xgb_ret{suffix}_model.json"
                dir_path = dir_dir / f"xgb_dir{suffix}_model.json"

                if fallback is None:
                    fallback = (reg_path, dir_path)

                if reg_path.exists() and dir_path.exists():
                    dir_changed = dir_version != base_dir_version
                    reg_changed = reg_version != base_reg_version
                    if suffix_idx > 0 or reg_changed or dir_changed:
                        print(
                            "Info: using %s (reg=%s, dir=%s) model artifacts for %s horizon (fallback from %s (reg=%s, dir=%s))."
                            % (
                                suffix,
                                reg_version,
                                dir_version,
                                label,
                                base_suffix,
                                base_reg_version,
                                base_dir_version,
                            ),
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
    dir_transformer_path: str | None,
) -> List[DirectionModelConfig]:
    overrides = {
        "lstm": dir_lstm_path,
        "bilstm": dir_bilstm_path,
        "gru": dir_gru_path,
        "cnn_lstm": dir_cnn_lstm_path,
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
    horizon_label: str,
) -> tuple[List[DirectionModelConfig], Dict[str, float]]:
    configs = clone_direction_model_configs(base_configs)
    apply_path_overrides(configs, {"xgb": dir_model_path})
    log_direction_model_configs(configs, label=f"[run_refresh_and_predict] direction models ({horizon_label})")
    weight_map = direction_configs_to_weight_map(configs)
    return configs, weight_map


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
    dir_transformer_path: str | None = None,
    dir_model_config_json: str | None = None,
    dir_model_weights: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
    prepared_override: tuple[PreparedData, int, float, str] | None = None,
    trend_ignition: Mapping[str, Any] | None = None,
    direction_only_fallback: Mapping[str, Any] | None = None,
    adaptive_thresholds: Mapping[str, Any] | None = None,
    target_range_models: Mapping[str, Any] | None = None,
    latest_close: float | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    normalized_targets = sorted({_normalize_horizon_value(h) for h in targets})
    if not normalized_targets:
        return {}

    trend_payload = _resolve_trend_ignition_payload(trend_ignition)
    direction_fallback_policy = _resolve_direction_fallback_policy(direction_only_fallback)
    adaptive_policy = _resolve_adaptive_thresholds_policy(adaptive_thresholds)
    target_range_policy = _resolve_target_range_policy(target_range_models)

    # compute automatic direction threshold if requested
    if auto_direction_threshold and thresholds_by_horizon:
        # pick the largest calibrated p_up_min across selected targets
        try:
            direction_threshold = max(th["p_up_min"] for th in thresholds_by_horizon.values())
        except Exception:
            pass

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
        reg_path, dir_path = _model_paths_for_horizon(horizon)
        if not reg_path.exists() or not dir_path.exists():
            print(
                f"Warning: skipping {label} horizon because model files are missing",
                file=sys.stderr,
            )
            continue

        direction_configs, dir_weight_map = _direction_configs_for_horizon(
            base_direction_configs,
            dir_model_path=str(dir_path),
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
        regime_state = REGIME_NEUTRAL
        regime_score = None
        adaptive_scale = 1.0
        if adaptive_policy and adaptive_policy.get("enabled"):
            profile_score = breakout_scores.get(profile_key)
            if profile_score is not None:
                regime_score = profile_score
                regime_state = _classify_regime_from_score(profile_score, adaptive_policy)
                horizon_p_up, horizon_ret, adaptive_scale = _apply_adaptive_thresholds(
                    adaptive_policy,
                    horizon_p_up,
                    horizon_ret,
                    regime_state,
                )
        signal = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=horizon_p_up,
            ret_min=horizon_ret,
            dir_model_weights=dir_weight_map,
            volatility_snapshot=volatility_snapshot,
            volatility_policy=horizon_thresholds,
        )
        # override direction-only signal using configurable threshold
        try:
            p_val = float(signal.get("p_up", 0.0))
        except Exception:
            p_val = 0.0
        # some older versions of this file may not define direction_threshold as a
        # parameter; fallback to 0.5 if it's missing.
        thresh = locals().get("direction_threshold", 0.5)
        signal["signal_dir_only"] = int(p_val >= thresh)
        if latest_close is not None:
            signal['close'] = latest_close
            close = latest_close

        ret_pred = float(signal.get("ret_pred", 0.0))
        p_up = float(signal.get("p_up", 0.0))
        signal_ts = str(signal.get("ts", ts_iso))
        residual_std = float(residual_std_by_horizon.get(horizon, DEFAULT_RESIDUAL_STD))
        stop_loss_price = _project_price(close, ret_pred - residual_std)
        take_profit_price = _project_price(close, ret_pred + residual_std)
        expected_value = p_up * ret_pred - (1 - p_up) * residual_std
        ev_multiplier = float(horizon_thresholds.get("expected_value_multiplier", 1.0))
        expected_value *= ev_multiplier
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
            "p_up": p_up,
            "p_trend_ignition": trend_prob,
            "ignition_state": ignition_state,
            "ignition_cooldown_active": cooldown_active if trend_payload else False,
            "ret_pred": ret_pred,
            "projected_price": _project_price(close, ret_pred),
            "signal_ensemble": int(signal.get("signal_ensemble", 0)),
            "signal_dir_only": int(signal.get("signal_dir_only", 0)),
            "p_up_components": signal.get("p_up_components", {}),
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "expected_value": expected_value,
            "thresholds": horizon_thresholds,
            "regime_state": regime_state,
            "projected_high": target_projection.get("projected_high") if target_projection else None,
            "projected_low": target_projection.get("projected_low") if target_projection else None,
            "projected_high_confidence": target_projection.get("projected_high_confidence", 0.0)
            if target_projection
            else 0.0,
            "projected_low_confidence": target_projection.get("projected_low_confidence", 0.0)
            if target_projection
            else 0.0,
            "volatility": signal.get("volatility", {
                "snapshot": volatility_snapshot,
                "metric": None,
                "ceiling": horizon_thresholds.get("volatility_ceiling"),
                "triggered": False,
            }),
            "volatility_flag": bool(signal.get("volatility_flag")),
        }
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
        if fallback_triggered:
            pending_direction_fallback_ts = signal_ts
        summary[label] = result
    if trend_payload and pending_trend_ts:
        _write_trend_ignition_state(pending_trend_ts)
    if direction_fallback_policy and pending_direction_fallback_ts:
        _write_direction_fallback_state(pending_direction_fallback_ts)

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


def write_summary(summary: Dict[str, Dict[str, float | str | int]]) -> dict[str, Any]:
    LATEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    json_payload = {
        "generated_at": generated_at,
        "predictions": summary,
    }
    LATEST_PREDICTION_PATH.write_text(json.dumps(json_payload, indent=2))
    print(json.dumps(json_payload, indent=2))

    history_entry = {
        "generated_at": generated_at,
        "predictions": summary,
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
    return json_payload


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
    }
    metadata = getattr(args, "local_feature_metadata", None)
    if metadata:
        request["local_feature_overrides"] = metadata
    return {
        "generated_at": predictions_payload.get("generated_at"),
        "source": "run_refresh_and_predict",
        "request": request,
        "horizons": horizons,
    }


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
        "--dry-run",
        action="store_true",
        help="Skip network-dependent steps and reuse cached datasets/models for smoke testing.",
    )
    parser.add_argument(
        "--spot-provider",
        choices=("binanceus", "tiingo"),
        default="binanceus",
        help="Spot ingestion provider for hourly candles (default: binanceus).",
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
        "--cryptoquant-path",
        type=str,
        default=None,
        help="Optional CryptoQuant parquet/CSV used only for metadata when --use-local-features is enabled.",
    )
    parser.add_argument(
        "--funding-path",
        type=str,
        default=None,
        help="Optional funding parquet/CSV used only for metadata when --use-local-features is enabled.",
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
    if not hasattr(args, "trend_ignition"):
        args.trend_ignition = None
    if not hasattr(args, "direction_only_fallback"):
        args.direction_only_fallback = None
    if not hasattr(args, "adaptive_thresholds"):
        args.adaptive_thresholds = None
    if not hasattr(args, "target_range_models"):
        args.target_range_models = None
    prepared_override: tuple[PreparedData, int, float, str] | None = None
    args.local_feature_metadata = None

    if getattr(args, "config", None):
        print(f"Loaded CLI defaults from config: {args.config}")

    if args.use_local_features and args.dry_run:
        print("Error: --use-local-features cannot be combined with --dry-run.", file=sys.stderr)
        sys.exit(2)
    if args.use_local_features and not args.features_path:
        print("Error: --features-path is required when --use-local-features is enabled.", file=sys.stderr)
        sys.exit(2)

    latest_close: float | None = None
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
    elif args.dry_run:
        print("Dry run enabled: using cached datasets and skipping ingestion, feature rebuild, and dataset regeneration.")
    else:
        try:
            output_path = run_ingestion(hours=args.hours, provider=args.spot_provider)
            # Save latest price data to spot_klines for dataset building
            if output_path and output_path.exists():
                df = pd.read_parquet(output_path)
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

        try:
            run_feature_builders()
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Feature rebuild failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            rebuild_datasets(args.targets)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Dataset build failed: {exc}", file=sys.stderr)
            sys.exit(1)

    env_dir_lstm = os.getenv("DIR_LSTM_PATH") or args.dir_lstm_path
    env_dir_bilstm = os.getenv("DIR_BILSTM_PATH") or args.dir_bilstm_path
    env_dir_gru = os.getenv("DIR_GRU_PATH") or args.dir_gru_path
    env_dir_cnn_lstm = os.getenv("DIR_CNN_LSTM_PATH") or args.dir_cnn_lstm_path
    env_dir_transformer = os.getenv("DIR_TRANSFORMER_PATH") or args.dir_transformer_path
    if any([env_dir_lstm, env_dir_bilstm, env_dir_gru, env_dir_cnn_lstm, env_dir_transformer]):
        print(
            "Sequence ensemble directories:"
            f" LSTM={env_dir_lstm or 'None'}"
            f", BiLSTM={env_dir_bilstm or 'None'}"
            f", GRU={env_dir_gru or 'None'}"
            f", CNN-LSTM={env_dir_cnn_lstm or 'None'}"
            f", transformer={env_dir_transformer or 'None'}",
        )

    thresholds_path = args.thresholds_json or None
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
            dir_transformer_path=env_dir_transformer,
            dir_model_config_json=args.dir_model_config_json or None,
            dir_model_weights=args.dir_model_weights,
            thresholds_by_horizon=thresholds_by_horizon,
            prepared_override=prepared_override,
            trend_ignition=getattr(args, "trend_ignition", None),
            direction_only_fallback=getattr(args, "direction_only_fallback", None),
            adaptive_thresholds=getattr(args, "adaptive_thresholds", None),
            target_range_models=getattr(args, "target_range_models", None),
            latest_close=latest_close,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Prediction step failed: {exc}", file=sys.stderr)
        sys.exit(1)

    predictions_payload = write_summary(summary)

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
