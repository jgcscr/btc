from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.runtime.horizon_support import coerce_numeric_horizon, format_horizon_label, normalize_horizon_value
from src.scripts import run_refresh_and_predict as legacy
from src.trading.thresholds import load_calibrated_thresholds


@dataclass(frozen=True)
class PredictionInputBundle:
    direction_output_cfg: dict[str, Any]
    thresholds_by_horizon: Mapping[float, Mapping[str, float]]
    platt_calibration: Mapping[str, Any]


@dataclass(frozen=True)
class SequenceModelDirs:
    dir_lstm_path: str | None
    dir_bilstm_path: str | None
    dir_gru_path: str | None
    dir_cnn_lstm_path: str | None
    dir_cnn_bilstm_path: str | None
    dir_garch_lstm_path: str | None
    dir_transformer_path: str | None

    def has_any(self) -> bool:
        return any(
            [
                self.dir_lstm_path,
                self.dir_bilstm_path,
                self.dir_gru_path,
                self.dir_cnn_lstm_path,
                self.dir_cnn_bilstm_path,
                self.dir_garch_lstm_path,
                self.dir_transformer_path,
            ]
        )


def load_cli_config(
    path: str | None,
    *,
    config_allowed_keys: Sequence[str],
    normalize_config_value: Callable[[str, Any], Any],
    yaml_safe_load: Callable[[str], Any],
    stderr_write: Callable[[str], None],
) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    try:
        raw_data = yaml_safe_load(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse config file {resolved}: {exc}") from exc
    if raw_data is None:
        return {"config": str(resolved)}
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Config file must contain a mapping/dict (got {type(raw_data).__name__}).")
    normalized: dict[str, Any] = {}
    allowed_keys = set(config_allowed_keys)
    for raw_key, value in raw_data.items():
        if not isinstance(raw_key, str):
            stderr_write(f"Ignoring non-string config key: {raw_key}\n")
            continue
        key = raw_key.replace("-", "_")
        if key not in allowed_keys:
            stderr_write(f"Warning: Unknown config key '{raw_key}' ignored.\n")
            continue
        normalized[key] = normalize_config_value(key, value)
    normalized["config"] = str(resolved)
    return normalized


def dataset_profile_for_horizon(
    horizon: float,
    *,
    dataset_multi_path: Path,
    dataset_1h_path: Path,
    dataset_15m_path: Path,
    dataset_candidate_type: Callable[..., Any],
    dataset_profile_type: Callable[..., Any],
) -> Any:
    hourly_candidates = (
        dataset_candidate_type(dataset_multi_path, "ret_1h", 1.0, offline_only=False),
        dataset_candidate_type(dataset_1h_path, "ret_1h", 1.0, offline_only=False),
    )
    if horizon < 1.0:
        sub_candidates = (
            dataset_candidate_type(dataset_15m_path, "ret_15m", 0.25, offline_only=True),
            *hourly_candidates,
        )
        return dataset_profile_type(key="15m", candidates=sub_candidates)
    return dataset_profile_type(key="hourly", candidates=hourly_candidates)


def select_dataset_candidate(profile: Any) -> tuple[Any, bool]:
    candidates = getattr(profile, "candidates", ())
    profile_key = getattr(profile, "key", "unknown")
    if not candidates:
        raise RuntimeError(f"Dataset profile {profile_key} does not define any candidates.")
    for idx, candidate in enumerate(candidates):
        if Path(getattr(candidate, "path")).exists():
            return candidate, idx > 0
    return candidates[-1], True


def warn_missing_thresholds(
    targets: Sequence[float],
    thresholds: Mapping[int | float | str, Mapping[str, float]] | None,
    source_path: str | None,
    *,
    normalize_horizon_value: Callable[[float], float],
    coerce_numeric_horizon: Callable[[int | float | str], float | None],
    format_horizon_label: Callable[[float], str],
    stderr_write: Callable[[str], None],
) -> None:
    if not thresholds:
        return
    requested = {normalize_horizon_value(h) for h in targets}
    available: set[float] = set()
    for key in thresholds.keys():
        numeric = coerce_numeric_horizon(key)
        if numeric is not None:
            available.add(numeric)
    missing = sorted(requested - available)
    if missing:
        label = ", ".join(format_horizon_label(h) for h in missing)
        source = source_path or "provided thresholds JSON"
        stderr_write(
            f"Warning: {source} is missing calibrated entries for horizons {label}; falling back to CLI defaults.\n"
        )


def load_probability_calibration(
    path: str | None,
    *,
    json_loads: Callable[[str], Any] = json.loads,
    stderr_write: Callable[[str], None],
) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        stderr_write(f"Warning: Platt calibration file not found at {path_obj}; skipping.\n")
        return {}
    payload = json_loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Platt calibration file must contain a JSON object keyed by horizon.")
    result: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        method = str(value.get("method", "platt")).lower()
        if method == "platt" and "a" in value and "b" in value:
            result[str(key)] = {"method": "platt", "a": float(value["a"]), "b": float(value["b"])}
            continue
        if method == "beta" and all(name in value for name in ("a", "b", "c")):
            result[str(key)] = {
                "method": "beta",
                "a": float(value["a"]),
                "b": float(value["b"]),
                "c": float(value["c"]),
            }
            continue
        if method == "isotonic" and all(name in value for name in ("x", "y")):
            x = [float(item) for item in value.get("x", [])]
            y = [float(item) for item in value.get("y", [])]
            if x and y and len(x) == len(y):
                result[str(key)] = {"method": "isotonic", "x": x, "y": y}
                continue
        if "a" in value and "b" in value:
            result[str(key)] = {"method": "platt", "a": float(value["a"]), "b": float(value["b"])}
    return result


def base_horizon_for_target_column(target_column: str) -> float:
    lowered = str(target_column).strip().lower()
    if lowered == "ret_15m":
        return 0.25
    return 1.0


def periods_per_hour_for_base_horizon(base_horizon: float) -> int:
    if float(base_horizon) >= 1.0:
        return 1
    return max(int(round(1.0 / float(base_horizon))), 1)


def load_prepared(
    dataset_path: Path,
    *,
    target_column: str,
    offline: bool = False,
    load_prepared_offline_fn: Callable[..., Any],
    prepare_data_for_signals_fn: Callable[..., Any],
    format_ts_iso_fn: Callable[[Any], str],
) -> tuple[Any, int, float, str]:
    if offline:
        return load_prepared_offline_fn(dataset_path, base_horizon=base_horizon_for_target_column(target_column))

    prepared = prepare_data_for_signals_fn(str(dataset_path), target_column=target_column)
    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Prepared dataset has no rows.")
    ts_value = prepared.df_all["ts"].iloc[index]
    close = float(prepared.df_all["close"].iloc[index])
    ts_iso = format_ts_iso_fn(ts_value)
    return prepared, index, close, ts_iso


def load_prepared_offline(
    dataset_path: Path,
    *,
    base_horizon: float,
    prepare_data_for_signals_from_ohlcv_fn: Callable[..., Any],
    format_ts_iso_fn: Callable[[Any], str],
    stderr_write: Callable[[str], None],
) -> tuple[Any, int, float, str]:
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

    x_all = np.concatenate(arrays, axis=0)
    if x_all.size == 0:
        raise RuntimeError("Dataset NPZ is empty after concatenation; cannot build offline prepared data.")

    df_features = pd.DataFrame(x_all, columns=feature_names)
    if "close" not in df_features.columns:
        raise RuntimeError("Offline dataset must include a 'close' feature column.")

    periods = len(df_features)
    expected_freq = pd.Timedelta(hours=float(base_horizon))
    freq_alias = expected_freq if expected_freq != pd.Timedelta(hours=1) else "H"
    ts_index = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq=freq_alias)
    df_features.insert(0, "ts", ts_index)

    prepared = prepare_data_for_signals_from_ohlcv_fn(
        df_features,
        feature_names=feature_names,
        train_frac=0.7,
        expected_freq=expected_freq,
        periods_per_hour=periods_per_hour_for_base_horizon(base_horizon),
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
            stderr_write(
                "Warning: close_all array length mismatch in offline dataset; falling back to scaled close values.\n"
            )
    ts_iso = format_ts_iso_fn(ts_value)
    return prepared, index, close, ts_iso


def project_price(close: float, log_return: float) -> float:
    return close * math.exp(log_return)


def normalize_refresh_args(args: argparse.Namespace) -> None:
    if int(getattr(args, "replay_offset_bars", 0) or 0) < 0:
        raise ValueError("--replay-offset-bars must be >= 0")
    for attr in [
        "trend_ignition",
        "direction_only_fallback",
        "adaptive_thresholds",
        "target_range_models",
        "data_quality",
        "abstention_policy",
        "uncertainty_policy",
        "regime_model_weights",
        "regime_model_dirs",
        "trade_decision_policy",
        "intrabar_aggregation",
        "feature_coverage_policy",
        "confluence_policy",
        "execution_policy",
        "forecast_coherence_policy",
        "direction_output_policy",
        "position_size_cap_by_horizon",
        "degradation_monitoring",
        "direction_ensemble_policy",
        "confidence_min_by_horizon_regime",
        "disabled_horizons",
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, None)
    if args.data_quality is None:
        args.data_quality = {}
    if getattr(args, "data_quality_enabled", False):
        args.data_quality["enabled"] = True
    args.data_quality["max_staleness_hours"] = args.max_staleness_hours
    args.data_quality["max_missing_ratio"] = args.max_missing_ratio
    args.data_quality["max_zero_volume_ratio"] = args.max_zero_volume_ratio
    args.data_quality["min_rows"] = args.min_rows
    args.local_feature_metadata = None

    intrabar_cfg = dict(getattr(args, "intrabar_aggregation", {}) or {})
    if getattr(args, "intrabar_enabled", False):
        intrabar_cfg["enabled"] = True
    if getattr(args, "intrabar_interval", None):
        intrabar_cfg.setdefault("interval", args.intrabar_interval)
    intrabar_cfg.setdefault("hours_multiplier", getattr(args, "intrabar_hours_multiplier", 4))
    intrabar_cfg.setdefault("max_rows", getattr(args, "intrabar_max_rows", 4000))
    args._intrabar_enabled = bool(intrabar_cfg.get("enabled", False))
    args._intrabar_cfg = intrabar_cfg

    trade_decision_cfg = dict(getattr(args, "trade_decision_policy", {}) or {})
    if getattr(args, "trade_decision_disabled", False):
        trade_decision_cfg["enabled"] = False
    if getattr(args, "trade_decision_enabled", False):
        trade_decision_cfg["enabled"] = True
    if getattr(args, "trade_decision_model", None):
        trade_decision_cfg["model_path"] = str(args.trade_decision_model)
    if getattr(args, "trade_decision_threshold", None) is not None:
        trade_decision_cfg["threshold"] = float(args.trade_decision_threshold)
    args.trade_decision_policy = trade_decision_cfg

    replay_offset_bars = int(getattr(args, "replay_offset_bars", 0) or 0)
    if replay_offset_bars > 0:
        if any(float(target) < 1.0 for target in getattr(args, "targets", []) or []):
            raise ValueError("--replay-offset-bars currently supports hourly horizons only")
        if getattr(args, "use_local_features", False):
            raise ValueError("--replay-offset-bars cannot be combined with --use-local-features")
        if not getattr(args, "dry_run", False):
            args.dry_run = True

    if getattr(args, "use_local_features", False) and getattr(args, "dry_run", False):
        raise ValueError("--use-local-features cannot be combined with --dry-run")
    if getattr(args, "use_local_features", False) and not getattr(args, "features_path", None):
        raise ValueError("--features-path is required when --use-local-features is enabled")


def resolve_prediction_inputs(args: argparse.Namespace) -> PredictionInputBundle:
    thresholds_path = args.thresholds_json or None
    platt_calibration = load_probability_calibration(
        getattr(args, "platt_calibration", None),
        stderr_write=legacy.sys.stderr.write,
    )
    direction_output_cfg = dict(getattr(args, "direction_output_policy", {}) or {})
    direction_output_calibration_path = direction_output_cfg.get("calibration_path")
    direction_output_cfg["calibration_map"] = load_probability_calibration(
        direction_output_calibration_path,
        stderr_write=legacy.sys.stderr.write,
    )
    if args.target_range_models is None:
        target_range_meta = legacy.TARGET_RANGE_MODEL_DIR / "metadata.json"
        if target_range_meta.exists():
            args.target_range_models = {
                "enabled": True,
                "model_dir": str(legacy.TARGET_RANGE_MODEL_DIR),
            }
    thresholds_by_horizon = load_calibrated_thresholds(thresholds_path)
    if thresholds_by_horizon:
        print(
            "Loaded calibrated thresholds for horizons"
            f" {sorted(thresholds_by_horizon.keys())}"
            f" from {thresholds_path}.",
        )
    warn_missing_thresholds(
        args.targets,
        thresholds_by_horizon,
        thresholds_path,
        normalize_horizon_value=normalize_horizon_value,
        coerce_numeric_horizon=coerce_numeric_horizon,
        format_horizon_label=format_horizon_label,
        stderr_write=legacy.sys.stderr.write,
    )
    return PredictionInputBundle(
        direction_output_cfg=direction_output_cfg,
        thresholds_by_horizon=thresholds_by_horizon,
        platt_calibration=platt_calibration,
    )


def resolve_sequence_model_dirs(args: argparse.Namespace) -> SequenceModelDirs:
    return SequenceModelDirs(
        dir_lstm_path=os.getenv("DIR_LSTM_PATH") or getattr(args, "dir_lstm_path", None),
        dir_bilstm_path=os.getenv("DIR_BILSTM_PATH") or getattr(args, "dir_bilstm_path", None),
        dir_gru_path=os.getenv("DIR_GRU_PATH") or getattr(args, "dir_gru_path", None),
        dir_cnn_lstm_path=os.getenv("DIR_CNN_LSTM_PATH") or getattr(args, "dir_cnn_lstm_path", None),
        dir_cnn_bilstm_path=os.getenv("DIR_CNN_BILSTM_PATH") or getattr(args, "dir_cnn_bilstm_path", None),
        dir_garch_lstm_path=os.getenv("DIR_GARCH_LSTM_PATH") or getattr(args, "dir_garch_lstm_path", None),
        dir_transformer_path=os.getenv("DIR_TRANSFORMER_PATH") or getattr(args, "dir_transformer_path", None),
    )
