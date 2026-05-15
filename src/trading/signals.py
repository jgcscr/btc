import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import mlflow
import mlflow.pytorch
from mlflow.models import get_model_info
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from src.trading.feature_engineering import (
    apply_funding_rate_features as _shared_apply_funding_rate_features,
    augment_hourly_price_features as _shared_augment_hourly_price_features,
)
from src.utils.component_diversity_support import (
    summarize_component_probabilities,
    summarize_pairwise_history,
)


def _augment_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    return _shared_augment_hourly_price_features(
        frame,
        strict_missing=False,
        warn=_warn_placeholder_once,
    )

from xgboost import XGBClassifier, XGBRegressor

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import (
    enforce_unique_hourly_index,
    make_features_and_target,
    repair_hourly_continuity,
)
from src.data.targets_multi_horizon import add_multi_horizon_targets
from src.data.targets_multi_horizon import add_trend_ignition_label
from src.scripts.build_training_dataset import (
    PROCESSED_PATHS as REG_PROCESSED_PATHS,
    _drop_non_binance_breakout_features,
    _merge_processed_features as merge_curated_features,
)
from src.training.cnn_bilstm import CNNBiLSTMDirectionClassifier
from src.training.cnn_lstm import CNNLSTMDirectionClassifier
from src.training.garch_lstm import GarchLSTMDirectionClassifier
from src.training.lstm_model import BiLSTMDirectionClassifier, GRUDirectionClassifier, LSTMDirectionClassifier
from src.models.transformer_classifier import TransformerDirectionClassifier
from src.trading.ensembles import select_diverse_models, simple_average, weighted_average
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    latest_volatility_snapshot,
)


EXCLUDED_FEATURES = {
    "fut_volume_delta_1h",
    "fut_volume_pct_change_1h",
    "cq_daily_fallback_active",
    "cq_daily_fallback_complete",
}

DEFAULT_RESIDUAL_STD = 0.01
MIN_RESIDUAL_STD = 1e-6
_RESIDUAL_STD_WARNED = False
_MISSING_FEATURE_WARNINGS: Dict[str, Set[str]] = {}
_EXTRA_FEATURE_PLACEHOLDER_WARNINGS: Set[str] = set()

_SEQUENCE_MODEL_ITER_ORDER = (
    "lstm",
    "bilstm",
    "gru",
    "cnn_lstm",
    "cnn_bilstm",
    "garch_lstm",
    "transformer",
    "transformer_large",
)
_SEQUENCE_MODEL_TYPES = set(_SEQUENCE_MODEL_ITER_ORDER)
_TREE_MODEL_TYPES = {"xgb", "lgbm", "regime_logit"}
_VOLATILITY_METRIC_DEFAULT = "volatility_realized_24h"
_VOLATILITY_MULT_DEFAULT = 1.25
_HORIZON_PRECISION = 6


def _warn_placeholder_once(key: str, message: str) -> None:
    if key in _EXTRA_FEATURE_PLACEHOLDER_WARNINGS:
        return
    print(message, file=sys.stderr)
    _EXTRA_FEATURE_PLACEHOLDER_WARNINGS.add(key)


def _ensure_feature_columns(frame: pd.DataFrame, required: List[str], context: str) -> pd.DataFrame:
    if not required:
        return frame

    missing = [col for col in required if col not in frame.columns]
    if not missing:
        return frame

    # Reindex once so pandas doesn't repeatedly insert columns and trigger fragmentation warnings.
    ordered_columns = list(frame.columns) + missing
    frame = frame.reindex(columns=ordered_columns, fill_value=0.0)

    warned = _MISSING_FEATURE_WARNINGS.setdefault(context, set())
    unseen = [col for col in missing if col not in warned]
    if unseen:
        preview = ", ".join(sorted(unseen)[:5])
        suffix = "..." if len(unseen) > 5 else ""
        print(
            f"Warning: {context} missing model columns {preview}{suffix}; filled with zeros for inference.",
            file=sys.stderr,
        )
        warned.update(unseen)

    return frame


def _apply_platt_calibration(p_up: float, params: Mapping[str, Any]) -> float:
    try:
        a = float(params.get("a"))
        b = float(params.get("b"))
    except (TypeError, ValueError):
        return p_up

    p = min(max(float(p_up), 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + math.exp(-(a * logit + b)))
    return float(calibrated)


def _apply_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    return _shared_apply_funding_rate_features(
        df,
        strict_missing=False,
        warn=_warn_placeholder_once,
    )


def _recompute_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "close" not in result.columns:
        return result

    log_close = np.log(result["close"].astype(float))
    result["ret_1h"] = log_close.diff()
    result["ret_fwd_3h"] = log_close.shift(-3) - log_close
    result["ret_4h"] = log_close.shift(-4) - log_close
    result["ret_8h"] = log_close.shift(-8) - log_close
    result["ret_12h"] = log_close.shift(-12) - log_close
    return result


@dataclass
class PreparedData:
    df_all: pd.DataFrame
    X_all_ordered: pd.DataFrame
    scaler: StandardScaler
    feature_names: List[str]
    volatility_columns: Optional[List[str]] = None


def _load_full_features_df() -> pd.DataFrame:
    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )
    if df.empty:
        raise RuntimeError(
            "Loaded empty DataFrame from BigQuery; check that the curated table has data.",
        )
    return df


def _load_feature_names_from_npz(path: str) -> Optional[List[str]]:
    try:
        data = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return None

    if "feature_names" not in data.files:
        return None

    return data["feature_names"].tolist()


def _infer_required_target_horizons(feature_names: Sequence[str]) -> List[int]:
    horizons: set[int] = {1, 4}
    for name in feature_names:
        text = str(name)
        if text.startswith("ret_") and text.endswith("h"):
            body = text[len("ret_") : -1]
            if body.isdigit():
                horizons.add(int(body))
    return sorted(h for h in horizons if h > 0)


def _build_scaler_from_training(X_all_ordered: pd.DataFrame) -> StandardScaler:
    n = len(X_all_ordered)
    if n == 0:
        raise ValueError("Empty feature matrix; cannot build scaler.")

    n_train = int(n * 0.7)
    if n_train <= 0:
        raise ValueError("Not enough samples to define a training split.")

    X_train = X_all_ordered.iloc[:n_train]

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def _build_features_from_csv(
    features_path: str,
    target_column: str,
    horizons: List[int],
    onchain_path: Optional[str],
    feature_names: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    path_obj = Path(features_path)
    if path_obj.suffix.lower() == ".parquet":
        df = pd.read_parquet(path_obj)
    else:
        df = pd.read_csv(features_path, parse_dates=["ts"])
    if "ts" not in df.columns:
        raise ValueError("Features CSV must include a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df, _, gap_csv = enforce_unique_hourly_index(
        df,
        label="offline_features",
        raise_on_gap=False,
    )
    if gap_csv:
        print(f"[offline_features] Logged {gap_csv} non-hourly intervals; proceeding with gaps.")

    if onchain_path:
        print(
            "Ignoring deprecated --onchain-path input; Binance-only breakout features do not merge on-chain feeds.",
            file=sys.stderr,
        )

    df = _drop_non_binance_breakout_features(df)
    df, volatility_columns = add_volatility_columns(
        df,
        realized_windows=DEFAULT_REALIZED_WINDOWS,
    )
    required_horizons = _infer_required_target_horizons(feature_names or horizons)
    df_targets = add_multi_horizon_targets(df, horizons=required_horizons, price_col="close")
    if feature_names and "trend_ignition_6h" in set(feature_names):
        df_targets = add_trend_ignition_label(
            df_targets,
            horizon_hours=6,
            threshold=0.01,
            price_col="close",
            label_col="trend_ignition_6h",
        )
    ret_cols = [f"ret_{h}h" for h in horizons]
    df_targets = df_targets.dropna(subset=ret_cols)

    X, _ = make_features_and_target(df_targets, target_column=target_column, dropna=False)

    drop_cols = [f"ret_{h}h" for h in horizons if f"ret_{h}h" in X.columns and f"ret_{h}h" != target_column]
    drop_cols.extend([f"dir_{h}h" for h in horizons if f"dir_{h}h" in X.columns])
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    present_volatility_columns = [col for col in volatility_columns if col in df_targets.columns]

    return df_targets.reset_index(drop=True), X.reset_index(drop=True), present_volatility_columns


def prepare_data_for_signals(
    dataset_npz_path: str,
    target_column: str = "ret_1h",
    features_path: Optional[str] = None,
    onchain_path: Optional[str] = None,
) -> PreparedData:
    """Load full features from BigQuery and prepare ordered features + scaler.

    This mirrors the logic used in training and in the live signal script:
    - sort by ts
    - build X using make_features_and_target
    - enforce feature order from the NPZ dataset (if available)
    - fit a StandardScaler on the train split only
    """
    horizons: List[int] = [1, 4]
    feature_names = _load_feature_names_from_npz(dataset_npz_path)
    with np.load(dataset_npz_path, allow_pickle=True) as dataset_npz:
        if "horizons" in dataset_npz.files:
            horizons_arr = dataset_npz["horizons"].tolist()
            if isinstance(horizons_arr, list):
                horizons = [int(h) for h in horizons_arr]

    volatility_columns: List[str] = []

    if features_path:
        df_all, X_all, volatility_columns = _build_features_from_csv(
            features_path=features_path,
            target_column=target_column,
            horizons=horizons,
            onchain_path=onchain_path,
            feature_names=feature_names,
        )
        if feature_names is None:
            feature_names = list(X_all.columns)
    else:
        df_all_raw = _load_full_features_df()
        df_all_raw, _, gap_live = enforce_unique_hourly_index(
            df_all_raw,
            label="curated_features_live",
            raise_on_gap=False,
            normalize_to_hour=True,
        )
        if gap_live:
            print(f"[curated_features_live] Logged {gap_live} non-hourly intervals; upstream feed has gaps.")
        if "ts" not in df_all_raw.columns:
            raise ValueError("Expected a 'ts' column in the curated features table.")

        df_all_sorted = df_all_raw.sort_values("ts").reset_index(drop=True)
        df_all_augmented = merge_curated_features(df_all_sorted, REG_PROCESSED_PATHS)
        df_all_augmented = _drop_non_binance_breakout_features(df_all_augmented)
        df_all_augmented = _augment_price_features(df_all_augmented)
        df_all_augmented, volatility_columns = add_volatility_columns(
            df_all_augmented,
            realized_windows=DEFAULT_REALIZED_WINDOWS,
        )
        df_all_augmented, backfilled_live = repair_hourly_continuity(
            df_all_augmented,
            label="curated_features_live_reindexed",
            expected_freq=pd.Timedelta(hours=1),
        )
        if backfilled_live:
            print(
                f"[curated_features_live_reindexed] Backfilled {backfilled_live} hourly gaps via forward/back fill.",
            )
        df_all_augmented = _recompute_return_targets(df_all_augmented)
        df_all_augmented, _, gap_live_merged = enforce_unique_hourly_index(
            df_all_augmented,
            label="curated_features_live_merged",
            raise_on_gap=False,
            normalize_to_hour=True,
        )
        if gap_live_merged:
            print(
                f"[curated_features_live_merged] Logged {gap_live_merged} non-hourly intervals after merge; upstream feed has gaps."
            )
        df_all = df_all_augmented.dropna(subset=[target_column]).reset_index(drop=True)

        non_feature_cols = {"ts", target_column, "ret_fwd_3h"}
        feature_cols = [c for c in df_all.columns if c not in non_feature_cols]
        X_all = df_all[feature_cols].copy()

        if feature_names is None:
            feature_names = list(X_all.columns)

    X_all = _drop_non_binance_breakout_features(X_all)
    excluded_in_frame = [col for col in EXCLUDED_FEATURES if col in X_all.columns]
    if excluded_in_frame:
        X_all = X_all.drop(columns=excluded_in_frame)
        print(
            "Removed excluded features from live feature matrix:",
            ", ".join(sorted(excluded_in_frame)),
        )

    if feature_names is not None:
        feature_names = [col for col in feature_names if col not in EXCLUDED_FEATURES]

    if feature_names is None:
        feature_names = list(X_all.columns)
    else:
        ordered: List[str] = []
        seen = set()
        for column in feature_names:
            if column in seen:
                continue
            seen.add(column)
            ordered.append(column)
        feature_names = ordered

    missing_in_all = set(feature_names) - set(X_all.columns)
    if missing_in_all:
        missing_sorted = sorted(missing_in_all)
        print(
            "Warning: curated features missing model columns "
            f"{missing_sorted}; filling with zeros for backtest/inference."
        )
        for column in missing_sorted:
            X_all[column] = 0.0
            if column not in feature_names:
                feature_names.append(column)

    X_all_ordered = X_all[feature_names].copy()

    n_total = len(X_all_ordered)
    if n_total == 0:
        raise RuntimeError("Feature matrix is empty after ordering columns.")

    n_train = int(n_total * 0.7)
    if n_train <= 0:
        raise RuntimeError("Not enough samples to compute training statistics for scaling.")

    col_means = X_all_ordered.iloc[:n_train].mean(axis=0, skipna=True)
    X_all_ordered = X_all_ordered.fillna(col_means)
    X_all_ordered = X_all_ordered.fillna(0.0)

    scaler = _build_scaler_from_training(X_all_ordered)

    return PreparedData(
        df_all=df_all,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names,
        volatility_columns=volatility_columns,
    )


def prepare_data_for_signals_from_ohlcv(
    df_features: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    train_frac: float = 0.7,
    *,
    expected_freq: pd.Timedelta | str = pd.Timedelta(hours=1),
    periods_per_hour: int = 1,
) -> PreparedData:
    """Build a ``PreparedData`` bundle directly from an OHLCV-derived dataframe.

    This is used for fallback realtime predictions when BigQuery-curated rows are
    unavailable; callers must supply a dataframe containing the same feature
    columns expected by the 1h models. Scaling is refit on the earliest portion
    of the data (``train_frac``) so the ensemble logic can reuse
    ``compute_signal_for_index`` unchanged.
    """

    if "ts" not in df_features.columns:
        raise ValueError("Expected dataframe to include a 'ts' column.")

    if feature_names is None:
        non_feature_cols = {"ts"}
        feature_names = [c for c in df_features.columns if c not in non_feature_cols]

    missing = set(feature_names) - set(df_features.columns)
    if missing:
        raise ValueError(f"Dataframe missing required feature columns: {sorted(missing)}")

    df_all = df_features.sort_values("ts").reset_index(drop=True)
    freq = pd.Timedelta(expected_freq)
    normalize_to_hour = freq >= pd.Timedelta(hours=1)
    df_all, _, _ = enforce_unique_hourly_index(
        df_all,
        label="realtime_features",
        expected_freq=freq,
        normalize_to_hour=normalize_to_hour,
    )
    df_all, volatility_columns = add_volatility_columns(
        df_all,
        realized_windows=DEFAULT_REALIZED_WINDOWS,
        periods_per_hour=max(int(periods_per_hour), 1),
    )
    X_all_ordered = df_all[feature_names].copy()

    n_rows = len(X_all_ordered)
    if n_rows == 0:
        raise ValueError("Empty dataframe; cannot build PreparedData.")

    n_train = max(int(n_rows * train_frac), 1)
    scaler = StandardScaler()
    scaler.fit(X_all_ordered.iloc[:n_train])

    return PreparedData(
        df_all=df_all,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names,
        volatility_columns=volatility_columns,
    )


def format_ts_iso(ts_value: Any) -> str:
    """Format a timestamp-like value as an RFC3339-like string with Z suffix.

    The curated table stores ``ts`` as an integer nanosecond timestamp. This
    helper accepts either a pandas ``Timestamp`` or an integer-like value and
    normalizes to UTC.
    """
    if isinstance(ts_value, pd.Timestamp):
        dt = ts_value.to_pydatetime().astimezone(timezone.utc)
    else:
        ts = pd.to_datetime(ts_value, unit="ns", utc=True)
        dt = ts.to_pydatetime().astimezone(timezone.utc)

    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def find_row_index_for_ts(df_all: pd.DataFrame, ts_str: str) -> int:
    """Find the row index for a given timestamp string.

    The ts column is stored as integer nanoseconds; parse the input and
    compare on that basis.
    """
    ts_parsed = pd.to_datetime(ts_str, utc=True)
    target_ns = int(ts_parsed.value)

    matches = np.where(df_all["ts"].to_numpy() == target_ns)[0]
    if matches.size == 0:
        raise ValueError(f"No row found with ts = {ts_str!r}")
    return int(matches[-1])


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_mlflow_artifact_file(artifact_path: str, filename: str, *, label: str) -> str:
    if os.path.isdir(artifact_path):
        candidate = os.path.join(artifact_path, filename)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"{label} artifact not found at {candidate}")
    if os.path.isfile(artifact_path):
        return artifact_path
    raise FileNotFoundError(f"{label} artifact not found at {artifact_path}")


def _ensure_estimator_type(model: Any, estimator_type: str) -> None:
    if model is None:
        return
    if not getattr(model, "_estimator_type", None):
        model._estimator_type = estimator_type
    if estimator_type == "classifier" and not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])


def _extract_xgb_feature_names(model: Any) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        try:
            booster = model.get_booster()
            names = booster.feature_names
        except Exception:
            names = None
    if names:
        return [str(name) for name in list(names)]
    return None


def _find_xgb_model_file(model_path: str, *, label: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"{label} model artifact not found at {model_path}")

    mlmodel_path = os.path.join(model_path, "MLmodel")
    if os.path.exists(mlmodel_path):
        try:
            model_cfg = mlflow.models.Model.load(model_path)
            xgb_flavor = model_cfg.flavors.get("xgboost", {})
            data_path = xgb_flavor.get("data")
            if data_path:
                candidate = os.path.join(model_path, data_path)
                if os.path.exists(candidate):
                    return candidate
        except Exception:
            pass

    candidates = ("model.json", "model.xgb", "model.bin")
    for name in candidates:
        candidate = os.path.join(model_path, name)
        if os.path.exists(candidate):
            return candidate

    nested = os.path.join(model_path, "model")
    if os.path.isdir(nested):
        for name in candidates:
            candidate = os.path.join(nested, name)
            if os.path.exists(candidate):
                return candidate

    for root, _, files in os.walk(model_path):
        for filename in files:
            if not filename.startswith("model"):
                continue
            if filename.endswith((".json", ".xgb", ".bin")):
                return os.path.join(root, filename)

    raise FileNotFoundError(f"{label} model file not found under {model_path}")


def _load_xgb_registry_model(model_uri: str, *, estimator: str, label: str) -> Any:
    model_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    model_file = _find_xgb_model_file(model_dir, label=label)
    if estimator == "classifier":
        model = XGBClassifier()
        if not getattr(model, "_estimator_type", None):
            model._estimator_type = "classifier"
        model.load_model(model_file)
        if not hasattr(model, "classes_"):
            model.classes_ = np.array([0, 1])
        return model
    if estimator == "regressor":
        model = XGBRegressor()
        if not getattr(model, "_estimator_type", None):
            model._estimator_type = "regressor"
        model.load_model(model_file)
        return model
    raise ValueError(f"Unsupported estimator type '{estimator}' for {label}.")


def _load_recurrent_direction_model(
    model_dir: str,
    device: Optional[str],
    *,
    model_cls: type[nn.Module],
    model_label: str,
) -> Dict[str, Any]:
    if model_dir.startswith("models:/"):
        # Load from MLflow registry
        model_uri = model_dir
        model_info = get_model_info(model_uri)
        run_id = model_info.run_id
        model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
        
        # Download artifacts
        summary_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="summary.json")
        summary_file = _resolve_mlflow_artifact_file(summary_path, "summary.json", label=f"{model_label} summary")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.joblib")
        scaler_file = _resolve_mlflow_artifact_file(scaler_path, "scaler.joblib", label=f"{model_label} scaler")
        scaler_payload = joblib_load(scaler_file)
        
        seq_len = int(summary.get("seq_len"))
        feature_names = summary.get("feature_names", [])
        if not feature_names:
            raise ValueError(f"{model_label} summary missing feature_names")
        
        torch_device = _resolve_device(device)
        model.to(torch_device)
        model.eval()
        
        scaler_mean = scaler_payload.get("mean")
        scaler_std = scaler_payload.get("std")
        
        return {
            "model": model,
            "device": torch_device,
            "seq_len": seq_len,
            "feature_names": feature_names,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }
    else:
        # Load from local directory
        resolved_dir = os.path.abspath(model_dir)
        summary_path = os.path.join(resolved_dir, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"{model_label} summary not found at {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        seq_len = int(summary.get("seq_len"))
        feature_names = summary.get("feature_names", [])
        if not feature_names:
            raise ValueError(f"{model_label} summary missing feature_names")
        hyperparams = summary.get("hyperparams", {})
        hidden_size = int(hyperparams.get("hidden_size"))
        num_layers = int(hyperparams.get("num_layers"))
        dropout = float(hyperparams.get("dropout", 0.0))
        norm_type = str(hyperparams.get("norm_type", "none"))

        model_path = os.path.join(resolved_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_label} weights not found at {model_path}")

        torch_device = _resolve_device(device)
        checkpoint = torch.load(model_path, map_location=torch_device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        input_size = int(checkpoint.get("input_size", len(feature_names)))

        classifier = model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
        )
        classifier.load_state_dict(state_dict)
        classifier.to(torch_device)
        classifier.eval()

        scaler_mean = None
        scaler_std = None
        scaler_path = summary.get("scaler_path")
        if scaler_path:
            resolved_scaler = scaler_path
            if not os.path.isabs(resolved_scaler):
                resolved_scaler = os.path.join(resolved_dir, os.path.basename(resolved_scaler))
            if os.path.exists(resolved_scaler):
                if resolved_scaler.endswith(".joblib"):
                    scaler_payload = joblib_load(resolved_scaler)
                    scaler_mean = scaler_payload.get("mean")
                    scaler_std = scaler_payload.get("std")
                else:
                    with np.load(resolved_scaler) as scaler_npz:
                        scaler_mean = scaler_npz.get("mean")
                        scaler_std = scaler_npz.get("std")

        return {
            "model": classifier,
            "device": torch_device,
            "seq_len": seq_len,
            "feature_names": feature_names,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }


def _load_lstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    return _load_recurrent_direction_model(
        model_dir,
        device,
        model_cls=LSTMDirectionClassifier,
        model_label="LSTM",
    )


def _load_bilstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    return _load_recurrent_direction_model(
        model_dir,
        device,
        model_cls=BiLSTMDirectionClassifier,
        model_label="BiLSTM",
    )


def _load_gru_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    return _load_recurrent_direction_model(
        model_dir,
        device,
        model_cls=GRUDirectionClassifier,
        model_label="GRU",
    )


def _load_cnn_lstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    resolved_dir = os.path.abspath(model_dir)
    summary_path = os.path.join(resolved_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"CNN-LSTM summary not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seq_len = int(summary.get("seq_len"))
    feature_names = summary.get("feature_names", [])
    if not feature_names:
        raise ValueError("CNN-LSTM summary missing feature_names")
    hyperparams = summary.get("hyperparams", {})

    conv_channels = hyperparams.get("conv_channels")
    conv_kernel_sizes = hyperparams.get("conv_kernel_sizes")
    conv_strides = hyperparams.get("conv_strides")
    if not (conv_channels and conv_kernel_sizes and conv_strides):
        raise ValueError("CNN-LSTM summary missing convolution hyperparameters")

    hidden_size = int(hyperparams.get("hidden_size"))
    num_layers = int(hyperparams.get("num_layers"))
    dropout = float(hyperparams.get("dropout", 0.0))
    norm_type = str(hyperparams.get("norm_type", "none"))
    conv_activation = str(hyperparams.get("conv_activation", "relu"))
    conv_dropout = float(hyperparams.get("conv_dropout", 0.0))

    model_path = os.path.join(resolved_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN-LSTM weights not found at {model_path}")

    torch_device = _resolve_device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    input_size = int(checkpoint.get("input_size", len(feature_names)))

    classifier = CNNLSTMDirectionClassifier(
        input_size=input_size,
        conv_channels=[int(value) for value in conv_channels],
        conv_kernel_sizes=[int(value) for value in conv_kernel_sizes],
        conv_strides=[int(value) for value in conv_strides],
        lstm_hidden_size=hidden_size,
        lstm_num_layers=num_layers,
        dropout=dropout,
        norm_type=norm_type,
        conv_activation=conv_activation,
        conv_dropout=conv_dropout,
    )
    classifier.load_state_dict(state_dict)
    classifier.to(torch_device)
    classifier.eval()

    scaler_mean = None
    scaler_std = None
    scaler_path = summary.get("scaler_path")
    if scaler_path:
        resolved_scaler = scaler_path
        if not os.path.isabs(resolved_scaler):
            resolved_scaler = os.path.join(resolved_dir, os.path.basename(resolved_scaler))
        if os.path.exists(resolved_scaler):
            if resolved_scaler.endswith(".joblib"):
                scaler_payload = joblib_load(resolved_scaler)
                scaler_mean = scaler_payload.get("mean")
                scaler_std = scaler_payload.get("std")
            else:
                with np.load(resolved_scaler) as scaler_npz:
                    scaler_mean = scaler_npz.get("mean")
                    scaler_std = scaler_npz.get("std")

    return {
        "model": classifier,
        "device": torch_device,
        "seq_len": seq_len,
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def _load_cnn_bilstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    resolved_dir = os.path.abspath(model_dir)
    summary_path = os.path.join(resolved_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"CNN-BiLSTM summary not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seq_len = int(summary.get("seq_len"))
    feature_names = summary.get("feature_names", [])
    if not feature_names:
        raise ValueError("CNN-BiLSTM summary missing feature_names")
    hyperparams = summary.get("hyperparams", {})

    conv_channels = hyperparams.get("conv_channels")
    conv_kernel_sizes = hyperparams.get("conv_kernel_sizes")
    conv_strides = hyperparams.get("conv_strides")
    if not (conv_channels and conv_kernel_sizes and conv_strides):
        raise ValueError("CNN-BiLSTM summary missing convolution hyperparameters")

    hidden_size = int(hyperparams.get("hidden_size"))
    num_layers = int(hyperparams.get("num_layers"))
    dropout = float(hyperparams.get("dropout", 0.0))
    norm_type = str(hyperparams.get("norm_type", "none"))
    conv_activation = str(hyperparams.get("conv_activation", "relu"))
    conv_dropout = float(hyperparams.get("conv_dropout", 0.0))

    model_path = os.path.join(resolved_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN-BiLSTM weights not found at {model_path}")

    torch_device = _resolve_device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    input_size = int(checkpoint.get("input_size", len(feature_names)))

    classifier = CNNBiLSTMDirectionClassifier(
        input_size=input_size,
        conv_channels=[int(value) for value in conv_channels],
        conv_kernel_sizes=[int(value) for value in conv_kernel_sizes],
        conv_strides=[int(value) for value in conv_strides],
        lstm_hidden_size=hidden_size,
        lstm_num_layers=num_layers,
        dropout=dropout,
        norm_type=norm_type,
        conv_activation=conv_activation,
        conv_dropout=conv_dropout,
    )
    classifier.load_state_dict(state_dict)
    classifier.to(torch_device)
    classifier.eval()

    scaler_mean = None
    scaler_std = None
    scaler_path = summary.get("scaler_path")
    if scaler_path:
        resolved_scaler = scaler_path
        if not os.path.isabs(resolved_scaler):
            resolved_scaler = os.path.join(resolved_dir, os.path.basename(resolved_scaler))
        if os.path.exists(resolved_scaler):
            if resolved_scaler.endswith(".joblib"):
                scaler_payload = joblib_load(resolved_scaler)
                scaler_mean = scaler_payload.get("mean")
                scaler_std = scaler_payload.get("std")
            else:
                with np.load(resolved_scaler) as scaler_npz:
                    scaler_mean = scaler_npz.get("mean")
                    scaler_std = scaler_npz.get("std")

    return {
        "model": classifier,
        "device": torch_device,
        "seq_len": seq_len,
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def _load_garch_lstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    resolved_dir = os.path.abspath(model_dir)
    summary_path = os.path.join(resolved_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"GARCH-LSTM summary not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seq_len = int(summary.get("seq_len"))
    feature_names = summary.get("feature_names", [])
    if not feature_names:
        raise ValueError("GARCH-LSTM summary missing feature_names")
    hyperparams = summary.get("hyperparams", {})

    hidden_size = int(hyperparams.get("hidden_size"))
    num_layers = int(hyperparams.get("num_layers"))
    dropout = float(hyperparams.get("dropout", 0.0))
    norm_type = str(hyperparams.get("norm_type", "none"))
    garch_feature = str(hyperparams.get("garch_feature", "volatility_garch_like"))
    garch_index = hyperparams.get("garch_feature_index")
    if garch_index is None:
        if garch_feature in feature_names:
            garch_index = feature_names.index(garch_feature)
        else:
            raise ValueError("GARCH-LSTM summary missing garch_feature_index")

    model_path = os.path.join(resolved_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GARCH-LSTM weights not found at {model_path}")

    torch_device = _resolve_device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    input_size = int(checkpoint.get("input_size", len(feature_names)))

    classifier = GarchLSTMDirectionClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        garch_feature_index=int(garch_index),
        norm_type=norm_type,
    )
    classifier.load_state_dict(state_dict)
    classifier.to(torch_device)
    classifier.eval()

    scaler_mean = None
    scaler_std = None
    scaler_path = summary.get("scaler_path")
    if scaler_path:
        resolved_scaler = scaler_path
        if not os.path.isabs(resolved_scaler):
            resolved_scaler = os.path.join(resolved_dir, os.path.basename(resolved_scaler))
        if os.path.exists(resolved_scaler):
            if resolved_scaler.endswith(".joblib"):
                scaler_payload = joblib_load(resolved_scaler)
                scaler_mean = scaler_payload.get("mean")
                scaler_std = scaler_payload.get("std")
            else:
                with np.load(resolved_scaler) as scaler_npz:
                    scaler_mean = scaler_npz.get("mean")
                    scaler_std = scaler_npz.get("std")

    return {
        "model": classifier,
        "device": torch_device,
        "seq_len": seq_len,
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def _load_transformer_direction_model(
    model_dir: str,
    device: Optional[str],
    *,
    model_label: str = "Transformer",
) -> Dict[str, Any]:
    if model_dir.startswith("models:/"):
        # Load from MLflow registry
        model_uri = model_dir
        model_info = get_model_info(model_uri)
        run_id = model_info.run_id
        model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
        
        # Download artifacts
        summary_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="summary.json")
        summary_file = _resolve_mlflow_artifact_file(summary_path, "summary.json", label=f"{model_label} summary")
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.joblib")
        scaler_file = _resolve_mlflow_artifact_file(scaler_path, "scaler.joblib", label=f"{model_label} scaler")
        scaler_payload = joblib_load(scaler_file)
        
        seq_len = int(summary.get("seq_len"))
        feature_names = summary.get("feature_names", [])
        if not feature_names:
            raise ValueError(f"{model_label} summary missing feature_names")
        
        torch_device = _resolve_device(device)
        model.to(torch_device)
        model.eval()
        
        scaler_mean = scaler_payload.get("mean")
        scaler_std = scaler_payload.get("std")
        
        return {
            "model": model,
            "device": torch_device,
            "seq_len": seq_len,
            "feature_names": feature_names,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }
    else:
        # Load from local directory
        resolved_dir = os.path.abspath(model_dir)
        summary_path = os.path.join(resolved_dir, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"{model_label} summary not found at {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        seq_len = int(summary.get("seq_len"))
        feature_names = summary.get("feature_names", [])
        if not feature_names:
            raise ValueError(f"{model_label} summary missing feature_names")
        hyperparams = summary.get("hyperparams", {})

        model_path = os.path.join(resolved_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_label} weights not found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        input_size = int(checkpoint.get("input_size"))
        hidden_dim = int(checkpoint.get("hidden_dim", hyperparams.get("hidden_dim", 128)))
        num_heads = int(checkpoint.get("num_heads", hyperparams.get("num_heads", 4)))
        ffn_dim = int(checkpoint.get("ffn_dim", hyperparams.get("ffn_dim", hidden_dim * 2)))
        num_layers = int(checkpoint.get("num_layers", hyperparams.get("num_layers", 2)))
        dropout = float(checkpoint.get("dropout", hyperparams.get("dropout", 0.1)))
        use_layer_norm = bool(checkpoint.get("use_layer_norm", hyperparams.get("use_layer_norm", True)))

        torch_device = _resolve_device(device)
        transformer_model = TransformerDirectionClassifier(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=seq_len,
            use_layer_norm=use_layer_norm,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        transformer_model.load_state_dict(state_dict)
        transformer_model.to(torch_device)
        transformer_model.eval()

        scaler_mean = None
        scaler_std = None
        scaler_path = os.path.join(resolved_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler_payload = joblib_load(scaler_path)
            scaler_mean = scaler_payload.get("mean")
            scaler_std = scaler_payload.get("std")

        return {
            "model": transformer_model,
            "device": torch_device,
            "seq_len": seq_len,
            "feature_names": feature_names,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }


def _load_transformer_large_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    return _load_transformer_direction_model(
        model_dir,
        device,
        model_label="Transformer-Large",
    )


def _load_xgb_direction_model(model_path: str, _device: Optional[str] = None) -> Dict[str, Any]:
    if model_path.startswith("models:/"):
        # Load from MLflow registry
        model_info = get_model_info(model_path)
        run_id = model_info.run_id
        direction_model = _load_xgb_registry_model(
            model_path,
            estimator="classifier",
            label="Direction",
        )
        feature_names = _extract_xgb_feature_names(direction_model)
        if feature_names is None and run_id:
            try:
                meta_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="model_metadata_direction.json",
                )
                meta_file = _resolve_mlflow_artifact_file(
                    meta_path,
                    "model_metadata_direction.json",
                    label="Direction metadata",
                )
                metadata = json.loads(Path(meta_file).read_text())
            except Exception:
                metadata = {}
            feature_names = metadata.get("feature_names")
            if isinstance(feature_names, list):
                feature_names = [str(name) for name in feature_names]
            else:
                feature_names = None
    else:
        # Load from local file
        resolved_path = os.path.abspath(model_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Direction model not found: {resolved_path}")

        direction_model = XGBClassifier()
        if not getattr(direction_model, "_estimator_type", None):
            direction_model._estimator_type = "classifier"
        direction_model.load_model(resolved_path)

        feature_names = None
        meta_path = Path(resolved_path).with_name("model_metadata_direction.json")
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
            feature_names = metadata.get("feature_names")
            if isinstance(feature_names, list):
                feature_names = [str(name) for name in feature_names]
            else:
                feature_names = None

    return {
        "model": direction_model,
        "feature_names": feature_names,
    }


def _load_lgbm_direction_model(model_path: str, _device: Optional[str] = None) -> Dict[str, Any]:
    resolved_path = Path(model_path).expanduser()
    if resolved_path.is_dir():
        candidate = resolved_path / "lgbm_dir_model.joblib"
        if candidate.exists():
            resolved_path = candidate
        else:
            raise FileNotFoundError(f"LightGBM direction model not found in {resolved_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"LightGBM direction model not found: {resolved_path}")

    payload = joblib_load(resolved_path)
    model = payload
    feature_names = None
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_names = payload.get("feature_names")

    if model is None:
        raise ValueError(f"LightGBM payload at {resolved_path} is missing a model instance.")

    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "classifier"

    if feature_names is not None:
        feature_names = [str(name) for name in feature_names]

    return {
        "model": model,
        "feature_names": feature_names,
    }


def _load_regime_logit_direction_model(model_path: str, _device: Optional[str] = None) -> Dict[str, Any]:
    resolved_path = Path(model_path).expanduser()
    if resolved_path.is_dir():
        candidate = next(iter(sorted(resolved_path.glob("regime_logit_dir*_model.joblib"))), None)
        if candidate is not None:
            resolved_path = candidate
        else:
            raise FileNotFoundError(f"Regime logistic direction model not found in {resolved_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Regime logistic direction model not found: {resolved_path}")

    payload = joblib_load(resolved_path)
    model = payload
    feature_names = None
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_names = payload.get("feature_names")

    if model is None:
        raise ValueError(f"Regime logistic payload at {resolved_path} is missing a model instance.")

    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "classifier"

    if feature_names is not None:
        feature_names = [str(name) for name in feature_names]

    return {
        "model": model,
        "feature_names": feature_names,
    }


def load_trend_ignition_classifier(model_path: str) -> Dict[str, Any]:
    resolved_path = Path(model_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Trend ignition model not found: {resolved_path}")

    payload = joblib_load(resolved_path)
    model = payload
    feature_names = None
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_names = payload.get("feature_names")

    if model is None:
        raise ValueError(f"Trend ignition payload at {resolved_path} is missing a model instance.")

    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "classifier"

    if feature_names is not None:
        feature_names = [str(name) for name in feature_names]

    return {
        "model": model,
        "feature_names": feature_names,
        "path": str(resolved_path),
    }


_DIRECTION_MODEL_LOADERS = {
    "xgb": _load_xgb_direction_model,
    "lstm": _load_lstm_direction_model,
    "bilstm": _load_bilstm_direction_model,
    "gru": _load_gru_direction_model,
    "cnn_lstm": _load_cnn_lstm_direction_model,
    "cnn_bilstm": _load_cnn_bilstm_direction_model,
    "garch_lstm": _load_garch_lstm_direction_model,
    "transformer": _load_transformer_direction_model,
    "transformer_large": _load_transformer_large_direction_model,
    "lgbm": _load_lgbm_direction_model,
    "regime_logit": _load_regime_logit_direction_model,
}


def load_models(
    reg_model_path: str,
    dir_model_path: Optional[str] = None,
    lstm_model_dir: Optional[str] = None,
    transformer_model_dir: Optional[str] = None,
    direction_model_configs: Optional[Sequence[Mapping[str, Any]]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    if reg_model_path.startswith("models:/"):
        reg_info = get_model_info(reg_model_path)
        reg = _load_xgb_registry_model(
            reg_model_path,
            estimator="regressor",
            label="Return",
        )
        models["reg"] = reg
        reg_feature_names = _extract_xgb_feature_names(reg)
        if reg_feature_names is None and reg_info.run_id:
            try:
                meta_path = mlflow.artifacts.download_artifacts(
                    run_id=reg_info.run_id,
                    artifact_path="model_metadata.json",
                )
                meta_file = _resolve_mlflow_artifact_file(
                    meta_path,
                    "model_metadata.json",
                    label="Return metadata",
                )
                metadata = json.loads(Path(meta_file).read_text())
            except Exception:
                metadata = {}
            reg_feature_names = metadata.get("feature_names")
            if isinstance(reg_feature_names, list):
                reg_feature_names = [str(name) for name in reg_feature_names]
            else:
                reg_feature_names = None
            if metadata:
                target_scale = float(metadata.get("target_scale", 1.0)) or 1.0
                models["reg_target_scale"] = target_scale
        if reg_feature_names:
            models["reg_feature_names"] = reg_feature_names
    else:
        reg = XGBRegressor()
        if not getattr(reg, "_estimator_type", None):
            reg._estimator_type = "regressor"
        reg.load_model(reg_model_path)
        models["reg"] = reg
        reg_meta_path = Path(reg_model_path).with_name("model_metadata.json")
        if reg_meta_path.exists():
            try:
                metadata = json.loads(reg_meta_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
            feature_names = metadata.get("feature_names")
            if isinstance(feature_names, list) and feature_names:
                models["reg_feature_names"] = [str(name) for name in feature_names]
            target_scale = float(metadata.get("target_scale", 1.0)) or 1.0
            models["reg_target_scale"] = target_scale

    direction_entries: List[Dict[str, Any]] = []

    def _register_direction_entry(
        name: str,
        model_type: str,
        info: Dict[str, Any],
        *,
        weight: float = 1.0,
        label: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "name": name,
            "type": model_type,
            "weight": weight,
            "info": info,
        }
        if label is not None:
            entry["label"] = label
        direction_entries.append(entry)

    if direction_model_configs:
        for cfg in direction_model_configs:
            cfg_type = str(cfg.get("type", "")).strip().lower()
            loader = _DIRECTION_MODEL_LOADERS.get(cfg_type)
            if loader is None:
                raise ValueError(f"Unsupported direction model type '{cfg_type}'.")
            path = str(cfg.get("path", "")).strip()
            if not path:
                raise ValueError(f"Direction model '{cfg.get('name') or cfg_type}' is missing a path.")
            optional = bool(cfg.get("optional"))
            try:
                info = loader(path, device)
            except FileNotFoundError as exc:
                if optional:
                    print(
                        f"Warning: optional direction model '{cfg.get('name') or cfg_type}' skipped ({exc}).",
                        file=sys.stderr,
                    )
                    continue
                raise
            name = str(cfg.get("name") or cfg_type)
            weight = float(cfg.get("weight", 1.0))
            label = cfg.get("label")

            if cfg_type == "xgb":
                models["dir"] = info["model"]
                models["dir_xgb"] = info
                feature_names = info.get("feature_names")
                if feature_names:
                    models["dir_feature_names"] = list(feature_names)
            elif cfg_type == "lgbm":
                models["dir_lgbm"] = info
            elif cfg_type == "regime_logit":
                models["dir_regime_logit"] = info
            elif cfg_type in _SEQUENCE_MODEL_TYPES:
                models[f"dir_{cfg_type}"] = info

            _register_direction_entry(name, cfg_type, info, weight=weight, label=label)
    else:
        if dir_model_path:
            info = _load_xgb_direction_model(dir_model_path)
            models["dir"] = info["model"]
            feature_names = info.get("feature_names")
            if feature_names:
                models["dir_feature_names"] = list(feature_names)
            _register_direction_entry("xgb", "xgb", info)

        if lstm_model_dir:
            info = _load_lstm_direction_model(lstm_model_dir, device)
            models["dir_lstm"] = info
            _register_direction_entry("lstm", "lstm", info)

        if transformer_model_dir:
            info = _load_transformer_direction_model(transformer_model_dir, device)
            models["dir_transformer"] = info
            _register_direction_entry("transformer", "transformer", info)

    if not direction_entries:
        raise ValueError("At least one direction model must be provided.")

    models["direction_models"] = direction_entries
    return models


def _normalize_dataset_horizon(value: int | float) -> float:
    numeric = float(value)
    if numeric <= 0 or math.isnan(numeric):
        raise ValueError(f"Invalid horizon {value}")
    return round(numeric, _HORIZON_PRECISION)


def _residual_suffix(horizon: float) -> str:
    if float(horizon).is_integer():
        return f"{int(horizon)}h"
    return f"{horizon:g}h"


def _extract_residual_series(
    dataset_npz: np.lib.npyio.NpzFile,
    horizon: float,
    base_horizon: float,
) -> Optional[np.ndarray]:
    if math.isclose(horizon, base_horizon, abs_tol=10 ** (-_HORIZON_PRECISION)):
        candidates = ("y_val", "y_train", "y_test")
    else:
        prefix = f"y_ret{_residual_suffix(horizon)}"
        candidates = (f"{prefix}_val", f"{prefix}_train", f"{prefix}_test")

    for key in candidates:
        if key in dataset_npz.files:
            values = np.asarray(dataset_npz[key], dtype=np.float64)
            if values.size > 1:
                return values
    return None


def load_residual_std_from_dataset(
    dataset_npz_path: str,
    horizons: Iterable[int | float],
    fallback_std: float = DEFAULT_RESIDUAL_STD,
    base_horizon: float = 1.0,
) -> Dict[float, float]:
    if not os.path.exists(dataset_npz_path):
        raise FileNotFoundError(f"Dataset NPZ not found at {dataset_npz_path}")

    resolved_horizons = sorted(
        {
            _normalize_dataset_horizon(h)
            for h in horizons
            if float(h) > 0
        }
    )
    if not resolved_horizons:
        return {}

    residuals: Dict[float, float] = {}
    fallback_triggered = False

    with np.load(dataset_npz_path, allow_pickle=True) as dataset_npz:
        available = set(dataset_npz.files)
        for horizon in resolved_horizons:
            suffix = _residual_suffix(horizon)
            metric_key = f"metrics_ret_std_{suffix}"
            residual_std: Optional[float] = None

            if metric_key in available:
                metric_value = np.asarray(dataset_npz[metric_key], dtype=np.float64)
                if metric_value.size:
                    residual_std = float(metric_value.reshape(-1)[0])

            if residual_std is None:
                series = _extract_residual_series(dataset_npz, horizon, base_horizon)
                if series is not None:
                    residual_std = float(np.std(series, ddof=1))

            if residual_std is None or residual_std <= 0.0 or math.isnan(residual_std):
                residual_std = float(fallback_std)
                fallback_triggered = True

            residuals[horizon] = max(residual_std, MIN_RESIDUAL_STD)

    global _RESIDUAL_STD_WARNED
    if fallback_triggered and not _RESIDUAL_STD_WARNED:
        print(
            f"Warning: missing residual std metrics in {dataset_npz_path}; using fallback {fallback_std:.4f}.",
            file=sys.stderr,
        )
        _RESIDUAL_STD_WARNED = True

    return residuals


def _iter_sequence_model_infos(models: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    seen: Set[int] = set()
    for entry in models.get("direction_models", []):
        if entry.get("type") not in _SEQUENCE_MODEL_TYPES:
            continue
        info = entry.get("info")
        if info is None:
            continue
        ident = id(info)
        if ident in seen:
            continue
        seen.add(ident)
        yield info

    for model_type in _SEQUENCE_MODEL_TYPES:
        key = f"dir_{model_type}"
        info = models.get(key)
        if info is None:
            continue
        ident = id(info)
        if ident in seen:
            continue
        seen.add(ident)
        yield info


def populate_sequence_cache_from_prepared(prepared: PreparedData, models: Dict[str, Any]) -> None:
    """Populate cached scaled feature matrices required for sequence models."""

    sequence_models = list(_iter_sequence_model_infos(models))
    if not sequence_models:
        return

    base_features = prepared.X_all_ordered
    default_scaled = prepared.scaler.transform(base_features).astype(np.float32)
    default_scaled_df = pd.DataFrame(default_scaled, columns=prepared.feature_names)

    for model_info in sequence_models:
        model_feature_names = list(model_info.get("feature_names", []))
        context = f"sequence_model_{model_info.get('seq_len', 'unknown')}"

        scaler_mean = model_info.get("scaler_mean")
        scaler_std = model_info.get("scaler_std")

        if scaler_mean is not None and scaler_std is not None:
            feature_frame = base_features.copy()
            if model_feature_names:
                feature_frame = _ensure_feature_columns(feature_frame, model_feature_names, context)
                feature_frame = feature_frame.reindex(columns=model_feature_names)

            mean_arr = np.asarray(scaler_mean, dtype=np.float32)
            std_arr = np.asarray(scaler_std, dtype=np.float32)
            std_arr[std_arr == 0.0] = 1.0
            matrix = feature_frame.to_numpy(dtype=np.float32, copy=False)
            scaled_matrix = (matrix - mean_arr) / std_arr
        else:
            scaled_frame = default_scaled_df.copy()
            if model_feature_names:
                scaled_frame = _ensure_feature_columns(scaled_frame, model_feature_names, context)
                scaled_frame = scaled_frame.reindex(columns=model_feature_names)
            scaled_matrix = scaled_frame.to_numpy(dtype=np.float32, copy=False)

        model_info["scaled_features"] = scaled_matrix


def _sequence_model_probability(model_info: Dict[str, Any], index: int) -> Optional[float]:
    seq_len = int(model_info.get("seq_len", 0))
    if seq_len <= 0:
        return None

    scaled_features = model_info.get("scaled_features")
    if scaled_features is None:
        raise RuntimeError("Sequence model missing precomputed feature matrix.")

    if index + 1 < seq_len:
        return None

    start = index + 1 - seq_len
    window = scaled_features[start : index + 1].astype(np.float32, copy=False)
    tensor = torch.from_numpy(window).unsqueeze(0).to(model_info["device"])
    model: torch.nn.Module = model_info["model"]
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def _tree_model_probabilities(
    info: Dict[str, Any],
    scaled_frame: pd.DataFrame,
    context: str,
) -> np.ndarray:
    model = info.get("model")
    _ensure_estimator_type(model, "classifier")
    feature_names = info.get("feature_names")
    model_frame = scaled_frame
    if feature_names:
        model_frame = _ensure_feature_columns(model_frame.copy(), list(feature_names), context)
        model_input = model_frame[list(feature_names)].to_numpy()
    else:
        model_input = model_frame.to_numpy()
    return np.asarray(model.predict_proba(model_input)[:, 1], dtype=np.float64)


def _direction_entry_probability(
    entry: Mapping[str, Any],
    index: int,
    *,
    scaled_row: pd.DataFrame,
) -> Optional[float]:
    model_type = str(entry.get("type", "")).lower()
    info = entry.get("info")
    if not isinstance(info, Mapping):
        return None
    if model_type in _TREE_MODEL_TYPES:
        probs = _tree_model_probabilities(dict(info), scaled_row, f"direction_model_{entry.get('name', model_type)}")
        return float(probs[0]) if probs.size else None
    if model_type in _SEQUENCE_MODEL_TYPES:
        return _sequence_model_probability(dict(info), index)
    return None


def _build_direction_history(
    prepared: PreparedData,
    index: int,
    direction_entries: Sequence[Mapping[str, Any]],
    *,
    lookback_bars: int,
) -> Dict[str, np.ndarray]:
    if lookback_bars <= 1 or not direction_entries:
        return {}

    start = max(0, index - int(lookback_bars) + 1)
    if start >= index:
        return {}

    base_features = prepared.X_all_ordered.iloc[start : index + 1]
    scaled = prepared.scaler.transform(base_features)
    scaled_frame = pd.DataFrame(scaled, columns=prepared.feature_names)
    history: Dict[str, np.ndarray] = {}

    for entry in direction_entries:
        name = str(entry.get("name") or entry.get("type") or "")
        model_type = str(entry.get("type", "")).lower()
        info = entry.get("info")
        if not name or not isinstance(info, Mapping):
            continue
        if model_type in _TREE_MODEL_TYPES:
            history[name] = _tree_model_probabilities(dict(info), scaled_frame, f"direction_model_history_{name}")
            continue
        if model_type in _SEQUENCE_MODEL_TYPES:
            values: list[float] = []
            for history_index in range(start, index + 1):
                prob = _sequence_model_probability(dict(info), history_index)
                if prob is not None:
                    values.append(float(prob))
            if values:
                history[name] = np.asarray(values, dtype=np.float64)
    return history


def compute_signal_for_index(
    prepared: PreparedData,
    index: int,
    models: Dict[str, Any],
    p_up_min: float,
    ret_min: float,
    *,
    horizon: float | None = None,
    dir_model_weights: Optional[Dict[str, float]] = None,
    direction_ensemble_policy: Optional[Mapping[str, Any]] = None,
    volatility_snapshot: Optional[Mapping[str, float]] = None,
    volatility_policy: Optional[Mapping[str, Any]] = None,
    p_up_calibration: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if not (0 <= index < len(prepared.df_all)):
        raise IndexError("Index out of range for prepared data.")

    ts_value = prepared.df_all["ts"].iloc[index]
    X_row = prepared.X_all_ordered.iloc[[index]]
    X_scaled = prepared.scaler.transform(X_row)
    X_scaled_df = pd.DataFrame(X_scaled, columns=prepared.feature_names)

    reg = models["reg"]
    _ensure_estimator_type(reg, "regressor")
    target_scale = float(models.get("reg_target_scale", 1.0)) or 1.0
    dir_model = models.get("dir")
    if dir_model is not None:
        _ensure_estimator_type(dir_model, "classifier")

    reg_feature_names = models.get("reg_feature_names")
    if reg_feature_names:
        X_scaled_df = _ensure_feature_columns(X_scaled_df, reg_feature_names, "regression_model")
        reg_input = X_scaled_df[reg_feature_names].to_numpy()
    else:
        reg_input = X_scaled

    ret_pred_arr = reg.predict(reg_input)
    ret_pred = float(ret_pred_arr[0])
    if target_scale != 0:
        ret_pred /= target_scale

    probabilities: Dict[str, float] = {}
    display_labels = {
        "xgb": "xgboost",
        "lstm": "lstm",
        "bilstm": "bi-lstm",
        "gru": "gru",
        "cnn_lstm": "cnn-lstm",
        "cnn_bilstm": "cnn-bilstm",
        "garch_lstm": "garch-lstm",
        "transformer": "transformer",
        "lgbm": "lightgbm",
    }
    direction_entries = list(models.get("direction_models", []))
    if direction_entries:
        for entry in direction_entries:
            name = str(entry.get("name") or entry.get("type") or "").lower()
            prob = _direction_entry_probability(entry, index, scaled_row=X_scaled_df)
            if name and prob is not None:
                probabilities[name] = float(prob)
    else:
        if dir_model is not None:
            dir_feature_names = models.get("dir_feature_names")
            if dir_feature_names:
                X_scaled_df = _ensure_feature_columns(X_scaled_df, dir_feature_names, "direction_model")
                dir_input = X_scaled_df[dir_feature_names].to_numpy()
            else:
                dir_input = X_scaled
            p_up_arr = dir_model.predict_proba(dir_input)[:, 1]
            probabilities["xgb"] = float(p_up_arr[0])

        for model_type in _SEQUENCE_MODEL_ITER_ORDER:
            info = models.get(f"dir_{model_type}")
            if info is None:
                continue
            seq_prob = _sequence_model_probability(info, index)
            if seq_prob is not None:
                probabilities[model_type] = seq_prob

    p_up: Optional[float]
    direction_model_kind: Optional[str]
    ensemble_debug: Dict[str, Any] = {
        "selected_models": sorted(probabilities.keys()),
        "selected_groups": [],
        "missing_preferred_groups": [],
        "effective_weights": {},
        "base_weights": {},
        "rejected_models": [],
        "policy_applied": False,
    }

    if probabilities:
        if dir_model_weights:
            applicable_weights = {k: v for k, v in dir_model_weights.items() if k in probabilities}
        else:
            applicable_weights = {}

        selected_probabilities = dict(probabilities)
        effective_weights = dict(applicable_weights)

        if direction_ensemble_policy and bool(direction_ensemble_policy.get("enabled", False)) and len(probabilities) > 1:
            history = _build_direction_history(
                prepared,
                index,
                direction_entries,
                lookback_bars=int(direction_ensemble_policy.get("lookback_bars", 0) or 0),
            )
            selection = select_diverse_models(
                probabilities,
                applicable_weights,
                history=history,
                priority_order=direction_ensemble_policy.get("priority_order"),
                preferred_groups=direction_ensemble_policy.get("preferred_groups"),
                max_active_models=direction_ensemble_policy.get("max_active_models"),
                model_groups=direction_ensemble_policy.get("model_groups"),
                max_models_per_group=direction_ensemble_policy.get("max_models_per_group"),
                max_correlation=direction_ensemble_policy.get("max_correlation"),
                min_mean_abs_probability_gap=direction_ensemble_policy.get("min_mean_abs_probability_gap"),
                min_history_points=int(direction_ensemble_policy.get("min_history_points", 0) or 0),
            )
            selected_names = selection.get("selected_models") or list(probabilities.keys())
            selected_probabilities = {
                name: value
                for name, value in probabilities.items()
                if name in set(selected_names)
            }
            effective_weights = {
                name: float(value)
                for name, value in (selection.get("effective_weights") or {}).items()
                if name in selected_probabilities
            }
            ensemble_debug = {
                "selected_models": list(selected_names),
                "selected_groups": list(selection.get("selected_groups", [])),
                "missing_preferred_groups": list(selection.get("missing_preferred_groups", [])),
                "effective_weights": effective_weights,
                "base_weights": selection.get("base_weights", {}),
                "rejected_models": selection.get("rejected_models", []),
                "pairwise": selection.get("pairwise", []),
                "policy_applied": True,
            }
        else:
            ensemble_debug["selected_groups"] = []
            ensemble_debug["effective_weights"] = effective_weights
            ensemble_debug["base_weights"] = applicable_weights

        component_summary = summarize_component_probabilities(
            probabilities,
            model_groups=(direction_ensemble_policy or {}).get("model_groups") if isinstance(direction_ensemble_policy, Mapping) else None,
        )
        pairwise_summary = summarize_pairwise_history(ensemble_debug.get("pairwise"))
        component_summary.update(pairwise_summary)
        component_summary["direction_ensemble_selected_count"] = float(len(selected_probabilities))
        component_summary["direction_ensemble_rejected_count"] = float(len(ensemble_debug.get("rejected_models", [])))
        component_summary["direction_ensemble_missing_preferred_group_count"] = float(len(ensemble_debug.get("missing_preferred_groups", [])))
        ensemble_debug["component_summary"] = component_summary

        if effective_weights:
            try:
                p_up = weighted_average(selected_probabilities, effective_weights)
            except ValueError:
                p_up = simple_average(selected_probabilities.values())
        else:
            p_up = simple_average(selected_probabilities.values())

        direction_model_kind = (
            display_labels.get(next(iter(selected_probabilities)), next(iter(selected_probabilities)))
            if len(selected_probabilities) == 1
            else "ensemble"
        )
    else:
        if dir_model is None and any(models.get(f"dir_{model_type}") for model_type in _SEQUENCE_MODEL_ITER_ORDER):
            p_up = 0.5
            direction_model_kind = "fallback"
        else:
            raise RuntimeError("No direction model available to compute probabilities.")

    if p_up_calibration and horizon is not None:
        key = f"{horizon:g}h" if horizon >= 1 else f"{horizon * 60:g}m"
        params = p_up_calibration.get(key)
        if params:
            p_up = _apply_platt_calibration(float(p_up), params)

    effective_p_up_min = p_up_min
    ret_min_effective = ret_min
    volatility_block: Optional[Dict[str, Any]] = None
    block_trade = False

    if volatility_snapshot or volatility_policy:
        policy = dict(volatility_policy or {})
        mode = str(policy.get("mode") or "ceiling")
        metric_key = str(policy.get("volatility_metric") or _VOLATILITY_METRIC_DEFAULT)
        ceiling_raw = policy.get("volatility_ceiling")
        multiplier = float(policy.get("volatility_mult", _VOLATILITY_MULT_DEFAULT))
        metric_value = None
        if volatility_snapshot:
            metric_value = volatility_snapshot.get(metric_key)
            if metric_value is not None:
                try:
                    metric_value = float(metric_value)
                except (TypeError, ValueError):
                    metric_value = None

        percentile_value: Optional[float] = None
        triggered = False
        hard_block = False
        ceiling: Optional[float]

        if mode == "percentile":
            percentiles = policy.get("percentiles")
            if percentiles is not None and 0 <= index < len(percentiles):
                pct = percentiles[index]
                if pct is not None and not math.isnan(pct):
                    percentile_value = float(pct)
            calm_pct = float(policy.get("calm_pct", 0.7))
            extreme_pct = float(policy.get("extreme_pct", 0.9))
            elevated_scale = float(policy.get("elevated_scale", 0.5))
            extreme_scale = float(policy.get("extreme_scale", 1.0))
            ret_scale = float(policy.get("ret_scale", 0.0))
            block_extreme = bool(policy.get("block_extreme", True))
            ceiling = float(ceiling_raw) if isinstance(ceiling_raw, (int, float)) else None

            if percentile_value is not None:
                if percentile_value <= calm_pct:
                    pass
                elif percentile_value < extreme_pct:
                    span = max(extreme_pct - calm_pct, 1e-6)
                    progress = (percentile_value - calm_pct) / span
                    scale = 1.0 + elevated_scale * progress
                    effective_p_up_min = p_up_min * scale
                    if ret_scale:
                        ret_min_effective = ret_min * (1.0 + ret_scale * progress)
                else:
                    triggered = True
                    effective_p_up_min = p_up_min * (1.0 + extreme_scale)
                    if ret_scale:
                        ret_min_effective = ret_min * (1.0 + ret_scale)
                    hard_block = block_extreme
            else:
                percentile_value = math.nan
        else:
            triggered = False
            if metric_value is not None and ceiling_raw is not None:
                try:
                    ceiling = float(ceiling_raw)
                except (TypeError, ValueError):
                    ceiling = None
                if ceiling is not None and metric_value > ceiling:
                    triggered = True
                    effective_p_up_min = p_up_min * max(multiplier, 1.0)
            else:
                ceiling = float(ceiling_raw) if isinstance(ceiling_raw, (int, float)) else None

        snapshot_values = {}
        for key, value in (volatility_snapshot or {}).items():
            if value is None:
                continue
            try:
                snapshot_values[key] = float(value)
            except (TypeError, ValueError):
                continue

        block_trade = bool(hard_block and triggered)
        volatility_block = {
            "mode": mode,
            "metric": metric_key,
            "current": metric_value,
            "ceiling": ceiling,
            "multiplier": None if mode == "percentile" else multiplier,
            "percentile": percentile_value,
            "calm_percentile": policy.get("calm_pct"),
            "extreme_percentile": policy.get("extreme_pct"),
            "triggered": triggered,
            "hard_block": block_trade,
            "snapshot": snapshot_values,
        }
        if mode == "percentile":
            volatility_block.update(
                {
                    "elevated_scale": policy.get("elevated_scale"),
                    "extreme_scale": policy.get("extreme_scale"),
                    "ret_scale": policy.get("ret_scale"),
                }
            )

    signal_ensemble = int((p_up >= effective_p_up_min) and (ret_pred >= ret_min_effective))
    signal_dir_only = int(p_up >= 0.5)

    if block_trade:
        signal_ensemble = 0

    result = {
        "ts": format_ts_iso(ts_value),
        "p_up": p_up,
        "ret_pred": ret_pred,
        "signal_ensemble": signal_ensemble,
        "signal_dir_only": signal_dir_only,
    }

    if probabilities:
        result["p_up_components"] = probabilities.copy()
        for name, value in probabilities.items():
            result[f"p_up_{name}"] = value
        result["direction_ensemble"] = ensemble_debug
        component_summary = ensemble_debug.get("component_summary") if isinstance(ensemble_debug, Mapping) else None
        if isinstance(component_summary, Mapping):
            for key, value in component_summary.items():
                result[str(key)] = float(value)

    if direction_model_kind is not None:
        result["direction_model_kind"] = direction_model_kind

    if volatility_block is not None:
        volatility_block["p_up_min_effective"] = effective_p_up_min
        volatility_block["ret_min_effective"] = ret_min_effective
        result["volatility"] = volatility_block
        result["volatility_flag"] = bool(volatility_block["triggered"])

    trend_entry = models.get("trend_ignition")
    if trend_entry is not None:
        classifier = trend_entry.get("model")
        if classifier is not None:
            _ensure_estimator_type(classifier, "classifier")
            ti_feature_names = trend_entry.get("feature_names")
            if ti_feature_names:
                ti_frame = _ensure_feature_columns(X_scaled_df, ti_feature_names, "trend_ignition_model")
                ti_input = ti_frame[ti_feature_names].to_numpy()
            else:
                ti_input = X_scaled
            proba = classifier.predict_proba(ti_input)[:, 1]
            result["p_trend_ignition"] = float(proba[0])

    return result
