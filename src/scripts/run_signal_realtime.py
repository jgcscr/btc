import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from google.auth.exceptions import DefaultCredentialsError
from sklearn.preprocessing import StandardScaler

from src.config_trading import (
    DEFAULT_DIR_MODEL_DIR_1H,
    DEFAULT_LSTM_MODEL_DIR_1H,
    DEFAULT_P_UP_MIN,
    DEFAULT_REG_MODEL_DIR_1H,
    DEFAULT_RET_MIN,
    OPTUNA_DIR_MODEL_DIR_1H,
    OPTUNA_LSTM_MODEL_DIR_1H,
    OPTUNA_P_UP_MIN_1H,
    OPTUNA_REG_MODEL_DIR_1H,
    OPTUNA_RET_MIN_1H,
)
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    find_row_index_for_ts,
    load_models,
    populate_sequence_cache_from_prepared,
    prepare_data_for_signals,
)
from src.trading.ensembles import parse_weight_spec
from src.utils import cloud_io

DEFAULT_REG_MODEL_DIR = DEFAULT_REG_MODEL_DIR_1H
DEFAULT_DIR_MODEL_DIR = DEFAULT_DIR_MODEL_DIR_1H
DEFAULT_P_UP_MIN_4H_CONFIRM = 0.55
LEGACY_LOG_COLUMNS = [
    "ts",
    "p_up",
    "ret_pred",
    "signal_ensemble",
    "signal_dir_only",
    "created_at",
    "notes",
]
META_NET_COLUMNS = [
    "meta_net_fee_20_10",
    "meta_net_fee_25_12",
    "meta_net_fee_30_15",
]

META_LABEL_TO_COLUMN = {
    "fee_20_10": "meta_net_fee_20_10",
    "fee_25_12": "meta_net_fee_25_12",
    "fee_30_15": "meta_net_fee_30_15",
}

LOG_COLUMNS = [
    "ts",
    "p_up",
    "p_up_xgb",
    "p_up_lstm",
    "p_up_transformer",
    "p_up_meta",
    "ret_pred",
    "signal_meta",
    "signal_ensemble",
    "signal_dir_only",
    "p_up_4h",
    "ret_pred_4h",
    "signal_1h4h_confirm",
    *META_NET_COLUMNS,
    "created_at",
    "notes",
]

MODEL_ROOT = Path(DEFAULT_REG_MODEL_DIR_1H).parent if DEFAULT_REG_MODEL_DIR_1H else Path("artifacts/models")


def parse_targets(argument: str) -> List[int]:
    tokens = [token.strip() for token in argument.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("--targets requires at least one horizon value")
    horizons: List[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise argparse.ArgumentTypeError(f"Invalid horizon: {token!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("Horizon values must be positive integers")
        horizons.append(value)
    return horizons


def _resolve_horizon_model_paths(horizon: int) -> Optional[tuple[str, str]]:
    reg_dir = MODEL_ROOT / f"xgb_ret{horizon}h_v1"
    dir_dir = MODEL_ROOT / f"xgb_dir{horizon}h_v1"
    reg_path = reg_dir / f"xgb_ret{horizon}h_model.json"
    dir_path = dir_dir / f"xgb_dir{horizon}h_model.json"
    if not reg_path.exists() or not dir_path.exists():
        return None
    return str(reg_path), str(dir_path)


def _project_price(close_value: float, log_return: float) -> float:
    return close_value * math.exp(log_return)


def _compute_additional_horizon_signal(
    horizon: int,
    prepared: PreparedData,
    index: int,
    p_up_min: float,
    ret_min: float,
) -> Optional[Dict[str, Any]]:
    resolved = _resolve_horizon_model_paths(horizon)
    if resolved is None:
        return None
    reg_path, dir_path = resolved
    models = load_models(reg_model_path=reg_path, dir_model_path=dir_path)
    _ensure_model_feature_coverage(prepared, models)
    signal = compute_signal_for_index(
        prepared=prepared,
        index=index,
        models=models,
        p_up_min=p_up_min,
        ret_min=ret_min,
    )
    signal["thresholds"] = {
        "p_up_min": float(p_up_min),
        "ret_min": float(ret_min),
    }
    return signal


def _resolve_input_artifact(
    path: Optional[str],
    stack: ExitStack,
    *,
    required: bool = True,
    descriptor: str = "artifact",
) -> Optional[str]:
    if not path:
        return None
    if not cloud_io.is_gcs_uri(path):
        return path

    try:
        local_path, cleanup = cloud_io.resolve_to_local(path)
    except FileNotFoundError:
        if required:
            raise
        print(f"Warning: {descriptor} not found at {path}; skipping.", file=sys.stderr)
        return None
    if cleanup:
        stack.callback(cleanup)
    return local_path


def _materialize_directory(path: Optional[str], stack: ExitStack, label: str) -> Optional[str]:
    if not path:
        return None
    if not cloud_io.is_gcs_uri(path):
        return path

    temp_root = Path(tempfile.mkdtemp(prefix="btc_model_dir_"))
    stack.callback(lambda: shutil.rmtree(temp_root, ignore_errors=True))

    bucket_name, blob_name = cloud_io.split_gcs_uri(path)
    blob_name = blob_name.strip("/")
    if not blob_name:
        raise ValueError(f"{label} must include a concrete object prefix when using a GCS URI.")

    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required to read model artifacts from GCS."
        ) from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if blob.exists():
        destination = temp_root / Path(blob_name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination)
        return str(destination.parent)

    prefix = f"{blob_name}/"
    blobs_iter = client.list_blobs(bucket_name, prefix=prefix)
    found = False
    for candidate in blobs_iter:
        name = candidate.name
        if not name or not name.startswith(prefix) or name.endswith("/"):
            continue
        relative = name[len(prefix) :]
        if not relative:
            continue
        destination = temp_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate.download_to_filename(destination)
        found = True

    if not found:
        raise FileNotFoundError(f"No files found under {path} for {label}.")

    return str(temp_root)


def _prepared_data_from_npz(dataset_path: str) -> PreparedData:
    with np.load(dataset_path, allow_pickle=True) as data:
        feature_names = data.get("feature_names")
        if feature_names is None:
            raise KeyError("NPZ dataset missing feature_names array for offline inference.")
        feature_names_list = [str(name) for name in feature_names.tolist()]

        required_keys = [
            "X_train",
            "X_val",
            "X_test",
            "y_train",
            "y_val",
            "y_test",
            "ts_all",
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"NPZ dataset missing required arrays for offline inference: {missing_keys}")

        X_train = np.asarray(data["X_train"], dtype=np.float64)
        X_val = np.asarray(data["X_val"], dtype=np.float64)
        X_test = np.asarray(data["X_test"], dtype=np.float64)
        y_train = np.asarray(data["y_train"], dtype=np.float64)
        y_val = np.asarray(data["y_val"], dtype=np.float64)
        y_test = np.asarray(data["y_test"], dtype=np.float64)
        ts_all = data["ts_all"].tolist()

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    if len(ts_all) != X_all.shape[0]:
        raise ValueError("NPZ dataset has mismatched ts_all length and feature rows.")

    scaler = StandardScaler()
    scaler.fit(X_train)

    df_all = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts_all, utc=True),
            "ret_1h": y_all,
        },
    )
    X_all_ordered = pd.DataFrame(X_all, columns=feature_names_list)

    return PreparedData(
        df_all=df_all,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names_list,
    )


def _prepare_data(args: argparse.Namespace) -> PreparedData:
    force_dataset = os.environ.get("RUN_SIGNAL_FORCE_DATASET", "0").lower() in {"1", "true", "yes"}
    if force_dataset:
        print("RUN_SIGNAL_FORCE_DATASET set; using offline NPZ dataset for features.", file=sys.stderr)
        setattr(args, "_offline_dataset", True)
        return _prepared_data_from_npz(args.dataset_path)

    try:
        prepared = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")
        setattr(args, "_offline_dataset", False)
        return prepared
    except DefaultCredentialsError as exc:
        print(
            f"Warning: default credentials unavailable ({exc}); falling back to offline NPZ dataset.",
            file=sys.stderr,
        )
        setattr(args, "_offline_dataset", True)
        return _prepared_data_from_npz(args.dataset_path)


def _ensure_model_feature_coverage(prepared: PreparedData, models: Dict[str, Any]) -> None:
    required_columns: List[str] = []

    reg_feature_names = models.get("reg_feature_names")
    if reg_feature_names:
        required_columns.extend(reg_feature_names)

    dir_feature_names = models.get("dir_feature_names")
    if dir_feature_names:
        required_columns.extend(dir_feature_names)

    for key in ("dir_lstm", "dir_transformer"):
        model_info = models.get(key) or {}
        feature_names = model_info.get("feature_names")
        if feature_names:
            required_columns.extend(feature_names)

    ordered_columns = prepared.X_all_ordered.columns.tolist()
    missing = [column for column in dict.fromkeys(required_columns) if column not in ordered_columns]
    if not missing:
        return

    for column in missing:
        prepared.X_all_ordered[column] = 0.0

    # Ensure columns remain in deterministic order with new additions at the end.
    prepared.feature_names = prepared.X_all_ordered.columns.tolist()
    prepared.X_all_ordered = prepared.X_all_ordered[prepared.feature_names]

    scaler = prepared.scaler
    zeros = np.zeros(len(missing), dtype=np.float64)
    ones = np.ones(len(missing), dtype=np.float64)
    scaler.mean_ = np.concatenate([scaler.mean_, zeros])
    scaler.var_ = np.concatenate([scaler.var_, ones])
    scaler.scale_ = np.concatenate([scaler.scale_, ones])
    scaler.n_features_in_ = scaler.mean_.shape[0]
    if hasattr(scaler, "feature_names_in_"):
        scaler.feature_names_in_ = np.concatenate([
            scaler.feature_names_in_,
            np.asarray(missing, dtype=object),
        ])

    sys.stderr.write(
        "[run_signal_realtime] augmented feature matrix with missing columns: "
        + ", ".join(missing[:10])
        + ("..." if len(missing) > 10 else "")
        + "\n",
    )
    sys.stderr.flush()


def _load_feature_config_for_4h(dataset_path: str) -> tuple[List[str], np.ndarray, np.ndarray]:
    if not dataset_path:
        raise ValueError("dataset_path_4h must be provided for 4h inference.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"4h dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        feature_names = data.get("feature_names")
        x_train = data.get("X_train")

    if feature_names is None or x_train is None:
        raise KeyError("multi-horizon dataset missing feature_names or X_train")

    feature_names_list = feature_names.tolist()
    x_train_arr = np.asarray(x_train, dtype=np.float64)
    mean = x_train_arr.mean(axis=0)
    std = x_train_arr.std(axis=0)
    std[std == 0.0] = 1.0

    return feature_names_list, mean, std


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a live-like trading signal using the latest curated row "
            "and append it to a paper-trade log, suitable for hourly cron runs."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("DATASET_1H_URI", "artifacts/datasets/btc_features_1h_splits.npz"),
        help="Path to the regression NPZ file (used for feature names and scaler reconstruction).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default=os.environ.get("REG_MODEL_DIR_1H", DEFAULT_REG_MODEL_DIR),
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default=os.environ.get("DIR_MODEL_DIR_1H", DEFAULT_DIR_MODEL_DIR),
        help="Directory containing direction model JSON (xgb_dir1h_model.json).",
    )
    parser.add_argument(
        "--lstm-model-dir",
        type=str,
        default=DEFAULT_LSTM_MODEL_DIR_1H,
        help="Optional directory containing an LSTM direction model (model.pt, summary.json).",
    )
    parser.add_argument(
        "--transformer-dir-model",
        type=str,
        default=None,
        help="Optional directory containing a transformer direction model (model.pt, summary.json).",
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=None,
        help="Optional comma-separated weights for direction models (e.g. transformer:2,lstm:1,xgb:1).",
    )
    parser.add_argument(
        "--meta-ensemble-config",
        type=str,
        default=os.environ.get("META_ENSEMBLE_URI", "artifacts/backtests/meta_ensemble_config.json"),
        help="Optional path to logistic meta-ensemble coefficients for realtime blending.",
    )
    parser.add_argument(
        "--lstm-device",
        type=str,
        default=None,
        help="Optional torch device override for LSTM inference (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Optional minimum sequence length to validate against the loaded LSTM model.",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Ensemble threshold for P(up).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Ensemble threshold for predicted ret_1h.",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=None,
        help="Comma-separated prediction horizons in hours (default: 1).",
    )
    parser.add_argument(
        "--use-optuna-profile",
        action="store_true",
        help="Override default 1h model dirs and thresholds with the Optuna-tuned profile.",
    )
    parser.add_argument(
        "--p-up-min-4h-confirm",
        type=float,
        default=DEFAULT_P_UP_MIN_4H_CONFIRM,
        help="4h p_up threshold for 1h+4h confirmation (signal_1h4h_confirm).",
    )
    parser.add_argument(
        "--dataset-path-4h",
        type=str,
        default=os.environ.get("DATASET_4H_URI"),
        help="Optional NPZ dataset path used to align 4h features/scaler (expects ret_4h target).",
    )
    parser.add_argument(
        "--reg-model-dir-4h",
        type=str,
        default=os.environ.get("REG_MODEL_DIR_4H"),
        help="Optional directory containing xgb_ret4h_model.json for 4h regression inference.",
    )
    parser.add_argument(
        "--dir-model-dir-4h",
        type=str,
        default=os.environ.get("DIR_MODEL_DIR_4H"),
        help="Optional directory containing xgb_dir4h_model.json for 4h direction inference.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=os.environ.get("LIVE_LOG_URI", "artifacts/live/paper_trade_realtime.csv"),
        help="Path to the CSV log file for live/paper trading signals.",
    )
    parser.add_argument(
        "--ts",
        type=str,
        default=None,
        help=(
            "Optional timestamp to evaluate instead of the latest bar (ISO8601/RFC3339). "
            "If omitted, uses the most recent available row."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, skip appending the result to the realtime log (validation only).",
    )
    return parser.parse_args()


def _apply_optuna_profile(args: argparse.Namespace) -> None:
    if not getattr(args, "use_optuna_profile", False):
        return

    if args.reg_model_dir == DEFAULT_REG_MODEL_DIR:
        args.reg_model_dir = OPTUNA_REG_MODEL_DIR_1H

    if args.dir_model_dir == DEFAULT_DIR_MODEL_DIR:
        args.dir_model_dir = OPTUNA_DIR_MODEL_DIR_1H

    if args.lstm_model_dir in (None, DEFAULT_LSTM_MODEL_DIR_1H):
        args.lstm_model_dir = OPTUNA_LSTM_MODEL_DIR_1H

    if args.p_up_min == DEFAULT_P_UP_MIN:
        args.p_up_min = OPTUNA_P_UP_MIN_1H

    if args.ret_min == DEFAULT_RET_MIN:
        args.ret_min = OPTUNA_RET_MIN_1H

    print(
        (
            "Optuna profile active (reg_model_dir="
            f"{args.reg_model_dir}, dir_model_dir={args.dir_model_dir}, "
            f"p_up_min={args.p_up_min}, ret_min={args.ret_min})"
        ),
    )


def _now_utc_iso() -> str:
    dt = datetime.now(timezone.utc)
    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def _load_last_logged_ts(log_path: str) -> Optional[str]:
    if not os.path.exists(log_path):
        return None

    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None

    if df.empty or "ts" not in df.columns:
        return None

    return str(df["ts"].iloc[-1])


def _load_log_with_fallback(log_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    try:
        with open(log_path, newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # skip header
            for raw in reader:
                if not raw:
                    continue

                if len(raw) == len(columns):
                    mapping = {columns[idx]: raw[idx] for idx in range(len(columns))}
                elif len(raw) == len(LEGACY_LOG_COLUMNS):
                    mapping = {
                        LEGACY_LOG_COLUMNS[idx]: raw[idx]
                        for idx in range(len(LEGACY_LOG_COLUMNS))
                    }
                else:
                    padded = list(raw)
                    if len(padded) < len(columns):
                        padded.extend(["" * (len(columns) - len(padded))])
                    mapping = {columns[idx]: padded[idx] for idx in range(len(columns))}

                rows.append(mapping)
    except OSError:
        return None

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    df = df[columns]
    return df


def _ensure_log_schema(log_path: str, columns: List[str]) -> None:
    if not os.path.exists(log_path):
        return

    try:
        existing_columns = pd.read_csv(log_path, nrows=0).columns.tolist()
    except Exception:
        existing_columns = []

    if existing_columns == columns:
        return

    try:
        df_existing = pd.read_csv(log_path)
    except Exception:
        df_existing = _load_log_with_fallback(log_path, columns)
        if df_existing is None:
            return
    else:
        for column in columns:
            if column not in df_existing.columns:
                df_existing[column] = ""
        df_existing = df_existing[columns]

    df_existing.to_csv(log_path, index=False)


def _append_to_log(log_path: str, row: Dict[str, Any], columns: List[str]) -> bool:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    defaults = {column: "" for column in columns}
    defaults.update(row)
    df_row = pd.DataFrame([defaults])[columns]

    if not os.path.exists(log_path):
        df_row.to_csv(log_path, index=False)
        return True

    _ensure_log_schema(log_path, columns)

    # Append without rewriting the whole file header
    df_row.to_csv(log_path, mode="a", header=False, index=False)
    return True


def _load_meta_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to load meta-ensemble config ({exc}); skipping meta blend.", file=sys.stderr)
        return None


def _compute_meta_outputs(sig: Dict[str, Any], meta_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    components = sig.get("p_up_components")
    if not isinstance(components, dict) or not components:
        return None

    feature_columns = meta_config.get("feature_columns", [])
    coefficients = meta_config.get("coefficients", [])
    if not feature_columns or len(feature_columns) != len(coefficients):
        return None

    intercept = float(meta_config.get("intercept", 0.0))
    threshold = float(meta_config.get("threshold", 0.5))

    feature_values: List[float] = []
    for column, coef in zip(feature_columns, coefficients):
        key = column
        if column.startswith("p_up_"):
            key = column[len("p_up_") :]
        value = components.get(key)
        if value is None:
            return None
        feature_values.append(float(value))

    linear_term = intercept
    for value, coef in zip(feature_values, coefficients):
        linear_term += float(coef) * value

    try:
        p_up_meta = 1.0 / (1.0 + math.exp(-linear_term))
    except OverflowError:
        p_up_meta = 0.0 if linear_term < 0 else 1.0

    signal_meta = int(p_up_meta >= threshold)
    try:
        ret_pred = float(sig["ret_pred"])
    except (KeyError, TypeError, ValueError):
        ret_pred = 0.0

    schedules = meta_config.get("schedules") or [
        {"fee_bps": 2.0, "slippage_bps": 1.0, "label": "fee_20_10"},
        {"fee_bps": 2.5, "slippage_bps": 1.2, "label": "fee_25_12"},
        {"fee_bps": 3.0, "slippage_bps": 1.5, "label": "fee_30_15"},
    ]

    meta_net: Dict[str, float] = {}
    for schedule in schedules:
        try:
            fee_bps = float(schedule["fee_bps"])
            slippage_bps = float(schedule["slippage_bps"])
            label = str(schedule["label"])
        except (KeyError, TypeError, ValueError):
            continue
        cost = (fee_bps + slippage_bps) / 10_000.0
        meta_net[label] = (ret_pred - cost) * signal_meta

    return {
        "p_up_meta": p_up_meta,
        "signal_meta": signal_meta,
        "meta_net": meta_net,
    }


def run_signal_realtime(args: argparse.Namespace) -> None:
    _apply_optuna_profile(args)

    log_destination = args.log_path

    with ExitStack() as stack:
        sys.stderr.write("[run_signal_realtime] starting execution\n")
        sys.stderr.flush()
        # Stage remote inputs (datasets/models/log) locally for the duration of the run.
        args.dataset_path = _resolve_input_artifact(args.dataset_path, stack)
        args.meta_ensemble_config = _resolve_input_artifact(
            getattr(args, "meta_ensemble_config", None),
            stack,
            required=False,
            descriptor="meta ensemble config",
        )
        args.dataset_path_4h = _resolve_input_artifact(args.dataset_path_4h, stack)

        args.reg_model_dir = _materialize_directory(args.reg_model_dir, stack, "reg_model_dir")
        args.dir_model_dir = _materialize_directory(args.dir_model_dir, stack, "dir_model_dir")
        args.lstm_model_dir = _materialize_directory(args.lstm_model_dir, stack, "lstm_model_dir")
        args.transformer_dir_model = _materialize_directory(
            args.transformer_dir_model,
            stack,
            "transformer_dir_model",
        )
        args.reg_model_dir_4h = _materialize_directory(args.reg_model_dir_4h, stack, "reg_model_dir_4h")
        args.dir_model_dir_4h = _materialize_directory(args.dir_model_dir_4h, stack, "dir_model_dir_4h")

        if not args.reg_model_dir:
            raise ValueError("reg_model_dir must be provided.")

        args.log_path = stack.enter_context(cloud_io.local_artifact(log_destination))

        meta_config = _load_meta_config(getattr(args, "meta_ensemble_config", None))

        prepared: PreparedData = _prepare_data(args)
        if getattr(args, "_offline_dataset", False):
            args.lstm_model_dir = None
            args.transformer_dir_model = None

        if args.ts is None:
            index = len(prepared.df_all) - 1
        else:
            index = find_row_index_for_ts(prepared.df_all, args.ts)

        requested_targets = args.targets or [1]
        target_set = {int(value) for value in requested_targets}
        if 1 not in target_set:
            target_set.add(1)
        target_list = sorted(target_set)

        close_value = float("nan")
        if "close" in prepared.df_all.columns and index >= 0:
            close_value = float(prepared.df_all["close"].iloc[index])

        reg_model_path = os.path.join(args.reg_model_dir, "xgb_ret1h_model.json")
        dir_model_path: Optional[str] = None
        if args.dir_model_dir:
            dir_model_path = os.path.join(args.dir_model_dir, "xgb_dir1h_model.json")

        models = load_models(
            reg_model_path=reg_model_path,
            dir_model_path=dir_model_path,
            lstm_model_dir=args.lstm_model_dir,
            transformer_model_dir=args.transformer_dir_model,
            device=args.lstm_device,
        )

        _ensure_model_feature_coverage(prepared, models)

        populate_sequence_cache_from_prepared(prepared, models)

        dir_model_weights = None
        if args.dir_model_weights:
            dir_model_weights = parse_weight_spec(args.dir_model_weights)

        if args.seq_len is not None:
            if len(prepared.df_all) < args.seq_len:
                raise RuntimeError(
                    f"Prepared dataset has insufficient rows ({len(prepared.df_all)}) for seq-len={args.seq_len}.",
                )

            for key in ("dir_lstm", "dir_transformer"):
                model_info = models.get(key)
                if model_info is None:
                    continue
                model_seq_len = int(model_info.get("seq_len", args.seq_len))
                if model_seq_len != int(args.seq_len):
                    print(
                        (
                            "Warning: requested seq_len="
                            f"{args.seq_len} but {key} expects {model_seq_len}; proceeding with model setting."
                        ),
                        file=sys.stderr,
                    )

        sig = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=args.p_up_min,
            ret_min=args.ret_min,
            dir_model_weights=dir_model_weights,
        )

        sig["thresholds"] = {
            "p_up_min": float(args.p_up_min),
            "ret_min": float(args.ret_min),
        }

        if meta_config:
            meta_outputs = _compute_meta_outputs(sig, meta_config)
            if meta_outputs:
                sig["p_up_meta"] = meta_outputs["p_up_meta"]
                sig["signal_meta"] = meta_outputs["signal_meta"]
                sig["meta_expected_net"] = meta_outputs["meta_net"]
            else:
                sig.setdefault("p_up_meta", None)
                sig.setdefault("signal_meta", None)
                sig.setdefault("meta_expected_net", {})
        else:
            sig.setdefault("p_up_meta", None)
            sig.setdefault("signal_meta", None)
            sig.setdefault("meta_expected_net", {})

        signals_by_horizon: Dict[int, Dict[str, Any]] = {1: sig}
        additional_targets = [h for h in target_list if h != 1]
        for horizon in additional_targets:
            horizon_signal = _compute_additional_horizon_signal(
                horizon=horizon,
                prepared=prepared,
                index=index,
                p_up_min=args.p_up_min,
                ret_min=args.ret_min,
            )
            if horizon_signal is None:
                print(
                    f"Warning: skipping {horizon}h horizon because model artifacts are unavailable.",
                    file=sys.stderr,
                )
                continue
            signals_by_horizon[horizon] = horizon_signal

        if (
            4 not in signals_by_horizon
            and args.dataset_path_4h
            and args.reg_model_dir_4h
            and args.dir_model_dir_4h
        ):
            try:
                feature_names_4h, feature_mean_4h, feature_std_4h = _load_feature_config_for_4h(args.dataset_path_4h)
                ordered_features = prepared.X_all_ordered
                missing_cols = [column for column in feature_names_4h if column not in ordered_features.columns]
                if missing_cols:
                    raise KeyError(f"Missing required 4h feature columns: {missing_cols}")

                live_features = ordered_features.iloc[[index]][feature_names_4h].to_numpy(dtype=np.float64)
                live_scaled = (live_features - feature_mean_4h) / feature_std_4h

                reg_model_path_4h = os.path.join(args.reg_model_dir_4h, "xgb_ret4h_model.json")
                dir_model_path_4h = os.path.join(args.dir_model_dir_4h, "xgb_dir4h_model.json")
                if not os.path.exists(reg_model_path_4h):
                    raise FileNotFoundError(f"Regression model not found: {reg_model_path_4h}")
                if not os.path.exists(dir_model_path_4h):
                    raise FileNotFoundError(f"Direction model not found: {dir_model_path_4h}")

                models_4h = load_models(
                    reg_model_path=reg_model_path_4h,
                    dir_model_path=dir_model_path_4h,
                )
                ret_pred_4h = float(models_4h["reg"].predict(live_scaled)[0])
                p_up_4h = float(models_4h["dir"].predict_proba(live_scaled)[:, 1][0])
                fallback_signal = {
                    "ts": sig.get("ts"),
                    "p_up": p_up_4h,
                    "ret_pred": ret_pred_4h,
                    "signal_ensemble": int((p_up_4h >= args.p_up_min) and (ret_pred_4h >= args.ret_min)),
                    "signal_dir_only": int(p_up_4h >= 0.5),
                    "p_up_components": {"xgb": p_up_4h},
                    "thresholds": {
                        "p_up_min": float(args.p_up_min),
                        "ret_min": float(args.ret_min),
                    },
                }
                signals_by_horizon[4] = fallback_signal
            except Exception as exc:
                print(
                    f"Warning: failed to compute 4h prediction ({exc}); proceeding without 4h confirmation.",
                    file=sys.stderr,
                )

        signal_1h4h_confirm: Optional[int] = None
        signal_4h = signals_by_horizon.get(4)
        if signal_4h is not None:
            sig["p_up_4h"] = signal_4h.get("p_up")
            sig["ret_pred_4h"] = signal_4h.get("ret_pred")

        p_up_4h = sig.get("p_up_4h")
        if p_up_4h is not None:
            try:
                p_up_4h_float = float(p_up_4h)
            except (TypeError, ValueError):
                p_up_4h_float = None
            if p_up_4h_float is not None:
                filter_4h = p_up_4h_float >= args.p_up_min_4h_confirm
                signal_1h4h_confirm = int(int(sig["signal_ensemble"]) == 1 and filter_4h)
                sig["signal_1h4h_confirm"] = signal_1h4h_confirm
                sig["thresholds"]["p_up_min_4h_confirm"] = float(args.p_up_min_4h_confirm)

        # Print signal JSON-like summary
        sys.stderr.write("[run_signal_realtime] emitting signal summary\n")
        sys.stderr.flush()
        predictions_summary: Dict[str, Dict[str, Any]] = {}
        close_field = None if math.isnan(close_value) else close_value
        for horizon_key, signal_data in sorted(signals_by_horizon.items()):
            horizon_label = f"{horizon_key}h"
            entry: Dict[str, Any] = {
                "timestamp": signal_data.get("ts", sig.get("ts")),
                "horizon_hours": horizon_key,
                "close": close_field,
                "signal_ensemble": int(signal_data.get("signal_ensemble", 0)),
                "signal_dir_only": int(signal_data.get("signal_dir_only", 0)),
                "thresholds": signal_data.get(
                    "thresholds",
                    {
                        "p_up_min": float(args.p_up_min),
                        "ret_min": float(args.ret_min),
                    },
                ),
            }

            p_up_value = signal_data.get("p_up")
            entry["p_up"] = float(p_up_value) if p_up_value is not None else None

            ret_pred_value = signal_data.get("ret_pred")
            entry["ret_pred"] = float(ret_pred_value) if ret_pred_value is not None else None

            if close_field is not None and entry["ret_pred"] is not None:
                entry["projected_price"] = _project_price(close_field, float(entry["ret_pred"]))
            else:
                entry["projected_price"] = None

            comps = signal_data.get("p_up_components")
            if isinstance(comps, dict) and comps:
                entry["p_up_components"] = comps

            for optional_key in (
                "p_up_xgb",
                "p_up_lstm",
                "p_up_transformer",
                "p_up_meta",
                "signal_meta",
                "meta_expected_net",
                "direction_model_kind",
            ):
                if optional_key in signal_data and signal_data[optional_key] not in (None, {}):
                    entry[optional_key] = signal_data[optional_key]

            predictions_summary[horizon_label] = entry

        payload = {
            "generated_at": _now_utc_iso(),
            "requested_horizons": target_list,
            "predictions": predictions_summary,
            "signal": sig,
        }

        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        sys.stdout.flush()

        current_ts = sig["ts"]
        last_ts = _load_last_logged_ts(args.log_path)

        if last_ts is not None and last_ts == current_ts:
            print(
                f"No new bar; last ts={last_ts} equal to current ts={current_ts}; skipping append.",
            )
            return

        meta_net = sig.get("meta_expected_net") or {}

        log_row = {
            "ts": current_ts,
            "p_up": sig.get("p_up", ""),
            "p_up_xgb": sig.get("p_up_xgb", ""),
            "p_up_lstm": sig.get("p_up_lstm", ""),
            "p_up_transformer": sig.get("p_up_transformer", ""),
            "p_up_meta": sig.get("p_up_meta", ""),
            "ret_pred": sig.get("ret_pred", ""),
            "signal_meta": sig.get("signal_meta", ""),
            "signal_ensemble": sig.get("signal_ensemble", ""),
            "signal_dir_only": sig.get("signal_dir_only", ""),
            "p_up_4h": sig.get("p_up_4h", ""),
            "ret_pred_4h": sig.get("ret_pred_4h", ""),
            "signal_1h4h_confirm": signal_1h4h_confirm if signal_1h4h_confirm is not None else "",
            "created_at": _now_utc_iso(),
            "notes": "",
        }

        for label, column in META_LABEL_TO_COLUMN.items():
            log_row[column] = meta_net.get(label, "")

        if getattr(args, "dry_run", False):
            print("Dry run enabled; skipping append to realtime log.")
            return

        appended = _append_to_log(args.log_path, log_row, LOG_COLUMNS)
        if appended:
            print(f"Appended signal for ts={current_ts} to {log_destination}")


def main() -> None:
    args = _parse_args()
    run_signal_realtime(args)


if __name__ == "__main__":
    main()
