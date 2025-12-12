import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.config_trading import DEFAULT_P_UP_MIN, DEFAULT_RET_MIN
from src.trading.signals import PreparedData, compute_signal_for_index, find_row_index_for_ts, load_models, prepare_data_for_signals

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
LOG_COLUMNS = [
    "ts",
    "p_up",
    "ret_pred",
    "signal_ensemble",
    "signal_dir_only",
    "p_up_4h",
    "ret_pred_4h",
    "signal_1h4h_confirm",
    "created_at",
    "notes",
]


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
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression NPZ file (used for feature names and scaler reconstruction).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default="artifacts/models/xgb_ret1h_v1",
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default="artifacts/models/xgb_dir1h_v1",
        help="Directory containing direction model JSON (xgb_dir1h_model.json).",
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
        "--p-up-min-4h-confirm",
        type=float,
        default=DEFAULT_P_UP_MIN_4H_CONFIRM,
        help="4h p_up threshold for 1h+4h confirmation (signal_1h4h_confirm).",
    )
    parser.add_argument(
        "--dataset-path-4h",
        type=str,
        default=None,
        help="Optional NPZ dataset path used to align 4h features/scaler (expects ret_4h target).",
    )
    parser.add_argument(
        "--reg-model-dir-4h",
        type=str,
        default=None,
        help="Optional directory containing xgb_ret4h_model.json for 4h regression inference.",
    )
    parser.add_argument(
        "--dir-model-dir-4h",
        type=str,
        default=None,
        help="Optional directory containing xgb_dir4h_model.json for 4h direction inference.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="artifacts/live/paper_trade_realtime.csv",
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
    return parser.parse_args()


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
                        padded.extend([""] * (len(columns) - len(padded)))
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


def run_signal_realtime(args: argparse.Namespace) -> None:
    # Prepare data and models using the same helpers as run_signal_once/backtest
    prepared: PreparedData = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")

    if args.ts is None:
        index = len(prepared.df_all) - 1
    else:
        index = find_row_index_for_ts(prepared.df_all, args.ts)

    models = load_models(
        reg_model_path=os.path.join(args.reg_model_dir, "xgb_ret1h_model.json"),
        dir_model_path=os.path.join(args.dir_model_dir, "xgb_dir1h_model.json"),
    )

    sig = compute_signal_for_index(
        prepared=prepared,
        index=index,
        models=models,
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
    )

    if (
        args.dataset_path_4h
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

            sig["p_up_4h"] = p_up_4h
            sig["ret_pred_4h"] = ret_pred_4h
        except Exception as exc:
            print(
                f"Warning: failed to compute 4h prediction ({exc}); proceeding without 4h confirmation.",
                file=sys.stderr,
            )

    sig["thresholds"] = {
        "p_up_min": float(args.p_up_min),
        "ret_min": float(args.ret_min),
    }

    signal_1h4h_confirm: Optional[int] = None
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
    print(json.dumps(sig, indent=2))

    current_ts = sig["ts"]
    last_ts = _load_last_logged_ts(args.log_path)

    if last_ts is not None and last_ts == current_ts:
        print(
            f"No new bar; last ts={last_ts} equal to current ts={current_ts}; skipping append.",
        )
        return

    log_row = {
        "ts": current_ts,
        "p_up": sig["p_up"],
        "ret_pred": sig["ret_pred"],
        "signal_ensemble": sig["signal_ensemble"],
        "signal_dir_only": sig["signal_dir_only"],
        "p_up_4h": sig.get("p_up_4h", ""),
        "ret_pred_4h": sig.get("ret_pred_4h", ""),
        "signal_1h4h_confirm": signal_1h4h_confirm if signal_1h4h_confirm is not None else "",
        "created_at": _now_utc_iso(),
        "notes": "",
    }

    appended = _append_to_log(args.log_path, log_row, LOG_COLUMNS)
    if appended:
        print(f"Appended signal for ts={current_ts} to {args.log_path}")


def main() -> None:
    args = _parse_args()
    run_signal_realtime(args)


if __name__ == "__main__":
    main()
