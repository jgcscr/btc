import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config_trading import (
    DEFAULT_DIR_MODEL_DIR_1H,
    DEFAULT_LSTM_MODEL_DIR_1H,
    DEFAULT_TRANSFORMER_MODEL_DIR_1H,
    DEFAULT_P_UP_MIN,
    DEFAULT_REG_MODEL_DIR_1H,
    DEFAULT_RET_MIN,
    DEFAULT_DIR_MODEL_WEIGHTS_1H,
    OPTUNA_DIR_MODEL_DIR_1H,
    OPTUNA_LSTM_MODEL_DIR_1H,
    OPTUNA_DIR_MODEL_WEIGHTS_1H,
    OPTUNA_TRANSFORMER_MODEL_DIR_1H,
    OPTUNA_P_UP_MIN_1H,
    OPTUNA_REG_MODEL_DIR_1H,
    OPTUNA_RET_MIN_1H,
)
from src.data.binance_klines import (
    BinanceAPIError,
    fetch_funding_rates,
    fetch_futures_klines,
    fetch_open_interest,
    fetch_spot_klines,
)
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    load_models,
    populate_sequence_cache_from_prepared,
    prepare_data_for_signals_from_ohlcv,
)
from src.trading.ensembles import parse_weight_spec


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_N_BARS = 500
DEFAULT_LOG_PATH = "artifacts/live/paper_trade_realtime.csv"
DEFAULT_REG_MODEL_DIR = DEFAULT_REG_MODEL_DIR_1H
DEFAULT_DIR_MODEL_DIR = DEFAULT_DIR_MODEL_DIR_1H
DEFAULT_TRANSFORMER_MODEL_DIR = DEFAULT_TRANSFORMER_MODEL_DIR_1H
DEFAULT_DIR_MODEL_WEIGHTS = DEFAULT_DIR_MODEL_WEIGHTS_1H
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
DEFAULT_FEATURE_LIST = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent Binance candles, reconstruct 1h features, and log a realtime "
            "prediction without relying on the curated BigQuery table."
        ),
    )
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Symbol to fetch (default: BTCUSDT).")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL, help="Kline interval (default: 1h).")
    parser.add_argument(
        "--n-bars",
        type=int,
        default=DEFAULT_N_BARS,
        help="Number of recent bars to pull from Binance (default: 500).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default=DEFAULT_REG_MODEL_DIR,
        help="Directory containing xgb_ret1h_model.json.",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default=DEFAULT_DIR_MODEL_DIR,
        help="Directory containing xgb_dir1h_model.json.",
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
        default=DEFAULT_TRANSFORMER_MODEL_DIR,
        help="Optional directory containing a transformer direction model (model.pt, summary.json).",
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=DEFAULT_DIR_MODEL_WEIGHTS,
        help="Optional comma-separated weights for direction models (e.g. transformer:2,lstm:1,xgb:1).",
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
        "--log-path",
        type=str,
        default=DEFAULT_LOG_PATH,
        help="CSV path to append realtime predictions to (v1-compatible schema).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Optional NPZ dataset path to read feature ordering from (default points to v1 dataset).",
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
        default=None,
        help="Optional NPZ dataset path (multi-horizon splits) for 4h scaling.",
    )
    parser.add_argument(
        "--reg-model-dir-4h",
        type=str,
        default=None,
        help="Optional directory containing xgb_ret4h_model.json.",
    )
    parser.add_argument(
        "--dir-model-dir-4h",
        type=str,
        default=None,
        help="Optional directory containing xgb_dir4h_model.json.",
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

    if args.transformer_dir_model in (None, DEFAULT_TRANSFORMER_MODEL_DIR_1H):
        args.transformer_dir_model = OPTUNA_TRANSFORMER_MODEL_DIR_1H

    if args.dir_model_weights in (None, "", DEFAULT_DIR_MODEL_WEIGHTS):
        args.dir_model_weights = OPTUNA_DIR_MODEL_WEIGHTS_1H

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


def _load_feature_names(dataset_path: str) -> List[str]:
    if not os.path.exists(dataset_path):
        return DEFAULT_FEATURE_LIST

    with np.load(dataset_path, allow_pickle=True) as data:
        feature_names = data.get("feature_names")
        if feature_names is None:
            return DEFAULT_FEATURE_LIST
        return feature_names.tolist()


def _merge_market_data(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    spot = fetch_spot_klines(symbol=symbol, interval=interval, limit=limit)

    try:
        futures = fetch_futures_klines(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIError as exc:
        print(
            f"Warning: futures endpoint unavailable ({exc}); falling back to spot-derived futures columns.",
            file=sys.stderr,
        )
        futures = spot[["ts", "open", "high", "low", "close", "volume"]].rename(
            columns={
                "open": "fut_open",
                "high": "fut_high",
                "low": "fut_low",
                "close": "fut_close",
                "volume": "fut_volume",
            },
        )

    df = pd.merge(spot, futures, on="ts", how="inner")

    try:
        oi = fetch_open_interest(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIError as exc:
        print(f"Warning: open interest unavailable ({exc}); filling with NaN.", file=sys.stderr)
        oi = pd.DataFrame({"ts": df["ts"], "open_interest": np.nan})
    df = pd.merge(df, oi, on="ts", how="left")

    try:
        funding = fetch_funding_rates(symbol=symbol, limit=limit)
    except BinanceAPIError as exc:
        print(f"Warning: funding rates unavailable ({exc}); defaulting to 0.", file=sys.stderr)
        funding = pd.DataFrame({"ts": df["ts"], "funding_rate": 0.0})

    df = pd.merge_asof(
        df.sort_values("ts"),
        funding.sort_values("ts"),
        on="ts",
        direction="backward",
    )

    df["open_interest"] = df["open_interest"].ffill()
    df["funding_rate"] = df["funding_rate"].ffill()
    df["open_interest"] = df["open_interest"].bfill()
    df["funding_rate"] = df["funding_rate"].fillna(0.0)

    if df["open_interest"].isna().all():
        df["open_interest"] = 0.0

    return df.sort_values("ts").reset_index(drop=True)


def _compute_feature_frame(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = df.copy()

    df["ma_close_7h"] = df["close"].rolling(window=7, min_periods=7).mean()
    df["ma_close_24h"] = df["close"].rolling(window=24, min_periods=24).mean()
    df["ma_ratio_7_24"] = df["ma_close_7h"] / df["ma_close_24h"]
    df["vol_24h"] = df["volume"].rolling(window=24, min_periods=24).sum()

    df = df.dropna(subset=feature_names)
    if df.empty:
        raise RuntimeError(
            "Insufficient data to compute feature row; try increasing --n-bars so 24h rolling stats are available.",
        )

    df = df[["ts", *feature_names]].copy()
    return df


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


def _now_utc_iso() -> str:
    dt = datetime.now(timezone.utc)
    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def _load_last_logged_ts(log_path: str) -> str | None:
    if not os.path.exists(log_path):
        return None

    try:
        df = pd.read_csv(log_path)
    except Exception:  # pragma: no cover - malformed log
        return None

    if df.empty or "ts" not in df.columns:
        return None

    return str(df["ts"].iloc[-1])


def _load_log_with_fallback(log_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    try:
        with open(log_path, newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
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


def _append_to_log(log_path: str, row: Dict[str, Any], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    defaults = {column: "" for column in columns}
    defaults.update(row)
    df_row = pd.DataFrame([defaults])[columns]

    if not os.path.exists(log_path):
        df_row.to_csv(log_path, index=False)
        return

    _ensure_log_schema(log_path, columns)

    df_row.to_csv(log_path, mode="a", header=False, index=False)


def _load_models(
    reg_dir: str,
    dir_dir: Optional[str],
    lstm_dir: Optional[str],
    lstm_device: Optional[str],
    transformer_dir: Optional[str],
) -> Dict[str, Any]:
    reg_model_path = os.path.join(reg_dir, "xgb_ret1h_model.json")
    if not os.path.exists(reg_model_path):
        raise FileNotFoundError(f"Regression model not found: {reg_model_path}")

    dir_model_path: Optional[str] = None
    if dir_dir:
        candidate = os.path.join(dir_dir, "xgb_dir1h_model.json")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Direction model not found: {candidate}")
        dir_model_path = candidate

    return load_models(
        reg_model_path=reg_model_path,
        dir_model_path=dir_model_path,
        lstm_model_dir=lstm_dir,
        transformer_model_dir=transformer_dir,
        device=lstm_device,
    )


def run_realtime_from_binance(args: argparse.Namespace) -> None:
    _apply_optuna_profile(args)

    try:
        raw_df = _merge_market_data(symbol=args.symbol, interval=args.interval, limit=args.n_bars)
    except BinanceAPIError as exc:
        raise SystemExit(f"Failed to fetch Binance market data: {exc}")

    feature_names = _load_feature_names(args.dataset_path)

    df_features = _compute_feature_frame(raw_df, feature_names)
    prepared: PreparedData = prepare_data_for_signals_from_ohlcv(df_features, feature_names=feature_names)

    models = _load_models(
        reg_dir=args.reg_model_dir,
        dir_dir=args.dir_model_dir,
        lstm_dir=args.lstm_model_dir,
        lstm_device=args.lstm_device,
        transformer_dir=args.transformer_dir_model,
    )

    populate_sequence_cache_from_prepared(prepared, models)

    dir_model_weights = None
    if args.dir_model_weights:
        dir_model_weights = parse_weight_spec(args.dir_model_weights)

    if args.seq_len is not None:
        if len(prepared.df_all) < args.seq_len:
            raise SystemExit(
                f"Insufficient rows after preprocessing ({len(prepared.df_all)}) for seq-len={args.seq_len}.",
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

    last_index = len(prepared.df_all) - 1
    if last_index < 0:
        raise SystemExit("No feature rows available after preprocessing; cannot produce realtime signal.")

    signal = compute_signal_for_index(
        prepared=prepared,
        index=last_index,
        models=models,
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
        dir_model_weights=dir_model_weights,
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

            live_features = ordered_features.iloc[[last_index]][feature_names_4h].to_numpy(dtype=np.float64)
            live_scaled = (live_features - feature_mean_4h) / feature_std_4h

            reg_model_path_4h = os.path.join(args.reg_model_dir_4h, "xgb_ret4h_model.json")
            dir_model_path_4h = os.path.join(args.dir_model_dir_4h, "xgb_dir4h_model.json")
            if not os.path.exists(reg_model_path_4h):
                raise FileNotFoundError(f"Regression model not found: {reg_model_path_4h}")
            if not os.path.exists(dir_model_path_4h):
                raise FileNotFoundError(f"Direction model not found: {dir_model_path_4h}")

            models_4h = load_models(reg_model_path=reg_model_path_4h, dir_model_path=dir_model_path_4h)
            ret_pred_4h = float(models_4h["reg"].predict(live_scaled)[0])
            p_up_4h = float(models_4h["dir"].predict_proba(live_scaled)[:, 1][0])

            signal["p_up_4h"] = p_up_4h
            signal["ret_pred_4h"] = ret_pred_4h
        except Exception as exc:
            print(
                f"Warning: failed to compute 4h prediction ({exc}); proceeding without 4h confirmation.",
                file=sys.stderr,
            )

    signal_1h4h_confirm: Optional[int] = None
    p_up_4h = signal.get("p_up_4h")
    if p_up_4h is not None:
        try:
            p_up_4h_float = float(p_up_4h)
        except (TypeError, ValueError):
            p_up_4h_float = None
        if p_up_4h_float is not None:
            filter_4h = p_up_4h_float >= args.p_up_min_4h_confirm
            signal_1h4h_confirm = int(int(signal["signal_ensemble"]) == 1 and filter_4h)

    summary = {
        "ts": signal["ts"],
        "p_up": signal["p_up"],
        "ret_pred": signal["ret_pred"],
        "signal_ensemble": int(signal["signal_ensemble"]),
        "signal_dir_only": int(signal["signal_dir_only"]),
        "source": "binance_direct_v1",
    }

    direction_model_kind = signal.get("direction_model_kind")
    if direction_model_kind is not None:
        summary["direction_model_kind"] = direction_model_kind

    if "p_up_4h" in signal:
        summary["p_up_4h"] = signal["p_up_4h"]
    if "ret_pred_4h" in signal:
        summary["ret_pred_4h"] = signal["ret_pred_4h"]
    if signal_1h4h_confirm is not None:
        summary["signal_1h4h_confirm"] = signal_1h4h_confirm

    print(json.dumps(summary, indent=2))

    last_logged = _load_last_logged_ts(args.log_path)
    if last_logged is not None and last_logged == signal["ts"]:
        print(f"No new bar; last ts={last_logged} already logged. Skipping append.")
        return

    log_row = {
        "ts": signal["ts"],
        "p_up": signal["p_up"],
        "ret_pred": signal["ret_pred"],
        "signal_ensemble": int(signal["signal_ensemble"]),
        "signal_dir_only": int(signal["signal_dir_only"]),
        "p_up_4h": signal.get("p_up_4h", ""),
        "ret_pred_4h": signal.get("ret_pred_4h", ""),
        "signal_1h4h_confirm": signal_1h4h_confirm if signal_1h4h_confirm is not None else "",
        "created_at": _now_utc_iso(),
        "notes": "binance_direct_v1",
    }

    _append_to_log(args.log_path, log_row, LOG_COLUMNS)
    print(f"Appended signal for ts={signal['ts']} to {args.log_path}")


def main() -> None:
    args = _parse_args()
    run_realtime_from_binance(args)


if __name__ == "__main__":
    main()
