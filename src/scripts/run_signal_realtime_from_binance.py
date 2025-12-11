import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config_trading import DEFAULT_P_UP_MIN, DEFAULT_RET_MIN
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
    prepare_data_for_signals_from_ohlcv,
)


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_N_BARS = 500
DEFAULT_LOG_PATH = "artifacts/live/paper_trade_realtime.csv"
DEFAULT_REG_MODEL_DIR = "artifacts/models/xgb_ret1h_v1"
DEFAULT_DIR_MODEL_DIR = "artifacts/models/xgb_dir1h_v1"
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
    return parser.parse_args()


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


def _append_to_log(log_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    df_row = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df_row.to_csv(log_path, index=False)
        return

    df_row.to_csv(log_path, mode="a", header=False, index=False)


def _load_models(reg_dir: str, dir_dir: str) -> Dict[str, Any]:
    reg_model_path = os.path.join(reg_dir, "xgb_ret1h_model.json")
    dir_model_path = os.path.join(dir_dir, "xgb_dir1h_model.json")

    if not os.path.exists(reg_model_path):
        raise FileNotFoundError(f"Regression model not found: {reg_model_path}")
    if not os.path.exists(dir_model_path):
        raise FileNotFoundError(f"Direction model not found: {dir_model_path}")

    return load_models(reg_model_path=reg_model_path, dir_model_path=dir_model_path)


def run_realtime_from_binance(args: argparse.Namespace) -> None:
    try:
        raw_df = _merge_market_data(symbol=args.symbol, interval=args.interval, limit=args.n_bars)
    except BinanceAPIError as exc:
        raise SystemExit(f"Failed to fetch Binance market data: {exc}")

    feature_names = _load_feature_names(args.dataset_path)

    df_features = _compute_feature_frame(raw_df, feature_names)
    prepared: PreparedData = prepare_data_for_signals_from_ohlcv(df_features, feature_names=feature_names)

    models = _load_models(args.reg_model_dir, args.dir_model_dir)

    last_index = len(prepared.df_all) - 1
    if last_index < 0:
        raise SystemExit("No feature rows available after preprocessing; cannot produce realtime signal.")

    signal = compute_signal_for_index(
        prepared=prepared,
        index=last_index,
        models=models,
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
    )

    summary = {
        "ts": signal["ts"],
        "p_up": signal["p_up"],
        "ret_pred": signal["ret_pred"],
        "signal_ensemble": int(signal["signal_ensemble"]),
        "signal_dir_only": int(signal["signal_dir_only"]),
        "source": "binance_direct_v1",
    }

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
        "created_at": _now_utc_iso(),
        "notes": "binance_direct_v1",
    }

    _append_to_log(args.log_path, log_row)
    print(f"Appended signal for ts={signal['ts']} to {args.log_path}")


def main() -> None:
    args = _parse_args()
    run_realtime_from_binance(args)


if __name__ == "__main__":
    main()
