"""Compute classical technical indicators from hourly OHLCV data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H, PROJECT_ID
from src.data.bq_loader import load_btc_features_1h

OUTPUT_PATH = Path("data/processed/technical/hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/technical_summary.json")

PRICE_METRIC_MAP = {
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


def _load_price_data(price_source: Optional[Path]) -> pd.DataFrame:
    if price_source is not None:
        if price_source.exists():
            frame = pd.read_parquet(price_source)
            return _normalize_price_frame(frame)
        print(f"Price source {price_source} not found; attempting curated table.")

    try:
        curated = load_btc_features_1h(
            project_id=PROJECT_ID,
            dataset_id=BQ_DATASET_CURATED,
            table_id=BQ_TABLE_FEATURES_1H,
        )
    except Exception as exc:  # pragma: no cover - credential safeguard
        print(f"Failed to load curated features ({exc}); returning empty frame.")
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    return _normalize_price_frame(curated)


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    working = frame.copy()

    if {"metric", "value"}.issubset(working.columns):
        metric_map = {key: PRICE_METRIC_MAP.get(key, key) for key in working["metric"].unique()}
        pivot = (
            working[["ts", "metric", "value"]]
            .pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
            .rename(columns=metric_map)
        )
        pivot = pivot.sort_index().reset_index()
        pivot.columns.name = None
        working = pivot

    if "timestamp" in working.columns and "ts" not in working.columns:
        working = working.rename(columns={"timestamp": "ts"})

    working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
    working = working.dropna(subset=["ts"])
    working = working.sort_values("ts").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in working.columns]
    if missing:
        print(f"Price frame missing required columns {missing}; returning empty frame.")
        return pd.DataFrame(columns=["ts"] + required)

    working = working[["ts", "open", "high", "low", "close", "volume"]].copy()
    working[required] = working[required].astype(float)
    return working


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic_k(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    k = 100 * (close - lowest_low) / denominator
    return k


def _williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    wr = -100 * (highest_high - close) / denominator
    return wr


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def _bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def _keltner_channels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema = _ema(close, span=ema_period)
    atr = _atr(high, low, close, period=atr_period)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return ema, upper, lower


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, span=fast)
    ema_slow = _ema(close, span=slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def _donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    return upper, lower


def _summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "row_count": 0,
            "latest_timestamp": None,
            "columns": {},
        }
    summary = {
        "row_count": int(len(frame)),
        "latest_timestamp": frame["timestamp"].max().isoformat(),
        "columns": {},
    }
    for column in frame.columns:
        if column == "timestamp":
            continue
        series = frame[column]
        summary["columns"][column] = {
            "missing_ratio": float(series.isna().mean()),
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        }
    return summary


def process_technical_features(
    price_source: Optional[Path] = None,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    include_history: bool = False,
    history_limit: Optional[int] = 2000,
) -> Path:
    price_frame = _load_price_data(price_source)

    if include_history and price_source is not None:
        history_frame = _load_price_data(None)
        frames: list[pd.DataFrame] = []
        if not history_frame.empty:
            frames.append(history_frame)
        if not price_frame.empty:
            frames.append(price_frame)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged = merged.sort_values("ts").drop_duplicates(subset="ts", keep="last")
            if history_limit is not None and history_limit > 0:
                merged = merged.tail(history_limit).reset_index(drop=True)
            price_frame = merged

    if price_frame.empty:
        print("Technical feature computation skipped; price data unavailable.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["timestamp"])
        empty.to_parquet(output_path, index=False)
        summary = _summarize(empty)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        return output_path

    close = price_frame["close"].astype(float)
    high = price_frame["high"].astype(float)
    low = price_frame["low"].astype(float)

    rsi = _rsi(close, period=14)
    stoch_k = _stochastic_k(close, high, low, period=14)
    stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
    williams_r = _williams_r(close, high, low, period=14)
    macd, macd_signal, macd_hist = _macd(close, fast=12, slow=26, signal=9)
    bb_mid, bb_upper, bb_lower = _bollinger_bands(close, period=20, num_std=2.0)
    kel_mid, kel_upper, kel_lower = _keltner_channels(close, high, low, ema_period=20, atr_period=10, multiplier=2.0)
    true_range = _true_range(high, low, close)
    atr = _atr(high, low, close, period=14)
    donchian_high, donchian_low = _donchian(high, low, period=20)

    output = pd.DataFrame(
        {
            "timestamp": price_frame["ts"],
            "ta_rsi_14": rsi,
            "ta_stoch_k_14_3": stoch_k,
            "ta_stoch_d_14_3": stoch_d,
            "ta_williams_r_14": williams_r,
            "ta_macd_12_26_9": macd,
            "ta_macd_signal_12_26_9": macd_signal,
            "ta_macd_hist_12_26_9": macd_hist,
            "ta_bbands_mid_20": bb_mid,
            "ta_bbands_upper_20_2": bb_upper,
            "ta_bbands_lower_20_2": bb_lower,
            "ta_keltner_mid_20_atr10": kel_mid,
            "ta_keltner_upper_20_atr10": kel_upper,
            "ta_keltner_lower_20_atr10": kel_lower,
            "ta_true_range": true_range,
            "ta_atr_14": atr,
            "ta_donchian_high_20": donchian_high,
            "ta_donchian_low_20": donchian_low,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    summary = _summarize(output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        "Computed technical indicators (RSI, Stochastic, Williams %R, MACD, Bollinger, Keltner, ATR, Donchian) "
        f"for {len(output)} hourly rows; saved to {output_path}.",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly technical indicator parquet from price data.")
    parser.add_argument(
        "--price-source",
        type=Path,
        default=None,
        help="Optional path to an hourly OHLCV parquet; falls back to curated BigQuery table if omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output parquet path (default: data/processed/technical/hourly_features.parquet).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Monitoring summary JSON path (default: artifacts/monitoring/technical_summary.json).",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Append curated hourly price history before computing indicators to reduce warmup gaps.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=2000,
        help="Keep at most this many most-recent rows when combining history (default: 2000).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_technical_features(
        price_source=args.price_source,
        output_path=args.output,
        summary_path=args.summary,
        include_history=args.include_history,
        history_limit=args.history_limit,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
