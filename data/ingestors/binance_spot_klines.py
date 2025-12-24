import argparse
import datetime as dt
import os
from typing import Optional

import pandas as pd
import requests

from data.ingestors.gcs_utils import upload_file_to_gcs

# Use binance.us endpoint to reduce risk of geo-blocking
BINANCE_BASE_URL = "https://api.binance.us"


def _to_ms(ts: Optional[dt.datetime]) -> Optional[int]:
    if ts is None:
        return None
    return int(ts.timestamp() * 1000)


def _interval_to_freq(interval: str) -> str:
    suffix = interval[-1]
    amount = int(interval[:-1])
    if suffix == "m":
        return f"{amount}min"
    if suffix == "h":
        return f"{amount}h"
    if suffix == "d":
        return f"{amount}d"
    raise ValueError(f"Unsupported interval: {interval}")


def _normalize_hourly_frame(df: pd.DataFrame, interval: str, label: str) -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts")
    frame = frame.drop_duplicates(subset="ts", keep="last")

    freq = _interval_to_freq(interval)
    start = frame["ts"].iloc[0]
    end = frame["ts"].iloc[-1]
    full_index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")

    existing_index = frame.set_index("ts").index
    missing_index = full_index.difference(existing_index)

    indexed = frame.set_index("ts").reindex(full_index)

    numeric_cols = indexed.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        indexed[numeric_cols] = indexed[numeric_cols].ffill().bfill()

    non_numeric_cols = indexed.select_dtypes(exclude=["number"]).columns
    for column in non_numeric_cols:
        indexed[column] = indexed[column].ffill().bfill()

    if len(missing_index) > 0:
        start_gap = missing_index[0].isoformat()
        end_gap = missing_index[-1].isoformat()
        print(
            f"[{label}] Backfilled {len(missing_index)} missing bars between {start_gap} and {end_gap}.",
        )

    indexed = indexed.reset_index().rename(columns={"index": "ts"})
    indexed["ts"] = pd.to_datetime(indexed["ts"], utc=True)
    return indexed


def fetch_binance_spot_klines(
    symbol: str,
    interval: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch spot klines from Binance /api/v3/klines into a DataFrame.

    Timestamps are converted to UTC and we use close_time as end-of-bar ts.
    """
    endpoint = f"{BINANCE_BASE_URL}/api/v3/klines"

    start_ms = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    all_rows = []
    current_start = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if current_start is not None:
            params["startTime"] = current_start
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)

        last_open_time = batch[-1][0]
        next_start = last_open_time + 1
        if end_ms is not None and next_start >= end_ms:
            break
        if current_start == next_start:
            break
        current_start = next_start

        if len(batch) < limit:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    # Use close_time as end-of-bar timestamp in UTC
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype("float64")
    df["high"] = df["high"].astype("float64")
    df["low"] = df["low"].astype("float64")
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["quote_volume"] = df["quote_asset_volume"].astype("float64")
    df["num_trades"] = df["number_of_trades"].astype("int64")
    df["taker_buy_base_volume"] = df["taker_buy_base_asset_volume"].astype("float64")
    df["taker_buy_quote_volume"] = df["taker_buy_quote_asset_volume"].astype("float64")
    df["interval"] = interval

    df = _normalize_hourly_frame(df, interval, label="binance_spot")

    return df[[
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "interval",
    ]]


def save_spot_parquet(
    df: pd.DataFrame,
    local_dir: str,
    symbol: str,
    interval: str,
    date: dt.date,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    fname = f"{symbol.lower()}_spot_{interval}_{date.isoformat()}.parquet"
    path = os.path.join(local_dir, fname)
    df.to_parquet(path, index=False)
    return path


def ingest_spot_day_to_gcs(
    symbol: str,
    interval: str,
    date: dt.date,
    bucket_name: str,
    local_dir: str = "data/spot_klines",
) -> str:
    start_time = dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_time = start_time + dt.timedelta(days=1)

    df = fetch_binance_spot_klines(symbol, interval, start_time, end_time)
    if df.empty:
        raise ValueError("No spot klines returned for given day")

    local_path = save_spot_parquet(df, local_dir, symbol, interval, date)

    blob_path = (
        f"raw/spot_klines/interval={interval}/"
        f"yyyy={date.year:04d}/mm={date.month:02d}/dd={date.day:02d}/"
        f"{os.path.basename(local_path)}"
    )
    upload_file_to_gcs(local_path, bucket_name, blob_path)
    return blob_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BTCUSDT spot klines into GCS as Parquet.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, default BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h, 1d")
    parser.add_argument("--date", required=True, help="UTC date to ingest, format YYYY-MM-DD")
    parser.add_argument("--bucket", required=True, help="GCS bucket name for raw data")
    parser.add_argument("--local-dir", default="data/spot_klines", help="Local directory for temporary Parquet files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = dt.date.fromisoformat(args.date)
    blob_path = ingest_spot_day_to_gcs(
        symbol=args.symbol,
        interval=args.interval,
        date=date,
        bucket_name=args.bucket,
        local_dir=args.local_dir,
    )
    print(f"Ingested spot klines to gs://{args.bucket}/{blob_path}")


if __name__ == "__main__":
    main()
