import argparse
import datetime as dt
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data.ingestors.gcs_utils import upload_file_to_gcs

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


def _to_ms(ts: Optional[dt.datetime]) -> Optional[int]:
    if ts is None:
        return None
    return int(ts.timestamp() * 1000)


def fetch_binance_futures_klines(
    pair: str,
    interval: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch perpetual futures klines for BTCUSDT.

    Uses /fapi/v1/continuousKlines with contractType=PERPETUAL. For now we
    focus on OHLCV and leave open_interest and funding_rate as NaN so the
    schema matches btc_forecast_raw.futures_metrics.
    """
    endpoint = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/continuousKlines"

    start_ms = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    all_rows = []
    current_start = start_ms

    while True:
        params = {
            "pair": pair,
            "contractType": "PERPETUAL",
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
            "num_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignored",
        ],
    )

    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype("float64")
    df["high"] = df["high"].astype("float64")
    df["low"] = df["low"].astype("float64")
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")

    df["open_interest"] = np.nan
    df["funding_rate"] = np.nan
    df["interval"] = interval

    return df[[
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "funding_rate",
        "interval",
    ]]


def generate_dummy_futures_klines(
    start_time: dt.datetime,
    end_time: dt.datetime,
    interval: str,
) -> pd.DataFrame:
    """Generate synthetic futures OHLCV + placeholders for open_interest/funding_rate.

    This is used when Binance futures endpoints are unreachable (e.g. HTTP 451),
    so you can still populate the `futures_metrics` table for pipeline testing.
    """

    if interval.endswith("h"):
        hours = int(interval[:-1])
        freq = f"{hours}H"
    elif interval.endswith("d"):
        days = int(interval[:-1])
        freq = f"{days}D"
    else:
        raise ValueError(f"Unsupported interval for dummy futures data: {interval}")

    index = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left", tz="UTC")
    if index.empty:
        return pd.DataFrame()

    n = len(index)
    rng = np.random.default_rng(seed=123)

    base_price = 50000.0
    steps = rng.normal(loc=0.0, scale=60.0, size=n)
    close = base_price + np.cumsum(steps)
    high = close + rng.uniform(0, 40, size=n)
    low = close - rng.uniform(0, 40, size=n)
    open_ = close + rng.normal(0, 15, size=n)

    volume = rng.uniform(50, 500, size=n)

    df = pd.DataFrame(
        {
            "ts": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": rng.uniform(1000, 10000, size=n),
            "funding_rate": rng.normal(0.0, 0.0001, size=n),
            "interval": interval,
        }
    )

    df["open"] = df["open"].astype("float64")
    df["high"] = df["high"].astype("float64")
    df["low"] = df["low"].astype("float64")
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["open_interest"] = df["open_interest"].astype("float64")
    df["funding_rate"] = df["funding_rate"].astype("float64")

    return df


def save_futures_parquet(
    df: pd.DataFrame,
    local_dir: str,
    pair: str,
    interval: str,
    date: dt.date,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    fname = f"{pair.lower()}_futures_{interval}_{date.isoformat()}.parquet"
    path = os.path.join(local_dir, fname)
    df.to_parquet(path, index=False)
    return path


def ingest_futures_day_to_gcs(
    pair: str,
    interval: str,
    date: dt.date,
    bucket_name: str,
    local_dir: str = "data/futures_metrics",
) -> str:
    start_time = dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_time = start_time + dt.timedelta(days=1)

    df = fetch_binance_futures_klines(pair, interval, start_time, end_time)
    if df.empty:
        raise ValueError("No futures klines returned for given day")

    local_path = save_futures_parquet(df, local_dir, pair, interval, date)

    blob_path = (
        f"raw/futures_metrics/interval={interval}/"
        f"yyyy={date.year:04d}/mm={date.month:02d}/dd={date.day:02d}/"
        f"{os.path.basename(local_path)}"
    )
    upload_file_to_gcs(local_path, bucket_name, blob_path)
    return blob_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BTCUSDT futures metrics into GCS as Parquet.")
    parser.add_argument("--pair", default="BTCUSDT", help="Perpetual futures pair, default BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h")
    parser.add_argument("--date", required=True, help="UTC date to ingest, format YYYY-MM-DD")
    parser.add_argument("--bucket", required=True, help="GCS bucket name for raw data")
    parser.add_argument("--local-dir", default="data/futures_metrics", help="Local directory for temporary Parquet files")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy futures data instead of calling Binance API")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = dt.date.fromisoformat(args.date)
    start_time = dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_time = start_time + dt.timedelta(days=1)

    if args.dummy:
        df = generate_dummy_futures_klines(start_time, end_time, args.interval)
        if df.empty:
            raise ValueError("No dummy futures data generated")

        os.makedirs(args.local_dir, exist_ok=True)
        local_path = save_futures_parquet(df, args.local_dir, args.pair, args.interval, date)
        blob_path = (
            f"raw/futures_metrics/interval={args.interval}/"
            f"yyyy={date.year:04d}/mm={date.month:02d}/dd={date.day:02d}/"
            f"{os.path.basename(local_path)}"
        )
        upload_file_to_gcs(local_path, args.bucket, blob_path)
    else:
        try:
            blob_path = ingest_futures_day_to_gcs(
                pair=args.pair,
                interval=args.interval,
                date=date,
                bucket_name=args.bucket,
                local_dir=args.local_dir,
            )
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            print(
                f"Failed to fetch futures klines from Binance ({exc}). "
                "This environment may be geo-blocked. "
                "Re-run with --dummy to generate synthetic futures data.",
                file=sys.stderr,
            )
            raise

    print(f"Ingested futures metrics to gs://{args.bucket}/{blob_path}")


if __name__ == "__main__":
    main()
