import argparse
import datetime as dt
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests


BINANCE_BASE_URL = "https://api.binance.com"


def fetch_spot_klines(symbol: str, interval: str, start_time: Optional[dt.datetime], end_time: Optional[dt.datetime], limit: int = 1000) -> pd.DataFrame:
    """Fetch spot klines from Binance public REST API.

    This pulls klines in ascending time order between start_time and end_time
    (if provided) with the given interval. It may paginate if the range spans
    more than `limit` candles.
    """
    endpoint = f"{BINANCE_BASE_URL}/api/v3/klines"

    def to_ms(ts: Optional[dt.datetime]) -> Optional[int]:
        if ts is None:
            return None
        return int(ts.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    start_ms = to_ms(start_time)
    end_ms = to_ms(end_time)

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

        # Next start is last open time + 1ms
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

    # Convert to proper dtypes and rename to match BigQuery schema
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


def generate_dummy_klines(start_time: dt.datetime, end_time: dt.datetime, interval: str) -> pd.DataFrame:
    """Generate a synthetic OHLCV series matching the BigQuery schema.

    This avoids external API calls (e.g. when Binance returns HTTP 451) and
    lets you exercise the GCS/BigQuery pipeline with realistic-looking data.
    """

    # Map a Binance-style interval string to a pandas frequency
    if interval.endswith("h"):
        hours = int(interval[:-1])
        freq = f"{hours}H"
    elif interval.endswith("d"):
        days = int(interval[:-1])
        freq = f"{days}D"
    else:
        raise ValueError(f"Unsupported interval for dummy data: {interval}")

    # Generate left-inclusive datetime index
    index = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left", tz="UTC")
    if index.empty:
        return pd.DataFrame()

    n = len(index)

    # Simple random walk for prices
    rng = np.random.default_rng(seed=42)
    base_price = 50000.0
    steps = rng.normal(loc=0.0, scale=50.0, size=n)
    close = base_price + np.cumsum(steps)
    high = close + rng.uniform(0, 30, size=n)
    low = close - rng.uniform(0, 30, size=n)
    open_ = close + rng.normal(0, 10, size=n)

    volume = rng.uniform(10, 200, size=n)
    quote_volume = volume * close
    num_trades = rng.integers(100, 2000, size=n)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, size=n)
    taker_buy_quote = taker_buy_base * close

    df = pd.DataFrame(
        {
            "ts": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "num_trades": num_trades,
            "taker_buy_base_volume": taker_buy_base,
            "taker_buy_quote_volume": taker_buy_quote,
            "interval": interval,
        }
    )

    # Ensure types are exactly as expected
    df["open"] = df["open"].astype("float64")
    df["high"] = df["high"].astype("float64")
    df["low"] = df["low"].astype("float64")
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["quote_volume"] = df["quote_volume"].astype("float64")
    df["num_trades"] = df["num_trades"].astype("int64")
    df["taker_buy_base_volume"] = df["taker_buy_base_volume"].astype("float64")
    df["taker_buy_quote_volume"] = df["taker_buy_quote_volume"].astype("float64")

    return df


def save_parquet(df: pd.DataFrame, output_dir: str, symbol: str, interval: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        raise ValueError("No rows fetched; nothing to save.")

    start_ts = df["ts"].min().strftime("%Y%m%dT%H%M%S")
    end_ts = df["ts"].max().strftime("%Y%m%dT%H%M%S")
    fname = f"{symbol.lower()}_{interval}_{start_ts}_{end_ts}.parquet"
    path = os.path.join(output_dir, fname)

    df.to_parquet(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BTCUSDT spot klines to Parquet.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, default BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h, 4h, 1d")
    parser.add_argument("--hours", type=int, default=24, help="Number of hours back from now to fetch")
    parser.add_argument("--output-dir", default="data/spot_klines", help="Local output directory for Parquet files")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data instead of calling Binance API")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = dt.datetime.now(tz=dt.timezone.utc)
    start_time = now - dt.timedelta(hours=args.hours)

    if args.dummy:
        df = generate_dummy_klines(
            start_time=start_time,
            end_time=now,
            interval=args.interval,
        )
    else:
        try:
            df = fetch_spot_klines(
                symbol=args.symbol,
                interval=args.interval,
                start_time=start_time,
                end_time=now,
            )
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            print(
                f"Failed to fetch klines from Binance ({exc}). "
                "This environment may be geo-blocked. "
                "Re-run with --dummy to generate synthetic data.",
                file=sys.stderr,
            )
            raise

    path = save_parquet(df, args.output_dir, args.symbol, args.interval)
    print(f"Saved Parquet to {path}")


if __name__ == "__main__":
    main()
