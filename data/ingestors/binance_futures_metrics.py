import argparse
import datetime as dt
import os
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from pandas.tseries.frequencies import to_offset

from data.ingestors.gcs_utils import upload_file_to_gcs

CRYPTOCOMPARE_BASE_URL = "https://data-api.cryptocompare.com"
CRYPTOCOMPARE_DEFAULT_MARKET = os.environ.get("CRYPTOCOMPARE_MARKET", "binance")
CRYPTOCOMPARE_DEFAULT_INSTRUMENT = os.environ.get(
    "CRYPTOCOMPARE_INSTRUMENT",
    "BTC-USDT-VANILLA-PERPETUAL",
)
CRYPTOCOMPARE_MAX_LIMIT = 2000


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


def _timestamp_to_period_end(timestamp: int, interval: str) -> pd.Timestamp:
    base = pd.to_datetime(timestamp, unit="s", utc=True)
    offset = to_offset(_interval_to_freq(interval))
    return base + offset - pd.Timedelta(milliseconds=1)


def _resolve_endpoint(interval: str, resource: Optional[str]) -> tuple[str, int]:
    suffix = interval[-1]
    amount = int(interval[:-1])
    if amount < 1:
        raise ValueError(f"Interval amount must be positive: {interval}")

    base = "/futures/v1/historical"
    if resource:
        base = f"{base}/{resource}"

    if suffix == "m":
        return (f"{base}/minutes", amount)
    if suffix == "h":
        return (f"{base}/hours", amount)
    if suffix == "d":
        return (f"{base}/days", amount)
    raise ValueError(f"Unsupported interval: {interval}")


def _require_api_key(explicit_key: Optional[str]) -> str:
    key = explicit_key or os.environ.get("CRYPTOCOMPARE_API_KEY")
    if not key:
        raise RuntimeError("Missing CryptoCompare API key. Set CRYPTOCOMPARE_API_KEY or pass --api-key.")
    return key


def _cc_request(path: str, params: Dict[str, Any], api_key: str) -> Iterable[Dict[str, Any]]:
    merged = dict(params)
    merged["api_key"] = api_key
    response = requests.get(f"{CRYPTOCOMPARE_BASE_URL}{path}", params=merged, timeout=10)
    response.raise_for_status()

    payload = response.json()
    err = payload.get("Err")
    if isinstance(err, dict) and any(err.values()):
        message = err.get("message") or err
        raise RuntimeError(f"CryptoCompare request failed for {path}: {message}")

    data = payload.get("Data")
    if not data:
        return []
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CryptoCompare payload for {path}: {type(data)!r}")
    return data


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_utc_timestamp(value: dt.datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_hourly_frame(
    df: pd.DataFrame,
    interval: str,
    label: str,
    *,
    expected_start: Optional[dt.datetime] = None,
    expected_end: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts")

    freq_str = _interval_to_freq(interval)
    frame["ts"] = frame["ts"].dt.ceil(freq_str)
    frame = frame.drop_duplicates(subset="ts", keep="last")

    freq = _interval_to_freq(interval)
    freq_offset = to_offset(freq)

    start = frame["ts"].iloc[0]
    if expected_start is not None:
        anchor_start = _as_utc_timestamp(expected_start)
        start = anchor_start + freq_offset

    end = frame["ts"].iloc[-1]
    if expected_end is not None:
        end = _as_utc_timestamp(expected_end)

    if start > end:
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


def _fetch_open_interest_series(
    market: str,
    instrument: str,
    interval: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    api_key: str,
) -> pd.DataFrame:
    endpoint, aggregate = _resolve_endpoint(interval, "open-interest")
    params: Dict[str, Any] = {
        "market": market,
        "instrument": instrument,
        "limit": CRYPTOCOMPARE_MAX_LIMIT,
        "to_ts": int(end_time.timestamp()),
    }
    if aggregate > 1:
        params["aggregate"] = aggregate

    start_ts = _as_utc_timestamp(start_time)
    end_ts = _as_utc_timestamp(end_time)
    rows = _cc_request(endpoint, params, api_key)
    records = []
    for row in rows:
        timestamp = row.get("TIMESTAMP")
        close_settlement = row.get("CLOSE_SETTLEMENT")
        if timestamp is None or close_settlement is None:
            continue
        ts = _timestamp_to_period_end(int(timestamp), interval)
        if ts < start_ts or ts >= end_ts:
            continue
        records.append({"ts": ts, "open_interest": _float_or_nan(close_settlement)})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    return df


def _fetch_funding_series(
    market: str,
    instrument: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    interval: str,
    api_key: str,
) -> pd.DataFrame:
    endpoint, aggregate = _resolve_endpoint(interval, "funding-rate")
    params: Dict[str, Any] = {
        "market": market,
        "instrument": instrument,
        "limit": CRYPTOCOMPARE_MAX_LIMIT,
        "to_ts": int(end_time.timestamp()),
    }
    if aggregate > 1:
        params["aggregate"] = aggregate

    start_ts = _as_utc_timestamp(start_time)
    end_ts = _as_utc_timestamp(end_time)
    rows = _cc_request(endpoint, params, api_key)
    records = []
    for row in rows:
        timestamp = row.get("TIMESTAMP")
        close_rate = row.get("CLOSE")
        if timestamp is None or close_rate is None:
            continue
        ts = _timestamp_to_period_end(int(timestamp), interval)
        if ts < start_ts or ts >= end_ts:
            continue
        records.append({"ts": ts, "funding_rate": _float_or_nan(close_rate)})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df


def _fetch_price_series(
    market: str,
    instrument: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    interval: str,
    api_key: str,
) -> pd.DataFrame:
    endpoint, aggregate = _resolve_endpoint(interval, None)
    params: Dict[str, Any] = {
        "market": market,
        "instrument": instrument,
        "limit": CRYPTOCOMPARE_MAX_LIMIT,
        "to_ts": int(end_time.timestamp()),
        "groups": "OHLC,VOLUME",
    }
    if aggregate > 1:
        params["aggregate"] = aggregate

    start_ts = _as_utc_timestamp(start_time)
    end_ts = _as_utc_timestamp(end_time)
    rows = _cc_request(endpoint, params, api_key)
    records = []
    for row in rows:
        timestamp = row.get("TIMESTAMP")
        if timestamp is None:
            continue
        ts = _timestamp_to_period_end(int(timestamp), interval)
        if ts < start_ts or ts >= end_ts:
            continue

        records.append(
            {
                "ts": ts,
                "open": _float_or_nan(row.get("OPEN")),
                "high": _float_or_nan(row.get("HIGH")),
                "low": _float_or_nan(row.get("LOW")),
                "close": _float_or_nan(row.get("CLOSE")),
                "volume": _float_or_nan(row.get("VOLUME")),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def fetch_binance_futures_klines(
    pair: str,
    interval: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    *,
    market: Optional[str] = None,
    instrument: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch perpetual futures metrics via CryptoCompare for the requested interval."""

    # The pair parameter is retained for compatibility with downstream file naming.
    _ = pair

    api_key = _require_api_key(api_key)
    market = market or CRYPTOCOMPARE_DEFAULT_MARKET
    instrument = instrument or CRYPTOCOMPARE_DEFAULT_INSTRUMENT

    price_df = _fetch_price_series(market, instrument, start_time, end_time, interval, api_key)
    if price_df.empty:
        return price_df

    oi_df = _fetch_open_interest_series(market, instrument, interval, start_time, end_time, api_key)
    if not oi_df.empty:
        price_df = price_df.merge(oi_df, on="ts", how="left")
    else:
        price_df["open_interest"] = np.nan

    funding_df = _fetch_funding_series(market, instrument, start_time, end_time, interval, api_key)
    if not funding_df.empty:
        price_df = price_df.merge(funding_df, on="ts", how="left")
    else:
        price_df["funding_rate"] = np.nan

    price_df["interval"] = interval

    normalized = _normalize_hourly_frame(
        price_df,
        interval,
        label="cryptocompare_futures",
        expected_start=start_time,
        expected_end=end_time,
    )

    return normalized[[
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

    This is used when the upstream futures provider is unreachable, so you can
    still populate the `futures_metrics` table for pipeline testing.
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
    *,
    market: Optional[str] = None,
    instrument: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    start_time = dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_time = start_time + dt.timedelta(days=1)

    df = fetch_binance_futures_klines(
        pair,
        interval,
        start_time,
        end_time,
        market=market,
        instrument=instrument,
        api_key=api_key,
    )
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
    parser.add_argument(
        "--market",
        default=CRYPTOCOMPARE_DEFAULT_MARKET,
        help="CryptoCompare market identifier, defaults to env CRYPTOCOMPARE_MARKET or 'binance'",
    )
    parser.add_argument(
        "--instrument",
        default=CRYPTOCOMPARE_DEFAULT_INSTRUMENT,
        help="CryptoCompare instrument identifier (e.g. BTC-USDT-VANILLA-PERPETUAL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CRYPTOCOMPARE_API_KEY"),
        help="CryptoCompare API key. You may also set CRYPTOCOMPARE_API_KEY env var.",
    )
    parser.add_argument("--dummy", action="store_true", help="Generate dummy futures data instead of calling CryptoCompare API")
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
                market=args.market,
                instrument=args.instrument,
                api_key=args.api_key,
            )
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            print(
                f"Failed to fetch futures klines from CryptoCompare ({exc}). "
                "This environment may be geo-blocked or the API key may be invalid. "
                "Re-run with --dummy to generate synthetic futures data.",
                file=sys.stderr,
            )
            raise
        except RuntimeError as exc:
            print(
                f"Failed to fetch futures klines from CryptoCompare ({exc}). "
                "Re-run with --dummy to generate synthetic futures data.",
                file=sys.stderr,
            )
            raise

    print(f"Ingested futures metrics to gs://{args.bucket}/{blob_path}")


if __name__ == "__main__":
    main()
