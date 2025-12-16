"""Binance US spot ingestor producing tidy spot metrics."""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import requests

BINANCE_US_BASE_URL = "https://api.binance.us"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT = 720
RAW_MARKET_ROOT = Path("data/raw/market/binanceus")
SOURCE_NAME = "binanceus"
DEFAULT_SYMBOL_ID = "BINANCEUS_SPOT_BTC_USDT"


@dataclass(slots=True)
class Candle:
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    num_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @property
    def close_timestamp(self) -> pd.Timestamp:
        return pd.to_datetime(self.close_time, unit="ms", utc=True)


def _parse_time(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    suffix = interval[-1]
    amount = int(interval[:-1])
    if suffix == "m":
        return pd.Timedelta(minutes=amount)
    if suffix == "h":
        return pd.Timedelta(hours=amount)
    if suffix == "d":
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported interval: {interval}")


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


def _fetch_klines(
    symbol: str,
    interval: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    limit: int,
) -> List[Candle]:
    endpoint = f"{BINANCE_US_BASE_URL}/api/v3/klines"
    start_ms = int(start.timestamp() * 1000) if start is not None else None
    end_ms = int(end.timestamp() * 1000) if end is not None else None

    candles: List[Candle] = []
    current_start = start_ms
    remaining = limit if start_ms is not None else None

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
        }
        if current_start is not None:
            params["startTime"] = current_start
        if end_ms is not None:
            params["endTime"] = end_ms
        if remaining is not None:
            params["limit"] = min(params["limit"], max(remaining, 1))

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network safeguard
            raise RuntimeError(f"Failed to fetch Binance US klines: {exc}") from exc

        batch = response.json()
        if not batch:
            break

        for row in batch:
            candle = Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
                quote_volume=float(row[7]),
                num_trades=int(row[8]),
                taker_buy_base_volume=float(row[9]),
                taker_buy_quote_volume=float(row[10]),
            )
            candles.append(candle)

        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                break

        last_open_time = int(batch[-1][0])
        next_start = last_open_time + 1
        if end_ms is not None and next_start > end_ms:
            break
        if current_start == next_start:
            break
        current_start = next_start

        if len(batch) < params["limit"]:
            break

    return candles


def _candles_to_tidy(
    candles: Iterator[Candle],
    symbol_id: str,
    interval_delta: pd.Timedelta,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candle in candles:
        ts = pd.to_datetime(candle.open_time, unit="ms", utc=True) + interval_delta
        metrics = {
            "spot_open": candle.open,
            "spot_high": candle.high,
            "spot_low": candle.low,
            "spot_close": candle.close,
            "spot_volume": candle.volume,
            "spot_quote_volume": candle.quote_volume,
            "spot_num_trades": float(candle.num_trades),
            "spot_taker_buy_base_volume": candle.taker_buy_base_volume,
            "spot_taker_buy_quote_volume": candle.taker_buy_quote_volume,
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "ts": ts,
                    "metric": metric,
                    "value": value,
                    "source": SOURCE_NAME,
                    "symbol_id": symbol_id,
                },
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance US returned no klines to persist.")
    frame = frame.sort_values("ts").reset_index(drop=True)
    return frame


def _write_parquet(frame: pd.DataFrame, output_root: Path, symbol_id: str) -> Path:
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / "entity=spot" / f"symbol={symbol_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"binanceus_spot_{symbol_id}_{timestamp}.parquet"
    frame.to_parquet(output_path, index=False)
    return output_path


def ingest_binance_us_spot(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    output_root: Path = RAW_MARKET_ROOT,
    symbol_id: str = DEFAULT_SYMBOL_ID,
) -> Path:
    interval_delta = _interval_to_timedelta(interval)
    freq = _interval_to_freq(interval)

    if end:
        end_ts = _parse_time(end)
    else:
        now = pd.Timestamp.now(tz="UTC")
        end_ts = (now - interval_delta).floor(freq)

    if start:
        start_ts = _parse_time(start)
    else:
        start_ts = end_ts - interval_delta * (limit - 1)

    if start_ts >= end_ts:
        raise ValueError("start timestamp must be before end timestamp")

    candles = _fetch_klines(symbol, interval, start_ts, end_ts, limit)
    frame = _candles_to_tidy(candles, symbol_id, interval_delta)
    path = _write_parquet(frame, output_root, symbol_id)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance US spot klines and write tidy parquet.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol, default BTCUSDT.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval, default 1h.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Approximate number of klines to capture when start is omitted (default: 720).",
    )
    parser.add_argument("--start", default=None, help="ISO timestamp for start (UTC). Overrides limit window.")
    parser.add_argument("--end", default=None, help="ISO timestamp for end (UTC). Default: now.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_MARKET_ROOT,
        help="Root directory for tidy parquet output.",
    )
    parser.add_argument(
        "--symbol-id",
        default=DEFAULT_SYMBOL_ID,
        help="Symbol identifier saved into the parquet metadata.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_args()
    path = ingest_binance_us_spot(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        limit=args.limit,
        output_root=args.output_root,
        symbol_id=args.symbol_id,
    )
    print(f"Saved Binance US spot tidy parquet to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
