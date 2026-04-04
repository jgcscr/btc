"""Helpers for fetching Binance klines and derivatives needed for realtime signals."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

_BINANCE_API_BASE = os.getenv("BINANCE_SPOT_API_BASE", "https://api.binance.us")
_BINANCE_FUTURES_API_BASE = os.getenv("BINANCE_FUTURES_API_BASE", "https://fapi.binance.com")


class BinanceAPIError(RuntimeError):
    """Raised when a Binance HTTP request fails."""


def _request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Iterable:
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise BinanceAPIError(f"Binance request failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - malformed payload
        raise BinanceAPIError(f"Binance response not valid JSON for {url}") from exc


def _to_millis(value: Any) -> int | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _paginate_json(
    url: str,
    *,
    base_params: Dict[str, Any],
    limit: int,
    timestamp_key: str | int,
    start_time: Any = None,
    end_time: Any = None,
) -> List[Any]:
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)
    current_start = start_ms
    rows: List[Any] = []

    while True:
        params = dict(base_params)
        params["limit"] = int(limit)
        if current_start is not None:
            params["startTime"] = current_start
        if end_ms is not None:
            params["endTime"] = end_ms

        batch = list(_request_json(url, params=params))
        if not batch:
            break

        rows.extend(batch)
        last = batch[-1]
        last_ts = int(last[timestamp_key] if isinstance(timestamp_key, int) else last[timestamp_key])
        next_start = last_ts + 1
        if end_ms is not None and next_start >= end_ms:
            break
        if len(batch) < limit or current_start == next_start:
            break
        current_start = next_start

    return rows


def fetch_spot_klines(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch recent spot klines for *symbol*/*interval* from Binance REST API."""
    url = f"{_BINANCE_API_BASE}/api/v3/klines"
    raw = _request_json(url, params={"symbol": symbol, "interval": interval, "limit": limit})

    if not raw:
        raise BinanceAPIError("Empty kline response from Binance spot endpoint.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    now = pd.Timestamp.utcnow()
    df = df[df["close_time"] <= now]

    df["ts"] = df["close_time"] - pd.Timedelta(milliseconds=1)

    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "num_trades"]].sort_values("ts").reset_index(drop=True)


def fetch_futures_klines(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch recent futures klines for *symbol*/*interval* from Binance Futures API."""
    url = f"{_BINANCE_FUTURES_API_BASE}/fapi/v1/klines"
    raw = _request_json(url, params={"symbol": symbol, "interval": interval, "limit": limit})

    if not raw:
        raise BinanceAPIError("Empty kline response from Binance futures endpoint.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    now = pd.Timestamp.utcnow()
    df = df[df["close_time"] <= now]

    df["ts"] = df["close_time"] - pd.Timedelta(milliseconds=1)

    return df[["ts", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "fut_open",
            "high": "fut_high",
            "low": "fut_low",
            "close": "fut_close",
            "volume": "fut_volume",
        },
    ).sort_values("ts").reset_index(drop=True)


def fetch_open_interest(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch open interest history for *symbol*/*interval* from Binance Futures."""
    url = f"{_BINANCE_FUTURES_API_BASE}/futures/data/openInterestHist"
    raw = _request_json(url, params={"symbol": symbol, "period": interval, "limit": limit})

    if not raw:
        raise BinanceAPIError("Empty open interest response from Binance.")

    records: List[Dict[str, str]] = list(raw)
    df = pd.DataFrame.from_records(records)
    if "timestamp" not in df.columns or "sumOpenInterest" not in df.columns:
        raise BinanceAPIError("Unexpected payload for open interest history.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ts"] = df["timestamp"] - pd.Timedelta(milliseconds=1)
    df["open_interest"] = df["sumOpenInterest"].astype(float)

    return df[["ts", "open_interest"]].sort_values("ts").reset_index(drop=True)


def fetch_funding_rates(
    symbol: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch recent funding rates for *symbol* from Binance Futures."""
    url = f"{_BINANCE_FUTURES_API_BASE}/fapi/v1/fundingRate"
    raw = _request_json(url, params={"symbol": symbol, "limit": limit})

    if not raw:
        raise BinanceAPIError("Empty funding rate response from Binance.")

    df = pd.DataFrame(raw)
    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        raise BinanceAPIError("Unexpected payload for funding rates.")

    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["ts"] = df["funding_time"] - pd.Timedelta(milliseconds=1)
    df["funding_rate"] = df["fundingRate"].astype(float)

    return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)


def fetch_futures_klines_range(
    symbol: str,
    interval: str,
    *,
    start_time: Any = None,
    end_time: Any = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch futures klines over a time range using Binance Futures pagination."""

    url = f"{_BINANCE_FUTURES_API_BASE}/fapi/v1/klines"
    raw = _paginate_json(
        url,
        base_params={"symbol": symbol, "interval": interval},
        limit=limit,
        timestamp_key=0,
        start_time=start_time,
        end_time=end_time,
    )
    if not raw:
        return pd.DataFrame(columns=["ts", "fut_open", "fut_high", "fut_low", "fut_close", "fut_volume"])

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    now = pd.Timestamp.utcnow()
    df = df[df["close_time"] <= now].copy()
    df["ts"] = df["close_time"] - pd.Timedelta(milliseconds=1)
    return df[["ts", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "fut_open",
            "high": "fut_high",
            "low": "fut_low",
            "close": "fut_close",
            "volume": "fut_volume",
        }
    ).sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)


def fetch_open_interest_range(
    symbol: str,
    interval: str,
    *,
    start_time: Any = None,
    end_time: Any = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch open interest history over a time range using Binance Futures pagination."""

    url = f"{_BINANCE_FUTURES_API_BASE}/futures/data/openInterestHist"
    raw = _paginate_json(
        url,
        base_params={"symbol": symbol, "period": interval},
        limit=limit,
        timestamp_key="timestamp",
        start_time=start_time,
        end_time=end_time,
    )
    if not raw:
        return pd.DataFrame(columns=["ts", "open_interest"])

    df = pd.DataFrame.from_records(raw)
    if "timestamp" not in df.columns or "sumOpenInterest" not in df.columns:
        raise BinanceAPIError("Unexpected payload for open interest history.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ts"] = df["timestamp"] - pd.Timedelta(milliseconds=1)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    return df[["ts", "open_interest"]].sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)


def fetch_funding_rates_range(
    symbol: str,
    *,
    start_time: Any = None,
    end_time: Any = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch funding rates over a time range using Binance Futures pagination."""

    url = f"{_BINANCE_FUTURES_API_BASE}/fapi/v1/fundingRate"
    raw = _paginate_json(
        url,
        base_params={"symbol": symbol},
        limit=limit,
        timestamp_key="fundingTime",
        start_time=start_time,
        end_time=end_time,
    )
    if not raw:
        return pd.DataFrame(columns=["ts", "funding_rate"])

    df = pd.DataFrame(raw)
    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        raise BinanceAPIError("Unexpected payload for funding rates.")
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["ts"] = df["funding_time"] - pd.Timedelta(milliseconds=1)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)