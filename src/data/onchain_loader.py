import os
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from src.config import (
    ONCHAIN_API_BASE_URL,
    ONCHAIN_API_KEY,
    ONCHAIN_DEFAULT_INTERVAL,
    ONCHAIN_METRICS,
)


class OnchainAPIError(RuntimeError):
    """Raised when the on-chain API returns an error or malformed payload."""


def _to_iso8601(value: pd.Timestamp | str | float | int) -> str:
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return ts.isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s", tz="UTC").isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        ts = pd.Timestamp(value, tz="UTC")
        return ts.isoformat().replace("+00:00", "Z")
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _normalize_payload(records: Iterable[Dict[str, object]], metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        ts_raw = record.get("ts") or record.get("timestamp") or record.get("time")
        if ts_raw is None:
            continue
        try:
            ts = pd.Timestamp(ts_raw, tz="UTC")
        except Exception as exc:
            raise OnchainAPIError(f"Unable to parse timestamp from record: {record}") from exc

        row = {"ts": ts}
        for metric in metrics:
            value = record.get(metric)
            if value is None and isinstance(record.get("metrics"), dict):
                value = record["metrics"].get(metric)
            row[metric] = value
        rows.append(row)

    if not rows:
        raise OnchainAPIError("API response did not contain any usable records.")

    df = pd.DataFrame(rows)
    df = df.sort_values("ts").drop_duplicates("ts")
    return df.reset_index(drop=True)


def fetch_onchain_metrics(
    start_ts: pd.Timestamp | str,
    end_ts: pd.Timestamp | str,
    interval: str = ONCHAIN_DEFAULT_INTERVAL,
    metrics: Optional[List[str]] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
) -> pd.DataFrame:
    """Fetch BTC on-chain metrics for the requested window."""

    if metrics is None:
        metrics = ONCHAIN_METRICS

    if not ONCHAIN_API_BASE_URL:
        raise OnchainAPIError("ONCHAIN_API_BASE_URL is not configured; set env or config.")

    params = {
        "asset": "BTC",
        "metrics": ",".join(metrics),
        "start": _to_iso8601(start_ts),
        "end": _to_iso8601(end_ts),
        "interval": interval,
    }
    headers: Dict[str, str] = {}
    api_key = ONCHAIN_API_KEY or os.getenv("ONCHAIN_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                ONCHAIN_API_BASE_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code != 200:
                last_error = OnchainAPIError(
                    f"On-chain API returned status {response.status_code}: {response.text}",
                )
            else:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise OnchainAPIError("Failed to decode JSON payload from on-chain API.") from exc
                return _normalize_payload(payload, metrics)

        if attempt < max_retries:
            time.sleep(backoff_seconds * attempt)

    raise OnchainAPIError(f"Failed to fetch on-chain metrics after {max_retries} attempts") from last_error


def load_onchain_cached(path: str) -> pd.DataFrame:
    """Load on-chain metrics from a cached CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached on-chain metrics not found at {path}")

    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError("Cached on-chain CSV must include a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    metric_cols = [col for col in df.columns if col != "ts"]
    for column in metric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("ts").drop_duplicates("ts")
    return df.reset_index(drop=True)
