import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    ONCHAIN_API_BASE_URL,
    ONCHAIN_API_KEY,
    ONCHAIN_DEFAULT_INTERVAL,
    ONCHAIN_METRICS,
)


DEFAULT_ONCHAIN_OUTPUT_PATH = Path("data/processed/onchain/hourly_features.parquet")
DEFAULT_ONCHAIN_METADATA_PATH = Path("data/processed/onchain/source_manifest.json")
DEFAULT_ONCHAIN_START_DATE = "2018-01-01"
DEFAULT_ONCHAIN_REFRESH_OVERLAP_HOURS = 72
DEFAULT_PUBLIC_ONCHAIN_TIMESPAN = "all"

BLOCKCHAIN_CHART_METRIC_MAP = {
    "active_addresses": "n-unique-addresses",
    "new_addresses": "n-new-addresses",
    "transaction_count": "n-transactions",
    "hashrate": "hash-rate",
    "difficulty": "difficulty",
}

ONCHAIN_RAW_COLUMNS = tuple(f"onchain_{metric}" for metric in ONCHAIN_METRICS)
ONCHAIN_FEATURE_COLUMNS = tuple(
    value
    for metric in ONCHAIN_METRICS
    for value in (
        f"onchain_{metric}",
        f"onchain_{metric}_change_1h",
        f"onchain_{metric}_zscore_24h",
        f"onchain_{metric}_trend_6h",
    )
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


def _fetch_blockchain_chart_series(chart_name: str) -> pd.DataFrame:
    url = f"https://api.blockchain.info/charts/{chart_name}"
    try:
        response = requests.get(
            url,
            params={"timespan": DEFAULT_PUBLIC_ONCHAIN_TIMESPAN, "format": "json"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OnchainAPIError(f"Failed to fetch public on-chain chart {chart_name}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise OnchainAPIError(f"Public on-chain chart {chart_name} returned invalid JSON.") from exc

    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise OnchainAPIError(f"Public on-chain chart {chart_name} returned no values.")

    rows = []
    for point in values:
        if not isinstance(point, dict) or "x" not in point:
            continue
        ts = pd.Timestamp(point["x"], unit="s", tz="UTC")
        rows.append({"ts": ts, "value": point.get("y")})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise OnchainAPIError(f"Public on-chain chart {chart_name} returned no usable rows.")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)


def _fetch_public_onchain_metrics(
    start_ts: pd.Timestamp | str,
    end_ts: pd.Timestamp | str,
    metrics: List[str],
) -> pd.DataFrame:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    combined: pd.DataFrame | None = None

    for metric in metrics:
        chart_name = BLOCKCHAIN_CHART_METRIC_MAP.get(metric)
        if not chart_name:
            continue
        try:
            series = _fetch_blockchain_chart_series(chart_name).rename(columns={"value": metric})
        except OnchainAPIError:
            continue
        combined = series if combined is None else combined.merge(series, on="ts", how="outer")

    if combined is None or combined.empty:
        raise OnchainAPIError("No public on-chain metrics were available from fallback sources.")

    combined = combined.sort_values("ts").reset_index(drop=True)
    combined = combined[(combined["ts"] >= start.floor("D")) & (combined["ts"] <= end.ceil("D"))].copy()
    if combined.empty:
        raise OnchainAPIError("Public on-chain fallback returned no rows in the requested window.")

    combined = combined.set_index("ts").sort_index()
    hourly_index = pd.date_range(combined.index.min(), combined.index.max() + pd.Timedelta(hours=23), freq="h", tz="UTC")
    combined = combined.reindex(hourly_index).ffill()
    combined.index.name = "ts"
    combined = combined.reset_index()
    for metric in metrics:
        if metric not in combined.columns:
            combined[metric] = np.nan
    return combined.loc[:, ["ts", *metrics]].reset_index(drop=True)


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
        return _fetch_public_onchain_metrics(start_ts, end_ts, metrics)

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


def _prefix_onchain_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map = {}
    for metric in ONCHAIN_METRICS:
        if metric in result.columns:
            rename_map[metric] = f"onchain_{metric}"
    result = result.rename(columns=rename_map)
    return result


def _extract_existing_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "ts" not in result.columns:
        if "timestamp" in result.columns:
            result = result.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError("Existing on-chain frame must include 'ts' or 'timestamp'.")
    result["ts"] = pd.to_datetime(result["ts"], utc=True, errors="coerce")
    result = _prefix_onchain_columns(result)
    available = ["ts", *[column for column in ONCHAIN_RAW_COLUMNS if column in result.columns]]
    result = result.loc[:, available].dropna(subset=["ts"])
    result = result.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    return result.reset_index(drop=True)


def _add_derived_onchain_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = _extract_existing_raw_frame(frame)
    for column in ONCHAIN_RAW_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        series = pd.to_numeric(result[column], errors="coerce").ffill()
        result[column] = series
        result[f"{column}_change_1h"] = series.diff().fillna(0.0)
        rolling_mean = series.rolling(window=24, min_periods=8).mean()
        rolling_std = series.rolling(window=24, min_periods=8).std(ddof=0).replace(0.0, np.nan)
        zscore = ((series - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
        result[f"{column}_zscore_24h"] = zscore.clip(lower=-10.0, upper=10.0).fillna(0.0)
        trend = series.pct_change(periods=6, fill_method=None).replace([np.inf, -np.inf], np.nan)
        result[f"{column}_trend_6h"] = trend.clip(lower=-5.0, upper=5.0).fillna(0.0)
    ordered = ["ts", *ONCHAIN_FEATURE_COLUMNS]
    for column in ordered[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.loc[:, ordered].sort_values("ts").reset_index(drop=True)


def build_onchain_feature_frame(
    *,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
    existing: pd.DataFrame | None = None,
    raw_frame: pd.DataFrame | None = None,
    interval: str = ONCHAIN_DEFAULT_INTERVAL,
) -> pd.DataFrame:
    if raw_frame is None:
        raw = fetch_onchain_metrics(
            start_ts or DEFAULT_ONCHAIN_START_DATE,
            end_ts or pd.Timestamp.utcnow(),
            interval=interval,
        )
    else:
        raw = raw_frame.copy()
    raw = _prefix_onchain_columns(raw)
    if existing is not None and not existing.empty:
        combined = pd.concat([_extract_existing_raw_frame(existing), _extract_existing_raw_frame(raw)], ignore_index=True)
        combined = combined.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    else:
        combined = _extract_existing_raw_frame(raw)
    return _add_derived_onchain_features(combined)


def load_onchain_features(path: str | Path = DEFAULT_ONCHAIN_OUTPUT_PATH) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"On-chain feature file not found at {resolved}")
    frame = pd.read_parquet(resolved)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    return frame.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)


def resolve_incremental_start_timestamp(
    existing: pd.DataFrame | None,
    *,
    default_start: str = DEFAULT_ONCHAIN_START_DATE,
    overlap_hours: int = DEFAULT_ONCHAIN_REFRESH_OVERLAP_HOURS,
) -> str:
    if existing is None or existing.empty:
        return default_start
    history = _extract_existing_raw_frame(existing)
    if history.empty:
        return default_start
    last_ts = pd.to_datetime(history["ts"], utc=True, errors="coerce").max()
    if pd.isna(last_ts):
        return default_start
    return (last_ts - pd.Timedelta(hours=max(overlap_hours, 0))).isoformat().replace("+00:00", "Z")


def build_onchain_source_manifest() -> dict[str, object]:
    return {
        "metrics": list(ONCHAIN_METRICS),
        "interval": ONCHAIN_DEFAULT_INTERVAL,
        "output_columns": list(ONCHAIN_FEATURE_COLUMNS),
        "provider": "configured_onchain_api_or_cached_frame_or_public_blockchain_chart_fallback",
    }


def write_onchain_source_manifest(path: str | Path, payload: dict[str, object]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
