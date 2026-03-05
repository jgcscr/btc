"""CryptoCompare on-chain metrics ingestion helper."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import requests

BASE_URL = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
DEFAULT_LIMIT = 720
DEFAULT_OUTPUT_ROOT = Path("data/raw/onchain/cryptocompare")
SOURCE_NAME = "cryptocompare"
API_METRIC_BY_CANONICAL = {
    "active_addresses": "activeaddresses",
    "new_addresses": "newaddresses",
    "transaction_count": "transactioncount",
    "hashrate": "hashrate",
    "difficulty": "difficulty",
}
CANONICAL_BY_INPUT = {
    "active_addresses": "active_addresses",
    "new_addresses": "new_addresses",
    "transaction_count": "transaction_count",
    "hashrate": "hashrate",
    "hash_rate": "hashrate",
    "difficulty": "difficulty",
}
SUPPORTED_METRICS = set(CANONICAL_BY_INPUT.keys())


class CryptoCompareIngestionError(RuntimeError):
    """Raised when the CryptoCompare request fails or returns malformed data."""


def _build_params(metric: str, limit: int, api_key: Optional[str]) -> dict[str, object]:
    params: dict[str, object] = {
        "fsym": "BTC",
        "metric": metric,
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def fetch_hourly_metrics(metric: str, limit: int = DEFAULT_LIMIT, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch on-chain metrics for BTC from CryptoCompare.

    The CryptoCompare blockchain API currently exposes daily aggregates for the
    requested metric. We still resample the data to hourly cadence downstream,
    so callers can continue to request an hourly observation limit here.
    """

    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise CryptoCompareIngestionError(f"Unsupported metric '{metric}'. Supported: {supported}")

    canonical_metric = CANONICAL_BY_INPUT[metric]
    api_metric = API_METRIC_BY_CANONICAL[canonical_metric]
    params = _build_params(metric=api_metric, limit=limit, api_key=api_key)
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network safeguard
        raise CryptoCompareIngestionError(f"CryptoCompare request failed: {exc}") from exc

    if response.status_code != 200:
        raise CryptoCompareIngestionError(
            f"CryptoCompare request returned status {response.status_code}: {response.text[:256]}",
        )

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - malformed payload safeguard
        raise CryptoCompareIngestionError("Failed to decode JSON payload from CryptoCompare.") from exc

    if not isinstance(payload, dict):
        raise CryptoCompareIngestionError("CryptoCompare payload is not an object.")

    if payload.get("Response") != "Success":
        message = payload.get("Message") or payload.get("MessageText") or "Unknown error"
        raise CryptoCompareIngestionError(f"CryptoCompare reported failure: {message}")

    data_section = payload.get("Data")
    if not isinstance(data_section, dict):
        raise CryptoCompareIngestionError("CryptoCompare payload missing 'Data' section.")

    rows = data_section.get("Data")
    if not isinstance(rows, Iterable):
        raise CryptoCompareIngestionError("CryptoCompare payload missing 'Data.Data' records.")

    tidy_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts_raw = row.get("time")
        value = row.get("value")
        if ts_raw is None:
            continue
        ts = pd.to_datetime(ts_raw, unit="s", utc=True)
        tidy_rows.append({
            "ts": ts,
            "metric": canonical_metric,
            "value": float(value) if value is not None else None,
            "source": SOURCE_NAME,
        })

    if not tidy_rows:
        raise CryptoCompareIngestionError("CryptoCompare response did not include any records.")

    frame = pd.DataFrame(tidy_rows).sort_values("ts").reset_index(drop=True)
    return frame


def ingest_metrics(
    metrics: Sequence[str],
    limit: int = DEFAULT_LIMIT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    api_key: Optional[str] = None,
) -> list[Path]:
    api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    output_root.mkdir(parents=True, exist_ok=True)

    stored_paths: list[Path] = []
    for metric in metrics:
        frame = fetch_hourly_metrics(metric=metric, limit=limit, api_key=api_key)
        metric_dir = output_root / f"metric={metric}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_path = metric_dir / f"{SOURCE_NAME}_{metric}_{timestamp}.parquet"
        frame.to_parquet(output_path, index=False)
        stored_paths.append(output_path)

    return stored_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Bitcoin on-chain metrics from CryptoCompare.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more CryptoCompare metric identifiers (e.g., active_addresses, new_addresses).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of hourly records to fetch (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for raw CryptoCompare parquet files (default: data/raw/onchain/cryptocompare).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="CryptoCompare API key. Falls back to CRYPTOCOMPARE_API_KEY environment variable if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = ingest_metrics(
        metrics=args.metrics,
        limit=args.limit,
        output_root=args.output_root,
        api_key=args.api_key,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
