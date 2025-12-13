"""Blockchain.com on-chain metrics ingestion helper."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from io import StringIO

BLOCKCHAIN_URL_TEMPLATE = "https://api.blockchain.info/charts/{metric}?timespan={timespan}&format=csv"
RAW_ROOT = Path("data/raw/onchain/blockchain")
SUPPORTED_METRICS = {
    "activeaddresses",
    "hash-rate",
    "difficulty",
    "market-price",
    "n-unique-addresses",
}
SUPPORTED_TIMESPANS = {
    "30days",
    "60days",
    "1year",
    "2years",
    "all",
}


class BlockchainIngestionError(RuntimeError):
    """Raised when the Blockchain.com request fails."""


def _fetch_csv(metric: str, timespan: str) -> pd.DataFrame:
    url = BLOCKCHAIN_URL_TEMPLATE.format(metric=metric, timespan=timespan)
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise BlockchainIngestionError(f"Blockchain.com request failed: {exc}") from exc

    if response.status_code != 200:
        raise BlockchainIngestionError(
            f"Blockchain.com request returned status {response.status_code}: {response.text[:256]}",
        )

    csv_text = response.text
    frame = pd.read_csv(StringIO(csv_text))
    if frame.empty:
        raise BlockchainIngestionError("Blockchain.com response returned no data.")
    return frame


def _to_tidy(metric: str, frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {"Timestamp", "Value"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise BlockchainIngestionError(f"Unexpected CSV schema, missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        ts = pd.Timestamp(row["Timestamp"], unit="s", tz="UTC")
        value = row["Value"]
        rows.append({
            "ts": ts,
            "metric": metric,
            "value": float(value) if pd.notna(value) else None,
            "source": "blockchain",
        })

    tidy = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return tidy


def ingest_metric(metric: str, timespan: str = "1year", output_root: Path = RAW_ROOT) -> Path:
    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise BlockchainIngestionError(f"Unsupported metric '{metric}'. Supported: {supported}")
    if timespan not in SUPPORTED_TIMESPANS:
        supported = ", ".join(sorted(SUPPORTED_TIMESPANS))
        raise BlockchainIngestionError(f"Unsupported timespan '{timespan}'. Supported: {supported}")

    raw_frame = _fetch_csv(metric, timespan)
    tidy_frame = _to_tidy(metric, raw_frame)

    output_dir = output_root / f"metric={metric}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"blockchain_{metric}_{timestamp}.parquet"
    tidy_frame.to_parquet(output_path, index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch blockchain.com on-chain metrics into Parquet.")
    parser.add_argument("metric", choices=sorted(SUPPORTED_METRICS), help="Blockchain.com metric identifier.")
    parser.add_argument(
        "--timespan",
        default="1year",
        choices=sorted(SUPPORTED_TIMESPANS),
        help="Timespan for the metric (default: 1year).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw Parquet output (default: data/raw/onchain/blockchain).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = ingest_metric(metric=args.metric, timespan=args.timespan, output_root=args.output_root)
    print(f"Saved blockchain.com metric {args.metric} ({args.timespan}) to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
