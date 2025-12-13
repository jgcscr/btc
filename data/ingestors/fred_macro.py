"""FRED macroeconomic data ingestion helper."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
RAW_ROOT = Path("data/raw/macro/fred")


class FREDIngestionError(RuntimeError):
    """Raised when the FRED API call fails."""


def _require_api_key() -> str:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise FREDIngestionError("FRED_API_KEY is not set; export the key before running the loader.")
    return api_key


def _fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
) -> List[dict[str, str]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "frequency": "d",
        "sort_order": "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    try:
        response = requests.get(FRED_API_URL, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise FREDIngestionError(f"FRED request failed: {exc}") from exc

    if response.status_code != 200:
        raise FREDIngestionError(
            f"FRED request returned status {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()
    observations: Iterable[dict[str, str]] = payload.get("observations", [])  # type: ignore[arg-type]
    return list(observations)


def _observations_to_frame(series_id: str, records: Iterable[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        date_str = record.get("date")
        value_str = record.get("value")
        if not date_str or value_str is None:
            continue
        if value_str == ".":
            value = None
        else:
            try:
                value = float(value_str)
            except (TypeError, ValueError):
                value = None

        ts = pd.Timestamp(date_str, tz="UTC")
        rows.append({
            "ts": ts,
            "metric": series_id,
            "value": value,
            "source": "fred",
        })

    if not rows:
        raise FREDIngestionError(f"No observations returned for FRED series {series_id}.")

    frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return frame


def _write_parquet(df: pd.DataFrame, series_id: str, output_root: Path) -> Path:
    output_dir = output_root / f"series={series_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"fred_{series_id}_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def ingest_series(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    output_root: Path = RAW_ROOT,
) -> Path:
    api_key = _require_api_key()
    records = _fetch_fred_series(series_id, api_key, observation_start, observation_end)
    frame = _observations_to_frame(series_id, records)
    return _write_parquet(frame, series_id, output_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FRED series observations into Parquet.")
    parser.add_argument("series_id", help="FRED series identifier (e.g. DTWEXBGS, DGS10).")
    parser.add_argument("--start", dest="observation_start", default=None, help="Observation start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="observation_end", default=None, help="Observation end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw Parquet output (default: data/raw/macro/fred).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = ingest_series(
        series_id=args.series_id,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        output_root=args.output_root,
    )
    print(f"Saved FRED series {args.series_id} to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
