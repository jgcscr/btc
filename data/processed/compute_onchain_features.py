"""Aggregate on-chain features from raw blockchain metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.ingestors.cryptoquant_stub import skip_cryptoquant_ingestion

RAW_ROOT = Path("data/raw/onchain")
OUTPUT_PATH = Path("data/processed/onchain/hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/onchain_summary.json")


def _load_tidy_frames(root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not root.exists():
        return frames
    for parquet_path in root.rglob("*.parquet"):
        frame = pd.read_parquet(parquet_path)
        if {"ts", "metric", "value"}.issubset(frame.columns):
            frames.append(frame[["ts", "metric", "value"]])
    return frames


def _pivot_hourly(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    if not frames:
        skip_cryptoquant_ingestion()
        return pd.DataFrame(columns=["timestamp"]).astype({"timestamp": "datetime64[ns, UTC]"})

    tidy = pd.concat(frames, ignore_index=True)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True)
    tidy = tidy.sort_values("ts")

    pivot = tidy.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    hourly = pivot.resample("1h").ffill()
    hourly.columns = [f"onchain_{col}" for col in hourly.columns]
    hourly = hourly.reset_index().rename(columns={"ts": "timestamp"})
    return hourly


def _write_summary(frame: pd.DataFrame, path: Path) -> None:
    summary = {
        "row_count": int(len(frame)),
        "latest_timestamp": frame["timestamp"].max().isoformat() if not frame.empty else None,
        "columns": {},
    }
    for column in frame.columns:
        if column == "timestamp":
            continue
        column_data = frame[column]
        missing_ratio = float(column_data.isna().mean()) if not frame.empty else 1.0
        summary["columns"][column] = {
            "missing_ratio": missing_ratio,
            "min": float(column_data.min()) if column_data.notna().any() else None,
            "max": float(column_data.max()) if column_data.notna().any() else None,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def process_onchain_features(raw_root: Path = RAW_ROOT, output_path: Path = OUTPUT_PATH) -> Path:
    frames = _load_tidy_frames(raw_root)
    hourly = _pivot_hourly(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hourly.empty:
        if output_path.exists():
            output_path.unlink()
        _write_summary(hourly, SUMMARY_PATH)
        return output_path

    hourly.to_parquet(output_path, index=False)
    _write_summary(hourly, SUMMARY_PATH)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly on-chain feature parquet from raw inputs.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory containing tidy on-chain parquet files (default: data/raw/onchain).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output parquet path (default: data/processed/onchain/hourly_features.parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = process_onchain_features(raw_root=args.raw_root, output_path=args.output)
    print(f"Wrote on-chain features to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
