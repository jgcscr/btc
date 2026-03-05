"""Resample CryptoQuant daily metrics to hourly fallback features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_ROOT = Path("data/raw/cryptoquant_daily")
OUTPUT_PATH = Path("data/processed/cryptoquant/hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/cryptoquant_daily_summary.json")
ZSCORE_WINDOW = 30
ZSCORE_MIN_PERIODS = 5


class CryptoQuantProcessingError(RuntimeError):
    """Raised when expected CryptoQuant raw data is missing."""


def _load_tidy_frames(root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not root.exists():
        return frames
    for parquet_path in root.rglob("*.parquet"):
        frame = pd.read_parquet(parquet_path)
        if {"ts", "metric", "value"}.issubset(frame.columns):
            frames.append(frame[["ts", "metric", "value"]])
    return frames


def _pivot_daily(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(frames), ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
    combined = combined.sort_values("ts")
    pivot = combined.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    return pivot


def _resample_hourly(pivot: pd.DataFrame) -> pd.DataFrame:
    base_columns = [str(col) for col in pivot.columns]

    hourly = pivot.resample("1h").ffill()
    hourly.columns = [f"cq_daily_{col}" for col in base_columns]

    daily_delta = pivot.diff().resample("1h").ffill()
    daily_delta.columns = [f"cq_daily_delta_{col}" for col in base_columns]

    daily_pct = pivot.pct_change()
    daily_pct = daily_pct.replace([float("inf"), float("-inf")], pd.NA)
    daily_pct = daily_pct.resample("1h").ffill()
    daily_pct.columns = [f"cq_daily_pct_{col}" for col in base_columns]

    rolling_mean = pivot.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
    rolling_std = pivot.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std(ddof=0)
    rolling_std = rolling_std.replace(0, pd.NA)
    zscore_daily = (pivot - rolling_mean) / rolling_std
    zscore_hourly = zscore_daily.resample("1h").ffill()
    zscore_hourly.columns = [f"cq_daily_zscore_{col}" for col in base_columns]

    merged = pd.concat([hourly, daily_delta, daily_pct, zscore_hourly], axis=1)
    merged = merged.reset_index().rename(columns={"ts": "timestamp"})
    return merged


def _summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "row_count": 0,
            "first_timestamp": None,
            "latest_timestamp": None,
            "zero_valued_columns": {},
            "note": "No CryptoQuant daily data available; hourly feed pending support ticket CQ-2025-1213.",
        }

    summary = {
        "row_count": int(len(frame)),
        "first_timestamp": frame["timestamp"].min().isoformat(),
        "latest_timestamp": frame["timestamp"].max().isoformat(),
        "zero_valued_columns": {},
        "feature_groups": {
            "value": len(
                [
                    c
                    for c in frame.columns
                    if c.startswith("cq_daily_") and "delta" not in c and "zscore" not in c and "pct_" not in c
                ]
            ),
            "delta": len([c for c in frame.columns if c.startswith("cq_daily_delta_")]),
            "pct": len([c for c in frame.columns if c.startswith("cq_daily_pct_")]),
            "zscore": len([c for c in frame.columns if c.startswith("cq_daily_zscore_")]),
        },
        "note": (
            "Derived from CryptoQuant daily metrics via forward-fill; true hourly data pending support ticket "
            "CQ-2025-1213."
        ),
    }
    for column in frame.columns:
        if column == "timestamp":
            continue
        series = frame[column]
        zero_ratio = float((series == 0).mean()) if len(series) else 0.0
        summary["zero_valued_columns"][column] = {
            "fraction_zero": zero_ratio,
            "missing_ratio": float(series.isna().mean()),
        }
    return summary


def process_cryptoquant_resampled(
    raw_root: Path = RAW_ROOT,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> Path:
    frames = _load_tidy_frames(raw_root)
    if not frames:
        summary = _summarize(pd.DataFrame())
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print("No CryptoQuant daily raw data found; summary recorded with pending support note.")
        return output_path

    pivot = _pivot_daily(frames)
    hourly = _resample_hourly(pivot)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False)

    summary = _summarize(hourly)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        "Generated CryptoQuant hourly fallback features "
        f"({len(hourly)} rows, latest {summary['latest_timestamp']}).",
    )
    print("Note: true hourly data remains pending support authorization.")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample CryptoQuant daily metrics into hourly fallback features.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Root directory of raw daily parquet files.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output parquet path for hourly features.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Monitoring summary JSON path (default: artifacts/monitoring/cryptoquant_daily_summary.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_cryptoquant_resampled(
        raw_root=args.raw_root,
        output_path=args.output,
        summary_path=args.summary,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
