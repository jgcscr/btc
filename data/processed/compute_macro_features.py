"""Aggregate macroeconomic features from raw ingested data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/macro")
OUTPUT_PATH = Path("data/processed/macro/hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/macro_summary.json")


class MacroProcessingError(RuntimeError):
    """Raised when required raw inputs are missing."""


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
        raise MacroProcessingError("No macro raw data found; run the ingestors first.")

    tidy = pd.concat(frames, ignore_index=True)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True)
    tidy = tidy.sort_values("ts")

    pivot = tidy.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index().ffill()  # Carry forward sparse symbols before resampling.
    hourly = pivot.resample("1h").asfreq().ffill()
    hourly.columns = [f"macro_{col}" for col in hourly.columns]
    hourly = hourly.reset_index().rename(columns={"ts": "timestamp"})
    hourly = _add_realized_volatility(hourly)
    return hourly


def _add_realized_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        if column == "timestamp" or not column.endswith("_close"):
            continue
        series = result[column].astype(float)
        safe_series = series.clip(lower=1e-9)
        log_return = np.log(safe_series).diff().fillna(0.0)
        result[f"{column}_realized_vol_1h"] = log_return.abs()
        rv_24h = log_return.rolling(window=24, min_periods=1).std(ddof=0).fillna(0.0)
        result[f"{column}_realized_vol_24h"] = rv_24h
    if any(col.endswith("_realized_vol_1h") for col in result.columns):
        print(
            "Macro realized volatility columns computed using log returns "
            "(1h absolute move, 24h rolling std, NaNs filled with 0).",
        )
    return result


def _write_summary(
    frame: pd.DataFrame,
    path: Path,
    lookback_duration: pd.Timedelta | None,
) -> None:
    window = frame.copy()
    if lookback_duration is not None and not window.empty:
        latest_timestamp = window["timestamp"].max()
        cutoff = latest_timestamp - lookback_duration
        window = window.loc[window["timestamp"] >= cutoff].copy()
    summary = {
        "row_count": int(len(window)),
        "latest_timestamp": window["timestamp"].max().isoformat() if not window.empty else None,
        "columns": {},
    }
    for column in frame.columns:
        if column == "timestamp":
            continue
        column_data = window[column] if column in window else pd.Series(dtype=float)
        if column_data.empty:
            missing_ratio = 1.0
        else:
            missing_ratio = float(column_data.isna().mean())
        summary["columns"][column] = {
            "missing_ratio": missing_ratio,
            "min": float(column_data.min()) if column_data.notna().any() else None,
            "max": float(column_data.max()) if column_data.notna().any() else None,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def process_macro_features(
    raw_root: Path = RAW_ROOT,
    output_path: Path = OUTPUT_PATH,
    summary_lookback_days: int | None = 90,
) -> Path:
    frames = _load_tidy_frames(raw_root)
    hourly = _pivot_hourly(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False)
    lookback_duration = None
    if summary_lookback_days is not None:
        lookback_duration = pd.Timedelta(days=int(summary_lookback_days))
    _write_summary(hourly, SUMMARY_PATH, lookback_duration)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly macro feature parquet from raw inputs.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory containing tidy macro parquet files (default: data/raw/macro).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output parquet path (default: data/processed/macro/hourly_features.parquet).",
    )
    parser.add_argument(
        "--summary-lookback-days",
        type=int,
        default=90,
        help="Window in days for monitoring summary statistics (default: 90).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = process_macro_features(
        raw_root=args.raw_root,
        output_path=args.output,
        summary_lookback_days=args.summary_lookback_days,
    )
    print(f"Wrote macro features to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
