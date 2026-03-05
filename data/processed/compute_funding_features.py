"""Build hourly Binance funding rate features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.binance_klines import fetch_funding_rates

RAW_ROOT = Path("data/raw/funding/binance")
OUTPUT_PATH = Path("data/processed/funding/hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/funding_summary.json")


def _is_env_truthy(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class FundingProcessingError(RuntimeError):
    """Raised when no funding data is available."""


def _fetch_live(pair: str, limit: int) -> pd.DataFrame:
    frame = fetch_funding_rates(symbol=pair, limit=limit)
    if frame.empty:
        raise FundingProcessingError(f"No funding data returned for pair {pair}.")
    tidy = frame.rename(columns={"funding_rate": "value"})[["ts", "value"]]
    tidy["metric"] = f"{pair}_funding_rate"
    tidy["source"] = "binance"
    return tidy


def _write_raw(df: pd.DataFrame, pair: str, output_root: Path) -> Path:
    output_dir = output_root / f"pair={pair}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"binance_{pair}_funding_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


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
        raise FundingProcessingError("No funding raw data found; run with --fetch or stage raw files.")

    tidy = pd.concat(frames, ignore_index=True)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True)
    tidy = tidy.sort_values("ts")

    pivot = tidy.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    hourly = pivot.resample("1h").ffill()
    hourly.columns = [f"funding_{col}" for col in hourly.columns]
    derived = hourly.copy()
    for column in hourly.columns:
        derived[f"{column}_annualized"] = hourly[column] * 3 * 365  # funding occurs every 8 hours
    derived = derived.reset_index().rename(columns={"ts": "timestamp"})
    return derived


def _empty_hourly_frame(pair: str) -> pd.DataFrame:
    metric_col = f"funding_{pair}_funding_rate"
    annualized_col = f"{metric_col}_annualized"
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([], tz="UTC"),
            metric_col: pd.Series(dtype="float64"),
            annualized_col: pd.Series(dtype="float64"),
        }
    )


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


def process_funding_features(
    pair: str,
    live_fetch: bool,
    live_limit: int,
    allow_missing: bool,
    raw_root: Path = RAW_ROOT,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    if live_fetch:
        live_frame = _fetch_live(pair, live_limit)
        _write_raw(live_frame, pair, raw_root)

    frames = _load_tidy_frames(raw_root)
    if not frames:
        if not allow_missing:
            raise FundingProcessingError("No funding raw data found; enable --allow-missing or RUN_WITHOUT_FUNDING to proceed.")

        print(
            "Funding features: no raw inputs detected but RUN_WITHOUT_FUNDING/--allow-missing is set; "
            "writing empty funding feature parquet.",
        )
        hourly = _empty_hourly_frame(pair)
    else:
        hourly = _pivot_hourly(frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False)
    _write_summary(hourly, SUMMARY_PATH)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly funding feature parquet from raw inputs.")
    parser.add_argument("--pair", default="BTCUSDT", help="Futures pair to load funding for (default: BTCUSDT).")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch latest funding data from Binance before processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of funding events to request when fetching live data (default: 1000).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory containing tidy funding parquet files (default: data/raw/funding/binance).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output parquet path (default: data/processed/funding/hourly_features.parquet).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing funding inputs and emit an empty parquet instead of raising.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    allow_missing = args.allow_missing or _is_env_truthy("RUN_WITHOUT_FUNDING")
    output_path = process_funding_features(
        pair=args.pair,
        live_fetch=args.fetch,
        live_limit=args.limit,
        allow_missing=allow_missing,
        raw_root=args.raw_root,
        output_path=args.output,
    )
    print(f"Wrote funding features to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
