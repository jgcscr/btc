"""Aggregate CoinAPI market and funding features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MARKET_RAW_ROOT = Path("data/raw/market/coinapi")
BINANCEUS_MARKET_ROOT = Path("data/raw/market/binanceus")
FUNDING_RAW_ROOT = Path("data/raw/funding/coinapi")
MARKET_OUTPUT = Path("data/processed/coinapi/market_hourly_features.parquet")
FUNDING_OUTPUT = Path("data/processed/coinapi/funding_hourly_features.parquet")
SUMMARY_PATH = Path("artifacts/monitoring/coinapi_summary.json")


class CoinAPIProcessingError(RuntimeError):
    """Raised when no CoinAPI raw data is available."""


def _load_tidy_frames(root: Path, allowed_prefixes: tuple[str, ...] | None = None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not root.exists():
        return frames
    for parquet_path in root.rglob("*.parquet"):
        frame = pd.read_parquet(parquet_path)
        if {"ts", "metric", "value"}.issubset(frame.columns):
            subset = frame[["ts", "metric", "value"]]
            if allowed_prefixes:
                mask = subset["metric"].astype(str).str.startswith(allowed_prefixes)
                subset = subset[mask]
            if not subset.empty:
                frames.append(subset)
    return frames


def _pivot(frames: Iterable[pd.DataFrame], prefix: str) -> pd.DataFrame:
    frames = list(frames)
    if not frames:
        raise CoinAPIProcessingError(f"No CoinAPI raw data found for prefix {prefix}; run coinapi loader first.")

    tidy = pd.concat(frames, ignore_index=True)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True)
    tidy = tidy.sort_values("ts")

    pivot = tidy.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    resampled = pivot.resample("1h").ffill()
    resampled.columns = [f"{prefix}_{col}" for col in resampled.columns]
    resampled = resampled.reset_index().rename(columns={"ts": "timestamp"})
    if prefix == "coinapi":
        resampled = _add_market_derivatives(resampled)
    if prefix == "coinapi_funding":
        resampled = _add_funding_derivatives(resampled)
    return resampled


def _add_market_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    spot_close = result.get("coinapi_spot_close")
    futures_close = result.get("coinapi_futures_close")
    if spot_close is not None and futures_close is not None:
        spot_safe = spot_close.astype(float).clip(lower=1e-9)
        fut_safe = futures_close.astype(float).clip(lower=1e-9)

        spot_log_ret = np.log(spot_safe).diff().fillna(0.0)
        fut_log_ret = np.log(fut_safe).diff().fillna(0.0)

        result["coinapi_spot_realized_vol_1h"] = spot_log_ret.abs()
        result["coinapi_futures_realized_vol_1h"] = fut_log_ret.abs()

        result["coinapi_spot_realized_vol_24h"] = (
            spot_log_ret.rolling(window=24, min_periods=1).std(ddof=0).fillna(0.0)
        )
        result["coinapi_futures_realized_vol_24h"] = (
            fut_log_ret.rolling(window=24, min_periods=1).std(ddof=0).fillna(0.0)
        )

        basis = futures_close.astype(float) - spot_close.astype(float)
        result["coinapi_basis_spread_1h"] = basis
        basis_mean = basis.rolling(window=24, min_periods=1).mean()
        result["coinapi_basis_spread_24h"] = basis_mean.ffill().fillna(0.0)

        print(
            "CoinAPI market derivatives computed: realized volatility from log returns, "
            "basis spread (perp minus spot) with 24h rolling mean; initial gaps filled with 0.",
        )
    return result


def _add_funding_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rate = result.get("coinapi_funding_funding_rate")
    if rate is not None:
        rate_series = rate.astype(float)
        result["coinapi_funding_funding_rate_delta_1h"] = rate_series.diff().fillna(0.0)
        result["coinapi_funding_funding_rate_delta_24h"] = rate_series.diff(periods=24).fillna(0.0)
        print(
            "CoinAPI funding deltas computed: 1h diff and 24h lag diff; NaNs filled with 0 for early rows.",
        )
    return result


def _write_summary(market: pd.DataFrame | None, funding: pd.DataFrame | None) -> None:
    summary = {
        "market": None,
        "funding": None,
    }
    for name, frame in ("market", market), ("funding", funding):
        if frame is None or frame.empty:
            summary[name] = {
                "row_count": 0,
                "latest_timestamp": None,
                "columns": {},
            }
            continue
        columns = {}
        for column in frame.columns:
            if column == "timestamp":
                continue
            series = frame[column]
            columns[column] = {
                "missing_ratio": float(series.isna().mean()),
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
            }
        summary[name] = {
            "row_count": int(len(frame)),
            "latest_timestamp": frame["timestamp"].max().isoformat(),
            "columns": columns,
        }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def process_coinapi_features(
    market_root: Path = MARKET_RAW_ROOT,
    funding_root: Path = FUNDING_RAW_ROOT,
    market_output: Path = MARKET_OUTPUT,
    funding_output: Path = FUNDING_OUTPUT,
) -> tuple[Path, Path | None]:
    spot_frames = _load_tidy_frames(BINANCEUS_MARKET_ROOT, allowed_prefixes=("spot_",))
    if not spot_frames:
        spot_frames = _load_tidy_frames(market_root, allowed_prefixes=("spot_",))
    futures_frames = _load_tidy_frames(market_root, allowed_prefixes=("futures_",))
    market_frames = spot_frames + futures_frames
    market = _pivot(market_frames, "coinapi")
    market_output.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(market_output, index=False)

    funding_frames = _load_tidy_frames(funding_root)
    funding: pd.DataFrame | None
    funding_path: Path | None = None
    if funding_frames:
        funding = _pivot(funding_frames, "coinapi_funding")
        funding_output.parent.mkdir(parents=True, exist_ok=True)
        funding.to_parquet(funding_output, index=False)
        funding_path = funding_output
    else:
        funding = None
        print("CoinAPI funding frames not found; skipping funding output.")

    _write_summary(market, funding)
    return market_output, funding_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoinAPI hourly market and funding features.")
    parser.add_argument("--market-root", type=Path, default=MARKET_RAW_ROOT, help="Raw market parquet root.")
    parser.add_argument("--funding-root", type=Path, default=FUNDING_RAW_ROOT, help="Raw funding parquet root.")
    parser.add_argument("--market-output", type=Path, default=MARKET_OUTPUT, help="Output parquet for market features.")
    parser.add_argument("--funding-output", type=Path, default=FUNDING_OUTPUT, help="Output parquet for funding features.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_coinapi_features(
        market_root=args.market_root,
        funding_root=args.funding_root,
        market_output=args.market_output,
        funding_output=args.funding_output,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
