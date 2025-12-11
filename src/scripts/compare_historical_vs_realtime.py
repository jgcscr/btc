"""Compare historical paper-trade signals with realtime logged signals.

This helper CLI reports divergences between the historical paper-trading loop
and the realtime signal logger when both have records for the same timestamps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare historical paper-trade results with realtime signals."
    )
    parser.add_argument(
        "--hist-path",
        default="artifacts/analysis/paper_trade_v1/paper_trade.csv",
        help="Path to historical paper-trade CSV (default: artifacts/analysis/paper_trade_v1/paper_trade.csv).",
    )
    parser.add_argument(
        "--live-path",
        default="artifacts/live/paper_trade_realtime.csv",
        help="Path to realtime signal CSV (default: artifacts/live/paper_trade_realtime.csv).",
    )
    return parser.parse_args()


def ensure_file(path_str: str) -> Path | None:
    path = Path(path_str)
    if not path.exists():
        print(f"Missing CSV at {path}", file=sys.stderr)
        return None
    if not path.is_file():
        print(f"Expected file but found non-file path at {path}", file=sys.stderr)
        return None
    return path


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=["ts"])
    except FileNotFoundError:
        print(f"Unable to open CSV at {path}", file=sys.stderr)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
    return None


def summarize_differences(merged: pd.DataFrame) -> None:
    count = len(merged)
    print(f"Overlapping bars: {count}")

    metrics = [
        ("p_up", merged["p_up_hist"], merged["p_up_live"]),
        ("ret_pred", merged["ret_pred_hist"], merged["ret_pred_live"]),
    ]

    for field, hist_series, live_series in metrics:
        abs_diff = (hist_series - live_series).abs()
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        print(
            f"{field}: max abs diff = {max_diff:.6g}, mean abs diff = {mean_diff:.6g}"
            if count
            else f"{field}: no overlapping data"
        )

    mismatches = (merged["signal_ensemble_hist"] != merged["signal_ensemble_live"]).sum()
    if mismatches == 0:
        print(
            "Signals match exactly for all overlapping bars." if count else "No overlap to compare signals."
        )
    else:
        print(
            "Signals differ on "
            f"{mismatches}/{count} bars; inspect signal_ensemble columns for details."
        )


def main() -> int:
    args = parse_args()

    hist_path = ensure_file(args.hist_path)
    live_path = ensure_file(args.live_path)
    if hist_path is None or live_path is None:
        return 1

    hist_df = load_csv(hist_path)
    live_df = load_csv(live_path)
    if hist_df is None or live_df is None:
        return 1

    overlap = pd.merge(
        hist_df,
        live_df,
        on="ts",
        how="inner",
        suffixes=("_hist", "_live"),
    )

    if overlap.empty:
        print("No overlapping timestamps found between historical and realtime data.")
        return 0

    required_cols = {
        "p_up_hist",
        "p_up_live",
        "ret_pred_hist",
        "ret_pred_live",
        "signal_ensemble_hist",
        "signal_ensemble_live",
    }
    missing_cols = required_cols - set(overlap.columns)
    if missing_cols:
        print(
            "Missing expected columns in merged data: " + ", ".join(sorted(missing_cols)),
            file=sys.stderr,
        )
        return 1

    summarize_differences(overlap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
