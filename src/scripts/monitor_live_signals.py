"""Monitor recent live BTC signal behavior against historical expectations."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

EXPECTED_FRACTION_LONG = 0.35
FRACTION_LONG_TOL = 0.10


@dataclass
class MonitoringResult:
    n_rows: int
    n_long: int
    n_dir_only: int
    n_overlap: int
    fraction_long: float
    p_up_stats: dict[str, float]
    ret_pred_stats: dict[str, float]
    fraction_evaluation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect recent live signals and compare to v1 expectations.",
    )
    parser.add_argument(
        "--live-path",
        default="artifacts/live/paper_trade_realtime.csv",
        help="Path to the live signal log CSV (default: artifacts/live/paper_trade_realtime.csv).",
    )
    parser.add_argument(
        "--window-trades",
        type=int,
        default=300,
        help="Number of most recent rows to consider when computing diagnostics (default: 300).",
    )
    parser.add_argument(
        "--p-up-threshold",
        type=float,
        default=0.45,
        help="Reference ensemble p_up threshold (default: 0.45).",
    )
    parser.add_argument(
        "--ret-threshold",
        type=float,
        default=0.0,
        help="Reference ensemble ret_pred threshold (default: 0.0).",
    )
    return parser.parse_args()


def ensure_live_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing live signal CSV: {path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if df.empty:
        print(f"Live signal CSV at {path} is empty.", file=sys.stderr)
        raise SystemExit(1)

    required_cols = {"ts", "p_up", "ret_pred", "signal_ensemble", "signal_dir_only"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            "Live signal CSV is missing required columns: " + ", ".join(sorted(missing_cols)),
            file=sys.stderr,
        )
        raise SystemExit(1)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].isna().any():
        print("Found unparseable timestamps in live signal CSV.", file=sys.stderr)
        raise SystemExit(1)

    return df.sort_values("ts")


def compute_stats(df: pd.DataFrame, window_trades: int) -> MonitoringResult:
    window = df.tail(window_trades)
    n_rows = len(window)
    if n_rows == 0:
        print("Recent window is empty after filtering.", file=sys.stderr)
        raise SystemExit(1)

    signal_ensemble = window["signal_ensemble"].astype(int)
    signal_dir_only = window["signal_dir_only"].astype(int)

    n_long = int(signal_ensemble.sum())
    n_dir_only = int(signal_dir_only.sum())
    n_overlap = int(((signal_ensemble == 1) & (signal_dir_only == 1)).sum())
    fraction_long = float(n_long / n_rows)

    p_up_stats = {
        "mean": float(window["p_up"].mean()),
        "std": float(window["p_up"].std(ddof=0)),
        "min": float(window["p_up"].min()),
        "max": float(window["p_up"].max()),
    }
    ret_pred_stats = {
        "mean": float(window["ret_pred"].mean()),
        "std": float(window["ret_pred"].std(ddof=0)),
        "min": float(window["ret_pred"].min()),
        "max": float(window["ret_pred"].max()),
    }

    diff = abs(fraction_long - EXPECTED_FRACTION_LONG)
    if diff <= FRACTION_LONG_TOL:
        evaluation = (
            f"Fraction_long vs expected (~{EXPECTED_FRACTION_LONG:.2f}): OK "
            f"(within +/- {FRACTION_LONG_TOL:.2f} band)."
        )
    else:
        direction = "higher" if fraction_long > EXPECTED_FRACTION_LONG else "lower"
        threshold = EXPECTED_FRACTION_LONG + (FRACTION_LONG_TOL if direction == "higher" else -FRACTION_LONG_TOL)
        evaluation = (
            "Fraction_long vs expected (~{expected:.2f}): WARNING – current {current:.2f} is "
            "{direction} than {threshold:.2f}."
        ).format(
            expected=EXPECTED_FRACTION_LONG,
            current=fraction_long,
            direction="above" if direction == "higher" else "below",
            threshold=threshold,
        )

    return MonitoringResult(
        n_rows=n_rows,
        n_long=n_long,
        n_dir_only=n_dir_only,
        n_overlap=n_overlap,
        fraction_long=fraction_long,
        p_up_stats=p_up_stats,
        ret_pred_stats=ret_pred_stats,
        fraction_evaluation=evaluation,
    )


def print_summary(
    result: MonitoringResult,
    window_trades: int,
    p_up_threshold: float,
    ret_threshold: float,
    live_path: Path,
) -> None:
    print(
        "Live signal window: last {window} rows from {path}".format(
            window=min(window_trades, result.n_rows),
            path=live_path,
        )
    )
    print(f"n_rows: {result.n_rows}")
    print(
        f"n_long: {result.n_long} (fraction_long = {result.fraction_long:.2f})"
    )
    print(
        "p_up (threshold {thr:.2f}): mean={mean:.4f}, std={std:.4f}, min={min_v:.4f}, max={max_v:.4f}".format(
            thr=p_up_threshold,
            mean=result.p_up_stats["mean"],
            std=result.p_up_stats["std"],
            min_v=result.p_up_stats["min"],
            max_v=result.p_up_stats["max"],
        )
    )
    print(
        "ret_pred (threshold {thr:.5f}): mean={mean:.5f}, std={std:.5f}, min={min_v:.5f}, max={max_v:.5f}".format(
            thr=ret_threshold,
            mean=result.ret_pred_stats["mean"],
            std=result.ret_pred_stats["std"],
            min_v=result.ret_pred_stats["min"],
            max_v=result.ret_pred_stats["max"],
        )
    )
    print(
        f"Direction-only longs: {result.n_dir_only}, overlap with ensemble: {result.n_overlap}"
    )
    print()
    print(result.fraction_evaluation)


def main() -> None:
    args = parse_args()
    live_path = Path(args.live_path)
    df_live = ensure_live_dataframe(live_path)
    result = compute_stats(df_live, args.window_trades)
    print_summary(result, args.window_trades, args.p_up_threshold, args.ret_threshold, live_path)


if __name__ == "__main__":
    main()
