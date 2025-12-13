"""CLI to compute monitoring baselines from live signal logs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.monitoring.metrics import summary_stats

DEFAULT_COLUMNS = ("p_up", "signal_ensemble", "signal_dir_only", "ret_pred")
DEFAULT_INPUT = Path("artifacts/live/paper_trade_realtime.csv")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the live signal CSV log.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=list(DEFAULT_COLUMNS),
        help="Columns to summarize. Defaults to common probability and signal columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If provided, only use the most recent N rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the resulting JSON baseline (defaults to stdout).",
    )
    return parser.parse_args(argv)


def load_dataframe(path: Path, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV path does not exist: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    if limit and limit > 0:
        df = df.tail(limit)

    return df


def compute_baseline(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, object]:
    baseline: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(df.shape[0]),
        "columns": {},
    }

    for column in columns:
        if column not in df.columns:
            baseline["columns"][column] = {"error": "missing_column"}
            continue

        stats = summary_stats(df[column])
        if stats:
            baseline["columns"][column] = stats
        else:
            baseline["columns"][column] = {"error": "no_valid_values"}

    return baseline


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df = load_dataframe(args.csv, args.limit)
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return 1

    if df.empty:
        print(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": 0,
            "columns": {},
        }, indent=2), flush=True)
        return 0

    baseline = compute_baseline(df, args.columns)
    output = json.dumps(baseline, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
