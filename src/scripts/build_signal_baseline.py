"""CLI to compute monitoring baselines from live signal logs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.monitoring.metrics import ensure_meta_metrics, summary_stats

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
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=None,
        help="Optional path to write a tabular baseline snapshot as Parquet.",
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

    return ensure_meta_metrics(df)


def _append_detected_meta_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    column_list = list(dict.fromkeys(columns))

    meta_candidates = [
        "p_up_meta",
        "signal_meta",
        "meta_hit_rate",
    ]
    meta_candidates.extend(sorted(col for col in df.columns if col.startswith("meta_net_fee_")))

    for column in meta_candidates:
        if column in df.columns and column not in column_list:
            column_list.append(column)

    return column_list


def compute_baseline(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, object]:
    baseline: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(df.shape[0]),
        "columns": {},
        "column_order": list(columns),
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


def baseline_to_dataframe(baseline: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, stats in baseline.get("columns", {}).items():
        if not isinstance(stats, dict) or "error" in stats:
            continue

        row = {"metric": metric}
        for key, value in stats.items():
            if key == "quantiles" and isinstance(value, dict):
                row[key] = json.dumps(value, sort_keys=True)
            else:
                row[key] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["metric"])

    return pd.DataFrame(rows)


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

    columns = _append_detected_meta_columns(df, args.columns)
    baseline = compute_baseline(df, columns)
    output = json.dumps(baseline, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, flush=True)

    if args.parquet_output:
        args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
        baseline_df = baseline_to_dataframe(baseline)
        baseline_df.to_parquet(args.parquet_output, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
