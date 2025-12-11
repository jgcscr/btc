import argparse
import math
import os
import sys

import pandas as pd

REQUIRED_COLUMNS_1H = ["ts", "p_up", "ret_pred", "signal_ensemble", "signal_dir_only"]
OPTIONAL_COLUMNS_4H = ["p_up_4h", "ret_pred_4h"]
OPTIONAL_NOTES_COLUMNS = ["notes", "source"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the most recent logged realtime signal (1h and optional 4h).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="artifacts/live/paper_trade_realtime.csv",
        help="Path to the realtime signal log CSV.",
    )
    return parser.parse_args()


def _load_last_row(path: str) -> pd.Series:
    if not os.path.exists(path):
        print(f"Log file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - malformed csv
        print(f"Failed to read CSV {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print(f"Log file {path} is empty; nothing to describe.", file=sys.stderr)
        sys.exit(1)

    missing = [col for col in REQUIRED_COLUMNS_1H if col not in df.columns]
    if missing:
        print(f"Log file is missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    return df.iloc[-1]


def _format_pct(ret_value: float) -> str:
    try:
        approx_pct = (math.exp(ret_value) - 1.0) * 100.0
    except OverflowError:  # extremely large values
        approx_pct = ret_value * 100.0
    return f"{approx_pct:.2f}%"


def describe_latest_signal(args: argparse.Namespace) -> None:
    row = _load_last_row(args.log_path)

    ts = row["ts"]
    p_up = float(row["p_up"])
    ret_pred = float(row["ret_pred"])
    signal_ens = int(row["signal_ensemble"])
    signal_dir_only = int(row["signal_dir_only"])

    notes_value = None
    for col in OPTIONAL_NOTES_COLUMNS:
        if col in row and not (pd.isna(row[col])):
            notes_value = str(row[col])
            break

    print(f"Latest signal (from {args.log_path}):\n")
    print(f"Time: {ts}")
    direction = "LONG" if signal_ens == 1 else "FLAT"
    print(
        "Horizon: 1h\n"
        f"Direction: {direction} (ensemble={signal_ens}, dir-only={signal_dir_only})\n"
        f"p_up_1h: {p_up:.4f}\n"
        f"ret_pred_1h (log): {ret_pred:.6f} (~{_format_pct(ret_pred)})",
    )

    has_4h = all(col in row for col in OPTIONAL_COLUMNS_4H)
    if has_4h:
        p_up_4h = float(row["p_up_4h"])
        ret_pred_4h = float(row["ret_pred_4h"])
        print(
            "\nHorizon: 4h\n"
            f"p_up_4h: {p_up_4h:.4f}\n"
            f"ret_pred_4h (log): {ret_pred_4h:.6f} (~{_format_pct(ret_pred_4h)})",
        )

    if notes_value:
        print(f"\nSource: {notes_value}")


def main() -> None:
    args = _parse_args()
    describe_latest_signal(args)


if __name__ == "__main__":
    main()
