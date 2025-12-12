import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

REQUIRED_COLUMNS_1H = ["ts", "p_up", "ret_pred", "signal_ensemble", "signal_dir_only"]
OPTIONAL_COLUMNS_4H = ["p_up_4h", "ret_pred_4h"]
OPTIONAL_CONFIRM_COLUMN = "signal_1h4h_confirm"
OPTIONAL_NOTES_COLUMNS = ["notes", "source"]
LEGACY_LOG_COLUMNS = [
    "ts",
    "p_up",
    "ret_pred",
    "signal_ensemble",
    "signal_dir_only",
    "created_at",
    "notes",
]
TARGET_LOG_COLUMNS = [
    "ts",
    "p_up",
    "ret_pred",
    "signal_ensemble",
    "signal_dir_only",
    "p_up_4h",
    "ret_pred_4h",
    "signal_1h4h_confirm",
    "created_at",
    "notes",
]


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
    except Exception:
        df = _load_with_fallback(path, TARGET_LOG_COLUMNS)
        if df is None:
            print(f"Failed to read CSV {path}: malformed contents.", file=sys.stderr)
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

    def _safe_float(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - unexpected format
            return None

    p_up_4h = _safe_float(row.get("p_up_4h"))
    ret_pred_4h = _safe_float(row.get("ret_pred_4h"))

    if p_up_4h is not None and ret_pred_4h is not None:
        print(
            "\nHorizon: 4h\n"
            f"p_up_4h: {p_up_4h:.4f}\n"
            f"ret_pred_4h (log): {ret_pred_4h:.6f} (~{_format_pct(ret_pred_4h)})",
        )

        confirm_raw = row.get(OPTIONAL_CONFIRM_COLUMN)
        confirm_val = None
        if confirm_raw not in (None, "") and not pd.isna(confirm_raw):
            try:
                confirm_val = int(float(confirm_raw))
            except (TypeError, ValueError):  # pragma: no cover - malformed value
                confirm_val = None

        if confirm_val is not None:
            status = "YES" if confirm_val == 1 else "NO"
            print(f"1h+4h confirmation: {confirm_val} ({status})")

    if notes_value:
        print(f"\nSource: {notes_value}")


def _load_with_fallback(path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    try:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            for raw in reader:
                if not raw:
                    continue

                if len(raw) == len(columns):
                    mapping = {columns[idx]: raw[idx] for idx in range(len(columns))}
                elif len(raw) == len(LEGACY_LOG_COLUMNS):
                    mapping = {
                        LEGACY_LOG_COLUMNS[idx]: raw[idx]
                        for idx in range(len(LEGACY_LOG_COLUMNS))
                    }
                else:
                    padded = list(raw)
                    if len(padded) < len(columns):
                        padded.extend([""] * (len(columns) - len(padded)))
                    mapping = {columns[idx]: padded[idx] for idx in range(len(columns))}

                rows.append(mapping)
    except OSError:
        return None

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    df = df[columns]
    return df


def main() -> None:
    args = _parse_args()
    describe_latest_signal(args)


if __name__ == "__main__":
    main()
