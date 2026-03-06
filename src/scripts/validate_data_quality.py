from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.trading.data_quality import DataQualityError, DataQualityPolicy, evaluate_ohlcv_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Binance OHLCV data freshness and continuity.")
    parser.add_argument("--input", type=Path, required=True, help="Parquet/CSV file with ts and volume columns.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/data_quality.json"))
    parser.add_argument("--max-staleness-hours", type=float, default=2.0)
    parser.add_argument("--max-missing-ratio", type=float, default=0.01)
    parser.add_argument("--max-zero-volume-ratio", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=120)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.input.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(args.input)
    else:
        frame = pd.read_parquet(args.input)

    policy = DataQualityPolicy(
        max_staleness_hours=args.max_staleness_hours,
        max_missing_ratio=args.max_missing_ratio,
        max_zero_volume_ratio=args.max_zero_volume_ratio,
        min_rows=args.min_rows,
    )

    payload = {
        "ok": True,
        "input": str(args.input),
    }
    try:
        payload.update(evaluate_ohlcv_quality(frame, policy))
    except DataQualityError as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload.update({
            "row_count": int(len(frame)),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if not payload["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
