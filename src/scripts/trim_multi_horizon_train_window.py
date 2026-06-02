from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim the train split of a multi-horizon NPZ to its most recent rows while keeping val/test unchanged."
    )
    parser.add_argument(
        "--input-path",
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--last-train-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".json")

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if int(args.last_train_rows) <= 0:
        raise ValueError("--last-train-rows must be positive.")

    with np.load(input_path, allow_pickle=True) as data:
        if "X_train" not in data.files:
            raise KeyError("Dataset missing X_train")
        original_train_rows = int(data["X_train"].shape[0])
        keep_train_rows = min(int(args.last_train_rows), original_train_rows)
        train_slice = slice(original_train_rows - keep_train_rows, original_train_rows)

        payload: Dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if key.endswith("_train"):
                payload[key] = value[train_slice]
            else:
                payload[key] = value

        if {"ts_train", "ts_val", "ts_test"}.issubset(data.files):
            payload["ts_train"] = data["ts_train"][train_slice]
            payload["ts_all"] = np.concatenate([payload["ts_train"], payload["ts_val"], payload["ts_test"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_train_rows": original_train_rows,
        "kept_train_rows": keep_train_rows,
        "dropped_train_rows": original_train_rows - keep_train_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote trimmed dataset: {output_path}")
    print(f"Kept last {keep_train_rows} of {original_train_rows} train rows.")
    print(f"Wrote trim report: {report_path}")


if __name__ == "__main__":
    main()