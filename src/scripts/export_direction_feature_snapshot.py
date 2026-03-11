from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scripts.build_training_dataset_direction import prepare_direction_feature_frame


def _load_dataset_timestamps(dataset_path: Path) -> tuple[pd.DatetimeIndex, list[str]]:
    with np.load(dataset_path, allow_pickle=True) as data:
        ts = pd.to_datetime(data["ts_all"], utc=True, errors="coerce")
        feature_names = [str(name) for name in data["feature_names"].tolist()]
    return pd.DatetimeIndex(ts).sort_values(), feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the raw pre-normalization feature frame for the exact timestamps present in a direction dataset NPZ.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Direction dataset NPZ path.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--meta-output", type=Path, required=True, help="Output JSON metadata path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    dataset_ts, feature_names = _load_dataset_timestamps(args.dataset)
    raw_frame, _, _ = prepare_direction_feature_frame(threshold=0.0, labeling_scheme="triple_barrier")
    raw_frame = raw_frame.copy()
    raw_frame["ts"] = pd.to_datetime(raw_frame["ts"], utc=True, errors="coerce")
    raw_frame = raw_frame.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset="ts", keep="last")
    filtered = raw_frame[raw_frame["ts"].isin(dataset_ts)].copy()
    filtered = filtered.sort_values("ts").reset_index(drop=True)

    ordered_cols = ["ts"] + [feature for feature in feature_names if feature in filtered.columns]
    if "ret_1h" in filtered.columns:
        ordered_cols.append("ret_1h")
    filtered = filtered[ordered_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export = filtered.copy()
    export["ts"] = export["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    export.to_csv(args.output, index=False)

    meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": str(args.dataset),
        "rows_requested": int(len(dataset_ts)),
        "rows_written": int(len(filtered)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "output": str(args.output),
        "ts_min": export["ts"].iloc[0] if not export.empty else None,
        "ts_max": export["ts"].iloc[-1] if not export.empty else None,
        "missing_timestamps": [
            ts.isoformat().replace("+00:00", "Z")
            for ts in dataset_ts.difference(pd.DatetimeIndex(filtered["ts"]))
        ],
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()