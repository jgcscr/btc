from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.scripts.build_training_dataset_direction import build_direction_splits


def _load_ts_set(csv_path: Path, ts_col: str) -> pd.DatetimeIndex:
    df = pd.read_csv(csv_path, usecols=[ts_col])
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna().dt.floor("h")
    return pd.DatetimeIndex(ts.unique()).sort_values()


def _stack_split_arrays(data: np.lib.npyio.NpzFile, base: str) -> np.ndarray:
    keys = (f"{base}_train", f"{base}_val", f"{base}_test")
    if not all(k in data.files for k in keys):
        raise KeyError(f"Missing split keys for base '{base}': {keys}")
    return np.concatenate([data[keys[0]], data[keys[1]], data[keys[2]]], axis=0)


def _slice_dataset(dataset_path: Path, ts_keep: pd.DatetimeIndex) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    with np.load(dataset_path, allow_pickle=True) as data:
        ts_all = pd.to_datetime(data["ts_all"], utc=True, errors="coerce") if "ts_all" in data.files else None
        if ts_all is None:
            raise KeyError(f"Dataset missing ts_all: {dataset_path}")

        mask = pd.Series(ts_all).isin(ts_keep).to_numpy(dtype=bool)
        overlap_rows = int(mask.sum())

        split_bases: List[str] = []
        for key in data.files:
            if key.endswith("_train"):
                split_bases.append(key[:-6])
        split_bases = sorted(set(split_bases))

        stacked: Dict[str, np.ndarray] = {}
        for base in split_bases:
            arr = _stack_split_arrays(data, base)
            if len(arr) != len(mask):
                raise RuntimeError(f"Length mismatch for base '{base}': {len(arr)} vs mask {len(mask)}")
            stacked[base] = arr[mask]

        static: Dict[str, np.ndarray] = {}
        for key in ("feature_names", "threshold", "scaler_mean", "scaler_scale"):
            if key in data.files:
                static[key] = data[key]

    return stacked, {
        "dataset_path": str(dataset_path),
        "rows_source": int(len(mask)),
        "rows_overlap": overlap_rows,
        "rows_removed": int(len(mask) - overlap_rows),
        "coverage_ratio": float(overlap_rows / max(len(ts_keep), 1)),
        "ts_all": np.asarray(ts_all[mask], dtype="datetime64[ns]"),
        "static": static,
    }


def _resplit_indices(n_rows: int) -> Tuple[slice, slice, slice]:
    if n_rows < 12:
        # Tiny fallback; keep at least one row in each split when possible.
        n_train = max(1, n_rows - 2)
        n_val = 1 if n_rows >= 2 else 0
        n_test = n_rows - n_train - n_val
    else:
        n_train = int(round(n_rows * 0.7))
        n_val = int(round(n_rows * 0.15))
        n_test = n_rows - n_train - n_val
        if n_val <= 0:
            n_val = 1
            n_train = max(1, n_train - 1)
            n_test = n_rows - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n_train - 1)
            n_val = n_rows - n_train - n_test
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n_rows)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice direction dataset NPZ by timestamp overlap with a labeled CSV.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labeled-csv", type=Path, required=True)
    parser.add_argument("--ts-col", type=str, default="ts")
    parser.add_argument("--min-rows", type=int, default=80)
    parser.add_argument(
        "--fallback-labeling-scheme",
        choices=("binary", "binary_no_trade", "triple_barrier"),
        default=None,
        help="Optional fallback direction-labeling scheme to rebuild a temporary dataset when timestamp overlap is too sparse.",
    )
    parser.add_argument(
        "--fallback-min-coverage-ratio",
        type=float,
        default=0.0,
        help="If primary overlap coverage is below this ratio of labeled timestamps, rebuild a temporary fallback dataset.",
    )
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    args = parser.parse_args()

    ts_keep = _load_ts_set(args.labeled_csv, args.ts_col)

    stacked, selected_meta = _slice_dataset(args.dataset, ts_keep)
    fallback_used = False
    fallback_dataset = None
    if args.fallback_labeling_scheme:
        primary_coverage = float(selected_meta["coverage_ratio"])
        if primary_coverage < float(args.fallback_min_coverage_ratio):
            fallback_dir = args.output_dataset.parent / f"_overlap_fallback_{args.fallback_labeling_scheme}"
            fallback_meta_path = fallback_dir / "btc_features_1h_direction_meta.json"
            fallback_dataset = Path(
                build_direction_splits(
                    str(fallback_dir),
                    0.0,
                    labeling_scheme=str(args.fallback_labeling_scheme),
                    meta_path=str(fallback_meta_path),
                )
            )
            fallback_stacked, fallback_meta = _slice_dataset(fallback_dataset, ts_keep)
            if int(fallback_meta["rows_overlap"]) > int(selected_meta["rows_overlap"]):
                stacked = fallback_stacked
                selected_meta = fallback_meta
                fallback_used = True

    overlap_rows = int(selected_meta["rows_overlap"])
    if overlap_rows < int(args.min_rows):
        raise RuntimeError(
            f"Overlap rows {overlap_rows} below min_rows={int(args.min_rows)} for dataset {selected_meta['dataset_path']} and labeled CSV {args.labeled_csv}",
        )

    split_train, split_val, split_test = _resplit_indices(overlap_rows)
    out: Dict[str, np.ndarray] = {}
    for base, arr in stacked.items():
        out[f"{base}_train"] = arr[split_train]
        out[f"{base}_val"] = arr[split_val]
        out[f"{base}_test"] = arr[split_test]

    out["ts_all"] = selected_meta["ts_all"]
    static = selected_meta["static"] if isinstance(selected_meta.get("static"), dict) else {}
    for key, value in static.items():
        out[key] = value

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output_dataset), **out)

    payload = {
        "source_dataset": str(selected_meta["dataset_path"]),
        "primary_source_dataset": str(args.dataset),
        "labeled_csv": str(args.labeled_csv),
        "rows_source": int(selected_meta["rows_source"]),
        "rows_overlap": overlap_rows,
        "rows_removed": int(selected_meta["rows_removed"]),
        "coverage_ratio": float(selected_meta["coverage_ratio"]),
        "fallback_labeling_scheme": args.fallback_labeling_scheme,
        "fallback_used": fallback_used,
        "fallback_dataset": str(fallback_dataset) if fallback_dataset is not None else None,
        "output_dataset": str(args.output_dataset),
        "split_rows": {
            "train": int(split_train.stop - split_train.start),
            "val": int(split_val.stop - split_val.start),
            "test": int(split_test.stop - split_test.start),
        },
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
