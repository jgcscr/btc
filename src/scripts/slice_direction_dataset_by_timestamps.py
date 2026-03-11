from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_ts_set(csv_path: Path, ts_col: str) -> pd.DatetimeIndex:
    df = pd.read_csv(csv_path, usecols=[ts_col])
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna().dt.floor("h")
    return pd.DatetimeIndex(ts.unique()).sort_values()


def _stack_split_arrays(data: np.lib.npyio.NpzFile, base: str) -> np.ndarray:
    keys = (f"{base}_train", f"{base}_val", f"{base}_test")
    if not all(k in data.files for k in keys):
        raise KeyError(f"Missing split keys for base '{base}': {keys}")
    return np.concatenate([data[keys[0]], data[keys[1]], data[keys[2]]], axis=0)


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
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    args = parser.parse_args()

    ts_keep = _load_ts_set(args.labeled_csv, args.ts_col)

    with np.load(args.dataset, allow_pickle=True) as data:
        ts_all = pd.to_datetime(data["ts_all"], utc=True, errors="coerce") if "ts_all" in data.files else None
        if ts_all is None:
            raise KeyError(f"Dataset missing ts_all: {args.dataset}")

        mask = pd.Series(ts_all).isin(ts_keep).to_numpy(dtype=bool)
        overlap_rows = int(mask.sum())
        if overlap_rows < int(args.min_rows):
            raise RuntimeError(
                f"Overlap rows {overlap_rows} below min_rows={int(args.min_rows)} for dataset {args.dataset} and labeled CSV {args.labeled_csv}",
            )

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

        split_train, split_val, split_test = _resplit_indices(overlap_rows)
        out: Dict[str, np.ndarray] = {}
        for base, arr in stacked.items():
            out[f"{base}_train"] = arr[split_train]
            out[f"{base}_val"] = arr[split_val]
            out[f"{base}_test"] = arr[split_test]

        out["ts_all"] = np.asarray(ts_all[mask], dtype="datetime64[ns]")

        # Keep static metadata arrays when present.
        for key in ("feature_names", "threshold", "scaler_mean", "scaler_scale"):
            if key in data.files:
                out[key] = data[key]

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output_dataset), **out)

    payload = {
        "source_dataset": str(args.dataset),
        "labeled_csv": str(args.labeled_csv),
        "rows_source": int(len(mask)),
        "rows_overlap": overlap_rows,
        "rows_removed": int(len(mask) - overlap_rows),
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
