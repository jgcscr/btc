from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.trading.signals import PreparedData, prepare_data_for_signals


def _load_dataset_payload(dataset_path: str) -> Dict[str, Any]:
    data = np.load(dataset_path, allow_pickle=True)
    required = {
        "feature_names",
        "scaler_mean",
        "scaler_scale",
        "ts_all",
        "X_train",
        "X_val",
        "X_test",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Dataset is missing required parity keys: {missing}")

    feature_names = [str(name) for name in data["feature_names"].tolist()]
    scaler_mean = np.asarray(data["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(data["scaler_scale"], dtype=float)
    if scaler_mean.shape[0] != len(feature_names) or scaler_scale.shape[0] != len(feature_names):
        raise ValueError("Scaler statistics do not match feature_names length.")

    X_all = np.concatenate([
        np.asarray(data["X_train"], dtype=float),
        np.asarray(data["X_val"], dtype=float),
        np.asarray(data["X_test"], dtype=float),
    ])
    ts_all = pd.to_datetime(data["ts_all"], utc=True)
    if len(ts_all) != len(X_all):
        raise ValueError("ts_all length does not match concatenated feature rows.")

    return {
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "X_all": X_all,
        "ts_all": ts_all,
    }


def _resolve_row_index(ts_all: Sequence[pd.Timestamp], *, timestamp: str | None, split: str, split_index: int) -> int:
    if timestamp:
        target = pd.to_datetime(timestamp, utc=True)
        matches = [idx for idx, value in enumerate(ts_all) if value == target]
        if not matches:
            raise ValueError(f"Timestamp {timestamp} not found in dataset.")
        return matches[-1]

    size = len(ts_all)
    if size == 0:
        raise ValueError("Dataset contains no rows.")
    if split_index < 0:
        return size + split_index
    return split_index


def audit_feature_parity(
    *,
    dataset_path: str,
    features_path: str,
    target_column: str,
    timestamp: str | None = None,
    split: str = "all",
    split_index: int = -1,
    tolerance: float = 1e-6,
    prepared: PreparedData | None = None,
) -> Dict[str, Any]:
    payload = _load_dataset_payload(dataset_path)
    feature_names = payload["feature_names"]
    scaler_mean = payload["scaler_mean"]
    scaler_scale = payload["scaler_scale"]
    X_all = payload["X_all"]
    ts_all = payload["ts_all"]
    row_index = _resolve_row_index(ts_all, timestamp=timestamp, split=split, split_index=split_index)

    if row_index < 0 or row_index >= len(ts_all):
        raise IndexError(f"Resolved row index {row_index} is outside dataset bounds.")

    prepared_bundle = prepared or prepare_data_for_signals(
        dataset_path,
        target_column=target_column,
        features_path=features_path,
    )
    live_ts = pd.to_datetime(prepared_bundle.df_all["ts"], utc=True)
    matching = np.where(live_ts.to_numpy(dtype="datetime64[ns]") == ts_all[row_index].to_datetime64())[0]
    if matching.size == 0:
        raise ValueError(f"Timestamp {ts_all[row_index].isoformat()} not found in live-prepared features.")
    live_index = int(matching[-1])

    missing_live_features = [name for name in feature_names if name not in prepared_bundle.X_all_ordered.columns]
    if missing_live_features:
        raise ValueError(
            "Live-prepared feature matrix is missing saved training features: "
            + ", ".join(missing_live_features)
        )
    extra_live_features = [name for name in prepared_bundle.X_all_ordered.columns if name not in feature_names]

    training_scaled = np.asarray(X_all[row_index], dtype=float)
    training_raw = training_scaled * scaler_scale + scaler_mean
    live_frame = prepared_bundle.X_all_ordered.loc[:, feature_names]
    live_raw = live_frame.iloc[live_index].to_numpy(dtype=float, copy=False)
    safe_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    live_scaled = (live_raw - scaler_mean) / safe_scale

    raw_diff = np.abs(training_raw - live_raw)
    scaled_diff = np.abs(training_scaled - live_scaled)
    mismatched = np.where(raw_diff > tolerance)[0]
    top_indices = np.argsort(raw_diff)[::-1][:10]
    top_diffs = [
        {
            "feature": feature_names[idx],
            "training_raw": float(training_raw[idx]),
            "live_raw": float(live_raw[idx]),
            "raw_abs_diff": float(raw_diff[idx]),
            "training_scaled": float(training_scaled[idx]),
            "live_scaled": float(live_scaled[idx]),
            "scaled_abs_diff": float(scaled_diff[idx]),
        }
        for idx in top_indices
    ]

    return {
        "dataset_path": dataset_path,
        "features_path": features_path,
        "target_column": target_column,
        "timestamp": ts_all[row_index].isoformat(),
        "row_index": int(row_index),
        "live_index": int(live_index),
        "feature_count": len(feature_names),
        "extra_live_feature_count": len(extra_live_features),
        "extra_live_features": extra_live_features,
        "tolerance": float(tolerance),
        "max_raw_abs_diff": float(raw_diff.max()) if raw_diff.size else 0.0,
        "max_scaled_abs_diff": float(scaled_diff.max()) if scaled_diff.size else 0.0,
        "mismatched_feature_count": int(mismatched.size),
        "ok": bool(mismatched.size == 0),
        "top_differences": top_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/serve feature parity for saved datasets.")
    parser.add_argument("--dataset-path", required=True, help="Path to NPZ dataset with scaler and timestamp metadata.")
    parser.add_argument("--features-path", required=True, help="CSV or parquet feature source used for live-style preparation.")
    parser.add_argument("--target-column", default="ret_1h", help="Target column used to prepare the feature bundle.")
    parser.add_argument("--timestamp", default=None, help="Optional exact UTC timestamp to audit.")
    parser.add_argument("--split", default="all", choices=("all",), help="Reserved for future split-specific selection.")
    parser.add_argument("--split-index", type=int, default=-1, help="Row index to audit when --timestamp is not provided.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Maximum allowed absolute raw-feature difference.")
    parser.add_argument("--output-path", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    report = audit_feature_parity(
        dataset_path=args.dataset_path,
        features_path=args.features_path,
        target_column=args.target_column,
        timestamp=args.timestamp,
        split=args.split,
        split_index=args.split_index,
        tolerance=args.tolerance,
    )
    payload = json.dumps(report, indent=2)
    if args.output_path:
        Path(args.output_path).write_text(payload, encoding="utf-8")
    print(payload)
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()