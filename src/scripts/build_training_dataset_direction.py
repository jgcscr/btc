import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split


PROCESSED_PATHS = [
    Path("data/processed/macro/hourly_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/coinapi/market_hourly_features.parquet"),
    Path("data/processed/coinapi/funding_hourly_features.parquet"),
    Path("data/processed/cryptoquant/hourly_features.parquet"),
]


def _add_cryptoquant_flags(df: pd.DataFrame) -> pd.DataFrame:
    cq_columns = [col for col in df.columns if col.startswith("cq_daily_")]
    if not cq_columns:
        df["cq_daily_fallback_active"] = 0
        df["cq_daily_fallback_complete"] = 0
        return df

    coverage = df[cq_columns].notna()
    df["cq_daily_fallback_active"] = coverage.any(axis=1).astype(int)
    df["cq_daily_fallback_complete"] = coverage.all(axis=1).astype(int)
    return df


def _merge_processed_features(df: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise RuntimeError("Expected 'ts' column in curated features for feature alignment.")

    augmented = df.copy()
    augmented["ts"] = pd.to_datetime(augmented["ts"], utc=True)
    augmented = augmented.sort_values("ts").reset_index(drop=True)

    for path in paths:
        if not path.exists():
            print(f"Processed features not found at {path}; skipping.")
            continue

        extra = pd.read_parquet(path)
        if extra.empty:
            print(f"Processed features at {path} are empty; skipping.")
            continue

        if "timestamp" in extra.columns:
            extra = extra.rename(columns={"timestamp": "ts"})

        extra["ts"] = pd.to_datetime(extra["ts"], utc=True)
        extra = extra.sort_values("ts").drop_duplicates(subset="ts", keep="last")

        columns_before = set(augmented.columns)
        merged = pd.merge_asof(
            augmented.sort_values("ts"),
            extra,
            on="ts",
            direction="backward",
            allow_exact_matches=True,
        )
        augmented = merged.sort_values("ts").reset_index(drop=True)
        new_columns = [col for col in augmented.columns if col not in columns_before]

        if new_columns:
            preview = ", ".join(new_columns[:5])
            suffix = "..." if len(new_columns) > 5 else ""
            print(f"Added {len(new_columns)} feature columns from {path}: {preview}{suffix}")
        else:
            print(f"No new columns merged from {path}; check schema overlap.")

    return _add_cryptoquant_flags(augmented)


def make_direction_labels(y_ret: pd.Series, threshold: float) -> pd.Series:
    """Convert continuous 1h returns into binary direction labels.

    Label is 1 if return > threshold (price up), else 0 (flat/down).
    """
    return (y_ret > threshold).astype(int)


def build_direction_splits(output_dir: str, threshold: float) -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )

    if df.empty:
        raise RuntimeError(
            "Loaded empty DataFrame from BigQuery; check that the curated table has data."
        )

    df = _merge_processed_features(df, PROCESSED_PATHS)

    X, y_ret = make_features_and_target(df, target_column="ret_1h")

    y_dir = make_direction_labels(y_ret, threshold=threshold)

    splits = time_series_train_val_test_split(X, y_dir)

    output_path = os.path.join(output_dir, "btc_features_1h_direction_splits.npz")
    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        feature_names=np.array(splits.feature_names),
        threshold=np.array([threshold], dtype=float),
    )
    print(f"Saved direction dataset splits to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BTCUSDT 1h direction training dataset from BigQuery curated features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared direction dataset splits.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Direction threshold theta: label is 1 if ret_1h > theta, else 0.",
    )
    args = parser.parse_args()

    build_direction_splits(args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
