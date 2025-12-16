import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import (
    enforce_unique_hourly_index,
    make_features_and_target,
    time_series_train_val_test_split,
)


PROCESSED_PATHS = [
    Path("data/processed/macro/hourly_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/coinapi/market_hourly_features.parquet"),
    Path("data/processed/coinapi/funding_hourly_features.parquet"),
    Path("data/processed/cryptoquant/hourly_features.parquet"),
]

CORE_MODEL_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
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


def _log_cryptoquant_features(df: pd.DataFrame) -> None:
    cq_columns = sorted(col for col in df.columns if col.startswith("cq_daily_"))
    if not cq_columns:
        return
    catalog_path = Path("artifacts/monitoring/cryptoquant_feature_catalog.json")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    feature_groups = {
        "value": [c for c in cq_columns if c.startswith("cq_daily_") and "delta" not in c and "zscore" not in c and "pct_" not in c],
        "delta": [c for c in cq_columns if c.startswith("cq_daily_delta_")],
        "pct": [c for c in cq_columns if c.startswith("cq_daily_pct_")],
        "zscore": [c for c in cq_columns if c.startswith("cq_daily_zscore_")],
    }
    payload = {
        "feature_count": len(cq_columns),
        "feature_groups": {name: len(columns) for name, columns in feature_groups.items()},
        "features": cq_columns,
    }
    catalog_path.write_text(json.dumps(payload, indent=2))
    print(
        "Logged {count} CryptoQuant fallback features ({value} values, {delta} deltas, {pct} pct, {zscore} z-scores) to {path}".format(
            count=len(cq_columns),
            value=payload["feature_groups"].get("value", 0),
            delta=payload["feature_groups"].get("delta", 0),
            pct=payload["feature_groups"].get("pct", 0),
            zscore=payload["feature_groups"].get("zscore", 0),
            path=catalog_path,
        )
    )


def _merge_processed_features(df: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise RuntimeError("Expected 'ts' column in curated features for feature alignment.")

    augmented = df.copy()
    augmented["ts"] = pd.to_datetime(augmented["ts"], utc=True)
    augmented["ts"] = augmented["ts"].dt.floor("h")

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
        extra["ts"] = extra["ts"].dt.floor("h")

        columns_before = set(augmented.columns)
        augmented = augmented.merge(extra, on="ts", how="left")
        new_columns = [col for col in augmented.columns if col not in columns_before]

        if new_columns:
            preview = ", ".join(new_columns[:5])
            suffix = "..." if len(new_columns) > 5 else ""
            print(f"Added {len(new_columns)} feature columns from {path}: {preview}{suffix}")
        else:
            print(f"No new columns merged from {path}; check schema overlap.")

    augmented = augmented.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    augmented = _add_cryptoquant_flags(augmented)
    _log_cryptoquant_features(augmented)
    return augmented


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check that the curated table has data.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df, dup_count, gap_count = enforce_unique_hourly_index(
        df,
        label="curated_features",
        raise_on_gap=False,
        normalize_to_hour=True,
    )
    if dup_count == 0 and gap_count == 0:
        print("[curated_features] Hourly spacing verified; no duplicates detected.")
    elif gap_count:
        print(f"[curated_features] Logged {gap_count} non-hourly intervals; upstream gaps remain.")

    df = _merge_processed_features(df, PROCESSED_PATHS)
    df, dup_after_merge, gap_after_merge = enforce_unique_hourly_index(
        df,
        label="curated_features_merged",
        raise_on_gap=False,
        normalize_to_hour=True,
    )
    if dup_after_merge:
        print(f"[curated_features_merged] Removed {dup_after_merge} duplicates introduced during merge.")
    if gap_after_merge:
        print(
            f"[curated_features_merged] Logged {gap_after_merge} non-hourly intervals after merge; "
            "downstream consumers should handle upstream gaps."
        )

    X, y = make_features_and_target(
        df,
        target_column="ret_1h",
        allowed_features=CORE_MODEL_FEATURES,
    )

    splits = time_series_train_val_test_split(X, y)

    output_path = os.path.join(output_dir, "btc_features_1h_splits.npz")
    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        feature_names=np.array(splits.feature_names),
    )
    print(f"Saved dataset splits to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BTCUSDT 1h training dataset from BigQuery curated features.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    args = parser.parse_args()
    main(args.output_dir)
