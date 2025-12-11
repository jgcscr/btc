import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split


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
