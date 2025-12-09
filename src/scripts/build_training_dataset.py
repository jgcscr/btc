import argparse
import os

import numpy as np

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check that the curated table has data.")

    X, y = make_features_and_target(df, target_column="ret_1h")

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
