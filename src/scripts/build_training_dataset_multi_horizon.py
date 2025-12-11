import argparse
import os
from typing import Iterable, List

import numpy as np

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split
from src.data.targets_multi_horizon import add_multi_horizon_targets


DEFAULT_HORIZONS: List[int] = [1, 4]


def _split_array(values: np.ndarray, n_train: int, n_val: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        values[:n_train],
        values[n_train:n_train + n_val],
        values[n_train + n_val :],
    )


def build_multi_horizon_dataset(
    output_dir: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check curated features table content.")

    df_targets = add_multi_horizon_targets(df, horizons=horizons, price_col="close")

    ret_cols = [f"ret_{h}h" for h in horizons]
    df_targets = df_targets.dropna(subset=ret_cols)

    X, y_ret1h = make_features_and_target(df_targets, target_column="ret_1h", dropna=False)
    remove_cols = [f"ret_{h}h" for h in horizons if h != 1] + [f"dir_{h}h" for h in horizons]
    X = X.drop(columns=remove_cols, errors="ignore")

    splits = time_series_train_val_test_split(X, y_ret1h, train_frac=train_frac, val_frac=val_frac)

    n_train = splits.X_train.shape[0]
    n_val = splits.X_val.shape[0]
    n_total = len(df_targets)
    if n_train + n_val + splits.X_test.shape[0] != n_total:
        raise RuntimeError("Split sizes do not sum to dataset length; check split configuration.")

    data_ret4h = {h: df_targets[f"ret_{h}h"].to_numpy(dtype=np.float32) for h in horizons if h != 1}
    data_dir = {h: df_targets[f"dir_{h}h"].to_numpy(dtype=np.int8) for h in horizons}

    output_path = os.path.join(output_dir, "btc_features_multi_horizon_splits.npz")

    save_kwargs = {
        "X_train": splits.X_train,
        "y_train": splits.y_train,
        "X_val": splits.X_val,
        "y_val": splits.y_val,
        "X_test": splits.X_test,
        "y_test": splits.y_test,
        "feature_names": np.array(splits.feature_names),
        "horizons": np.array(sorted({int(h) for h in horizons}), dtype=np.int32),
        "direction_threshold": np.array([0.0], dtype=np.float32),
    }

    for horizon, values in data_ret4h.items():
        train, val, test = _split_array(values, n_train, n_val)
        save_kwargs[f"y_ret{horizon}h_train"] = train
        save_kwargs[f"y_ret{horizon}h_val"] = val
        save_kwargs[f"y_ret{horizon}h_test"] = test

    for horizon, values in data_dir.items():
        train, val, test = _split_array(values, n_train, n_val)
        save_kwargs[f"y_dir{horizon}h_train"] = train
        save_kwargs[f"y_dir{horizon}h_val"] = val
        save_kwargs[f"y_dir{horizon}h_test"] = test

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved multi-horizon dataset splits to {output_path}")
    print("Stored horizons:", save_kwargs["horizons"].tolist())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-horizon BTC dataset (1h & 4h targets) from the curated BigQuery features. "
            "This keeps the legacy 1h dataset untouched and writes a separate NPZ with additional targets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Horizons (in hours) to include when computing targets (default: 1 4).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of samples allocated to the training split (default: 0.7).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of samples allocated to the validation split (default: 0.15).",
    )
    args = parser.parse_args()

    build_multi_horizon_dataset(
        output_dir=args.output_dir,
        horizons=args.horizons,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
