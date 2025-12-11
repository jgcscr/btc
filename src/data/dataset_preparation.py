from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def make_features_and_target(
    df: pd.DataFrame,
    target_column: str = "ret_1h",
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix X and target y from the curated features table.

    - Sorts by ts.
    - Drops rows with NA in the target (optional).
    - Uses all remaining numeric columns except ts and the target as features.
    """
    if "ts" not in df.columns:
        raise ValueError("Expected a 'ts' column in the dataframe.")

    df = df.sort_values("ts").reset_index(drop=True)

    if dropna:
        df = df.dropna(subset=[target_column])

    y = df[target_column].copy()

    non_feature_cols = {"ts", target_column, "ret_fwd_3h"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].copy()

    return X, y


def time_series_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> DatasetSplits:
    """Split X, y into time-ordered train/val/test sets and scale features.

    The split respects time ordering: earliest rows go to train, then val, then test.
    Standard scaling is fit on train only and applied to val/test to avoid leakage.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset; cannot create splits.")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]

    X_test = X.iloc[n_train + n_val:]
    y_test = y.iloc[n_train + n_val:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DatasetSplits(
        X_train=X_train_scaled,
        y_train=y_train.to_numpy(),
        X_val=X_val_scaled,
        y_val=y_val.to_numpy(),
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        feature_names=list(X.columns),
    )
