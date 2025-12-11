import numpy as np
import pandas as pd

from src.data.dataset_preparation import (
    DatasetSplits,
    make_features_and_target,
    time_series_train_val_test_split,
)


def test_make_features_and_target_excludes_ts_target_and_ret_fwd_3h() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="H"),
            "ret_1h": np.linspace(-0.01, 0.01, 5),
            "ret_fwd_3h": np.linspace(0.0, 0.02, 5),
            "feat_a": np.arange(5),
            "feat_b": np.arange(5, 10),
        },
    )

    X, y = make_features_and_target(df, target_column="ret_1h")

    assert "ts" not in X.columns
    assert "ret_1h" not in X.columns
    assert "ret_fwd_3h" not in X.columns

    # Features should include the other numeric columns
    assert set(X.columns) == {"feat_a", "feat_b"}
    assert y.name == "ret_1h"


def test_time_series_train_val_test_split_order_and_sizes() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="H"),
            "ret_1h": np.arange(n, dtype=float),
            "feat": np.arange(n, dtype=float) * 2.0,
        },
    )

    X, y = make_features_and_target(df, target_column="ret_1h")

    splits: DatasetSplits = time_series_train_val_test_split(X, y, train_frac=0.6, val_frac=0.2)

    # Sizes
    assert splits.X_train.shape[0] == 6
    assert splits.X_val.shape[0] == 2
    assert splits.X_test.shape[0] == 2

    # Order: first element of train corresponds to first element of original y
    assert splits.y_train[0] == y.iloc[0]
    # Last element of test corresponds to last element of original y
    assert splits.y_test[-1] == y.iloc[-1]
