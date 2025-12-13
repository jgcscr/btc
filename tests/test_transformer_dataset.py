import numpy as np
from numpy.random import default_rng

from src.training.lstm_data import estimate_feature_stats
from src.training.transformer_dataset import prepare_transformer_data


def _collect(loader):
    xs = []
    ys = []
    for batch_X, batch_y in loader:
        xs.append(batch_X.numpy())
        ys.append(batch_y.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def test_prepare_transformer_data_scales_splits(tmp_path):
    dataset_path = tmp_path / "direction.npz"

    X_train = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    X_val = np.array(
        [
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ],
        dtype=np.float32,
    )
    y_val = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    X_test = np.array(
        [
            [9.0, 10.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 13.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    np.savez(
        dataset_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=np.array(["feat_a", "feat_b"], dtype=object),
        threshold=np.array([0.0], dtype=float),
    )

    seq_len = 3
    batch_size = 4
    rng = default_rng(42)

    data, train_loader, val_loader, test_loader = prepare_transformer_data(
        str(dataset_path),
        seq_len=seq_len,
        batch_size=batch_size,
        generator=rng,
    )

    assert data.splits.seq_len == seq_len

    splits = data.splits
    expected_mean, expected_std = estimate_feature_stats(
        splits.X_train_seq.reshape(-1, splits.X_train_seq.shape[-1]),
    )

    assert np.allclose(data.scaler_mean, expected_mean)
    assert np.allclose(data.scaler_std, expected_std)

    train_X, train_y = _collect(train_loader)
    flattened = train_X.reshape(-1, train_X.shape[-1])
    assert np.allclose(flattened.mean(axis=0), np.zeros_like(expected_mean))
    assert np.allclose(flattened.std(axis=0), np.ones_like(expected_std))
    assert np.array_equal(np.sort(train_y), np.sort(splits.y_train_seq.astype(np.float32)))

    val_X, val_y = _collect(val_loader)
    expected_val = (splits.X_val_seq - data.scaler_mean) / data.scaler_std
    assert np.allclose(val_X, expected_val.astype(np.float32))
    assert np.allclose(val_y, splits.y_val_seq.astype(np.float32))

    test_X, test_y = _collect(test_loader)
    expected_test = (splits.X_test_seq - data.scaler_mean) / data.scaler_std
    assert np.allclose(test_X, expected_test.astype(np.float32))
    assert np.allclose(test_y, splits.y_test_seq.astype(np.float32))
