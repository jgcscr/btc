import argparse
import os
from typing import Tuple

import numpy as np


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat features into sliding windows of length seq_len.

    X: shape [N, F]
    y: shape [N]
    Returns X_seq: [N_seq, seq_len, F], y_seq: [N_seq]
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape [N, F], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to have shape [N], got {y.shape}")

    n, _ = X.shape
    if n != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if n < seq_len:
        raise ValueError("Not enough samples to create at least one sequence")

    windows = []
    labels = []
    for i in range(seq_len - 1, n):
        start = i - seq_len + 1
        end = i + 1
        windows.append(X[start:end, :])
        labels.append(y[i])

    X_seq = np.stack(windows, axis=0)
    y_seq = np.asarray(labels)
    return X_seq, y_seq


def build_sequence_dataset(input_path: str, output_path: str, seq_len: int) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    data = np.load(input_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    threshold_arr = data.get("threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0

    X_train_seq, y_train_seq = _make_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = _make_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = _make_sequences(X_test, y_test, seq_len)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        feature_names=np.array(feature_names),
        seq_len=np.array([seq_len], dtype=int),
        threshold=np.array([threshold], dtype=float),
    )

    print(f"Saved sequence direction dataset to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sequence (temporal) direction dataset from flat direction splits.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the flat direction npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_seq_len24.npz",
        help="Path to save the sequence npz file.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="Sequence length (number of timesteps) per example.",
    )
    args = parser.parse_args()

    build_sequence_dataset(args.input_path, args.output_path, args.seq_len)


if __name__ == "__main__":
    main()
