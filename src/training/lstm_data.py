from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat features into sliding windows of length ``seq_len``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``[N, F]``.
    y : np.ndarray
        Label vector of shape ``[N]``.
    seq_len : int
        Number of timesteps per sequence.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape [N, F], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to have shape [N], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    n = X.shape[0]
    if n < seq_len:
        raise ValueError("Not enough samples to create at least one sequence")

    windows = []
    labels = []
    for end in range(seq_len, n + 1):
        start = end - seq_len
        windows.append(X[start:end])
        labels.append(y[end - 1])

    X_seq = np.stack(windows, axis=0)
    y_seq = np.asarray(labels)
    return X_seq, y_seq


@dataclass(frozen=True)
class SequenceSplits:
    X_train_seq: np.ndarray
    y_train_seq: np.ndarray
    X_val_seq: np.ndarray
    y_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_test_seq: np.ndarray
    feature_names: list[str]
    threshold: float
    seq_len: int


class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray) -> None:
        if X_seq.ndim != 3:
            raise ValueError("Expected X_seq to have shape [N, T, F]")
        if y_seq.ndim != 1:
            raise ValueError("Expected y_seq to have shape [N]")
        if X_seq.shape[0] != y_seq.shape[0]:
            raise ValueError("Mismatched sequence and label counts")
        self.X = torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_seq, dtype=np.float32))

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


def load_direction_npz(
    dataset_path: str,
    *,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Direction dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    required = {"X_train", "X_val", "X_test", "feature_names"}
    if horizon == 1 and {"y_train", "y_val", "y_test"}.issubset(data.files):
        y_train_key = "y_train"
        y_val_key = "y_val"
        y_test_key = "y_test"
    else:
        y_train_key = f"y_dir{horizon}h_train"
        y_val_key = f"y_dir{horizon}h_val"
        y_test_key = f"y_dir{horizon}h_test"

    required |= {y_train_key, y_val_key, y_test_key}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Direction dataset missing keys for {horizon}h: {sorted(missing)}")

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data[y_train_key]
    y_val = data[y_val_key]
    y_test = data[y_test_key]
    feature_names = data["feature_names"].tolist()
    threshold_arr = data.get("threshold")
    if threshold_arr is None:
        threshold_arr = data.get("direction_threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, threshold


def load_direction_npz_full(
    dataset_path: str,
    *,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, list[str], float]:
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, threshold = load_direction_npz(
        dataset_path,
        horizon=horizon,
    )
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    return X_all, y_all, feature_names, threshold


def build_sequence_splits(dataset_path: str, seq_len: int, *, horizon: int = 1) -> SequenceSplits:
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, threshold = load_direction_npz(
        dataset_path,
        horizon=horizon,
    )

    return build_sequence_splits_from_arrays(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,
        threshold,
        seq_len,
    )


def build_sequence_splits_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    threshold: float,
    seq_len: int,
) -> SequenceSplits:

    # Replace NaNs using training column means to avoid degenerate losses during training.
    mask = ~np.isnan(X_train)
    sums = np.nan_to_num(X_train, nan=0.0).sum(axis=0)
    counts = mask.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        train_means = sums / counts
    train_means = np.where(counts == 0, 0.0, train_means)

    def _fill_nan(arr: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(arr), train_means, arr)
        return filled.astype(np.float32, copy=False)

    X_train = _fill_nan(X_train)
    X_val = _fill_nan(X_val)
    X_test = _fill_nan(X_test)

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq_len)

    return SequenceSplits(
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        feature_names=feature_names,
        threshold=threshold,
        seq_len=seq_len,
    )


def create_dataloader(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    dataset = SequenceDataset(X_seq, y_seq)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def save_sequence_dataset(input_path: str, output_path: str, seq_len: int) -> None:
    splits = build_sequence_splits(input_path, seq_len)
    volatility_arrays = _load_volatility_arrays(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train_seq=splits.X_train_seq,
        y_train_seq=splits.y_train_seq,
        X_val_seq=splits.X_val_seq,
        y_val_seq=splits.y_val_seq,
        X_test_seq=splits.X_test_seq,
        y_test_seq=splits.y_test_seq,
        feature_names=np.array(splits.feature_names),
        seq_len=np.array([splits.seq_len], dtype=int),
        threshold=np.array([splits.threshold], dtype=float),
        **volatility_arrays,
    )
    print(f"Saved sequence direction dataset to {output_path}")


def estimate_feature_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std <= 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


__all__ = [
    "SequenceDataset",
    "SequenceSplits",
    "build_sequence_splits",
    "create_dataloader",
    "estimate_feature_stats",
    "load_direction_npz",
    "make_sequences",
    "save_sequence_dataset",
]


def _load_volatility_arrays(dataset_path: str) -> dict[str, np.ndarray]:
    volatility_arrays: dict[str, np.ndarray] = {}
    with np.load(dataset_path, allow_pickle=True) as data:
        for key in data.files:
            if key.startswith("volatility_"):
                volatility_arrays[key] = np.asarray(data[key]).copy()
    return volatility_arrays
