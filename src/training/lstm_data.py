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


def load_direction_npz(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Direction dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    required = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "feature_names"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Direction dataset missing keys: {sorted(missing)}")

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()
    threshold_arr = data.get("threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, threshold


def build_sequence_splits(dataset_path: str, seq_len: int) -> SequenceSplits:
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, threshold = load_direction_npz(dataset_path)

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
