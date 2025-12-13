from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from joblib import dump as joblib_dump
from numpy.random import Generator as NpGenerator
from torch.utils.data import DataLoader

from src.training.lstm_data import (
    SequenceDataset,
    SequenceSplits,
    build_sequence_splits,
    estimate_feature_stats,
)


@dataclass
class TransformerData:
    splits: SequenceSplits
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


def _scale_sequences(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    original_shape = arr.shape
    reshaped = arr.reshape(-1, arr.shape[-1])
    scaled = (reshaped - mean) / std
    return scaled.reshape(original_shape)


def prepare_transformer_data(
    dataset_path: str,
    seq_len: int,
    batch_size: int,
    generator: Optional[NpGenerator] = None,
) -> tuple[TransformerData, DataLoader, DataLoader, DataLoader]:
    splits = build_sequence_splits(dataset_path, seq_len)

    train_mean, train_std = estimate_feature_stats(splits.X_train_seq.reshape(-1, splits.X_train_seq.shape[-1]))

    X_train_scaled = _scale_sequences(splits.X_train_seq, train_mean, train_std)
    X_val_scaled = _scale_sequences(splits.X_val_seq, train_mean, train_std)
    X_test_scaled = _scale_sequences(splits.X_test_seq, train_mean, train_std)

    train_dataset = SequenceDataset(X_train_scaled, splits.y_train_seq)
    val_dataset = SequenceDataset(X_val_scaled, splits.y_val_seq)
    test_dataset = SequenceDataset(X_test_scaled, splits.y_test_seq)

    torch_generator = None
    if generator is not None:
        torch_generator = torch.Generator()
        torch_generator.manual_seed(int(generator.integers(0, 2**31 - 1)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=torch_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    data = TransformerData(splits=splits, scaler_mean=train_mean, scaler_std=train_std)
    return data, train_loader, val_loader, test_loader


def save_scaler(mean: np.ndarray, std: np.ndarray, path: str) -> None:
    payload = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    joblib_dump(payload, path)


__all__ = [
    "TransformerData",
    "prepare_transformer_data",
    "save_scaler",
]
