from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Metrics:
    loss: float
    accuracy: float
    f1: float
    auc: float


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, losses: float, n: int) -> Metrics:
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float(accuracy_score(y_true, preds)) if y_true.size > 0 else 0.0
    f1 = float(f1_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    return Metrics(loss=losses / max(n, 1), accuracy=accuracy, f1=f1, auc=auc)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.detach().cpu().numpy())
    probs_arr = np.concatenate(all_probs) if all_probs else np.empty(0)
    targets_arr = np.concatenate(all_targets) if all_targets else np.empty(0)
    return compute_metrics(targets_arr, probs_arr, total_loss, total_count)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
    return running_loss / max(total, 1)


__all__ = [
    "Metrics",
    "resolve_device",
    "compute_metrics",
    "evaluate",
    "train_epoch",
]
