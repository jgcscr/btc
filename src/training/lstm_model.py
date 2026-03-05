from __future__ import annotations

import torch
import torch.nn as nn


def _validate_recurrent_args(hidden_size: int, num_layers: int, norm_type: str) -> None:
    if num_layers <= 0:
        raise ValueError("num_layers must be >= 1")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0")
    valid_norms = {"none", "layer", "batch"}
    if norm_type not in valid_norms:
        raise ValueError(f"norm_type must be one of {sorted(valid_norms)}")


def _build_norm(norm_type: str, hidden_dim: int) -> nn.Module | None:
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    return None


class LSTMDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        _validate_recurrent_args(hidden_size, num_layers, norm_type)
        dropout_prob = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type
        self.hidden_dim = hidden_size
        self.norm = _build_norm(norm_type, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm is not None:
            last_hidden = self.norm(last_hidden)
        logits = self.fc(last_hidden)
        return logits.squeeze(1)


class BiLSTMDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        _validate_recurrent_args(hidden_size, num_layers, norm_type)
        dropout_prob = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type
        self.hidden_dim = hidden_size * 2
        self.norm = _build_norm(norm_type, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm is not None:
            last_hidden = self.norm(last_hidden)
        logits = self.fc(last_hidden)
        return logits.squeeze(1)


class GRUDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        _validate_recurrent_args(hidden_size, num_layers, norm_type)
        dropout_prob = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type
        self.hidden_dim = hidden_size
        self.norm = _build_norm(norm_type, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm is not None:
            last_hidden = self.norm(last_hidden)
        logits = self.fc(last_hidden)
        return logits.squeeze(1)


__all__ = [
    "BiLSTMDirectionClassifier",
    "GRUDirectionClassifier",
    "LSTMDirectionClassifier",
]
