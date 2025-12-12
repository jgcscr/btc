from __future__ import annotations

import torch
import torch.nn as nn


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
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        valid_norms = {"none", "layer", "batch"}
        if norm_type not in valid_norms:
            raise ValueError(f"norm_type must be one of {sorted(valid_norms)}")
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
        if norm_type == "layer":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(hidden_size)
        else:
            self.norm = None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm_type == "layer" and self.norm is not None:
            last_hidden = self.norm(last_hidden)
        elif self.norm_type == "batch" and self.norm is not None:
            last_hidden = self.norm(last_hidden)
        logits = self.fc(last_hidden)
        return logits.squeeze(1)


__all__ = ["LSTMDirectionClassifier"]
