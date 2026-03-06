from __future__ import annotations

import torch
import torch.nn as nn

from .lstm_model import _build_norm, _validate_recurrent_args


class GarchLSTMDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        garch_feature_index: int,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        _validate_recurrent_args(hidden_size, num_layers, norm_type)
        if garch_feature_index < 0 or garch_feature_index >= input_size:
            raise ValueError("garch_feature_index out of bounds for input size")

        dropout_prob = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = _build_norm(norm_type, hidden_size)
        self.hidden_dim = hidden_size
        self.garch_feature_index = int(garch_feature_index)
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm is not None:
            last_hidden = self.norm(last_hidden)
        garch_feature = x[:, -1, self.garch_feature_index].unsqueeze(1)
        combined = torch.cat([last_hidden, garch_feature], dim=1)
        logits = self.fc(combined)
        return logits.squeeze(1)


__all__ = ["GarchLSTMDirectionClassifier"]
