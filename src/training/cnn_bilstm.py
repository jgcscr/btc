from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .lstm_model import _build_norm, _validate_recurrent_args


def _validate_conv_params(
    channels: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
) -> None:
    if not channels:
        raise ValueError("conv_channels must include at least one layer")
    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError("conv parameter lists must be the same length")
    if len(channels) > 2:
        raise ValueError("cnn-bilstm classifier supports at most two conv layers")
    for kernel in kernel_sizes:
        if kernel <= 0:
            raise ValueError("conv kernel sizes must be positive")
    for stride in strides:
        if stride <= 0:
            raise ValueError("conv strides must be positive")
    for ch in channels:
        if ch <= 0:
            raise ValueError("conv channel counts must be positive")


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported conv activation '{name}'. Use 'relu' or 'gelu'.")


class CNNBiLSTMDirectionClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        conv_channels: Sequence[int],
        conv_kernel_sizes: Sequence[int],
        conv_strides: Sequence[int],
        lstm_hidden_size: int,
        lstm_num_layers: int,
        dropout: float,
        norm_type: str = "none",
        conv_activation: str = "relu",
        conv_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_conv_params(conv_channels, conv_kernel_sizes, conv_strides)
        _validate_recurrent_args(lstm_hidden_size, lstm_num_layers, norm_type)

        self.conv_channels = list(int(c) for c in conv_channels)
        self.conv_kernel_sizes = list(int(k) for k in conv_kernel_sizes)
        self.conv_strides = list(int(s) for s in conv_strides)
        self.conv_dropout = float(conv_dropout)
        self.conv_activation = conv_activation

        layers: list[nn.Module] = []
        in_channels = input_size
        for out_channels, kernel_size, stride in zip(
            self.conv_channels,
            self.conv_kernel_sizes,
            self.conv_strides,
        ):
            padding = max(kernel_size // 2, 0)
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            layers.append(_activation(self.conv_activation))
            if self.conv_dropout > 0:
                layers.append(nn.Dropout(self.conv_dropout))
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*layers)

        dropout_prob = dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = _build_norm(norm_type, lstm_hidden_size * 2)
        self.hidden_dim = lstm_hidden_size * 2
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, input_size]")
        conv_input = x.permute(0, 2, 1)
        conv_output = self.conv_stack(conv_input)
        lstm_input = conv_output.permute(0, 2, 1)
        output, _ = self.lstm(lstm_input)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.norm is not None:
            last_hidden = self.norm(last_hidden)
        logits = self.fc(last_hidden)
        return logits.squeeze(1)


__all__ = ["CNNBiLSTMDirectionClassifier"]
