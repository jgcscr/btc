from __future__ import annotations

import argparse
import json
from typing import List

import torch.nn as nn

from src.scripts.train_lstm_dir1h import (
    _apply_params_overrides,
    build_arg_parser,
    train_model,
)
from src.training.cnn_lstm import CNNLSTMDirectionClassifier


def _parse_int_list(raw_value: str | None, label: str) -> List[int]:
    if raw_value is None:
        raise ValueError(f"{label} must be provided")
    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{label} cannot be empty")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be a comma-separated list of integers") from exc


def _resolve_conv_params(args: argparse.Namespace) -> tuple[List[int], List[int], List[int]]:
    channels = _parse_int_list(args.conv_channels, "conv_channels")
    kernels = _parse_int_list(args.conv_kernel_sizes, "conv_kernel_sizes")
    strides = _parse_int_list(args.conv_strides, "conv_strides")

    if not (len(channels) == len(kernels) == len(strides)):
        raise ValueError("conv_channels, conv_kernel_sizes, and conv_strides must have the same length")
    if len(channels) > 2:
        raise ValueError("CNN-LSTM trainer supports at most two convolutional layers")
    return channels, kernels, strides


def _build_cnn_lstm_classifier(input_size: int, args: argparse.Namespace) -> nn.Module:
    return CNNLSTMDirectionClassifier(
        input_size=input_size,
        conv_channels=args.conv_channels_list,
        conv_kernel_sizes=args.conv_kernel_sizes_list,
        conv_strides=args.conv_strides_list,
        lstm_hidden_size=args.hidden_size,
        lstm_num_layers=args.num_layers,
        dropout=args.dropout,
        norm_type=args.norm_type,
        conv_activation=args.conv_activation,
        conv_dropout=args.conv_dropout,
    )


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description="Train a CNN-LSTM hybrid classifier for 1h BTC direction.",
        default_output_dir="artifacts/models/cnn_lstm_dir1h_v1",
    )
    parser.add_argument(
        "--conv-channels",
        type=str,
        default="64,64",
        help="Comma-separated list of Conv1d output channels (1-2 layers supported).",
    )
    parser.add_argument(
        "--conv-kernel-sizes",
        type=str,
        default="3,3",
        help="Comma-separated list of Conv1d kernel sizes (match conv-channels length).",
    )
    parser.add_argument(
        "--conv-strides",
        type=str,
        default="1,1",
        help="Comma-separated list of Conv1d strides (match conv-channels length).",
    )
    parser.add_argument(
        "--conv-dropout",
        type=float,
        default=0.1,
        help="Dropout applied after each Conv1d block (default: 0.1).",
    )
    parser.add_argument(
        "--conv-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="Activation used after each Conv1d block (default: relu).",
    )
    return parser.parse_args()


def train_cnn_lstm(args: argparse.Namespace) -> None:
    channels, kernels, strides = _resolve_conv_params(args)
    args.conv_channels_list = channels
    args.conv_kernel_sizes_list = kernels
    args.conv_strides_list = strides

    resolved = {
        "conv_channels": channels,
        "conv_kernel_sizes": kernels,
        "conv_strides": strides,
        "conv_dropout": args.conv_dropout,
        "conv_activation": args.conv_activation,
    }
    print("Resolved CNN-LSTM hyperparameters:")
    print(json.dumps(resolved, indent=2))

    train_model(
        args,
        model_builder=_build_cnn_lstm_classifier,
        model_label="cnn_lstm_direction_classifier",
        extra_hyperparams=resolved,
    )


def main() -> None:
    args = parse_args()
    _apply_params_overrides(args)
    train_cnn_lstm(args)


if __name__ == "__main__":
    main()
