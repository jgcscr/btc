from __future__ import annotations

import argparse

import torch.nn as nn

from src.scripts.train_lstm_dir1h import (
    _apply_params_overrides,
    build_arg_parser,
    train_model,
)
from src.training.lstm_model import BiLSTMDirectionClassifier


def _build_bilstm_classifier(input_size: int, args: argparse.Namespace) -> nn.Module:
    return BiLSTMDirectionClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm_type=args.norm_type,
    )


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description="Train a bidirectional LSTM classifier for 1h BTC direction.",
        default_output_dir="artifacts/models/bilstm_dir1h_v1",
    )
    return parser.parse_args()


def train_bilstm(args: argparse.Namespace) -> None:
    train_model(
        args,
        model_builder=_build_bilstm_classifier,
        model_label="bilstm_direction_classifier",
    )


def main() -> None:
    args = parse_args()
    _apply_params_overrides(args)
    train_bilstm(args)


if __name__ == "__main__":
    main()
