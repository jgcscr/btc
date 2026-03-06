from __future__ import annotations

import argparse

import torch.nn as nn

from src.scripts.train_lstm_dir1h import (
    _apply_params_overrides,
    build_arg_parser,
    train_model,
)
from src.training.lstm_model import GRUDirectionClassifier


def _build_gru_classifier(input_size: int, args: argparse.Namespace) -> nn.Module:
    return GRUDirectionClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm_type=args.norm_type,
    )


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description="Train a GRU classifier for 1h BTC direction.",
        default_output_dir="artifacts/models/gru_dir1h_v1",
    )
    return parser.parse_args()


def train_gru(args: argparse.Namespace) -> None:
    train_model(
        args,
        model_builder=_build_gru_classifier,
        model_label="gru_direction_classifier",
    )


def main() -> None:
    args = parse_args()
    _apply_params_overrides(args)
    train_gru(args)


if __name__ == "__main__":
    main()
