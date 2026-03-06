from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch.nn as nn

from src.scripts.train_lstm_dir1h import (
    _apply_params_overrides,
    build_arg_parser,
    train_model,
)
from src.training.garch_lstm import GarchLSTMDirectionClassifier


DEFAULT_GARCH_FEATURE = "volatility_garch_like"


def _resolve_garch_feature_index(dataset_path: str, feature_name: str) -> int:
    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(data_path, allow_pickle=True) as data:
        if "feature_names" not in data.files:
            raise KeyError("Dataset missing feature_names for GARCH hybrid training")
        feature_names = [str(name) for name in data["feature_names"].tolist()]
    if feature_name not in feature_names:
        raise ValueError(
            f"GARCH feature '{feature_name}' not found in dataset. Available count: {len(feature_names)}",
        )
    return feature_names.index(feature_name)


def _build_garch_lstm_classifier(input_size: int, args: argparse.Namespace) -> nn.Module:
    return GarchLSTMDirectionClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        garch_feature_index=args.garch_feature_index,
        norm_type=args.norm_type,
    )


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description="Train a GARCH-LSTM hybrid classifier for 1h BTC direction.",
        default_output_dir="artifacts/models/garch_lstm_dir1h_v1",
    )
    parser.add_argument(
        "--garch-feature",
        type=str,
        default=DEFAULT_GARCH_FEATURE,
        help="Feature name to use as the GARCH volatility input.",
    )
    return parser.parse_args()


def train_garch_lstm(args: argparse.Namespace) -> None:
    garch_feature = str(args.garch_feature).strip()
    if not garch_feature:
        raise ValueError("--garch-feature cannot be empty")

    garch_index = _resolve_garch_feature_index(args.dataset_path, garch_feature)
    args.garch_feature_index = garch_index

    resolved: Dict[str, object] = {
        "garch_feature": garch_feature,
        "garch_feature_index": garch_index,
    }
    print("Resolved GARCH-LSTM hyperparameters:")
    print(json.dumps(resolved, indent=2))

    train_model(
        args,
        model_builder=_build_garch_lstm_classifier,
        model_label="garch_lstm_direction_classifier",
        extra_hyperparams=resolved,
    )


def main() -> None:
    args = parse_args()
    _apply_params_overrides(args)
    train_garch_lstm(args)


if __name__ == "__main__":
    main()
