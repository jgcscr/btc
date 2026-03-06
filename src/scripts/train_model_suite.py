from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _parse_targets(value: str) -> List[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one horizon must be provided.")
    targets: List[float] = []
    for part in parts:
        try:
            targets.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid horizon: {part}") from exc
    return targets


def _label_for_horizon(horizon: float) -> str:
    if horizon < 1.0:
        minutes = int(round(horizon * 60))
        return f"{minutes}m"
    if float(horizon).is_integer():
        return f"{int(horizon)}h"
    return f"{horizon:g}h"


@dataclass(frozen=True)
class DatasetConfig:
    regression_path: str
    direction_path: str
    use_flat_labels: bool


DATASET_15M = DatasetConfig(
    regression_path="artifacts/datasets/btc_features_15m_splits.npz",
    direction_path="artifacts/datasets/btc_features_15m_direction_splits.npz",
    use_flat_labels=True,
)
DATASET_1H = DatasetConfig(
    regression_path="artifacts/datasets/btc_features_1h_splits.npz",
    direction_path="artifacts/datasets/btc_features_1h_direction_splits.npz",
    use_flat_labels=True,
)
DATASET_MULTI = DatasetConfig(
    regression_path="artifacts/datasets/btc_features_multi_horizon_splits.npz",
    direction_path="artifacts/datasets/btc_features_multi_horizon_splits.npz",
    use_flat_labels=False,
)


def _dataset_for_horizon(horizon: float) -> DatasetConfig:
    if horizon < 1.0:
        return DATASET_15M
    if float(horizon).is_integer() and int(round(horizon)) == 1:
        return DATASET_1H
    return DATASET_MULTI


def _run_command(args: Sequence[str]) -> None:
    cmd = [sys.executable, "-m", *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_datasets(targets: Iterable[float]) -> None:
    if any(h < 1.0 for h in targets):
        _run_command(["src.scripts.build_training_dataset_15m", "--output-dir", "artifacts/datasets"])
        _run_command(["src.scripts.build_training_dataset_direction_15m", "--output-dir", "artifacts/datasets"])
    if any(h >= 1.0 for h in targets):
        _run_command(["src.scripts.build_training_dataset", "--output-dir", "artifacts/datasets"])
        _run_command(["src.scripts.build_training_dataset_direction", "--output-dir", "artifacts/datasets"])
        _run_command(
            [
                "src.scripts.build_training_dataset_multi_horizon",
                "--output-dir",
                "artifacts/datasets",
                "--horizons",
                "1",
                "4",
                "8",
                "12",
            ]
        )


def _train_xgb_regression(horizon: float, dataset: DatasetConfig) -> None:
    suffix = _label_for_horizon(horizon)
    horizon_int = int(round(horizon)) if horizon >= 1.0 else 1
    output_dir = f"artifacts/models/xgb_ret{suffix}_v1"
    args = [
        "src.scripts.train_xgb_ret4h_v1",
        "--dataset-path",
        dataset.regression_path,
        "--output-dir",
        output_dir,
        "--horizon",
        str(horizon_int),
        "--suffix",
        suffix,
    ]
    if dataset.use_flat_labels:
        args.append("--use-flat-labels")
    _run_command(args)


def _train_xgb_direction(horizon: float, dataset: DatasetConfig) -> None:
    suffix = _label_for_horizon(horizon)
    if dataset.use_flat_labels:
        output_dir = f"artifacts/models/xgb_dir{suffix}_v1"
        _run_command(
            [
                "src.scripts.train_direction_model",
                "--dataset-path",
                dataset.direction_path,
                "--output-dir",
                output_dir,
                "--model-filename",
                f"xgb_dir{suffix}_model.json",
            ]
        )
        return

    horizon_int = int(round(horizon))
    output_dir = f"artifacts/models/xgb_dir{suffix}_v1"
    _run_command(
        [
            "src.scripts.train_xgb_dir4h_v1",
            "--dataset-path",
            dataset.direction_path,
            "--output-dir",
            output_dir,
            "--horizon",
            str(horizon_int),
        ]
    )


def _train_lgbm_direction(horizon: float, dataset: DatasetConfig) -> None:
    suffix = _label_for_horizon(horizon)
    horizon_int = int(round(horizon)) if horizon >= 1.0 else 1
    output_dir = f"artifacts/models/lgbm_dir{suffix}_v1"
    args = [
        "src.scripts.train_lgbm_dir",
        "--dataset-path",
        dataset.direction_path,
        "--output-dir",
        output_dir,
        "--horizon",
        str(horizon_int),
        "--suffix",
        suffix,
    ]
    if dataset.use_flat_labels:
        args.append("--use-flat-labels")
    _run_command(args)


def _train_sequence_models(horizon: float, dataset: DatasetConfig, seq_len: int) -> None:
    suffix = _label_for_horizon(horizon)
    horizon_int = int(round(horizon)) if horizon >= 1.0 else 1

    common = [
        "--dataset-path",
        dataset.direction_path,
        "--horizon",
        str(horizon_int),
        "--seq-len",
        str(seq_len),
    ]

    _run_command(
        [
            "src.scripts.train_lstm_dir1h",
            "--output-dir",
            f"artifacts/models/lstm_dir{suffix}_v1",
            *common,
        ]
    )
    _run_command(
        [
            "src.scripts.train_gru_dir1h",
            "--output-dir",
            f"artifacts/models/gru_dir{suffix}_v1",
            *common,
        ]
    )
    _run_command(
        [
            "src.scripts.train_bilstm_dir1h",
            "--output-dir",
            f"artifacts/models/bilstm_dir{suffix}_v1",
            *common,
        ]
    )
    _run_command(
        [
            "src.scripts.train_cnn_lstm_dir1h",
            "--output-dir",
            f"artifacts/models/cnn_lstm_dir{suffix}_v1",
            *common,
        ]
    )
    _run_command(
        [
            "src.scripts.train_cnn_bilstm_dir1h",
            "--output-dir",
            f"artifacts/models/cnn_bilstm_dir{suffix}_v1",
            *common,
        ]
    )
    _run_command(
        [
            "src.scripts.train_garch_lstm_dir1h",
            "--output-dir",
            f"artifacts/models/garch_lstm_dir{suffix}_v1",
            *common,
        ]
    )


def _train_transformer(horizon: float, dataset: DatasetConfig, seq_len: int, preset: str) -> None:
    suffix = _label_for_horizon(horizon)
    horizon_int = int(round(horizon)) if horizon >= 1.0 else 1
    _run_command(
        [
            "src.scripts.train_transformer_dir1h",
            "--dataset-path",
            dataset.direction_path,
            "--output-dir",
            f"artifacts/models/transformer_dir{suffix}_v1",
            "--horizon",
            str(horizon_int),
            "--seq-len",
            str(seq_len),
            "--preset",
            preset,
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the full BTC model suite across multiple horizons.",
    )
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=[0.25, 1, 4, 8, 12],
        help="Comma-separated horizons in hours (default: 0.25,1,4,8,12).",
    )
    parser.add_argument(
        "--rebuild-datasets",
        action="store_true",
        help="Rebuild dataset splits before training models.",
    )
    parser.add_argument(
        "--train-regression",
        action="store_true",
        help="Train regression models for each horizon.",
    )
    parser.add_argument(
        "--train-direction",
        action="store_true",
        help="Train XGBoost direction models for each horizon.",
    )
    parser.add_argument(
        "--train-lgbm",
        action="store_true",
        help="Train LightGBM direction models for each horizon.",
    )
    parser.add_argument(
        "--train-sequence",
        action="store_true",
        help="Train LSTM/GRU/BiLSTM/CNN-LSTM/CNN-BiLSTM/GARCH-LSTM direction models for each horizon.",
    )
    parser.add_argument(
        "--train-transformer",
        action="store_true",
        help="Train transformer direction models for each horizon.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="Sequence length for recurrent/transformer models (default: 24).",
    )
    parser.add_argument(
        "--transformer-preset",
        type=str,
        default="base",
        choices=["base", "large"],
        help="Transformer preset (default: base).",
    )
    args = parser.parse_args(argv)

    targets = sorted({float(h) for h in args.targets})
    if not targets:
        raise SystemExit("No horizons provided.")

    if args.rebuild_datasets:
        _build_datasets(targets)

    for horizon in targets:
        dataset = _dataset_for_horizon(horizon)
        if args.train_regression:
            _train_xgb_regression(horizon, dataset)
        if args.train_direction:
            _train_xgb_direction(horizon, dataset)
        if args.train_lgbm:
            _train_lgbm_direction(horizon, dataset)
        if args.train_sequence:
            _train_sequence_models(horizon, dataset, args.seq_len)
        if args.train_transformer:
            _train_transformer(horizon, dataset, args.seq_len, args.transformer_preset)


if __name__ == "__main__":
    main()
