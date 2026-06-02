from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from src.models.transformer_classifier import TransformerDirectionClassifier
from src.training.transformer_dataset import prepare_transformer_data, save_scaler
from src.training.transformer_training import Metrics, evaluate, resolve_device, train_epoch
from src.utils.model_summary import build_model_summary


TRANSFORMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {
        "seq_len": 12,
        "hidden_dim": 128,
        "num_heads": 4,
        "ffn_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "large": {
        "seq_len": 24,
        "hidden_dim": 192,
        "num_heads": 6,
        "ffn_dim": 384,
        "num_layers": 4,
        "dropout": 0.15,
    },
}

_PRESET_PARAMS: Sequence[str] = (
    "seq_len",
    "hidden_dim",
    "num_heads",
    "ffn_dim",
    "num_layers",
    "dropout",
)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def load_params_from_json(json_path: str) -> Dict[str, Any]:
    with open(json_path) as fp:
        return json.load(fp)


def build_parser(
    *,
    default_output_dir: str = "artifacts/models/transformer_dir1h_v1",
    preset_default: str = "base",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a transformer-based direction classifier for 1h BTC signals.")
    parser.add_argument("--dataset-path", type=str, default="artifacts/datasets/btc_features_1h_direction_splits.npz")
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Direction horizon in hours (default: 1).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(TRANSFORMER_PRESETS.keys()),
        default=preset_default,
        help="Preset controlling transformer depth/width (default: base).",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length (default: preset).")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Transformer hidden size (default: preset).")
    parser.add_argument("--num-heads", type=int, default=None, help="Multi-head attention heads (default: preset).")
    parser.add_argument("--ffn-dim", type=int, default=None, help="Feed-forward hidden dim (default: preset).")
    parser.add_argument("--num-layers", type=int, default=None, help="Transformer layers (default: preset).")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout prob (default: preset).")
    parser.add_argument("--use-layer-norm", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device (e.g. cpu, cuda:0).")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation F1 score.")
    parser.add_argument("--params-json", type=str, default=None, help="Optional path to JSON file containing hyperparameter overrides.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit on optimizer steps per epoch (useful for CI smoke tests).",
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_output_dir: str = "artifacts/models/transformer_dir1h_v1",
    preset_default: str = "base",
) -> argparse.Namespace:
    parser = build_parser(default_output_dir=default_output_dir, preset_default=preset_default)
    return parser.parse_args(argv)


def _apply_param_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Unknown hyperparameter in JSON overrides: {key}")
    return args


def apply_transformer_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset_key = getattr(args, "preset", "base")
    if preset_key not in TRANSFORMER_PRESETS:
        raise ValueError(f"Unknown transformer preset '{preset_key}'. Choices: {sorted(TRANSFORMER_PRESETS)}")

    defaults = TRANSFORMER_PRESETS[preset_key]
    for field in _PRESET_PARAMS:
        if getattr(args, field, None) is None:
            setattr(args, field, defaults[field])
    return args


def _prepare_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_summary(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def _build_model(args: argparse.Namespace, input_size: int) -> nn.Module:
    model = TransformerDirectionClassifier(
        input_size=input_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        use_layer_norm=args.use_layer_norm,
    )
    return model


def train_transformer(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("btc_transformer_direction")

    with mlflow.start_run():
        if args.params_json:
            overrides = load_params_from_json(args.params_json)
            args = _apply_param_overrides(args, overrides)
        args = apply_transformer_preset(args)

        if args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("--max-steps must be positive when provided.")

        device = resolve_device(args.device)
        print(f"Using device: {device}")

        # Log parameters
        mlflow.log_param("dataset_path", args.dataset_path)
        mlflow.log_param("horizon", args.horizon)
        mlflow.log_param("preset", args.preset)
        mlflow.log_param("seq_len", args.seq_len)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_param("num_heads", args.num_heads)
        mlflow.log_param("ffn_dim", args.ffn_dim)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("use_layer_norm", args.use_layer_norm)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("max_steps", args.max_steps)

        data_bundle, train_loader, val_loader, test_loader = prepare_transformer_data(
            dataset_path=args.dataset_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            horizon=args.horizon,
        )

        input_size = data_bundle.splits.X_train_seq.shape[-1]
        model = _build_model(args, input_size).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        history: list[Dict[str, Any]] = []
        best_state: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Metrics] = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                device,
                criterion,
                optimizer,
                max_steps=args.max_steps,
            )
            val_metrics = evaluate(model, val_loader, device, criterion)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_f1": val_metrics.f1,
                    "val_auc": val_metrics.auc,
                },
            )

            print(
                "Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc}".format(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics.loss,
                    val_acc=val_metrics.accuracy,
                    val_f1=val_metrics.f1,
                    val_auc=val_metrics.auc,
                ),
            )

            # Log metrics
            mlflow.log_metric(f"train_loss_epoch_{epoch}", train_loss)
            mlflow.log_metric(f"val_loss_epoch_{epoch}", val_metrics.loss)
            mlflow.log_metric(f"val_accuracy_epoch_{epoch}", val_metrics.accuracy)
            mlflow.log_metric(f"val_f1_epoch_{epoch}", val_metrics.f1)
            mlflow.log_metric(f"val_auc_epoch_{epoch}", val_metrics.auc)

            improved = False
            if best_metrics is None or val_metrics.f1 > best_metrics.f1:
                improved = True
            elif best_metrics is not None and np.isclose(val_metrics.f1, best_metrics.f1) and val_metrics.loss < best_metrics.loss:
                improved = True

            if improved:
                best_metrics = val_metrics
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        train_metrics = evaluate(model, train_loader, device, criterion)
        test_metrics = evaluate(model, test_loader, device, criterion)
        print(
            "Test metrics: loss={loss:.6f} acc={acc:.4f} f1={f1:.4f} auc={auc}".format(
                loss=test_metrics.loss,
                acc=test_metrics.accuracy,
                f1=test_metrics.f1,
                auc=test_metrics.auc,
            ),
        )

        # Log final metrics
        mlflow.log_metric("test_loss", test_metrics.loss)
        mlflow.log_metric("test_accuracy", test_metrics.accuracy)
        mlflow.log_metric("test_f1", test_metrics.f1)
        mlflow.log_metric("test_auc", test_metrics.auc)

        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "model.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "ffn_dim": args.ffn_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "use_layer_norm": args.use_layer_norm,
            },
            model_path,
        )

        scaler_path = os.path.join(output_dir, "scaler.joblib")
        save_scaler(data_bundle.scaler_mean, data_bundle.scaler_std, scaler_path)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Register model
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, f"transformer_dir{args.horizon}h")

        # Log summary as artifact
        hyperparams = {
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_layer_norm": args.use_layer_norm,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "max_steps": args.max_steps,
        }
        summary: Dict[str, Any] = build_model_summary(
            model_type="transformer_direction_classifier",
            target=f"direction_{int(args.horizon)}h",
            dataset_path=args.dataset_path,
            model_path=model_path,
            metrics={
                "train": asdict(train_metrics),
                "val": asdict(best_metrics) if best_metrics else {},
                "test": asdict(test_metrics),
            },
            feature_names=data_bundle.splits.feature_names,
            hyperparams=hyperparams,
            threshold=data_bundle.splits.threshold,
            horizon_hours=int(args.horizon),
            seq_len=args.seq_len,
            scaler_path=scaler_path,
            extra_fields={
                "preset": args.preset,
                "best_epoch": best_state["epoch"] if best_state else args.epochs,
                "train_metrics": asdict(train_metrics),
                "val_metrics": asdict(best_metrics) if best_metrics else None,
                "test_metrics": asdict(test_metrics),
                "history": history,
                "params_json": args.params_json,
            },
        )

        summary_path = os.path.join(output_dir, "summary.json")
        _save_summary(summary_path, summary)
        mlflow.log_artifact(summary_path)
        mlflow.log_artifact(scaler_path)
    print(f"Saved model artifacts to {args.output_dir}")


def main() -> None:
    args = parse_args()
    train_transformer(args)


if __name__ == "__main__":
    main()
