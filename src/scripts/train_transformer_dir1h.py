from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from src.models.transformer_classifier import TransformerDirectionClassifier
from src.training.transformer_dataset import prepare_transformer_data, save_scaler
from src.training.transformer_training import Metrics, evaluate, resolve_device, train_epoch


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer-based direction classifier for 1h BTC signals.")
    parser.add_argument("--dataset-path", type=str, default="artifacts/datasets/btc_features_1h_direction_splits.npz")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-layer-norm", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device (e.g. cpu, cuda:0).")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/transformer_dir1h_v1")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation F1 score.")
    parser.add_argument("--params-json", type=str, default=None, help="Optional path to JSON file containing hyperparameter overrides.")
    return parser.parse_args()


def _apply_param_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Unknown hyperparameter in JSON overrides: {key}")
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
    if args.params_json:
        overrides = load_params_from_json(args.params_json)
        args = _apply_param_overrides(args, overrides)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data_bundle, train_loader, val_loader, test_loader = prepare_transformer_data(
        dataset_path=args.dataset_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
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
        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)
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

    test_metrics = evaluate(model, test_loader, device, criterion)
    print(
        "Test metrics: loss={loss:.6f} acc={acc:.4f} f1={f1:.4f} auc={auc}".format(
            loss=test_metrics.loss,
            acc=test_metrics.accuracy,
            f1=test_metrics.f1,
            auc=test_metrics.auc,
        ),
    )

    _prepare_output_dir(args.output_dir)

    model_path = os.path.join(args.output_dir, "model.pt")
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
            "seq_len": args.seq_len,
        },
        model_path,
    )

    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    save_scaler(data_bundle.scaler_mean, data_bundle.scaler_std, scaler_path)

    summary: Dict[str, Any] = {
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "hyperparams": {
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
        },
        "feature_names": data_bundle.splits.feature_names,
        "threshold": data_bundle.splits.threshold,
        "best_epoch": best_state["epoch"] if best_state else args.epochs,
        "val_metrics": asdict(best_metrics) if best_metrics else None,
        "test_metrics": asdict(test_metrics),
        "history": history,
        "params_json": args.params_json,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    _save_summary(summary_path, summary)
    print(f"Saved model artifacts to {args.output_dir}")


def main() -> None:
    args = _parse_args()
    train_transformer(args)


if __name__ == "__main__":
    main()
