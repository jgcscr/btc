from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple
from joblib import dump
def _apply_params_overrides(args: argparse.Namespace) -> None:
    params_path = getattr(args, "params_json", None)
    if not params_path:
        return
    params_file = Path(params_path)
    if not params_file.exists():
        raise FileNotFoundError(f"params_json not found: {params_path}")
    with params_file.open("r", encoding="utf-8") as handle:
        overrides: Dict[str, Any] = json.load(handle)
    for key, value in overrides.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import Adam

from src.training.lstm_data import (
    build_sequence_splits,
    create_dataloader,
    estimate_feature_stats,
)
from src.training.lstm_model import LSTMDirectionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _effective_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _bce_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def _collect_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, preds)) if y_true.size else float("nan"),
        "precision": float(precision_score(y_true, preds, zero_division=0)) if y_true.size else float("nan"),
        "recall": float(recall_score(y_true, preds, zero_division=0)) if y_true.size else float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)) if y_true.size else float("nan"),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def _format_metrics(metrics: Dict[str, float]) -> str:
    ordered = [
        ("loss", metrics.get("loss")),
        ("accuracy", metrics.get("accuracy")),
        ("precision", metrics.get("precision")),
        ("recall", metrics.get("recall")),
        ("f1", metrics.get("f1")),
        ("auc", metrics.get("auc")),
    ]
    parts = []
    for key, value in ordered:
        if value is None or isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={value:.3f}")
    return "{" + ", ".join(parts) + "}"


def _evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            batch_size = X_batch.size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y_batch.cpu().numpy())
    avg_loss = loss_sum / max(count, 1)
    if all_probs:
        probs_np = np.concatenate(all_probs)
        targets_np = np.concatenate(all_targets)
    else:
        probs_np = np.array([], dtype=np.float32)
        targets_np = np.array([], dtype=np.float32)
    metrics = _collect_metrics(targets_np, probs_np)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = X_batch.size(0)
        loss_sum += loss.item() * batch_size
        count += batch_size
    return loss_sum / max(count, 1)


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _effective_device(args.device)

    splits = build_sequence_splits(args.dataset_path, args.seq_len)
    input_size = splits.X_train_seq.shape[-1]

    train_gen = _make_generator(args.seed)
    train_loader = create_dataloader(splits.X_train_seq, splits.y_train_seq, args.batch_size, shuffle=True, generator=train_gen)
    val_loader = create_dataloader(splits.X_val_seq, splits.y_val_seq, args.batch_size, shuffle=False)
    test_loader = create_dataloader(splits.X_test_seq, splits.y_test_seq, args.batch_size, shuffle=False)

    model = LSTMDirectionClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        norm_type=args.norm_type,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = _bce_loss()

    best_state = None
    best_val_loss = float("inf")
    best_metrics: Dict[str, Dict[str, float]] = {}
    patience_counter = 0

    print(f"Training on device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_metrics = _evaluate(model, val_loader, device, criterion)

        log_line = (
            f"Epoch {epoch:02d}/{args.epochs} - train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_auc={val_metrics['auc']:.3f} "
            f"val_f1={val_metrics['f1']:.3f}"
        )
        print(log_line)

        if epoch % args.metric_interval == 0 or epoch == args.epochs:
            _, train_metrics_snapshot = _evaluate(model, train_loader, device, criterion)
            _, test_metrics_snapshot = _evaluate(model, test_loader, device, criterion)
            print(
                f"Metrics @epoch={epoch:02d}: "
                f"train={_format_metrics(train_metrics_snapshot)} "
                f"val={_format_metrics(val_metrics)} "
                f"test={_format_metrics(test_metrics_snapshot)}"
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            best_metrics["val"] = val_metrics
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])  # type: ignore[arg-type]
    model.eval()

    # Final metrics on train/val/test
    train_loss, train_metrics = _evaluate(model, train_loader, device, criterion)
    val_loss, val_metrics = _evaluate(model, val_loader, device, criterion)
    test_loss, test_metrics = _evaluate(model, test_loader, device, criterion)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "input_size": input_size}, model_path)

    feature_mean, feature_std = estimate_feature_stats(splits.X_train_seq.reshape(-1, input_size))
    scaler_path = output_dir / "scaler.joblib"
    dump({"mean": feature_mean, "std": feature_std}, scaler_path)

    summary = {
        "model_type": "lstm_direction_classifier",
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "feature_names": splits.feature_names,
        "threshold": splits.threshold,
        "device": str(device),
        "hyperparams": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "norm_type": args.norm_type,
        },
        "metrics": {
            "train": {**train_metrics, "loss": train_loss},
            "val": {**val_metrics, "loss": val_loss},
            "test": {**test_metrics, "loss": test_loss},
        },
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved model weights to {model_path}")
    print(f"Saved scaler stats to {scaler_path}")
    print(f"Saved summary to {summary_path}")
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM classifier for 1h BTC direction.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the flat direction NPZ dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/lstm_dir1h_v2",
        help="Directory to store the trained model and summary.",
    )
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length (timesteps).")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for the LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate applied after the LSTM output.")
    parser.add_argument("--norm-type", type=str, default="none", choices=["none", "layer", "batch"], help="Optional normalization applied to the LSTM final hidden state.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--params-json", type=str, default=None, help="Optional JSON file with hyperparameter overrides.")
    parser.add_argument("--metric-interval", type=int, default=2, help="Epoch interval for detailed metric logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_params_overrides(args)
    train_model(args)


if __name__ == "__main__":
    main()
