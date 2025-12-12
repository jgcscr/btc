from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch

from src.training.lstm_data import build_sequence_splits, create_dataloader
from src.training.lstm_model import LSTMDirectionClassifier
from src.scripts.train_lstm_dir1h import (
    _bce_loss,
    _effective_device,
    _make_generator,
    _train_one_epoch,
    _evaluate,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna study for LSTM direction classifier.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the direction dataset NPZ.",
    )
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length used to build windows.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout (seconds).")
    parser.add_argument("--study-name", type=str, default=None, help="Optional Optuna study name.")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URI (e.g., sqlite:///optuna.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/analysis/optuna_lstm_dir1h_v1",
        help="Directory to write best params and summary.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    return parser.parse_args()


def _sample_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "hidden_size": trial.suggest_int("hidden_size", 64, 256, step=32),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "norm_type": trial.suggest_categorical("norm_type", ["none", "layer", "batch"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96, 128, 192, 256]),
        "epochs": trial.suggest_int("epochs", 6, 20),
        "patience": trial.suggest_int("patience", 3, 8),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


def _stateless_state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = _effective_device(args.device)

    print(f"Loading dataset from {args.dataset_path} with seq_len={args.seq_len}")
    splits = build_sequence_splits(args.dataset_path, args.seq_len)
    input_size = splits.X_train_seq.shape[-1]

    best_result: Dict[str, Any] = {"val_loss": float("inf")}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_result
        params = _sample_hyperparams(trial)

        trial_seed = args.seed + trial.number
        train_loader = create_dataloader(
            splits.X_train_seq,
            splits.y_train_seq,
            batch_size=params["batch_size"],
            shuffle=True,
            generator=_make_generator(trial_seed),
        )
        val_loader = create_dataloader(
            splits.X_val_seq,
            splits.y_val_seq,
            batch_size=params["batch_size"],
            shuffle=False,
        )
        test_loader = create_dataloader(
            splits.X_test_seq,
            splits.y_test_seq,
            batch_size=params["batch_size"],
            shuffle=False,
        )

        model = LSTMDirectionClassifier(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            norm_type=params["norm_type"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = _bce_loss()

        best_state: Optional[Dict[str, Any]] = None
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, params["epochs"] + 1):
            train_loss = _train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_metrics = _evaluate(model, val_loader, device, criterion)
            trial.report(val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    "model": _stateless_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
            else:
                patience_counter += 1
                if patience_counter >= params["patience"]:
                    break

        if best_state is not None:
            model.load_state_dict(best_state["model"])
        model.to(device)
        model.eval()

        train_loss, train_metrics = _evaluate(model, train_loader, device, criterion)
        val_loss, val_metrics = _evaluate(model, val_loader, device, criterion)
        test_loss, test_metrics = _evaluate(model, test_loader, device, criterion)

        metrics = {
            "train": {**train_metrics, "loss": train_loss},
            "val": {**val_metrics, "loss": val_loss},
            "test": {**test_metrics, "loss": test_loss},
        }

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("hyperparams", params)

        print(
            "Trial {num}: val_loss={val_loss:.5f} val_auc={val_auc:.3f} test_auc={test_auc:.3f} params={params}".format(
                num=trial.number,
                val_loss=val_loss,
                val_auc=val_metrics.get("auc", float("nan")),
                test_auc=test_metrics.get("auc", float("nan")),
                params=params,
            )
        )

        if val_loss < best_result["val_loss"]:
            best_result = {
                "val_loss": val_loss,
                "trial_number": trial.number,
                "hyperparams": params,
                "metrics": metrics,
                "state_dict": _stateless_state_dict(model),
            }

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if best_result["val_loss"] == float("inf"):
        raise RuntimeError("Optuna study finished without a valid best result.")

    best_summary = {
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "input_size": input_size,
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best_result["trial_number"],
            "value": best_result["val_loss"],
            "hyperparams": best_result["hyperparams"],
            "metrics": best_result["metrics"],
        },
    }

    summary_path = output_dir / "best_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    params_path = output_dir / "best_params.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(best_result["hyperparams"], handle, indent=2)

    weights_path = output_dir / "best_model.pt"
    torch.save({"state_dict": best_result["state_dict"], "input_size": input_size}, weights_path)

    print(f"Saved best summary to {summary_path}")
    print(f"Saved best params to {params_path}")
    print(f"Saved best weights to {weights_path}")


if __name__ == "__main__":
    main()
