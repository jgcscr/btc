from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
from torch import nn
from torch.optim import AdamW

from src.models.transformer_classifier import TransformerDirectionClassifier
from src.training.lstm_data import load_direction_npz_full
from src.training.time_series_cv import build_time_series_folds
from src.training.transformer_dataset import prepare_transformer_data, prepare_transformer_data_from_arrays
from src.training.transformer_training import Metrics, evaluate, resolve_device, train_epoch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for transformer direction classifier.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to NPZ dataset containing train/val/test direction splits.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Direction horizon in hours (default: 1).")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length used to build windows.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to execute.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument("--study-name", type=str, default="transformer_dir1h_optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g. sqlite:///study.db).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write best params and summary.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    parser.add_argument("--cv-folds", type=int, default=0, help="Enable time-series CV with this many folds.")
    parser.add_argument("--cv-train-size", type=int, default=0, help="Train window size for each fold.")
    parser.add_argument("--cv-val-size", type=int, default=0, help="Validation window size for each fold.")
    parser.add_argument("--cv-test-size", type=int, default=0, help="Test window size for each fold.")
    parser.add_argument("--cv-gap", type=int, default=0, help="Gap size between train/val/test windows.")
    parser.add_argument("--cv-step-size", type=int, default=None, help="Step size between folds (defaults to test size).")
    parser.add_argument("--cv-mode", type=str, default="expanding", choices=["expanding", "rolling"], help="Time-series CV mode.")
    return parser.parse_args()


def _suggest_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 160, 192, 224, 256])
    head_choices = [h for h in [2, 4, 8] if hidden_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", head_choices)
    ffn_multiplier = trial.suggest_categorical("ffn_multiplier", [2, 3, 4])

    params: Dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "ffn_dim": hidden_dim * ffn_multiplier,
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "use_layer_norm": trial.suggest_categorical("use_layer_norm", [True, False]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96, 128, 192, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 8, 28),
        "patience": trial.suggest_int("patience", 3, 8),
        "ffn_multiplier": ffn_multiplier,
    }
    return params


def _metrics_to_dict(metrics: Metrics) -> Dict[str, Optional[float]]:
    payload: Dict[str, Optional[float]] = asdict(metrics)
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
    return payload


def _is_better(candidate: Metrics, incumbent: Optional[Metrics]) -> bool:
    if incumbent is None:
        return True
    if candidate.loss < incumbent.loss - 1e-5:
        return True
    if math.isclose(candidate.loss, incumbent.loss, rel_tol=1e-5, abs_tol=1e-5):
        cand_auc = candidate.auc if candidate.auc is not None and math.isfinite(candidate.auc) else -float("inf")
        inc_auc = incumbent.auc if incumbent.auc is not None and math.isfinite(incumbent.auc) else -float("inf")
        if cand_auc > inc_auc + 1e-4:
            return True
    return False


def main() -> None:
    args = _parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(
        "Running Optuna search (dataset={dataset}, seq_len={seq_len}, n_trials={n_trials}, timeout={timeout})".format(
            dataset=args.dataset_path,
            seq_len=args.seq_len,
            n_trials=args.n_trials,
            timeout=args.timeout,
        )
    )

    X_all = None
    y_all = None
    feature_names = None
    threshold = 0.0

    if args.cv_folds > 0:
        X_all, y_all, feature_names, threshold = load_direction_npz_full(
            args.dataset_path,
            horizon=args.horizon,
        )
        input_size = int(X_all.shape[-1])
    else:
        # Prepare once to infer input size.
        data_bundle, _, _, _ = prepare_transformer_data(
            dataset_path=args.dataset_path,
            seq_len=args.seq_len,
            batch_size=128,
            horizon=args.horizon,
            generator=np.random.default_rng(args.seed),
        )
        input_size = data_bundle.splits.X_train_seq.shape[-1]

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
    )

    def _train_fold(
        *,
        train_loader,
        val_loader,
        test_loader,
        params: Dict[str, Any],
    ) -> tuple[Metrics, Metrics, Metrics, Optional[Dict[str, torch.Tensor]], int]:
        model = TransformerDirectionClassifier(
            input_size=input_size,
            hidden_dim=params["hidden_dim"],
            num_heads=params["num_heads"],
            ffn_dim=params["ffn_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            max_seq_len=args.seq_len,
            use_layer_norm=params["use_layer_norm"],
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

        best_state = None
        best_val_metrics: Optional[Metrics] = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, params["epochs"] + 1):
            train_epoch(model, train_loader, device, criterion, optimizer)
            val_metrics = evaluate(model, val_loader, device, criterion)

            if _is_better(val_metrics, best_val_metrics):
                best_val_metrics = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= params["patience"]:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()

        train_metrics = evaluate(model, train_loader, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        test_metrics = evaluate(model, test_loader, device, criterion)
        return train_metrics, val_metrics, test_metrics, best_state, best_epoch

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hyperparams(trial)

        if args.cv_folds > 0:
            if X_all is None or y_all is None:
                raise RuntimeError("CV requested but full dataset arrays are missing.")
            if args.cv_train_size <= 0 or args.cv_val_size <= 0 or args.cv_test_size <= 0:
                raise ValueError("CV sizes must be positive when --cv-folds is set.")
            folds = build_time_series_folds(
                len(y_all),
                n_splits=args.cv_folds,
                train_size=args.cv_train_size,
                val_size=args.cv_val_size,
                test_size=args.cv_test_size,
                gap=args.cv_gap,
                step_size=args.cv_step_size,
                mode=args.cv_mode,
            )

            val_losses = []
            fold_metrics: Dict[str, Any] = {}
            for fold_idx, fold in enumerate(folds, start=1):
                generator = np.random.default_rng(args.seed + trial.number + fold_idx)
                _, train_loader, val_loader, test_loader = prepare_transformer_data_from_arrays(
                    X_train=X_all[fold.train_slice],
                    y_train=y_all[fold.train_slice],
                    X_val=X_all[fold.val_slice],
                    y_val=y_all[fold.val_slice],
                    X_test=X_all[fold.test_slice],
                    y_test=y_all[fold.test_slice],
                    feature_names=feature_names or [],
                    threshold=threshold,
                    seq_len=args.seq_len,
                    batch_size=params["batch_size"],
                    generator=generator,
                )
                train_metrics, val_metrics, test_metrics, _, best_epoch = _train_fold(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    params=params,
                )
                val_losses.append(val_metrics.loss)
                fold_metrics[f"fold_{fold_idx}"] = {
                    "train_metrics": _metrics_to_dict(train_metrics),
                    "val_metrics": _metrics_to_dict(val_metrics),
                    "test_metrics": _metrics_to_dict(test_metrics),
                    "best_epoch": best_epoch,
                }

            mean_val = float(np.mean(val_losses))
            trial.set_user_attr("hyperparams", params)
            trial.set_user_attr("cv_metrics", {"folds": fold_metrics, "cv_mean": mean_val})
            return mean_val

        generator = np.random.default_rng(args.seed + trial.number)
        _, train_loader, val_loader, test_loader = prepare_transformer_data(
            dataset_path=args.dataset_path,
            seq_len=args.seq_len,
            batch_size=params["batch_size"],
            horizon=args.horizon,
            generator=generator,
        )

        train_metrics, val_metrics, test_metrics, _, best_epoch = _train_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            params=params,
        )

        val_dict = _metrics_to_dict(val_metrics)
        test_dict = _metrics_to_dict(test_metrics)
        train_dict = _metrics_to_dict(train_metrics)

        trial.set_user_attr("hyperparams", params)
        trial.set_user_attr("train_metrics", train_dict)
        trial.set_user_attr("val_metrics", val_dict)
        trial.set_user_attr("test_metrics", test_dict)
        trial.set_user_attr("best_epoch", best_epoch)

        print(
            "Trial {num}: val_loss={val_loss:.6f} val_auc={val_auc} test_auc={test_auc} params={params}".format(
                num=trial.number,
                val_loss=val_metrics.loss,
                val_auc=val_dict.get("auc"),
                test_auc=test_dict.get("auc"),
                params=params,
            )
        )

        return val_metrics.loss

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    if not study.trials:
        raise RuntimeError("Optuna study finished without completed trials.")

    best_trial = study.best_trial
    raw_params = dict(best_trial.params)
    ffn_multiplier = raw_params.pop("ffn_multiplier", 2)
    best_params = dict(raw_params)
    best_params["ffn_dim"] = best_params["hidden_dim"] * ffn_multiplier
    best_val_metrics = best_trial.user_attrs.get("val_metrics", {})
    best_test_metrics = best_trial.user_attrs.get("test_metrics", {})
    best_train_metrics = best_trial.user_attrs.get("train_metrics", {})
    best_epoch = best_trial.user_attrs.get("best_epoch")
    best_cv_metrics = best_trial.user_attrs.get("cv_metrics")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_summary = {
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "study_name": args.study_name,
        "n_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": {
            "number": best_trial.number,
            "value": float(best_trial.value),
            "params": {**raw_params, "ffn_multiplier": ffn_multiplier, "ffn_dim": best_params["ffn_dim"]},
            "train_metrics": best_train_metrics,
            "val_metrics": best_val_metrics,
            "test_metrics": best_test_metrics,
            "best_epoch": best_epoch,
        },
    }
    if args.cv_folds > 0:
        best_summary["cv"] = {
            "folds": args.cv_folds,
            "train_size": args.cv_train_size,
            "val_size": args.cv_val_size,
            "test_size": args.cv_test_size,
            "gap": args.cv_gap,
            "step_size": args.cv_step_size,
            "mode": args.cv_mode,
        }
        best_summary["best_trial"]["cv_metrics"] = best_cv_metrics

    summary_path = output_dir / "best_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, indent=2)

    params_path = output_dir / "best_params.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)

    trials_path = output_dir / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in study.trials:
            record = {
                "number": trial.number,
                "state": trial.state.name,
                "value": float(trial.value) if trial.value is not None else None,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            json.dump(record, handle)
            handle.write("\n")

    print(f"Best trial #{best_trial.number} value={best_trial.value:.6f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Saved best summary to {summary_path}")
    print(f"Saved best params to {params_path}")
    print(f"Wrote trial log to {trials_path}")


if __name__ == "__main__":
    main()
