from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.training.time_series_cv import build_time_series_folds


def _parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for p in value.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    if not out:
        raise ValueError("Expected non-empty comma-separated integer list.")
    return out


def _load_target(data: np.lib.npyio.NpzFile, y_key: str) -> np.ndarray:
    if y_key in data.files:
        return np.asarray(data[y_key], dtype=int)
    alt = f"{y_key}_val"
    if alt in data.files:
        raise KeyError(f"Use a full target array key. Found split key '{alt}' but expected full-series key '{y_key}'.")
    raise KeyError(f"Missing target key '{y_key}' in dataset.")


def _score_fold(X: np.ndarray, y: np.ndarray, train_slice: slice, val_slice: slice) -> Dict[str, float]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    X_train = X[train_slice]
    y_train = y[train_slice]
    X_val = X[val_slice]
    y_val = y[val_slice]
    model.fit(X_train, y_train)
    p = np.clip(model.predict_proba(X_val)[:, 1], 1e-8, 1.0 - 1e-8)
    return {
        "logloss": float(log_loss(y_val, p)),
        "auc": float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run purge/embargo CV stress sweep and report stability.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--purge-list", type=str, default="0,12,24")
    parser.add_argument("--embargo-list", type=str, default="0,12,24")
    parser.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/cv_stress_sweep.json"))
    args = parser.parse_args()

    with np.load(args.dataset_path, allow_pickle=True) as data:
        X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
        if args.y_key == "y":
            y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
        else:
            y = _load_target(data, args.y_key)

    purge_list = _parse_int_list(args.purge_list)
    embargo_list = _parse_int_list(args.embargo_list)

    rows: List[Dict[str, Any]] = []
    for purge in purge_list:
        for embargo in embargo_list:
            folds = build_time_series_folds(
                n_samples=len(y),
                n_splits=int(args.folds),
                train_size=int(args.train_size),
                val_size=int(args.val_size),
                test_size=int(args.test_size),
                gap=int(args.gap),
                purge_size=int(purge),
                embargo_size=int(embargo),
                mode=str(args.mode),
            )
            fold_scores: List[Dict[str, float]] = []
            for fold in folds:
                fold_scores.append(_score_fold(X, y, fold.train_slice, fold.val_slice))
            auc_vals = np.array([v["auc"] for v in fold_scores if np.isfinite(v["auc"])], dtype=float)
            ll_vals = np.array([v["logloss"] for v in fold_scores], dtype=float)
            rows.append(
                {
                    "purge_size": int(purge),
                    "embargo_size": int(embargo),
                    "fold_count": int(len(fold_scores)),
                    "auc_mean": float(np.nanmean(auc_vals)) if auc_vals.size else float("nan"),
                    "auc_std": float(np.nanstd(auc_vals)) if auc_vals.size else float("nan"),
                    "logloss_mean": float(np.mean(ll_vals)),
                    "logloss_std": float(np.std(ll_vals)),
                }
            )

    # Stable/high-performing settings rank: high auc, low auc std, low logloss.
    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["auc_mean"]) if np.isfinite(r["auc_mean"]) else float("inf"),
            float(r["auc_std"]),
            float(r["logloss_mean"]),
        ),
    )

    payload = {
        "dataset_path": str(args.dataset_path),
        "y_key": args.y_key,
        "settings": {
            "folds": int(args.folds),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "gap": int(args.gap),
            "mode": args.mode,
            "purge_list": purge_list,
            "embargo_list": embargo_list,
        },
        "results": rows,
        "recommended": ranked[0] if ranked else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
