from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.training.time_series_cv import build_time_series_folds


def _load_npz(path: Path, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "X_train" not in data or "X_val" not in data or "X_test" not in data:
        raise KeyError("NPZ missing split arrays")
    if y_key not in data and y_key != "y":
        raise KeyError(f"Missing target key {y_key}")

    X = np.vstack([data["X_train"], data["X_val"], data["X_test"]]).astype(np.float32)
    if y_key == "y":
        y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]]).astype(np.float32)
    else:
        y = np.concatenate([data[f"{y_key}_train"], data[f"{y_key}_val"], data[f"{y_key}_test"]]).astype(np.float32)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for direction model stability.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--y-key", type=str, default="y", help="'y' for flat labels or prefix like y_dir4h")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--purge-size", type=int, default=0)
    parser.add_argument("--embargo-size", type=int, default=0)
    parser.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/walkforward_validation.json"))
    args = parser.parse_args()

    X, y = _load_npz(args.dataset_path, args.y_key)
    folds = build_time_series_folds(
        len(y),
        n_splits=args.folds,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        gap=args.gap,
        purge_size=args.purge_size,
        embargo_size=args.embargo_size,
        mode=args.mode,
    )

    rows: List[dict] = []
    for i, fold in enumerate(folds, start=1):
        X_train = X[fold.train_slice]
        y_train = y[fold.train_slice].astype(int)
        X_test = X[fold.test_slice]
        y_test = y[fold.test_slice].astype(int)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            n_jobs=4,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train, verbose=False)
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)

        auc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan")
        acc = float(accuracy_score(y_test, pred))
        rows.append({"fold": i, "auc": auc, "acc": acc, "n_test": int(len(y_test))})

    auc_vals = np.asarray([r["auc"] for r in rows if np.isfinite(r["auc"])], dtype=float)
    payload = {
        "folds": rows,
        "auc_mean": float(np.nanmean(auc_vals)) if auc_vals.size else float("nan"),
        "auc_std": float(np.nanstd(auc_vals, ddof=0)) if auc_vals.size else float("nan"),
        "auc_cv": float(np.nanstd(auc_vals, ddof=0) / abs(np.nanmean(auc_vals))) if auc_vals.size and abs(np.nanmean(auc_vals)) > 1e-12 else float("nan"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
