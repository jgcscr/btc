from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.training.time_series_cv import TimeSeriesFold, build_time_series_folds


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "logloss",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 4h static train/val/test behavior against walk-forward folds on the same NPZ dataset."
    )
    parser.add_argument("--dataset-path", default="artifacts/datasets/btc_features_multi_horizon_splits.npz")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output-json", default="artifacts/analysis/4h_static_vs_walkforward_diagnostic_latest.json")
    parser.add_argument("--output-md", default="artifacts/analysis/4h_static_vs_walkforward_diagnostic_latest.md")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model = XGBClassifier(**DEFAULT_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_pred = (train_proba >= 0.5).astype(int)
    val_pred = (val_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    train_accuracy = float(accuracy_score(y_train, train_pred))
    val_accuracy = float(accuracy_score(y_val, val_pred))
    test_accuracy = float(accuracy_score(y_test, test_pred))
    return {
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "val_positive_rate": float(np.mean(y_val)),
        "test_positive_rate": float(np.mean(y_test)),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_val_gap": float(abs(train_accuracy - val_accuracy)),
        "val_auc": _safe_auc(y_val, val_proba),
        "test_auc": _safe_auc(y_test, test_proba),
    }


def _read_dataset(path: Path, horizon: int) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "X_train",
            "X_val",
            "X_test",
            f"y_dir{horizon}h_train",
            f"y_dir{horizon}h_val",
            f"y_dir{horizon}h_test",
        }
        missing = sorted(key for key in required if key not in data.files)
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")

        X_train = np.asarray(data["X_train"], dtype=np.float32)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_train = np.asarray(data[f"y_dir{horizon}h_train"], dtype=np.int32)
        y_val = np.asarray(data[f"y_dir{horizon}h_val"], dtype=np.int32)
        y_test = np.asarray(data[f"y_dir{horizon}h_test"], dtype=np.int32)
        ts_all = None
        if {"ts_train", "ts_val", "ts_test"}.issubset(data.files):
            ts_all = np.concatenate([data["ts_train"], data["ts_val"], data["ts_test"]])

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_all": np.concatenate([X_train, X_val, X_test], axis=0),
        "y_all": np.concatenate([y_train, y_val, y_test], axis=0),
        "ts_all": ts_all,
    }


def _fold_timestamp_summary(ts_all: np.ndarray | None, fold: TimeSeriesFold) -> Dict[str, str] | None:
    if ts_all is None:
        return None
    return {
        "train_start": str(ts_all[fold.train_start]),
        "train_end": str(ts_all[fold.train_end - 1]),
        "val_start": str(ts_all[fold.val_start]),
        "val_end": str(ts_all[fold.val_end - 1]),
        "test_start": str(ts_all[fold.test_start]),
        "test_end": str(ts_all[fold.test_end - 1]),
    }


def _percentile_rank(values: List[float], target: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr <= target))


def _render_markdown(payload: Mapping[str, Any]) -> str:
    static = payload["static_split"]
    comparison = payload["comparison"]
    lines = ["# 4h Static vs Walk-Forward Diagnostic", ""]
    lines.append("## Static Split")
    lines.append(f"- train_val_gap: {static['train_val_gap']}")
    lines.append(f"- test_accuracy: {static['test_accuracy']}")
    lines.append(f"- test_auc: {static['test_auc']}")
    lines.append("")
    lines.append("## Walk-Forward Aggregate")
    lines.append(f"- median_train_val_gap: {comparison['walkforward_median_train_val_gap']}")
    lines.append(f"- median_test_accuracy: {comparison['walkforward_median_test_accuracy']}")
    lines.append(f"- static_gap_percentile_vs_walkforward: {comparison['static_gap_percentile_vs_walkforward']}")
    lines.append(f"- static_test_accuracy_percentile_vs_walkforward: {comparison['static_test_accuracy_percentile_vs_walkforward']}")
    lines.append("")
    lines.append("## Recommendation")
    for note in payload.get("recommendations", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    dataset = _read_dataset(dataset_path, horizon=int(args.horizon))
    static_metrics = _fit_and_score(
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_val"],
        dataset["y_val"],
        dataset["X_test"],
        dataset["y_test"],
    )

    folds = build_time_series_folds(
        len(dataset["y_all"]),
        n_splits=int(args.n_splits),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        gap=int(args.gap),
        mode=str(args.mode),
    )
    fold_rows: List[Dict[str, Any]] = []
    for idx, fold in enumerate(folds, start=1):
        metrics = _fit_and_score(
            dataset["X_all"][fold.train_slice],
            dataset["y_all"][fold.train_slice],
            dataset["X_all"][fold.val_slice],
            dataset["y_all"][fold.val_slice],
            dataset["X_all"][fold.test_slice],
            dataset["y_all"][fold.test_slice],
        )
        row = {
            "fold": idx,
            **metrics,
        }
        ts_summary = _fold_timestamp_summary(dataset["ts_all"], fold)
        if ts_summary is not None:
            row["timestamps"] = ts_summary
        fold_rows.append(row)

    fold_gaps = [float(row["train_val_gap"]) for row in fold_rows]
    fold_test_acc = [float(row["test_accuracy"]) for row in fold_rows]
    fold_test_auc = [float(row["test_auc"]) for row in fold_rows if row.get("test_auc") is not None]

    comparison = {
        "walkforward_median_train_val_gap": float(np.median(fold_gaps)) if fold_gaps else None,
        "walkforward_median_test_accuracy": float(np.median(fold_test_acc)) if fold_test_acc else None,
        "walkforward_median_test_auc": float(np.median(fold_test_auc)) if fold_test_auc else None,
        "static_gap_percentile_vs_walkforward": _percentile_rank(fold_gaps, float(static_metrics["train_val_gap"])),
        "static_test_accuracy_percentile_vs_walkforward": _percentile_rank(fold_test_acc, float(static_metrics["test_accuracy"])),
    }

    recommendations: List[str] = []
    static_gap_pct = comparison["static_gap_percentile_vs_walkforward"]
    static_test_pct = comparison["static_test_accuracy_percentile_vs_walkforward"]
    if static_gap_pct is not None and static_gap_pct >= 0.75:
        recommendations.append(
            "Static 4h validation is harsher than most walk-forward folds on train/val gap; treat the current holdout boundary as a meaningful stress regime rather than a typical slice."
        )
    if static_test_pct is not None and static_test_pct <= 0.25:
        recommendations.append(
            "Static 4h test accuracy sits in the bottom quartile of walk-forward folds; prioritize regime-robust feature engineering and evaluation over additional family swaps."
        )
    if not recommendations:
        recommendations.append("Static 4h split looks broadly comparable to walk-forward folds under this XGB baseline.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(dataset_path),
            "horizon": int(args.horizon),
            "n_splits": int(args.n_splits),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "gap": int(args.gap),
            "mode": str(args.mode),
        },
        "static_split": static_metrics,
        "walkforward_folds": fold_rows,
        "comparison": comparison,
        "recommendations": recommendations,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote diagnostic JSON: {output_json}")
    print(f"Wrote diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()