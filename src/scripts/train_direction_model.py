import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    unique_vals = np.unique(y)
    if np.all(np.isin(unique_vals, [0, 1])):
        return y.astype(np.int32)
    return (y > 0).astype(np.int32)


def _load_params_json(params_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not params_path:
        return None
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params JSON not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("Params JSON must contain an object with hyperparameters.")
        return loaded


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = _ensure_binary_labels(data["y_train"])
    X_val = data["X_val"]
    y_val = _ensure_binary_labels(data["y_val"])
    X_test = data["X_test"]
    y_test = _ensure_binary_labels(data["y_test"])
    feature_names = data["feature_names"].tolist()

    # Threshold is stored as a length-1 array
    threshold_arr = data.get("threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "threshold": threshold,
    }


def _evaluate_split(model: XGBClassifier, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    return {
        "split": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def train_and_evaluate(dataset_path: str, output_dir: str, params_path: Optional[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]
    threshold = dataset["threshold"]

    params: Dict[str, Any] = {
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

    params_override = _load_params_json(params_path)
    if params_override:
        params.update(params_override)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)
    params.setdefault("eval_metric", "logloss")

    model = XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    metrics = [
        _evaluate_split(model, "train", X_train, y_train),
        _evaluate_split(model, "val", X_val, y_val),
        _evaluate_split(model, "test", X_test, y_test),
    ]

    metrics_by_split = {
        entry["split"]: {k: v for k, v in entry.items() if k != "split"}
        for entry in metrics
    }

    model_path = os.path.join(output_dir, "xgb_dir1h_model.json")
    model.save_model(model_path)

    metadata = {
        "model_type": "xgboost_classifier",
        "target": "direction_1h",
        "threshold": threshold,
        "feature_names": feature_names,
        "metrics": metrics,
        "params": params,
        "dataset_path": dataset_path,
    }

    meta_path = os.path.join(output_dir, "model_metadata_direction.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "model_type": "xgboost_classifier",
        "target": "direction_1h",
        "dataset_path": dataset_path,
        "threshold": threshold,
        "params": params,
        "metrics": metrics_by_split,
        "model_path": model_path,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved direction model to:", model_path)
    print("Saved metadata to:", meta_path)
    print("Saved summary to:", summary_path)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier for 1h BTC direction.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the npz file produced by build_training_dataset_direction.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/xgb_dir1h_v1",
        help="Directory to store the trained direction model and metadata",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional JSON file containing XGBoost hyperparameters to override defaults.",
    )
    args = parser.parse_args()

    train_and_evaluate(args.dataset_path, args.output_dir, args.params_json)


if __name__ == "__main__":
    main()
