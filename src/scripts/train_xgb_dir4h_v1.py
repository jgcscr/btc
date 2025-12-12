import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


REQUIRED_KEYS = {
    "X_train",
    "X_val",
    "X_test",
    "y_dir4h_train",
    "y_dir4h_val",
    "y_dir4h_test",
    "feature_names",
}


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
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required keys for 4h direction training: {missing}")

    threshold_arr = data.get("direction_threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0

    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_dir4h_train"],
        "y_val": data["y_dir4h_val"],
        "y_test": data["y_dir4h_test"],
        "feature_names": data["feature_names"].tolist(),
        "threshold": threshold,
    }


def _evaluate_split(model: XGBClassifier, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "split": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def train_and_evaluate(dataset_path: str, output_dir: str, params_json: Optional[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
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

    params_override = _load_params_json(params_json)
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

    model_path = os.path.join(output_dir, "xgb_dir4h_model.json")
    model.save_model(model_path)

    metadata = {
        "model_type": "xgboost_classifier",
        "target": "direction_4h",
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
        "target": "direction_4h",
        "dataset_path": dataset_path,
        "threshold": threshold,
        "feature_names": feature_names,
        "params": params,
        "metrics": metrics_by_split,
        "model_path": model_path,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved 4h direction model to:", model_path)
    print("Saved metadata to:", meta_path)
    print("Saved summary to:", summary_path)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier for 4h BTC direction.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        help="Path to the multi-horizon dataset produced by build_training_dataset_multi_horizon.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/xgb_dir4h_v1",
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
