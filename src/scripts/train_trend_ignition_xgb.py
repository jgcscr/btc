import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_1h_splits.npz"
DEFAULT_OUTPUT_DIR = Path("artifacts/models/trend_ignition")
MODEL_FILENAME = "xgb_trend_ignition.joblib"
META_FILENAME = "metadata.json"
SUMMARY_FILENAME = "summary.json"


def _load_params_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Params JSON not found: {payload_path}")
    with payload_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Params JSON must contain an object with hyperparameters.")
    return data


def _required_dataset_keys() -> set[str]:
    return {
        "X_train",
        "X_val",
        "X_test",
        "y_ignition_train",
        "y_ignition_val",
        "y_ignition_test",
        "feature_names",
    }


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    npz_path = Path(dataset_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    missing = [key for key in _required_dataset_keys() if key not in data]
    if missing:
        raise KeyError(f"Dataset missing required trend ignition keys: {missing}")

    feature_names = data["feature_names"].tolist()
    threshold = float(data.get("trend_ignition_threshold", np.array([0.01]))[0])
    horizon = float(data.get("trend_ignition_horizon", np.array([6.0]))[0])
    label_name_arr = data.get("trend_ignition_label")
    label_name = str(label_name_arr[0]) if label_name_arr is not None else "trend_ignition_6h"

    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_ignition_train"],
        "y_val": data["y_ignition_val"],
        "y_test": data["y_ignition_test"],
        "feature_names": feature_names,
        "threshold": threshold,
        "horizon": horizon,
        "label": label_name,
    }


def _evaluate_split(model: XGBClassifier, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if X.size == 0:
        return {
            "split": name,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
        }
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "split": name,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
    }


def train_and_evaluate(dataset_path: str, output_dir: str, params_json: Optional[str]) -> None:
    payload = _load_dataset(dataset_path)
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "logloss",
        "reg_lambda": 1.0,
    }
    params.update(_load_params_json(params_json))
    params.setdefault("objective", "binary:logistic")
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)
    params.setdefault("eval_metric", "logloss")

    model = XGBClassifier(**params)
    model.fit(
        payload["X_train"],
        payload["y_train"],
        eval_set=[
            (payload["X_train"], payload["y_train"]),
            (payload["X_val"], payload["y_val"]),
        ],
        verbose=False,
    )
    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "classifier"

    splits = [
        _evaluate_split(model, "train", payload["X_train"], payload["y_train"]),
        _evaluate_split(model, "val", payload["X_val"], payload["y_val"]),
        _evaluate_split(model, "test", payload["X_test"], payload["y_test"]),
    ]
    metrics = {
        entry["split"]: {k: v for k, v in entry.items() if k != "split"}
        for entry in splits
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "model": model,
        "feature_names": payload["feature_names"],
    }
    model_path = output_root / MODEL_FILENAME
    joblib.dump(model_payload, model_path)

    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_type": "xgboost_classifier",
        "label": payload["label"],
        "horizon_hours": payload["horizon"],
        "threshold": payload["threshold"],
        "feature_names": payload["feature_names"],
        "trained_at": trained_at,
        "dataset_path": dataset_path,
        "params": params,
        "metrics": metrics,
        "model_path": model_path.as_posix(),
    }
    meta_path = output_root / META_FILENAME
    meta_path.write_text(json.dumps(metadata, indent=2))

    summary = {
        "model": model_path.name,
        "positive_rate": float(payload["y_train"].mean()),
        "metrics": metrics,
        "label": payload["label"],
        "horizon_hours": payload["horizon"],
    }
    summary_path = output_root / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved trend ignition model to {model_path}")
    print(f"Wrote metadata to {meta_path}")
    print(json.dumps(summary, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier for the trend-ignition label.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to btc_features_1h_splits.npz produced by build_training_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store the trained model and metadata.",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional JSON file that overrides default XGBoost hyperparameters.",
    )
    args = parser.parse_args(argv)

    train_and_evaluate(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        params_json=args.params_json,
    )


if __name__ == "__main__":
    main()
