import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


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


def _required_keys(horizon: int, *, use_flat_labels: bool) -> set[str]:
    base = {
        "X_train",
        "X_val",
        "X_test",
        "feature_names",
    }
    if use_flat_labels:
        base.update({"y_train", "y_val", "y_test"})
    else:
        base.update(
            {
                f"y_ret{horizon}h_train",
                f"y_ret{horizon}h_val",
                f"y_ret{horizon}h_test",
            }
        )
    return base


def _load_dataset(dataset_path: str, horizon: int, *, use_flat_labels: bool) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    missing = [key for key in _required_keys(horizon, use_flat_labels=use_flat_labels) if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required keys for {horizon}h training: {missing}")

    if use_flat_labels:
        y_train_key = "y_train"
        y_val_key = "y_val"
        y_test_key = "y_test"
    else:
        y_train_key = f"y_ret{horizon}h_train"
        y_val_key = f"y_ret{horizon}h_val"
        y_test_key = f"y_ret{horizon}h_test"

    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data[y_train_key],
        "y_val": data[y_val_key],
        "y_test": data[y_test_key],
        "feature_names": data["feature_names"].tolist(),
    }


def _evaluate_split(model: XGBRegressor, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    return {"split": name, "rmse": rmse, "mae": mae}


def train_and_evaluate(
    dataset_path: str,
    output_dir: str,
    params_json: Optional[str],
    horizon: int,
    *,
    suffix: Optional[str] = None,
    use_flat_labels: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path, horizon=horizon, use_flat_labels=use_flat_labels)
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]

    params: Dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }

    params_override = _load_params_json(params_json)
    if params_override:
        params.update(params_override)
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "regressor"

    metrics = [
        _evaluate_split(model, "train", X_train, y_train),
        _evaluate_split(model, "val", X_val, y_val),
        _evaluate_split(model, "test", X_test, y_test),
    ]

    metrics_by_split = {
        entry["split"]: {k: v for k, v in entry.items() if k != "split"}
        for entry in metrics
    }

    resolved_suffix = suffix or f"{horizon}h"
    model_filename = f"xgb_ret{resolved_suffix}_model.json"
    model_path = os.path.join(output_dir, model_filename)
    model.save_model(model_path)

    metadata = {
        "model_type": "xgboost_regressor",
        "target": f"ret_{resolved_suffix}",
        "horizon_hours": horizon,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "metrics": metrics,
        "params": params,
        "dataset_path": dataset_path,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metadata_simple = {
        "model_type": "xgboost_regressor",
        "horizon_hours": horizon,
        "target": f"ret_{resolved_suffix}",
        "feature_names": feature_names,
        "trained_at": metadata["trained_at"],
        "model_path": model_path,
        "dataset_path": dataset_path,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata_simple, handle, indent=2)

    summary = {
        "model_type": "xgboost_regressor",
        "target": f"ret_{resolved_suffix}",
        "dataset_path": dataset_path,
        "feature_names": feature_names,
        "params": params,
        "metrics": metrics_by_split,
        "model_path": model_path,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved {resolved_suffix} regression model to:", model_path)
    print("Saved metadata to:", meta_path)
    print("Saved summary to:", summary_path)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost regressor for multi-horizon BTC returns.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        help="Path to the multi-horizon dataset produced by build_training_dataset_multi_horizon.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store the trained model and metadata",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional JSON file containing XGBoost hyperparameters to override defaults.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in hours (e.g., 1, 4, 8, 12).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for model naming (e.g., '15m' to produce xgb_ret15m_model.json).",
    )
    parser.add_argument(
        "--use-flat-labels",
        action="store_true",
        help="Use y_train/y_val/y_test labels instead of y_ret{h}h_* keys (for 1h/15m datasets).",
    )
    args = parser.parse_args()

    if args.horizon <= 0:
        raise SystemExit("--horizon must be a positive integer")

    output_dir = args.output_dir or f"artifacts/models/xgb_ret{args.horizon}h_v1"

    train_and_evaluate(
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        params_json=args.params_json,
        horizon=args.horizon,
        suffix=args.suffix,
        use_flat_labels=args.use_flat_labels,
    )


if __name__ == "__main__":
    main()
