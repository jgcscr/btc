import argparse
import json
import os
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


REQUIRED_KEYS = {
    "X_train",
    "X_val",
    "X_test",
    "y_ret4h_train",
    "y_ret4h_val",
    "y_ret4h_test",
    "feature_names",
}


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required keys for 4h training: {missing}")

    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_ret4h_train"],
        "y_val": data["y_ret4h_val"],
        "y_test": data["y_ret4h_test"],
        "feature_names": data["feature_names"].tolist(),
    }


def _evaluate_split(model: XGBRegressor, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    return {"split": name, "rmse": rmse, "mae": mae}


def train_and_evaluate(dataset_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

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

    model_filename = "xgb_ret4h_model.json"
    model_path = os.path.join(output_dir, model_filename)
    model.save_model(model_path)

    metadata = {
        "model_type": "xgboost_regressor",
        "target": "ret_4h",
        "feature_names": feature_names,
        "metrics": metrics,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved 4h regression model to:", model_path)
    print("Saved metadata to:", meta_path)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost regressor for 4h BTC returns.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        help="Path to the multi-horizon dataset produced by build_training_dataset_multi_horizon.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/xgb_ret4h_v1",
        help="Directory to store the trained model and metadata",
    )
    args = parser.parse_args()

    train_and_evaluate(args.dataset_path, args.output_dir)


if __name__ == "__main__":
    main()
