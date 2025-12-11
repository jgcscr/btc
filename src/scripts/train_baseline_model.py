import argparse
import json
import os
from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


def _create_model():
    if XGBOOST_AVAILABLE:
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
        model_type = "xgboost"
    else:
        # Fallback to RandomForestRegressor if XGBoost is not available
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
        model_type = "random_forest"
    return model_type, model


def _evaluate_split(model, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    preds = model.predict(X)
    # Compute RMSE in a way that is compatible with older/newer sklearn versions
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    return {"split": name, "rmse": rmse, "mae": mae}


def train_and_evaluate(dataset_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]

    model_type, model = _create_model()

    if model_type == "xgboost":
        # XGBoost supports eval_set for basic monitoring
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
    else:
        # RandomForestRegressor has no eval_set
        model.fit(X_train, y_train)

    metrics = [
        _evaluate_split(model, "train", X_train, y_train),
        _evaluate_split(model, "val", X_val, y_val),
        _evaluate_split(model, "test", X_test, y_test),
    ]

    # Save model
    if model_type == "xgboost":
        model_filename = "xgb_ret1h_model.json"
        model_path = os.path.join(output_dir, model_filename)
        model.save_model(model_path)
    else:
        # Fallback: sklearn model via joblib
        from joblib import dump  # type: ignore

        model_filename = "rf_ret1h_model.joblib"
        model_path = os.path.join(output_dir, model_filename)
        dump(model, model_path)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "target": "ret_1h",
        "feature_names": feature_names,
        "metrics": metrics,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved model to:", model_path)
    print("Saved metadata to:", meta_path)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline regressor on BTC 1h returns.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the npz file produced by build_training_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/xgb_ret1h_v1",
        help="Directory to store the trained model and metadata",
    )
    args = parser.parse_args()

    train_and_evaluate(args.dataset_path, args.output_dir)


if __name__ == "__main__":
    main()
