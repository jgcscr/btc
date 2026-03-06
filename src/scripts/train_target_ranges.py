"""Train regressors that forecast forward high/low log returns for BTC."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import joblib
import numpy as np
from xgboost import XGBRegressor

TARGET_RANGE_MODEL_DIR = Path("artifacts/models/target_ranges")
DEFAULT_HORIZONS: Tuple[int, ...] = (4, 8, 12)


def _load_dataset(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found at {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[XGBRegressor, Dict[str, float]]:
    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    residuals = y_val - val_pred
    metrics = {
        "val_mae": float(np.mean(np.abs(residuals))),
        "val_rmse": float(np.sqrt(np.mean(np.square(residuals))))
        if residuals.size
        else 0.0,
        "val_residual_std": float(np.std(residuals)) if residuals.size else 0.0,
    }
    return model, metrics


def _ensure_targets(data: Dict[str, np.ndarray], horizon: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_prefix = f"y_ret{kind}{horizon}h"
    required = [f"{key_prefix}_train", f"{key_prefix}_val", f"{key_prefix}_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset missing target arrays for horizon {horizon}h ({', '.join(missing)})")
    return data[required[0]], data[required[1]], data[required[2]]


def train_target_range_models(
    dataset_path: Path,
    output_dir: Path,
    horizons: Sequence[int],
) -> None:
    data = _load_dataset(dataset_path)
    feature_names = data.get("feature_names")
    if feature_names is None:
        raise KeyError("Dataset NPZ missing feature_names array")
    feature_names = [str(name) for name in feature_names]

    X_train = data["X_train"]
    X_val = data["X_val"]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Dict[str, Dict[str, float]]] = {}

    for horizon in horizons:
        train_max, val_max, _ = _ensure_targets(data, horizon, "max")
        train_min, val_min, _ = _ensure_targets(data, horizon, "min")

        max_model, max_metrics = _train_regressor(X_train, train_max, X_val, val_max)
        min_model, min_metrics = _train_regressor(X_train, train_min, X_val, val_min)

        label = f"{horizon}h"
        metadata[label] = {
            "high": max_metrics,
            "low": min_metrics,
        }

        payload_high = {
            "model": max_model,
            "feature_names": feature_names,
            "metrics": max_metrics,
        }
        payload_low = {
            "model": min_model,
            "feature_names": feature_names,
            "metrics": min_metrics,
        }

        joblib.dump(payload_high, output_dir / f"{label}_high.joblib")
        joblib.dump(payload_low, output_dir / f"{label}_low.joblib")
        print(f"Saved target-range models for {label} to {output_dir}")

    meta_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "feature_count": len(feature_names),
        "horizons": metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote target-range metadata to {output_dir / 'metadata.json'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train target-range regressors for BTC price projections.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("artifacts/datasets/btc_features_multi_horizon_splits.npz"),
        help="Path to the multi-horizon dataset NPZ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TARGET_RANGE_MODEL_DIR,
        help="Directory to store trained target-range models",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Horizons (hours) to train high/low regressors for",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_target_range_models(args.dataset_path, args.output_dir, args.horizons)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
