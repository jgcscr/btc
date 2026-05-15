from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
from joblib import dump as joblib_dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.model_summary import build_model_summary, write_model_summary


DEFAULT_FEATURE_PATTERNS: Sequence[str] = (
    "cvd_",
    "liquidity_",
    "distance_from_session_",
    "vwap_deviation_",
    "momentum_slope_",
    "macro_",
    "funding_",
    "onchain_",
    "volatility_",
    "interaction_chop_volatility_",
    "interaction_momentum_volatility_",
    "interaction_regime_",
    "intrabar_volume_regime_",
    "intrabar_return_dispersion_regime_",
    "range_compression_",
    "trend_path_efficiency_",
    "trend_regime_strength_",
    "volume_regime_",
)


def _required_keys(horizon: int, *, use_flat_labels: bool) -> set[str]:
    base = {"X_train", "X_val", "X_test", "feature_names"}
    if use_flat_labels:
        base.update({"y_train", "y_val", "y_test"})
    else:
        base.update(
            {
                f"y_dir{horizon}h_train",
                f"y_dir{horizon}h_val",
                f"y_dir{horizon}h_test",
            }
        )
    return base


def _load_dataset(dataset_path: str, horizon: int, *, use_flat_labels: bool) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    missing = [key for key in _required_keys(horizon, use_flat_labels=use_flat_labels) if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required keys for {horizon}h direction training: {missing}")

    threshold_arr = data.get("direction_threshold")
    if threshold_arr is None:
        threshold_arr = data.get("threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0

    if use_flat_labels:
        y_train_key = "y_train"
        y_val_key = "y_val"
        y_test_key = "y_test"
    else:
        y_train_key = f"y_dir{horizon}h_train"
        y_val_key = f"y_dir{horizon}h_val"
        y_test_key = f"y_dir{horizon}h_test"

    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": np.asarray(data[y_train_key], dtype=int),
        "y_val": np.asarray(data[y_val_key], dtype=int),
        "y_test": np.asarray(data[y_test_key], dtype=int),
        "feature_names": [str(name) for name in data["feature_names"].tolist()],
        "threshold": threshold,
    }


def _select_feature_indices(feature_names: Sequence[str], patterns: Sequence[str]) -> List[int]:
    lowered_patterns = [str(pattern).strip().lower() for pattern in patterns if str(pattern).strip()]
    selected: List[int] = []
    for index, raw_name in enumerate(feature_names):
        name = str(raw_name).strip()
        lowered = name.lower()
        if any(pattern in lowered for pattern in lowered_patterns):
            selected.append(index)
    return selected


def _evaluate_split(model: Pipeline, name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics: Dict[str, Any] = {
        "split": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(np.unique(y)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
    return metrics


def train_and_evaluate(
    dataset_path: str,
    output_dir: str,
    horizon: int,
    *,
    suffix: str | None = None,
    use_flat_labels: bool = False,
    feature_patterns: Sequence[str] = DEFAULT_FEATURE_PATTERNS,
    min_features: int = 8,
    c_value: float = 0.35,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset(dataset_path, horizon=horizon, use_flat_labels=use_flat_labels)
    feature_names = dataset["feature_names"]
    selected_indices = _select_feature_indices(feature_names, feature_patterns)
    if len(selected_indices) < min_features:
        raise ValueError(
            f"regime_logit selected only {len(selected_indices)} features from {len(feature_names)} total; "
            f"minimum required is {min_features}."
        )

    selected_feature_names = [feature_names[index] for index in selected_indices]
    X_train = np.asarray(dataset["X_train"][:, selected_indices], dtype=float)
    X_val = np.asarray(dataset["X_val"][:, selected_indices], dtype=float)
    X_test = np.asarray(dataset["X_test"][:, selected_indices], dtype=float)
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    y_test = dataset["y_test"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    solver="liblinear",
                    penalty="l1",
                    C=float(c_value),
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    metrics = [
        _evaluate_split(model, "train", X_train, y_train),
        _evaluate_split(model, "val", X_val, y_val),
        _evaluate_split(model, "test", X_test, y_test),
    ]
    metrics_by_split = {
        entry["split"]: {key: value for key, value in entry.items() if key != "split"}
        for entry in metrics
    }

    classifier = model.named_steps["classifier"]
    coef = classifier.coef_[0]
    nonzero_features = [
        {"feature": name, "coefficient": float(weight)}
        for name, weight in zip(selected_feature_names, coef)
        if abs(float(weight)) > 1e-9
    ]
    nonzero_features.sort(key=lambda entry: abs(entry["coefficient"]), reverse=True)

    resolved_suffix = suffix or f"{horizon}h"
    model_path = os.path.join(output_dir, f"regime_logit_dir{resolved_suffix}_model.joblib")
    joblib_dump({"model": model, "feature_names": selected_feature_names}, model_path)

    metadata = {
        "model_type": "regime_logit_classifier",
        "target": f"direction_{resolved_suffix}",
        "horizon_hours": horizon,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": dataset["threshold"],
        "feature_patterns": list(feature_patterns),
        "feature_names": selected_feature_names,
        "selected_feature_count": len(selected_feature_names),
        "nonzero_feature_count": len(nonzero_features),
        "nonzero_features": nonzero_features,
        "metrics": metrics,
        "params": {
            "model": "logistic_regression_l1",
            "C": float(c_value),
            "class_weight": "balanced",
            "solver": "liblinear",
        },
        "dataset_path": dataset_path,
    }
    meta_path = os.path.join(output_dir, "model_metadata_direction.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = build_model_summary(
        model_type="regime_logit_classifier",
        target=f"direction_{resolved_suffix}",
        dataset_path=dataset_path,
        model_path=model_path,
        metrics=metrics_by_split,
        feature_names=selected_feature_names,
        params=metadata["params"],
        threshold=dataset["threshold"],
        horizon_hours=horizon,
        trained_at=metadata["trained_at"],
    )
    summary_path = os.path.join(output_dir, "summary.json")
    write_model_summary(summary_path, summary)

    print(f"Saved {horizon}h regime_logit direction model to: {model_path}")
    print("Saved metadata to:", meta_path)
    print("Saved summary to:", summary_path)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a sparse regime-focused logistic direction model from the existing direction datasets.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the direction dataset NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/regime_logit_dir1h_v1",
        help="Directory to store the trained regime_logit model and metadata.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in hours (e.g. 1, 4, 8, 12).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for model naming (e.g. 15m or 4h).",
    )
    parser.add_argument(
        "--use-flat-labels",
        action="store_true",
        help="Use y_train/y_val/y_test labels instead of y_dir{h}h_* keys.",
    )
    parser.add_argument(
        "--feature-pattern",
        action="append",
        default=list(DEFAULT_FEATURE_PATTERNS),
        help="Substring pattern used to select regime features. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=8,
        help="Minimum number of selected features required to fit the model.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.35,
        help="Inverse regularization strength for the sparse logistic model.",
    )
    args = parser.parse_args()

    if args.horizon <= 0:
        raise SystemExit("--horizon must be a positive integer")

    train_and_evaluate(
        args.dataset_path,
        args.output_dir,
        args.horizon,
        suffix=args.suffix,
        use_flat_labels=bool(args.use_flat_labels),
        feature_patterns=args.feature_pattern,
        min_features=int(args.min_features),
        c_value=float(args.c_value),
    )


if __name__ == "__main__":
    main()