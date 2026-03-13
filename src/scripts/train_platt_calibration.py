from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.model_artifact_selection import resolve_best_versioned_model_file


def _format_horizon_label(horizon: float) -> str:
    if float(horizon).is_integer() and horizon >= 1.0:
        return f"{int(round(horizon))}h"
    if horizon < 1.0:
        return f"{int(round(horizon * 60))}m"
    return f"{horizon:g}h"


def _resolve_xgb_dir_model_path(model_root: str, horizon: float) -> str:
    suffix = _format_horizon_label(horizon)
    model_dir = Path(model_root) / f"xgb_dir{suffix}_v1"
    model_path = resolve_best_versioned_model_file(
        model_dir,
        expected_filename=f"xgb_dir{suffix}_model.json",
        version_priority=("v2", "v1"),
    )
    if model_path.exists():
        return str(model_path)
    raise FileNotFoundError(f"xgb_dir model not found for {suffix} under {model_root}")


def _load_dataset(dataset_path: str, horizon: float) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    if horizon <= 1 and {"X_val", "y_val"}.issubset(data.files):
        feature_names = [str(name) for name in data.get("feature_names", [])]
        return {
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "feature_names": feature_names,
        }
    label = _format_horizon_label(horizon)
    key = f"y_dir{label}_val"
    if key not in data.files:
        key = f"y_ret{label}_val"
    if key not in data.files:
        raise KeyError(f"Dataset missing {key}")
    feature_names = [str(name) for name in data.get("feature_names", [])]
    return {
        "X_val": data["X_val"],
        "y_val": data[key],
        "feature_names": feature_names,
    }


def _load_model_feature_names(model_path: str) -> List[str]:
    meta_path = Path(model_path).with_name("model_metadata_direction.json")
    if not meta_path.exists():
        return []
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _align_features(X: np.ndarray, dataset_names: List[str], model_names: List[str]) -> np.ndarray:
    if not model_names or not dataset_names:
        return X
    if len(dataset_names) == len(model_names) and dataset_names == model_names:
        return X

    name_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
    aligned = np.zeros((X.shape[0], len(model_names)), dtype=X.dtype)
    for out_idx, name in enumerate(model_names):
        in_idx = name_to_idx.get(name)
        if in_idx is not None and in_idx < X.shape[1]:
            aligned[:, out_idx] = X[:, in_idx]
    return aligned


def _fit_platt(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(logit, y.astype(int))
    a = float(clf.coef_.reshape(-1)[0])
    b = float(clf.intercept_.reshape(-1)[0])
    return {"a": a, "b": b}


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(p, y.astype(int))
    x = np.asarray(getattr(model, "X_thresholds_", []), dtype=float)
    yhat = np.asarray(getattr(model, "y_thresholds_", []), dtype=float)
    if x.size == 0 or yhat.size == 0:
        # Fallback to identity when thresholds are unavailable.
        x = np.array([0.0, 1.0], dtype=float)
        yhat = np.array([0.0, 1.0], dtype=float)
    return {"x": x.tolist(), "y": yhat.tolist()}


def _fit_beta_calibration(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    # Beta calibration: sigma(a*log(p) + b*log(1-p) + c)
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    X = np.column_stack([np.log(p), np.log(1.0 - p)])
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X, y.astype(int))
    coef = clf.coef_.reshape(-1)
    intercept = float(clf.intercept_.reshape(-1)[0])
    return {"a": float(coef[0]), "b": float(coef[1]), "c": intercept}


def _fit_calibrator(p: np.ndarray, y: np.ndarray, method: str) -> Dict[str, Any]:
    method_norm = str(method).strip().lower()
    if method_norm == "platt":
        params = _fit_platt(p, y)
        params["method"] = "platt"
        return params
    if method_norm == "isotonic":
        params = _fit_isotonic(p, y)
        params["method"] = "isotonic"
        return params
    if method_norm == "beta":
        params = _fit_beta_calibration(p, y)
        params["method"] = "beta"
        return params
    raise ValueError(f"Unsupported calibration method: {method}")


def _fit_regime_calibration_from_labeled_csv(
    path: str,
    *,
    regime_col: str,
    min_rows: int,
    method: str,
) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"p_up", "y_true"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Regime calibration input missing required columns: {missing}")

    has_regime = regime_col in df.columns
    if not has_regime:
        # Labeled-input augmentation is strictly for regime-specific keys.
        return {}

    if "horizon" not in df.columns:
        df["horizon"] = "1h"

    out: Dict[str, Dict[str, Any]] = {}
    for horizon, horizon_df in df.groupby("horizon"):
        for regime, g in horizon_df.groupby(regime_col):
            y_r = pd.to_numeric(g["y_true"], errors="coerce")
            p_r = pd.to_numeric(g["p_up"], errors="coerce")
            m_r = y_r.notna() & p_r.notna()
            yy_r = y_r[m_r].to_numpy(dtype=float).astype(int)
            pp_r = p_r[m_r].to_numpy(dtype=float)
            if yy_r.size < min_rows or np.unique(yy_r).size <= 1:
                continue
            out[f"{horizon}@{regime}"] = _fit_calibrator(pp_r, yy_r, method)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt scaling coefficients for XGBoost direction models.")
    parser.add_argument(
        "--model-root",
        type=str,
        default="artifacts/models",
        help="Root directory containing xgb_dir*h* model folders.",
    )
    parser.add_argument(
        "--dataset-1h",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Direction dataset path for 1h horizon.",
    )
    parser.add_argument(
        "--dataset-15m",
        type=str,
        default="artifacts/datasets/btc_features_15m_direction_splits.npz",
        help="Direction dataset path for 15m horizon.",
    )
    parser.add_argument(
        "--dataset-multi",
        type=str,
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        help="Multi-horizon dataset path for 4h/8h/12h horizons.",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[0.25, 1, 4, 8, 12],
        help="Horizons to calibrate (default: 1 4 8 12).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/models/platt_calibration.json",
        help="Output JSON path for calibration coefficients.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("platt", "isotonic", "beta"),
        default="platt",
        help="Calibration method to fit for horizon-wide and regime-aware entries.",
    )
    parser.add_argument(
        "--labeled-input",
        type=str,
        default=None,
        help=(
            "Optional labeled CSV (p_up,y_true,regime_state[,horizon]) used to fit "
            "additional regime-aware calibration entries like '1h@trend_ignition'."
        ),
    )
    parser.add_argument(
        "--regime-col",
        type=str,
        default="regime_state",
        help="Column in --labeled-input used as regime identifier.",
    )
    parser.add_argument(
        "--min-regime-rows",
        type=int,
        default=100,
        help="Minimum labeled rows required per horizon@regime for calibration fit.",
    )
    args = parser.parse_args()

    output: Dict[str, Dict[str, Any]] = {}
    for horizon in args.horizons:
        if horizon < 1:
            dataset_path = args.dataset_15m
        elif horizon == 1:
            dataset_path = args.dataset_1h
        else:
            dataset_path = args.dataset_multi
        model_path = _resolve_xgb_dir_model_path(args.model_root, horizon)

        dataset = _load_dataset(dataset_path, horizon)
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        y_val = np.asarray(y_val, dtype=float)
        unique_y = np.unique(y_val[~np.isnan(y_val)]) if y_val.size else np.asarray([])
        if unique_y.size > 0 and not np.all(np.isin(unique_y, [0.0, 1.0])):
            y_val = (y_val > 0.0).astype(int)
        dataset_feature_names = [str(name) for name in dataset.get("feature_names", [])]

        model = XGBClassifier()
        model._estimator_type = "classifier"
        model.load_model(model_path)
        if not hasattr(model, "classes_"):
            model.classes_ = np.array([0, 1])

        model_feature_names = _load_model_feature_names(model_path)
        X_val = _align_features(X_val, dataset_feature_names, model_feature_names)

        p_val = model.predict_proba(X_val)[:, 1]
        params = _fit_calibrator(p_val, y_val, args.method)
        label = _format_horizon_label(horizon)
        output[label] = params
        if args.method == "platt":
            print(f"Calibrated {label} (platt): a={params['a']:.4f} b={params['b']:.4f}")
        elif args.method == "beta":
            print(
                "Calibrated "
                f"{label} (beta): a={params['a']:.4f} b={params['b']:.4f} c={params['c']:.4f}"
            )
        else:
            print(f"Calibrated {label} (isotonic): {len(params.get('x', []))} knots")

    if args.labeled_input:
        extra = _fit_regime_calibration_from_labeled_csv(
            args.labeled_input,
            regime_col=args.regime_col,
            min_rows=max(int(args.min_regime_rows), 20),
            method=args.method,
        )
        if extra:
            output.update(extra)
            print(f"Added {len(extra)} horizon/regime calibration entries from labeled input.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved calibration to {args.output_path}")


if __name__ == "__main__":
    main()
