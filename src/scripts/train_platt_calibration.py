from __future__ import annotations

import argparse
import json
import math
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


def _apply_calibrator(p: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    method = str(params.get("method") or "platt").strip().lower()
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    if method == "platt":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        z = a * np.log(p_clip / (1.0 - p_clip)) + b
        return 1.0 / (1.0 + np.exp(-z))
    if method == "beta":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", -1.0))
        c = float(params.get("c", 0.0))
        z = a * np.log(p_clip) + b * np.log(1.0 - p_clip) + c
        return 1.0 / (1.0 + np.exp(-z))
    if method == "isotonic":
        x = np.asarray(params.get("x", []), dtype=float)
        y = np.asarray(params.get("y", []), dtype=float)
        if x.size >= 2 and y.size == x.size:
            return np.interp(p_clip, x, y, left=y[0], right=y[-1])
    return p_clip


def _coerce_horizon_hours(label: str) -> float | None:
    raw = str(label or "").strip().lower()
    if not raw:
        return None
    if raw.endswith("m"):
        try:
            minutes = float(raw[:-1])
        except ValueError:
            return None
        return minutes / 60.0
    if raw.endswith("h"):
        try:
            return float(raw[:-1])
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def _bearish_remap_guard(
    p: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    *,
    horizon_label: str,
    higher_horizon_min_hours: float,
    min_bearish_rows: int,
    neutral_band: float,
    max_allowed_uplift: float,
) -> Dict[str, Any]:
    horizon_hours = _coerce_horizon_hours(horizon_label)
    if horizon_hours is None or horizon_hours < float(higher_horizon_min_hours):
        return {"applied": False, "accepted": True, "reason": "below_higher_horizon_threshold"}

    raw_probability = np.asarray(p, dtype=float)
    labels = np.asarray(y, dtype=int)
    bearish_mask = (labels == 0) & (raw_probability < (0.5 - float(neutral_band)))
    bearish_rows = int(np.sum(bearish_mask))
    if bearish_rows < int(min_bearish_rows):
        return {
            "applied": False,
            "accepted": True,
            "reason": "insufficient_bearish_rows",
            "bearish_rows": bearish_rows,
        }

    calibrated_probability = _apply_calibrator(raw_probability[bearish_mask], params)
    bullish_flip_mask = calibrated_probability > (0.5 + float(neutral_band))
    bullish_flip_count = int(np.sum(bullish_flip_mask))
    uplift = calibrated_probability - raw_probability[bearish_mask]
    uplift_violation_mask = uplift > float(max_allowed_uplift)
    uplift_violation_count = int(np.sum(uplift_violation_mask))
    accepted = bullish_flip_count == 0 and uplift_violation_count == 0
    return {
        "applied": True,
        "accepted": accepted,
        "reason": (
            "pass"
            if accepted
            else "bullish_remap_on_bearish_aligned_rows"
            if bullish_flip_count > 0
            else "excessive_bullish_uplift_on_bearish_aligned_rows"
        ),
        "bearish_rows": bearish_rows,
        "bullish_flip_count": bullish_flip_count,
        "bullish_flip_rate": float(bullish_flip_count / bearish_rows) if bearish_rows else 0.0,
        "uplift_violation_count": uplift_violation_count,
        "uplift_violation_rate": float(uplift_violation_count / bearish_rows) if bearish_rows else 0.0,
        "max_uplift_on_bearish_rows": float(np.max(uplift)) if uplift.size else None,
        "mean_uplift_on_bearish_rows": float(np.mean(uplift)) if uplift.size else None,
        "max_calibrated_probability_on_bearish_rows": float(np.max(calibrated_probability)) if calibrated_probability.size else None,
    }


def _summarize_regime_calibration_coverage_from_labeled_csv(
    path: str,
    *,
    regime_col: str,
    min_rows: int,
) -> Dict[str, Any]:
    df = pd.read_csv(path)
    coverage: Dict[str, Any] = {
        "enabled": True,
        "path": path,
        "regime_col": regime_col,
        "min_rows": int(min_rows),
        "rows": int(len(df)),
        "has_regime_col": regime_col in df.columns,
        "has_horizon_col": "horizon" in df.columns,
        "default_horizon_applied": "horizon" not in df.columns,
        "eligible_entries": [],
        "ineligible_entries": [],
    }
    required = {"p_up", "y_true"}
    missing_required = sorted(column for column in required if column not in df.columns)
    coverage["missing_required_columns"] = missing_required
    if missing_required or regime_col not in df.columns:
        coverage["reason"] = "missing_required_columns" if missing_required else "missing_regime_col"
        return coverage

    working = df.copy()
    if "horizon" not in working.columns:
        working["horizon"] = "1h"

    for horizon, horizon_df in working.groupby("horizon"):
        for regime, group in horizon_df.groupby(regime_col):
            y_values = pd.to_numeric(group["y_true"], errors="coerce")
            p_values = pd.to_numeric(group["p_up"], errors="coerce")
            mask = y_values.notna() & p_values.notna()
            labels = y_values[mask].to_numpy(dtype=float).astype(int)
            row_summary = {
                "horizon": str(horizon),
                "regime_state": str(regime),
                "rows": int(labels.size),
                "class_count": int(np.unique(labels).size) if labels.size else 0,
            }
            if labels.size < min_rows:
                row_summary["eligible"] = False
                row_summary["reason"] = "insufficient_rows"
                coverage["ineligible_entries"].append(row_summary)
            elif np.unique(labels).size <= 1:
                row_summary["eligible"] = False
                row_summary["reason"] = "single_class"
                coverage["ineligible_entries"].append(row_summary)
            else:
                row_summary["eligible"] = True
                row_summary["reason"] = "ready"
                coverage["eligible_entries"].append(row_summary)

    coverage["eligible_entry_count"] = int(len(coverage["eligible_entries"]))
    coverage["ineligible_entry_count"] = int(len(coverage["ineligible_entries"]))
    coverage["reason"] = "ready" if coverage["eligible_entries"] else "no_eligible_entries"
    return coverage


def _fit_regime_calibration_from_labeled_csv(
    path: str,
    *,
    regime_col: str,
    min_rows: int,
    method: str,
    higher_horizon_min_hours: float = 4.0,
    bearish_guard_min_rows: int = 3,
    bearish_guard_neutral_band: float = 0.0,
    bearish_guard_max_allowed_uplift: float = 0.05,
    suppressed_entries: List[Dict[str, Any]] | None = None,
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
            params = _fit_calibrator(pp_r, yy_r, method)
            guard = _bearish_remap_guard(
                pp_r,
                yy_r,
                params,
                horizon_label=str(horizon),
                higher_horizon_min_hours=higher_horizon_min_hours,
                min_bearish_rows=bearish_guard_min_rows,
                neutral_band=bearish_guard_neutral_band,
                max_allowed_uplift=bearish_guard_max_allowed_uplift,
            )
            if not bool(guard.get("accepted", True)):
                if suppressed_entries is not None:
                    suppressed_entries.append(
                        {
                            "key": f"{horizon}@{regime}",
                            "scope": "regime",
                            "horizon": str(horizon),
                            "regime_state": str(regime),
                            **guard,
                        }
                    )
                continue
            out[f"{horizon}@{regime}"] = params
    return out


def _fit_base_horizon_calibration_from_labeled_csv(
    path: str,
    *,
    min_rows: int,
    method: str,
    higher_horizon_min_hours: float = 4.0,
    bearish_guard_min_rows: int = 3,
    bearish_guard_neutral_band: float = 0.0,
    bearish_guard_max_allowed_uplift: float = 0.05,
    suppressed_entries: List[Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"p_up", "y_true"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Labeled calibration input missing required columns: {missing}")

    if "horizon" not in df.columns:
        df["horizon"] = "1h"

    out: Dict[str, Dict[str, Any]] = {}
    for horizon, horizon_df in df.groupby("horizon"):
        y_r = pd.to_numeric(horizon_df["y_true"], errors="coerce")
        p_r = pd.to_numeric(horizon_df["p_up"], errors="coerce")
        mask = y_r.notna() & p_r.notna()
        yy_r = y_r[mask].to_numpy(dtype=float).astype(int)
        pp_r = p_r[mask].to_numpy(dtype=float)
        if yy_r.size < min_rows or np.unique(yy_r).size <= 1:
            continue
        params = _fit_calibrator(pp_r, yy_r, method)
        guard = _bearish_remap_guard(
            pp_r,
            yy_r,
            params,
            horizon_label=str(horizon),
            higher_horizon_min_hours=higher_horizon_min_hours,
            min_bearish_rows=bearish_guard_min_rows,
            neutral_band=bearish_guard_neutral_band,
            max_allowed_uplift=bearish_guard_max_allowed_uplift,
        )
        if not bool(guard.get("accepted", True)):
            if suppressed_entries is not None:
                suppressed_entries.append(
                    {
                        "key": str(horizon),
                        "scope": "base",
                        "horizon": str(horizon),
                        **guard,
                    }
                )
            continue
        out[str(horizon)] = params
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
        "--fit-base-horizons-from-labeled-input",
        action="store_true",
        help=(
            "Also fit base horizon keys like '1h' and '4h' from --labeled-input instead of only adding horizon@regime keys."
        ),
    )
    parser.add_argument(
        "--skip-model-fit",
        action="store_true",
        help="Skip model-validation calibration and emit only labeled-input-derived calibrators.",
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
    parser.add_argument(
        "--min-regime-rows-floor",
        type=int,
        default=20,
        help="Hard lower bound applied to --min-regime-rows (default: 20). Use lower values only for controlled experiments.",
    )
    parser.add_argument(
        "--coverage-output-path",
        type=str,
        default=None,
        help="Optional JSON path for regime-calibration coverage diagnostics.",
    )
    parser.add_argument(
        "--bearish-remap-guard-min-rows",
        type=int,
        default=3,
        help="Minimum bearish aligned rows (y_true=0 with raw p_up<0.5) required before higher-horizon remap suppression is enforced.",
    )
    parser.add_argument(
        "--bearish-remap-guard-neutral-band",
        type=float,
        default=0.0,
        help="Neutral band around 0.5 used by the higher-horizon bearish remap guard.",
    )
    parser.add_argument(
        "--bearish-remap-guard-min-hours",
        type=float,
        default=4.0,
        help="Minimum horizon in hours where the bearish remap guard is enforced for labeled-input calibration fits.",
    )
    parser.add_argument(
        "--bearish-remap-guard-max-allowed-uplift",
        type=float,
        default=0.05,
        help="Maximum allowed calibrated uplift on bearish aligned rows before a higher-horizon labeled-input calibration entry is suppressed.",
    )
    args = parser.parse_args()
    regime_min_rows = max(int(args.min_regime_rows), int(args.min_regime_rows_floor))

    output: Dict[str, Dict[str, Any]] = {}
    coverage_payload: Dict[str, Any] = {
        "enabled": bool(args.labeled_input),
        "reason": "no_labeled_input" if not args.labeled_input else "pending",
    }
    suppressed_entries: List[Dict[str, Any]] = []
    if not args.skip_model_fit:
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
        if args.fit_base_horizons_from_labeled_input:
            base_extra = _fit_base_horizon_calibration_from_labeled_csv(
                args.labeled_input,
                min_rows=regime_min_rows,
                method=args.method,
                higher_horizon_min_hours=float(args.bearish_remap_guard_min_hours),
                bearish_guard_min_rows=max(int(args.bearish_remap_guard_min_rows), 1),
                bearish_guard_neutral_band=max(float(args.bearish_remap_guard_neutral_band), 0.0),
                bearish_guard_max_allowed_uplift=max(float(args.bearish_remap_guard_max_allowed_uplift), 0.0),
                suppressed_entries=suppressed_entries,
            )
            if base_extra:
                output.update(base_extra)
                print(f"Added {len(base_extra)} base horizon calibration entries from labeled input.")
        coverage_payload = _summarize_regime_calibration_coverage_from_labeled_csv(
            args.labeled_input,
            regime_col=args.regime_col,
            min_rows=regime_min_rows,
        )
        extra = _fit_regime_calibration_from_labeled_csv(
            args.labeled_input,
            regime_col=args.regime_col,
            min_rows=regime_min_rows,
            method=args.method,
            higher_horizon_min_hours=float(args.bearish_remap_guard_min_hours),
            bearish_guard_min_rows=max(int(args.bearish_remap_guard_min_rows), 1),
            bearish_guard_neutral_band=max(float(args.bearish_remap_guard_neutral_band), 0.0),
            bearish_guard_max_allowed_uplift=max(float(args.bearish_remap_guard_max_allowed_uplift), 0.0),
            suppressed_entries=suppressed_entries,
        )
        if extra:
            output.update(extra)
            print(f"Added {len(extra)} horizon/regime calibration entries from labeled input.")
        elif coverage_payload.get("reason") == "ready":
            coverage_payload["reason"] = "ready_but_no_entries_emitted"
        coverage_payload["bearish_remap_guard"] = {
            "enabled": True,
            "min_rows": max(int(args.bearish_remap_guard_min_rows), 1),
            "neutral_band": max(float(args.bearish_remap_guard_neutral_band), 0.0),
            "min_hours": float(args.bearish_remap_guard_min_hours),
            "max_allowed_uplift": max(float(args.bearish_remap_guard_max_allowed_uplift), 0.0),
            "suppressed_entries": suppressed_entries,
            "suppressed_entry_count": int(len(suppressed_entries)),
        }
        if suppressed_entries:
            print(f"Suppressed {len(suppressed_entries)} labeled-input calibration entries via bearish remap guard.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved calibration to {args.output_path}")
    if args.coverage_output_path:
        os.makedirs(os.path.dirname(args.coverage_output_path) or ".", exist_ok=True)
        with open(args.coverage_output_path, "w", encoding="utf-8") as handle:
            json.dump(coverage_payload, handle, indent=2)
        print(f"Saved regime calibration coverage to {args.coverage_output_path}")


if __name__ == "__main__":
    main()
