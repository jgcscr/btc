from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


BASELINE_PARAMS: Dict[str, Any] = {
    "n_estimators": 649,
    "max_depth": 4,
    "learning_rate": 0.014025052676221455,
    "subsample": 0.9016994271646056,
    "colsample_bytree": 0.6958478455944792,
    "min_child_weight": 1.9979490168766945,
    "gamma": 2.0855702856219374,
    "reg_lambda": 5.129453176718993,
    "reg_alpha": 2.6651084934334337,
    "scale_pos_weight": 1.1984828864641477,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a constrained 4h XGB regularization sweep on a prepared multi-horizon dataset."
    )
    parser.add_argument("--dataset-path", default="artifacts/datasets/btc_features_multi_horizon_splits.raw_price_levels_ablated.npz")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--output-json", default="artifacts/analysis/4h_xgb_regularization_diagnostic_latest.json")
    parser.add_argument("--output-md", default="artifacts/analysis/4h_xgb_regularization_diagnostic_latest.md")
    parser.add_argument("--max-train-val-gap", type=float, default=0.03)
    parser.add_argument("--min-test-accuracy", type=float, default=0.58)
    return parser.parse_args()


def _read_dataset(path: Path, horizon: int) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "X_train",
            "X_val",
            "X_test",
            f"y_dir{horizon}h_train",
            f"y_dir{horizon}h_val",
            f"y_dir{horizon}h_test",
        }
        missing = sorted(key for key in required if key not in data.files)
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")
        return {
            "X_train": np.asarray(data["X_train"], dtype=np.float32),
            "X_val": np.asarray(data["X_val"], dtype=np.float32),
            "X_test": np.asarray(data["X_test"], dtype=np.float32),
            "y_train": np.asarray(data[f"y_dir{horizon}h_train"], dtype=np.int32),
            "y_val": np.asarray(data[f"y_dir{horizon}h_val"], dtype=np.int32),
            "y_test": np.asarray(data[f"y_dir{horizon}h_test"], dtype=np.int32),
        }


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def _score_split(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    proba = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1.0 - 1e-12)
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": _safe_auc(y, proba),
        "logloss": float(log_loss(y, proba)),
    }


def _evaluate_profile(name: str, params: Mapping[str, Any], dataset: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    model = XGBClassifier(**dict(params))
    model.fit(
        dataset["X_train"],
        dataset["y_train"],
        eval_set=[(dataset["X_val"], dataset["y_val"])],
        verbose=False,
    )
    train = _score_split(model, dataset["X_train"], dataset["y_train"])
    val = _score_split(model, dataset["X_val"], dataset["y_val"])
    test = _score_split(model, dataset["X_test"], dataset["y_test"])
    return {
        "profile": name,
        "params": dict(params),
        "train": train,
        "val": val,
        "test": test,
        "train_val_gap": float(abs(train["accuracy"] - val["accuracy"])),
        "test_minus_val_accuracy": float(test["accuracy"] - val["accuracy"]),
    }


def _profiles() -> List[Dict[str, Any]]:
    baseline = dict(BASELINE_PARAMS)
    return [
        {"name": "baseline_ablated_best", "params": baseline},
        {
            "name": "shallower_depth",
            "params": {**baseline, "max_depth": 3, "min_child_weight": 4.0, "n_estimators": 500},
        },
        {
            "name": "higher_l1_l2",
            "params": {**baseline, "reg_alpha": 8.0, "reg_lambda": 12.0, "n_estimators": 500},
        },
        {
            "name": "lower_column_sampling",
            "params": {**baseline, "colsample_bytree": 0.45, "subsample": 0.75, "n_estimators": 500},
        },
        {
            "name": "conservative_combo",
            "params": {
                **baseline,
                "max_depth": 3,
                "min_child_weight": 6.0,
                "reg_alpha": 10.0,
                "reg_lambda": 15.0,
                "gamma": 3.0,
                "subsample": 0.7,
                "colsample_bytree": 0.45,
                "n_estimators": 450,
            },
        },
        {
            "name": "ultra_conservative",
            "params": {
                **baseline,
                "max_depth": 2,
                "min_child_weight": 10.0,
                "reg_alpha": 15.0,
                "reg_lambda": 20.0,
                "gamma": 4.0,
                "subsample": 0.65,
                "colsample_bytree": 0.35,
                "n_estimators": 350,
                "learning_rate": 0.02,
            },
        },
    ]


def _sort_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    gap = float(row["train_val_gap"])
    test_acc = float((row.get("test") or {}).get("accuracy") or 0.0)
    test_auc = float((row.get("test") or {}).get("auc") or 0.0)
    return (gap, -test_acc, -test_auc)


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h XGB Regularization Diagnostic", ""]
    lines.append("## Recommendation")
    for note in payload.get("recommendations", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Top Profiles")
    for row in payload.get("profiles_ranked", [])[:3]:
        lines.append(
            f"- {row['profile']}: gap={row['train_val_gap']:.4f}, test_accuracy={row['test']['accuracy']:.4f}, test_auc={row['test']['auc']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    dataset = _read_dataset(Path(args.dataset_path), horizon=int(args.horizon))
    rows = [_evaluate_profile(profile["name"], profile["params"], dataset) for profile in _profiles()]
    ranked = sorted(rows, key=_sort_key)
    best = ranked[0]

    recommendations: List[str] = []
    if float(best["train_val_gap"]) <= float(args.max_train_val_gap) and float(best["test"]["accuracy"]) >= float(args.min_test_accuracy):
        recommendations.append(
            f"Profile '{best['profile']}' clears the requested trust-gap and minimum test-accuracy thresholds; use it as the next staged 4h candidate."
        )
    else:
        recommendations.append(
            f"No constrained XGB profile cleared the requested trust-gap threshold of {float(args.max_train_val_gap):.2f}; further gains likely require training/objective changes rather than stronger static regularization alone."
        )
    recommendations.append(
        f"Best gap profile was '{best['profile']}' with train_val_gap={float(best['train_val_gap']):.4f}, test_accuracy={float(best['test']['accuracy']):.4f}, test_auc={best['test']['auc']}."
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(args.dataset_path),
            "horizon": int(args.horizon),
            "max_train_val_gap": float(args.max_train_val_gap),
            "min_test_accuracy": float(args.min_test_accuracy),
        },
        "profiles_ranked": ranked,
        "recommendations": recommendations,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote diagnostic JSON: {output_json}")
    print(f"Wrote diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()