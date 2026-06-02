from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_META_PATH = "artifacts/datasets/btc_features_multi_horizon_meta.json"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_dataset_shift_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_dataset_shift_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose 4h train/val/test dataset shift and label drift from the multi-horizon NPZ."
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _describe_binary(y: np.ndarray) -> Dict[str, Any]:
    yf = np.asarray(y, dtype=float)
    pos_rate = float(np.mean(yf)) if yf.size else float("nan")
    return {
        "rows": int(yf.size),
        "positive_rate": pos_rate,
        "negative_rate": float(1.0 - pos_rate) if pos_rate == pos_rate else float("nan"),
    }


def _standardized_mean_shift(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_mean = np.nanmean(reference, axis=0)
    cand_mean = np.nanmean(candidate, axis=0)
    ref_std = np.nanstd(reference, axis=0)
    cand_std = np.nanstd(candidate, axis=0)
    pooled = np.sqrt(np.maximum((ref_std ** 2 + cand_std ** 2) / 2.0, 1e-12))
    return (cand_mean - ref_mean) / pooled


def _top_shift_rows(feature_names: List[str], shifts: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
    order = np.argsort(np.abs(shifts))[::-1]
    rows: List[Dict[str, Any]] = []
    for idx in order[:top_k]:
        rows.append(
            {
                "feature": str(feature_names[idx]),
                "standardized_mean_shift": float(shifts[idx]),
                "absolute_shift": float(abs(shifts[idx])),
            }
        )
    return rows


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["# 4h Dataset Shift Diagnostic", ""]
    lines.append("## Label Balance")
    for split_name in ("train", "val", "test"):
        split = payload["label_balance"][split_name]
        lines.append(
            f"- {split_name}: rows={split['rows']}, positive_rate={split['positive_rate']:.4f}, negative_rate={split['negative_rate']:.4f}"
        )
    lines.append("")
    lines.append("## Largest Train->Val Shifts")
    for row in payload["feature_shift"]["train_to_val_top"]:
        lines.append(f"- {row['feature']}: {row['standardized_mean_shift']:.4f}")
    lines.append("")
    lines.append("## Largest Train->Test Shifts")
    for row in payload["feature_shift"]["train_to_test_top"]:
        lines.append(f"- {row['feature']}: {row['standardized_mean_shift']:.4f}")
    lines.append("")
    lines.append("## Recommendations")
    for note in payload.get("recommendations", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path)
    meta_path = Path(args.meta_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    data = np.load(dataset_path, allow_pickle=True)
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_train = np.asarray(data["y_dir4h_train"], dtype=np.float32)
    y_val = np.asarray(data["y_dir4h_val"], dtype=np.float32)
    y_test = np.asarray(data["y_dir4h_test"], dtype=np.float32)
    feature_names = [str(name) for name in data["feature_names"].tolist()]

    train_to_val_shift = _standardized_mean_shift(X_train, X_val)
    train_to_test_shift = _standardized_mean_shift(X_train, X_test)

    meta = _read_json(meta_path) if meta_path.exists() else {}
    label_balance = {
        "train": _describe_binary(y_train),
        "val": _describe_binary(y_val),
        "test": _describe_binary(y_test),
    }
    positive_rate_drift_val = abs(label_balance["val"]["positive_rate"] - label_balance["train"]["positive_rate"])
    positive_rate_drift_test = abs(label_balance["test"]["positive_rate"] - label_balance["train"]["positive_rate"])

    recommendations: List[str] = []
    if positive_rate_drift_val > 0.05 or positive_rate_drift_test > 0.05:
        recommendations.append(
            "4h label prevalence drifts materially across contiguous splits; consider shorter or rolling training windows for trust-sensitive models."
        )
    if float(np.nanmax(np.abs(train_to_val_shift))) > 0.5 or float(np.nanmax(np.abs(train_to_test_shift))) > 0.5:
        recommendations.append(
            "Several 4h features show large standardized mean shifts between train and newer windows; audit those features or consider regime-aware retraining slices."
        )
    if not recommendations:
        recommendations.append("No major split-level drift was detected by this simple standardized-mean diagnostic.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(dataset_path),
            "meta_path": str(meta_path),
            "top_k": int(args.top_k),
        },
        "dataset_meta": {
            "row_count": meta.get("row_count"),
            "feature_count": meta.get("feature_count"),
            "splits": meta.get("splits"),
        },
        "label_balance": label_balance,
        "label_balance_drift": {
            "train_to_val_positive_rate_abs": float(positive_rate_drift_val),
            "train_to_test_positive_rate_abs": float(positive_rate_drift_test),
        },
        "feature_shift": {
            "train_to_val_top": _top_shift_rows(feature_names, train_to_val_shift, top_k=int(args.top_k)),
            "train_to_test_top": _top_shift_rows(feature_names, train_to_test_shift, top_k=int(args.top_k)),
            "train_to_val_max_abs": float(np.nanmax(np.abs(train_to_val_shift))),
            "train_to_test_max_abs": float(np.nanmax(np.abs(train_to_test_shift))),
        },
        "recommendations": recommendations,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote dataset shift JSON: {output_json}")
    print(f"Wrote dataset shift memo: {output_md}")


if __name__ == "__main__":
    main()