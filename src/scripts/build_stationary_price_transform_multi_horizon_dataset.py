from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


TRANSFORM_SPECS: Dict[str, tuple[str, str]] = {
    "open": ("open", "fut_open"),
    "high": ("high", "close"),
    "low": ("low", "close"),
    "close": ("close", "open"),
    "fut_open": ("fut_open", "fut_close"),
    "fut_high": ("fut_high", "fut_close"),
    "fut_low": ("fut_low", "fut_close"),
    "fut_close": ("fut_close", "close"),
    "intrabar_open_first": ("intrabar_open_first", "open"),
    "intrabar_close_last": ("intrabar_close_last", "close"),
    "intrabar_path_high": ("intrabar_path_high", "close"),
    "intrabar_path_low": ("intrabar_path_low", "close"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-horizon dataset where raw price-level columns are replaced by stationary log-ratio transforms."
    )
    parser.add_argument("--input-path", default="artifacts/datasets/btc_features_multi_horizon_splits.npz")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--eps", type=float, default=1e-9)
    return parser.parse_args()


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
    num = np.maximum(np.asarray(numerator, dtype=np.float64), eps)
    den = np.maximum(np.asarray(denominator, dtype=np.float64), eps)
    return np.log(num / den)


def _restore_raw(scaled: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(scaled, dtype=np.float64) * scale + mean


def _scale_from_train(raw: np.ndarray, train_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = raw[:train_rows]
    mean = np.nanmean(train_raw, axis=0)
    scale = np.nanstd(train_raw, axis=0)
    safe_scale = np.where(np.isclose(scale, 0.0), 1.0, scale)
    scaled = (raw - mean) / safe_scale
    return scaled.astype(np.float32), mean.astype(np.float32), safe_scale.astype(np.float32)


def _apply_transforms(raw: np.ndarray, feature_names: list[str], eps: float) -> tuple[np.ndarray, Dict[str, str]]:
    transformed = np.asarray(raw, dtype=np.float64).copy()
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    formulas: Dict[str, str] = {}

    missing = sorted(
        {
            required
            for target, (num_name, den_name) in TRANSFORM_SPECS.items()
            for required in (target, num_name, den_name)
            if required not in index_by_name
        }
    )
    if missing:
        raise KeyError(f"Dataset is missing required price-level features: {missing}")

    for target, (num_name, den_name) in TRANSFORM_SPECS.items():
        target_idx = index_by_name[target]
        num_idx = index_by_name[num_name]
        den_idx = index_by_name[den_name]
        transformed[:, target_idx] = _safe_log_ratio(raw[:, num_idx], raw[:, den_idx], eps)
        formulas[target] = f"log({num_name}/{den_name})"
    return transformed, formulas


def _load_static_payload(data: Mapping[str, Any], *, feature_count: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in data.keys():
        if key in {"X_train", "X_val", "X_test", "scaler_mean", "scaler_scale"}:
            continue
        value = data[key]
        if key == "feature_names":
            if len(value) != feature_count:
                raise ValueError("feature_names length does not match feature matrix width")
            payload[key] = value
        else:
            payload[key] = value
    return payload


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".json")

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    with np.load(input_path, allow_pickle=True) as data:
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        scaled_train = np.asarray(data["X_train"], dtype=np.float64)
        scaled_val = np.asarray(data["X_val"], dtype=np.float64)
        scaled_test = np.asarray(data["X_test"], dtype=np.float64)
        scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(data["scaler_scale"], dtype=np.float64)
        raw_all = _restore_raw(
            np.concatenate([scaled_train, scaled_val, scaled_test], axis=0),
            scaler_mean,
            scaler_scale,
        )

        transformed_raw, formulas = _apply_transforms(raw_all, feature_names, float(args.eps))
        rescaled_all, new_mean, new_scale = _scale_from_train(transformed_raw, train_rows=scaled_train.shape[0])
        n_train = scaled_train.shape[0]
        n_val = scaled_val.shape[0]

        payload = _load_static_payload(data, feature_count=len(feature_names))
        payload["X_train"] = rescaled_all[:n_train]
        payload["X_val"] = rescaled_all[n_train:n_train + n_val]
        payload["X_test"] = rescaled_all[n_train + n_val:]
        payload["scaler_mean"] = new_mean
        payload["scaler_scale"] = new_scale

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "transformed_feature_count": len(formulas),
        "transforms": formulas,
        "feature_count": len(feature_names),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote stationary-price dataset: {output_path}")
    print(f"Transformed {len(formulas)} price-level features.")
    print(f"Wrote transform report: {report_path}")


if __name__ == "__main__":
    main()