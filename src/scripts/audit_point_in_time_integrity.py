from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_TOKENS = (
    "future",
    "lead",
    "label",
    "target",
    "ret_1h",
    "ret_4h",
    "ret_8h",
    "ret_12h",
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _split_ts(ts_all: np.ndarray, n_train: int, n_val: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ts_series = pd.to_datetime(ts_all, utc=True, errors="coerce")
    ts_train = pd.Series(ts_series[:n_train])
    ts_val = pd.Series(ts_series[n_train : n_train + n_val])
    ts_test = pd.Series(ts_series[n_train + n_val :])
    return ts_train, ts_val, ts_test


def _audit(dataset_path: Path, y_key: str, leakage_corr_alert: float) -> Dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as data:
        required = {"X_train", "X_val", "X_test", "feature_names"}
        missing = [k for k in sorted(required) if k not in data.files]
        if missing:
            raise KeyError(f"Dataset missing keys: {missing}")

        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        if y_key in data.files:
            y_all = data[y_key]
        else:
            split_keys = [f"{y_key}_train", f"{y_key}_val", f"{y_key}_test"]
            if all(k in data.files for k in split_keys):
                y_all = np.concatenate([data[split_keys[0]], data[split_keys[1]], data[split_keys[2]]], axis=0)
            else:
                # Common fallback naming (y_train/y_val/y_test).
                fallback = ["y_train", "y_val", "y_test"]
                if all(k in data.files for k in fallback):
                    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
                else:
                    raise KeyError(
                        f"Dataset missing full target key '{y_key}' and split keys {split_keys}"
                    )
        feature_names = [str(v) for v in data["feature_names"].tolist()]

        X_all = np.concatenate([X_train, X_val, X_test], axis=0)
        n_train = int(X_train.shape[0])
        n_val = int(X_val.shape[0])

        ts_available = "ts_all" in data.files
        ts_report: Dict[str, Any] = {
            "available": bool(ts_available),
            "train_monotonic": None,
            "val_monotonic": None,
            "test_monotonic": None,
            "train_val_gap_hours": None,
            "val_test_gap_hours": None,
            "duplicates": None,
            "nonpositive_step_count": None,
        }
        if ts_available:
            ts_train, ts_val, ts_test = _split_ts(data["ts_all"], n_train, n_val)
            ts_full = pd.concat([ts_train, ts_val, ts_test], ignore_index=True)
            step_hours = ts_full.diff().dt.total_seconds() / 3600.0
            ts_report.update(
                {
                    "train_monotonic": bool(ts_train.dropna().is_monotonic_increasing),
                    "val_monotonic": bool(ts_val.dropna().is_monotonic_increasing),
                    "test_monotonic": bool(ts_test.dropna().is_monotonic_increasing),
                    "train_val_gap_hours": float((ts_val.iloc[0] - ts_train.iloc[-1]).total_seconds() / 3600.0)
                    if (len(ts_train) and len(ts_val) and pd.notna(ts_train.iloc[-1]) and pd.notna(ts_val.iloc[0]))
                    else None,
                    "val_test_gap_hours": float((ts_test.iloc[0] - ts_val.iloc[-1]).total_seconds() / 3600.0)
                    if (len(ts_val) and len(ts_test) and pd.notna(ts_val.iloc[-1]) and pd.notna(ts_test.iloc[0]))
                    else None,
                    "duplicates": int(ts_full.duplicated().sum()),
                    "nonpositive_step_count": int((step_hours.fillna(1.0) <= 0.0).sum()),
                }
            )

    suspicious_name_features = [
        name for name in feature_names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]

    y = np.asarray(y_all, dtype=float).reshape(-1)
    lead_corrs: List[Dict[str, Any]] = []
    if X_all.shape[0] == y.shape[0] and X_all.shape[0] >= 4:
        y_lead = y[1:]
        X_now = X_all[:-1]
        for idx, name in enumerate(feature_names):
            c = _safe_corr(np.asarray(X_now[:, idx], dtype=float), y_lead)
            if np.isfinite(c):
                lead_corrs.append({"feature": name, "abs_corr": abs(float(c)), "corr": float(c)})

    lead_corrs_sorted = sorted(lead_corrs, key=lambda d: d["abs_corr"], reverse=True)
    top_corrs = lead_corrs_sorted[:20]
    leakage_alerts = [d for d in top_corrs if float(d["abs_corr"]) >= leakage_corr_alert]

    alerts: List[str] = []
    if suspicious_name_features:
        alerts.append("suspicious_feature_names")
    if leakage_alerts:
        alerts.append("high_lead_correlation")
    if ts_report.get("duplicates", 0) and int(ts_report["duplicates"]) > 0:
        alerts.append("duplicate_timestamps")
    if ts_report.get("nonpositive_step_count", 0) and int(ts_report["nonpositive_step_count"]) > 0:
        alerts.append("nonpositive_time_steps")

    return {
        "dataset_path": str(dataset_path),
        "y_key": y_key,
        "rows": int(X_all.shape[0]),
        "feature_count": int(len(feature_names)),
        "timestamp_integrity": ts_report,
        "suspicious_feature_names": suspicious_name_features,
        "top_lead_correlations": top_corrs,
        "leakage_corr_alert_threshold": float(leakage_corr_alert),
        "leakage_alerts": leakage_alerts,
        "ok": len(alerts) == 0,
        "alerts": alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit point-in-time integrity and leakage risk on prepared datasets.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--leakage-corr-alert", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/point_in_time_audit.json"))
    args = parser.parse_args()

    report = _audit(args.dataset_path, y_key=args.y_key, leakage_corr_alert=float(args.leakage_corr_alert))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
