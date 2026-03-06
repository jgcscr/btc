from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def _expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += (np.sum(mask) / max(n, 1)) * abs(acc - conf)
    return float(ece)


def _metrics(df: pd.DataFrame, p_col: str, y_col: str) -> Dict[str, float]:
    p = np.clip(pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask].astype(int)
    if len(y) == 0:
        return {"rows": 0, "auc": float("nan"), "brier": float("nan"), "ece_10": float("nan")}

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p))
    ece = _expected_calibration_error(y, p, bins=10)
    return {"rows": int(len(y)), "auc": auc, "brier": brier, "ece_10": ece}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration robustness and drift by horizon/time windows.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical labeled CSV with p_up/y_true and ts.")
    parser.add_argument("--p-col", type=str, default="p_up")
    parser.add_argument("--y-col", type=str, default="y_true")
    parser.add_argument("--ts-col", type=str, default="ts")
    parser.add_argument("--horizon-col", type=str, default="horizon")
    parser.add_argument("--default-horizon", type=str, default="1h")
    parser.add_argument("--baseline-window", type=int, default=240)
    parser.add_argument("--recent-window", type=int, default=120)
    parser.add_argument("--max-ece-drift", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/calibration_robustness.json"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    for col in [args.p_col, args.y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if args.ts_col in df.columns:
        df[args.ts_col] = pd.to_datetime(df[args.ts_col], utc=True, errors="coerce")
    else:
        df[args.ts_col] = pd.NaT

    if args.horizon_col not in df.columns:
        df[args.horizon_col] = args.default_horizon

    df = df.dropna(subset=[args.p_col, args.y_col]).copy()
    if df.empty:
        raise RuntimeError("No valid rows for calibration robustness evaluation")

    horizon_reports: Dict[str, Dict[str, object]] = {}
    for horizon, group in df.groupby(args.horizon_col):
        g = group.sort_values(args.ts_col).copy()
        overall = _metrics(g, args.p_col, args.y_col)

        recent_n = min(int(args.recent_window), max(len(g) // 2, 1))
        recent = g.iloc[-recent_n:]
        baseline_pool = g.iloc[:-recent_n]
        baseline_n = min(int(args.baseline_window), len(baseline_pool))
        baseline = baseline_pool.iloc[-baseline_n:] if baseline_n > 0 else baseline_pool.iloc[0:0]

        baseline_m = _metrics(baseline, args.p_col, args.y_col) if not baseline.empty else {"rows": 0, "auc": float("nan"), "brier": float("nan"), "ece_10": float("nan")}
        recent_m = _metrics(recent, args.p_col, args.y_col) if not recent.empty else {"rows": 0, "auc": float("nan"), "brier": float("nan"), "ece_10": float("nan")}

        ece_drift = float(recent_m["ece_10"] - baseline_m["ece_10"]) if baseline_m["rows"] > 0 and recent_m["rows"] > 0 else float("nan")
        horizon_reports[str(horizon)] = {
            "overall": overall,
            "baseline": baseline_m,
            "recent": recent_m,
            "ece_drift": ece_drift,
            "ece_drift_alert": bool(np.isfinite(ece_drift) and ece_drift > float(args.max_ece_drift)),
        }

    payload = {
        "rows": int(len(df)),
        "settings": {
            "baseline_window": int(args.baseline_window),
            "recent_window": int(args.recent_window),
            "max_ece_drift": float(args.max_ece_drift),
        },
        "horizons": horizon_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
