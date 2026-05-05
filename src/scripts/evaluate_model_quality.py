from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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


def _fold_stability(metric_values: List[float]) -> Dict[str, float]:
    if not metric_values:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
    arr = np.asarray(metric_values, dtype=float)
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr, ddof=0))
    cv = float(s / abs(m)) if abs(m) > 1e-12 else float("inf")
    return {"mean": m, "std": s, "cv": cv}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibration and stability for direction probabilities.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with p_up and y_true columns.")
    parser.add_argument("--p-col", type=str, default="p_up")
    parser.add_argument("--y-col", type=str, default="y_true")
    parser.add_argument("--fold-col", type=str, default="fold")
    parser.add_argument(
        "--signal-col",
        type=str,
        default="signal_ensemble",
        help="Optional signal column used to compute trade_count. Falls back to p-col threshold if missing.",
    )
    parser.add_argument(
        "--trade-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold on p-col when signal-col is unavailable.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/model_quality.json"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input, low_memory=False)
    if args.p_col not in df.columns or args.y_col not in df.columns:
        raise ValueError(f"Input must contain {args.p_col} and {args.y_col}")

    p = pd.to_numeric(df[args.p_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[args.y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 1e-6, 1.0 - 1e-6)
    y = y[mask].astype(int)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p))
    nll = float(log_loss(y, p))
    ece = _expected_calibration_error(y, p, bins=10)

    fold_metrics: Dict[str, float] = {}
    if args.fold_col in df.columns:
        grouped = df.loc[mask].groupby(args.fold_col)
        auc_values: List[float] = []
        for _, g in grouped:
            yy = pd.to_numeric(g[args.y_col], errors="coerce").to_numpy(dtype=float).astype(int)
            pp = np.clip(pd.to_numeric(g[args.p_col], errors="coerce").to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            if len(np.unique(yy)) > 1:
                auc_values.append(float(roc_auc_score(yy, pp)))
        fold_metrics = _fold_stability(auc_values)

    trade_count = 0
    trade_count_method = "none"
    if args.signal_col in df.columns:
        signal = pd.to_numeric(df[args.signal_col], errors="coerce").fillna(0.0)
        trade_count = int((signal > 0).sum())
        trade_count_method = f"signal_col:{args.signal_col}"
    else:
        # Fallback proxy for datasets that only contain probabilities.
        trade_count = int((p >= float(args.trade_threshold)).sum())
        trade_count_method = f"p_col_threshold:{args.p_col}>={float(args.trade_threshold)}"

    payload = {
        "rows": int(len(y)),
        "trade_count": int(trade_count),
        "trade_count_method": trade_count_method,
        "auc": auc,
        "brier": brier,
        "log_loss": nll,
        "ece_10": ece,
        "fold_auc_stability": fold_metrics,
    }

    if "ret_ensemble_net" in df.columns:
        ret_net = pd.to_numeric(df["ret_ensemble_net"], errors="coerce").fillna(0.0)
        payload["net_return_total"] = float(ret_net.sum())
        payload["net_return_mean"] = float(ret_net.mean())
        if args.signal_col in df.columns:
            signal = pd.to_numeric(df[args.signal_col], errors="coerce").fillna(0.0)
            active = signal > 0
            if int(active.sum()) > 0:
                payload["net_return_per_trade_mean"] = float(ret_net[active].mean())
            else:
                payload["net_return_per_trade_mean"] = float("nan")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
