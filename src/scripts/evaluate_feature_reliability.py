from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _feature_score(series: pd.Series, baseline_window: int, recent_window: int) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
    n = len(s)
    if n < 4:
        return {
            "missing_ratio": float(s.isna().mean()),
            "drift": 999.0,
            "score": 0.0,
        }

    recent_n = min(int(recent_window), max(n // 2, 1))
    baseline_pool = s.iloc[:-recent_n]
    baseline_n = min(int(baseline_window), len(baseline_pool))
    baseline = baseline_pool.iloc[-baseline_n:] if baseline_n > 0 else baseline_pool
    recent = s.iloc[-recent_n:]
    if baseline.empty or recent.empty:
        return {
            "missing_ratio": float(s.isna().mean()),
            "drift": 999.0,
            "score": 0.0,
        }

    b_mean = _safe_float(baseline.mean(), 0.0)
    r_mean = _safe_float(recent.mean(), 0.0)
    b_std = max(_safe_float(baseline.std(ddof=0), 0.0), 1e-8)
    drift = abs(r_mean - b_mean) / b_std

    missing_ratio = float(s.isna().mean())

    # Reliability score in [0,1], penalizing missingness and normalized drift.
    score = max(0.0, 1.0 - min(1.0, 0.7 * missing_ratio + 0.3 * min(drift / 3.0, 1.0)))
    return {
        "missing_ratio": float(missing_ratio),
        "drift": float(drift),
        "score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute feature reliability scores from recent drift and missingness.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet/csv with feature columns.")
    parser.add_argument("--baseline-window", type=int, default=240)
    parser.add_argument("--recent-window", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--max-features", type=int, default=0, help="Optional cap on accepted feature count (0 disables).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/feature_reliability.json"))
    args = parser.parse_args()

    if args.input.suffix.lower() == ".npz":
        with np.load(args.input, allow_pickle=True) as data:
            required = {"X_train", "X_val", "X_test", "feature_names"}
            missing = required.difference(set(data.files))
            if missing:
                raise KeyError(f"Dataset NPZ missing required keys: {sorted(missing)}")
            X_all = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
            feature_names = [str(v) for v in data["feature_names"].tolist()]
            df = pd.DataFrame(X_all, columns=feature_names)
    elif args.input.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    feature_cols: List[str] = [
        c
        for c in df.columns
        if c not in {"ts", "timestamp", "y", "y_true", "ret_1h", "horizon", "signal_ensemble", "ret_pred", "p_up"}
    ]
    scores: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        scores[col] = _feature_score(
            df[col],
            baseline_window=int(args.baseline_window),
            recent_window=int(args.recent_window),
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    accepted = [name for name, meta in ranked if meta["score"] >= float(args.min_score)]
    if int(args.max_features) > 0:
        accepted = accepted[: int(args.max_features)]

    payload = {
        "rows": int(len(df)),
        "settings": {
            "baseline_window": int(args.baseline_window),
            "recent_window": int(args.recent_window),
            "min_score": float(args.min_score),
            "max_features": int(args.max_features),
        },
        "accepted_features": accepted,
        "feature_scores": scores,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
