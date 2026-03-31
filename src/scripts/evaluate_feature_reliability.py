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


def _format_horizon_key(value: Any) -> str | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    if numeric >= 1.0 and float(numeric).is_integer():
        return f"{int(numeric)}h"
    return f"{int(round(numeric * 60))}m"


def _derive_regime_state(df: pd.DataFrame) -> pd.Series:
    if "regime_state" in df.columns:
        return df["regime_state"].fillna("unknown").astype(str).str.strip().str.lower()

    volatility = pd.to_numeric(df.get("volatility_realized_24h"), errors="coerce")
    range_expansion = pd.to_numeric(df.get("range_expansion_1h"), errors="coerce")
    momentum = pd.to_numeric(df.get("momentum_slope_4h"), errors="coerce")
    ignition = pd.to_numeric(df.get("trend_ignition_6h"), errors="coerce")

    regime = pd.Series("neutral", index=df.index, dtype=object)
    trend_mask = (
        ignition.fillna(0.0) >= 0.55
    ) | (
        momentum.abs().fillna(0.0) >= volatility.fillna(0.0).clip(lower=0.001)
    )
    chop_mask = (
        range_expansion.fillna(0.0) <= 0.9
    ) & (
        momentum.abs().fillna(0.0) <= volatility.fillna(0.0).clip(lower=0.001) * 0.75
    )
    regime.loc[trend_mask] = "trend"
    regime.loc[chop_mask & ~trend_mask] = "chop"
    return regime.astype(str)


def _compute_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    baseline_window: int,
    recent_window: int,
) -> Dict[str, Dict[str, float]]:
    return {
        col: _feature_score(
            df[col],
            baseline_window=baseline_window,
            recent_window=recent_window,
        )
        for col in feature_cols
    }


def _accepted_from_scores(
    scores: Dict[str, Dict[str, float]],
    *,
    min_score: float,
    max_features: int,
) -> List[str]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    accepted = [name for name, meta in ranked if meta["score"] >= float(min_score)]
    if int(max_features) > 0:
        accepted = accepted[: int(max_features)]
    return accepted


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute feature reliability scores from recent drift and missingness.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet/csv with feature columns.")
    parser.add_argument("--baseline-window", type=int, default=240)
    parser.add_argument("--recent-window", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--max-features", type=int, default=0, help="Optional cap on accepted feature count (0 disables).")
    parser.add_argument("--horizon", type=float, default=None, help="Optional fixed horizon for the evaluated frame.")
    parser.add_argument("--horizon-col", type=str, default="horizon", help="Column containing horizon labels when evaluating tabular inputs.")
    parser.add_argument("--regime-col", type=str, default="regime_state", help="Column containing regime labels when available.")
    parser.add_argument("--derive-regime", action="store_true", help="Derive a simple trend/chop/neutral regime when no regime column exists.")
    parser.add_argument("--min-slice-rows", type=int, default=80, help="Minimum rows required to emit horizon/regime slice scores.")
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
    scores = _compute_scores(
        df,
        feature_cols,
        baseline_window=int(args.baseline_window),
        recent_window=int(args.recent_window),
    )
    accepted = _accepted_from_scores(
        scores,
        min_score=float(args.min_score),
        max_features=int(args.max_features),
    )

    accepted_by_horizon: Dict[str, List[str]] = {}
    horizon_feature_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    accepted_by_horizon_regime: Dict[str, Dict[str, List[str]]] = {}
    horizon_regime_feature_scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    horizon_series = None
    if args.horizon is not None:
        horizon_key = _format_horizon_key(args.horizon)
        if horizon_key is not None:
            horizon_series = pd.Series(horizon_key, index=df.index, dtype=object)
    elif args.horizon_col in df.columns:
        horizon_series = df[args.horizon_col].map(_format_horizon_key)

    regime_series = None
    if args.regime_col in df.columns:
        regime_series = df[args.regime_col].fillna("unknown").astype(str).str.strip().str.lower()
    elif args.derive_regime:
        regime_series = _derive_regime_state(df)

    if horizon_series is not None:
        working = df.copy()
        working["_horizon_key"] = horizon_series
        if regime_series is not None:
            working["_regime_key"] = regime_series

        for horizon_key, horizon_frame in working.groupby("_horizon_key"):
            if not isinstance(horizon_key, str) or not horizon_key or len(horizon_frame) < int(args.min_slice_rows):
                continue
            slice_scores = _compute_scores(
                horizon_frame,
                feature_cols,
                baseline_window=int(args.baseline_window),
                recent_window=int(args.recent_window),
            )
            horizon_feature_scores[horizon_key] = slice_scores
            accepted_by_horizon[horizon_key] = _accepted_from_scores(
                slice_scores,
                min_score=float(args.min_score),
                max_features=int(args.max_features),
            )

            if "_regime_key" not in horizon_frame.columns:
                continue
            regime_payload: Dict[str, List[str]] = {}
            regime_scores_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
            for regime_key, regime_frame in horizon_frame.groupby("_regime_key"):
                if not isinstance(regime_key, str) or not regime_key or len(regime_frame) < int(args.min_slice_rows):
                    continue
                regime_scores = _compute_scores(
                    regime_frame,
                    feature_cols,
                    baseline_window=int(args.baseline_window),
                    recent_window=int(args.recent_window),
                )
                regime_scores_payload[regime_key] = regime_scores
                regime_payload[regime_key] = _accepted_from_scores(
                    regime_scores,
                    min_score=float(args.min_score),
                    max_features=int(args.max_features),
                )
            if regime_payload:
                accepted_by_horizon_regime[horizon_key] = regime_payload
                horizon_regime_feature_scores[horizon_key] = regime_scores_payload

    payload = {
        "rows": int(len(df)),
        "settings": {
            "baseline_window": int(args.baseline_window),
            "recent_window": int(args.recent_window),
            "min_score": float(args.min_score),
            "max_features": int(args.max_features),
            "min_slice_rows": int(args.min_slice_rows),
        },
        "accepted_features": accepted,
        "feature_scores": scores,
        "accepted_features_by_horizon": accepted_by_horizon,
        "horizon_feature_scores": horizon_feature_scores,
        "accepted_features_by_horizon_regime": accepted_by_horizon_regime,
        "horizon_regime_feature_scores": horizon_regime_feature_scores,
    }

    safe_payload = _json_safe(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe_payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(safe_payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
