from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    valid = y_true.notna() & score.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    y = y_true.loc[valid].astype(int)
    if int(y.nunique()) < 2:
        return float("nan")
    return float(roc_auc_score(y, score.loc[valid].astype(float)))


def _ece_10(y_true: pd.Series, score: pd.Series) -> float:
    valid = y_true.notna() & score.notna()
    if int(valid.sum()) == 0:
        return float("nan")
    y = y_true.loc[valid].astype(float)
    p = score.loc[valid].astype(float).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, 11)
    total = float(len(y))
    err = 0.0
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < len(edges) - 2:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        err += (count / total) * abs(float(p.loc[mask].mean()) - float(y.loc[mask].mean()))
    return float(err)


def _normalize_regime(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text or "unknown"


def _summary(df: pd.DataFrame, *, label_col: str, score_col: str, return_col: str) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "auc": float("nan"),
            "ece_10": float("nan"),
            "net_return_total": 0.0,
            "hit_rate": float("nan"),
        }
    labels = pd.to_numeric(df[label_col], errors="coerce")
    scores = pd.to_numeric(df[score_col], errors="coerce")
    returns = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
    return {
        "rows": int(len(df)),
        "auc": _safe_auc(labels, scores),
        "ece_10": _ece_10(labels, scores),
        "net_return_total": float(returns.sum()),
        "hit_rate": float((returns > 0.0).mean()) if len(df) else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the recent signal-active slice of a candidate CSV and report regime and threshold-neighborhood diagnostics."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-window", type=int, default=288)
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--label-col", type=str, default="y_true")
    parser.add_argument("--score-col", type=str, default="p_up")
    parser.add_argument("--return-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--regime-col", type=str, default="regime_state")
    parser.add_argument("--threshold", type=float, default=0.555)
    parser.add_argument("--threshold-bands", type=str, default="0.54:0.57,0.57:0.60,0.60:")
    return parser.parse_args()


def _parse_threshold_bands(raw_value: str) -> List[Dict[str, float | None | str]]:
    bands: List[Dict[str, float | None | str]] = []
    for raw_band in str(raw_value).split(","):
        token = raw_band.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        lo = float(parts[0]) if parts[0].strip() else None
        hi = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        label = token.replace(":", "-") if hi is not None else f">={lo}"
        bands.append({"label": label, "low": lo, "high": hi})
    return bands


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    required = [args.signal_col, args.label_col, args.score_col, args.return_col]
    for column in required:
        if column not in df.columns:
            raise KeyError(f"Missing required column: {column}")

    recent_df = df.tail(max(0, int(args.recent_window))).copy()
    signal = pd.to_numeric(recent_df[args.signal_col], errors="coerce").fillna(0.0)
    recent_df["_signal_active"] = signal != 0.0
    active_df = recent_df.loc[recent_df["_signal_active"]].copy()
    active_df["_regime"] = active_df.get(args.regime_col, pd.Series(index=active_df.index, dtype=object)).map(_normalize_regime)
    score = pd.to_numeric(active_df[args.score_col], errors="coerce")

    threshold_bands = []
    for band in _parse_threshold_bands(args.threshold_bands):
        low = band["low"]
        high = band["high"]
        if low is None and high is None:
            mask = pd.Series(True, index=active_df.index)
        elif low is None:
            mask = score < float(high)
        elif high is None:
            mask = score >= float(low)
        else:
            mask = (score >= float(low)) & (score < float(high))
        threshold_bands.append(
            {
                "label": band["label"],
                "rows": int(mask.sum()),
                "trade_count": int((pd.to_numeric(active_df.loc[mask, args.signal_col], errors="coerce").fillna(0.0) != 0.0).sum()),
            }
        )

    near_threshold = {
        "threshold": float(args.threshold),
        "rows_ge_threshold": int((score >= float(args.threshold)).sum()),
        "rows_in_threshold_to_plus_0p005": int(((score >= float(args.threshold)) & (score < float(args.threshold) + 0.005)).sum()),
        "rows_in_threshold_to_plus_0p015": int(((score >= float(args.threshold)) & (score < float(args.threshold) + 0.015)).sum()),
    }

    by_regime: List[Dict[str, Any]] = []
    for regime_value, group in active_df.groupby("_regime", dropna=False):
        row = {"regime_state": str(regime_value)}
        row.update(_summary(group, label_col=args.label_col, score_col=args.score_col, return_col=args.return_col))
        by_regime.append(row)
    by_regime.sort(key=lambda item: (-item["rows"], str(item["regime_state"])))

    payload = {
        "input": str(args.input),
        "recent_window": int(args.recent_window),
        "signal_col": str(args.signal_col),
        "label_col": str(args.label_col),
        "score_col": str(args.score_col),
        "return_col": str(args.return_col),
        "regime_col": str(args.regime_col),
        "recent_scope": {
            "rows": int(len(recent_df)),
            "signal_active": _summary(active_df, label_col=args.label_col, score_col=args.score_col, return_col=args.return_col),
        },
        "by_regime": by_regime,
        "threshold_neighborhood": near_threshold,
        "threshold_bands": threshold_bands,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()