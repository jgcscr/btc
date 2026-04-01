from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _resolve_label_column(frame: pd.DataFrame, preferred: str | None) -> str:
    candidates = [preferred, "y", "y_true", "target_up", "label", "direction_target"]
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return candidate
    raise ValueError("Unable to resolve label column. Provide --label-col explicitly.")


def _resolve_horizon_series(frame: pd.DataFrame) -> pd.Series:
    if "horizon_hours" in frame.columns:
        return pd.to_numeric(frame["horizon_hours"], errors="coerce")
    if "horizon" in frame.columns:
        raw = frame["horizon"].astype(str).str.strip().str.lower()
        mapped = raw.str.replace("h", "", regex=False)
        return pd.to_numeric(mapped, errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _expected_calibration_error(y_true: np.ndarray, p_up: np.ndarray, bins: int = 10) -> float:
    if y_true.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = float(y_true.size)
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (p_up >= left) & (p_up <= right)
        else:
            mask = (p_up >= left) & (p_up < right)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p_up[mask]))
        ece += abs(acc - conf) * (count / total)
    return float(ece)


def _metric_bundle(frame: pd.DataFrame, *, prob_col: str, label_col: str, threshold: float) -> Dict[str, Any]:
    probs = pd.to_numeric(frame[prob_col], errors="coerce")
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    working = pd.DataFrame({"p": probs, "y": labels}).dropna()
    if working.empty:
        return {
            "rows": 0,
            "brier": None,
            "ece": None,
            "f1": None,
            "positive_rate": None,
        }
    y = working["y"].to_numpy(dtype=float)
    y = np.where(y > 0.0, 1.0, 0.0)
    p = working["p"].clip(0.0, 1.0).to_numpy(dtype=float)
    pred = (p >= float(threshold)).astype(int)
    f1_value = float(f1_score(y.astype(int), pred, zero_division=0))
    brier = float(np.mean((p - y) ** 2))
    ece = _expected_calibration_error(y.astype(int), p)
    return {
        "rows": int(y.size),
        "brier": float(brier),
        "ece": float(ece),
        "f1": float(f1_value),
        "positive_rate": float(np.mean(y)),
    }


def _evaluate_thresholds(metrics: Dict[str, Any], thresholds: Dict[str, float], *, prefix: str) -> list[str]:
    failures: list[str] = []
    rows = int(metrics.get("rows") or 0)
    min_rows = int(thresholds.get("min_rows", 0) or 0)
    if rows < min_rows:
        failures.append(f"{prefix}:rows_below_min")
        return failures

    max_brier = _safe_float(thresholds.get("max_brier"))
    max_ece = _safe_float(thresholds.get("max_ece"))
    min_f1 = _safe_float(thresholds.get("min_f1"))
    brier = _safe_float(metrics.get("brier"))
    ece = _safe_float(metrics.get("ece"))
    f1_value = _safe_float(metrics.get("f1"))

    if max_brier is not None and brier is not None and brier > max_brier:
        failures.append(f"{prefix}:brier_above_max")
    if max_ece is not None and ece is not None and ece > max_ece:
        failures.append(f"{prefix}:ece_above_max")
    if min_f1 is not None and f1_value is not None and f1_value < min_f1:
        failures.append(f"{prefix}:f1_below_min")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate directional quality objectives by horizon and regime.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prob-col", default="p_up")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--regime-col", default="regime_state")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--group-min-rows", type=int, default=80)
    parser.add_argument("--min-rows-by-regime", default="")
    parser.add_argument("--max-brier", type=float, default=0.25)
    parser.add_argument("--max-ece", type=float, default=0.08)
    parser.add_argument("--min-f1", type=float, default=0.45)
    parser.add_argument("--max-brier-by-horizon", default="")
    parser.add_argument("--max-ece-by-horizon", default="")
    parser.add_argument("--min-f1-by-horizon", default="")
    parser.add_argument("--max-brier-by-regime", default="")
    parser.add_argument("--max-ece-by-regime", default="")
    parser.add_argument("--min-f1-by-regime", default="")
    return parser


def _parse_threshold_map(raw: str) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    if not raw:
        return resolved
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        parsed = _safe_float(value.strip())
        if parsed is None:
            continue
        resolved[key.strip().lower()] = float(parsed)
    return resolved


def main() -> int:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input)
    if args.prob_col not in frame.columns:
        raise ValueError(f"Missing probability column: {args.prob_col}")
    label_col = _resolve_label_column(frame, args.label_col)

    horizon_series = _resolve_horizon_series(frame)
    if args.regime_col in frame.columns:
        regime_raw = frame[args.regime_col]
        regime_series = regime_raw.where(regime_raw.notna(), "unknown").astype(str).str.strip().str.lower()
        regime_series = regime_series.replace({"": "unknown", "nan": "unknown", "none": "unknown", "null": "unknown"})
    else:
        regime_series = pd.Series("unknown", index=frame.index)

    overall = _metric_bundle(frame, prob_col=args.prob_col, label_col=label_col, threshold=args.threshold)

    per_horizon: Dict[str, Dict[str, Any]] = {}
    for horizon in sorted(value for value in horizon_series.dropna().unique() if value > 0):
        horizon_mask = horizon_series == horizon
        label = f"{int(horizon)}h" if float(horizon).is_integer() else f"{horizon:g}h"
        per_horizon[label] = _metric_bundle(
            frame.loc[horizon_mask],
            prob_col=args.prob_col,
            label_col=label_col,
            threshold=args.threshold,
        )

    per_regime: Dict[str, Dict[str, Any]] = {}
    for regime in sorted(set(regime_series.dropna().tolist())):
        mask = regime_series == regime
        per_regime[str(regime)] = _metric_bundle(
            frame.loc[mask],
            prob_col=args.prob_col,
            label_col=label_col,
            threshold=args.threshold,
        )

    global_thresholds = {
        "min_rows": int(args.min_rows),
        "max_brier": float(args.max_brier),
        "max_ece": float(args.max_ece),
        "min_f1": float(args.min_f1),
    }
    group_thresholds = {
        "min_rows": int(args.group_min_rows),
        "max_brier": float(args.max_brier),
        "max_ece": float(args.max_ece),
        "min_f1": float(args.min_f1),
    }
    horizon_overrides = {
        "max_brier": _parse_threshold_map(args.max_brier_by_horizon),
        "max_ece": _parse_threshold_map(args.max_ece_by_horizon),
        "min_f1": _parse_threshold_map(args.min_f1_by_horizon),
    }
    regime_overrides = {
        "min_rows": _parse_threshold_map(args.min_rows_by_regime),
        "max_brier": _parse_threshold_map(args.max_brier_by_regime),
        "max_ece": _parse_threshold_map(args.max_ece_by_regime),
        "min_f1": _parse_threshold_map(args.min_f1_by_regime),
    }

    failed_checks: list[str] = []
    failed_checks.extend(_evaluate_thresholds(overall, global_thresholds, prefix="overall"))

    for key, metrics in per_horizon.items():
        scoped = dict(group_thresholds)
        lookup = str(key).strip().lower()
        for metric_name in ("max_brier", "max_ece", "min_f1"):
            if lookup in horizon_overrides[metric_name]:
                scoped[metric_name] = horizon_overrides[metric_name][lookup]
        failed_checks.extend(_evaluate_thresholds(metrics, scoped, prefix=f"horizon:{key}"))

    for key, metrics in per_regime.items():
        scoped = dict(group_thresholds)
        lookup = str(key).strip().lower()
        for metric_name in ("min_rows", "max_brier", "max_ece", "min_f1"):
            if lookup in regime_overrides[metric_name]:
                scoped[metric_name] = regime_overrides[metric_name][lookup]
        failed_checks.extend(_evaluate_thresholds(metrics, scoped, prefix=f"regime:{key}"))

    payload = {
        "input": str(args.input),
        "prob_col": args.prob_col,
        "label_col": label_col,
        "regime_col": args.regime_col,
        "threshold": float(args.threshold),
        "thresholds": {
            "overall": global_thresholds,
            "group_default": group_thresholds,
            "horizon_overrides": horizon_overrides,
            "regime_overrides": regime_overrides,
        },
        "overall": overall,
        "by_horizon": per_horizon,
        "by_regime": per_regime,
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failed_checks:
        print("Directional objectives failed:")
        for check in failed_checks:
            print(f"- {check}")
        return 2

    print("Directional objectives passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
