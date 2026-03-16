from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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


def _trade_metrics(df: pd.DataFrame, signal_col: str | None, return_col: str | None) -> Dict[str, Any]:
    if not signal_col or signal_col not in df.columns:
        return {"trade_count": 0, "net_return_total": None, "hit_rate": None}
    signal = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    active = signal > 0.0
    trade_count = int(active.sum())
    if not return_col or return_col not in df.columns:
        return {"trade_count": trade_count, "net_return_total": None, "hit_rate": None}
    returns = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
    active_returns = returns[active]
    return {
        "trade_count": trade_count,
        "net_return_total": float(active_returns.sum()) if trade_count else 0.0,
        "hit_rate": float((active_returns > 0.0).mean()) if trade_count else None,
    }


def _calibration_gap(df: pd.DataFrame, p_col: str, y_col: str) -> Dict[str, float | None]:
    p = pd.to_numeric(df[p_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = p.notna() & y.notna()
    if not bool(mask.any()):
        return {"mean_probability": None, "observed_rate": None, "gap": None, "abs_gap": None}
    mean_probability = float(p[mask].mean())
    observed_rate = float(y[mask].mean())
    gap = float(observed_rate - mean_probability)
    return {
        "mean_probability": mean_probability,
        "observed_rate": observed_rate,
        "gap": gap,
        "abs_gap": abs(gap),
    }


def _recent_probability_bin_summary(
    df: pd.DataFrame,
    *,
    p_col: str,
    y_col: str,
    signal_col: str | None,
    return_col: str | None,
    bins: int,
) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    out: List[Dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    prob = pd.to_numeric(df[p_col], errors="coerce")
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        mask = (prob >= lo) & (prob < hi if idx < len(edges) - 2 else prob <= hi)
        bucket = df.loc[mask].copy()
        if bucket.empty:
            continue
        metrics = _metrics(bucket, p_col, y_col)
        gap = _calibration_gap(bucket, p_col, y_col)
        trade = _trade_metrics(bucket, signal_col, return_col)
        out.append(
            {
                "bin": f"{lo:.2f}-{hi:.2f}",
                "rows": int(metrics["rows"]),
                "auc": metrics["auc"],
                "brier": metrics["brier"],
                "ece_10": metrics["ece_10"],
                **gap,
                **trade,
            }
        )
    return out


def _recent_group_summary(
    df: pd.DataFrame,
    *,
    group_col: str,
    p_col: str,
    y_col: str,
    signal_col: str | None,
    return_col: str | None,
) -> List[Dict[str, Any]]:
    if group_col not in df.columns or df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_col, dropna=False)
    for raw_value, group in grouped:
        label = "missing" if pd.isna(raw_value) else str(raw_value)
        metrics = _metrics(group, p_col, y_col)
        gap = _calibration_gap(group, p_col, y_col)
        trade = _trade_metrics(group, signal_col, return_col)
        rows.append(
            {
                group_col: label,
                "rows": int(metrics["rows"]),
                "auc": metrics["auc"],
                "brier": metrics["brier"],
                "ece_10": metrics["ece_10"],
                **gap,
                **trade,
            }
        )
    rows.sort(key=lambda item: (float(item.get("abs_gap") or 0.0), int(item.get("rows") or 0)), reverse=True)
    return rows


def _selection_scope_slice(
    df: pd.DataFrame,
    *,
    scope: str,
    signal_col: str | None,
) -> pd.DataFrame:
    resolved_scope = str(scope or "all").strip().lower()
    if resolved_scope != "signal_active" or not signal_col or signal_col not in df.columns:
        return df.copy()
    signal = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    return df.loc[signal > 0.0].copy()


def _resolve_selection_row_policy(
    *,
    recent_selection_rows: int,
    baseline_selection_rows: int,
    strict_min_selection_rows: int,
    adaptive_enabled: bool,
    adaptive_min_floor: int,
    adaptive_baseline_ratio: float,
    adaptive_max_shortfall: int,
) -> Dict[str, Any]:
    strict_min_rows = max(int(strict_min_selection_rows), 0)
    baseline_rows = max(int(baseline_selection_rows), 0)
    recent_rows = max(int(recent_selection_rows), 0)
    min_floor = max(int(adaptive_min_floor), 0)
    max_shortfall = max(int(adaptive_max_shortfall), 0)
    baseline_ratio = max(float(adaptive_baseline_ratio), 0.0)
    baseline_ratio_rows = int(np.ceil(baseline_rows * baseline_ratio)) if baseline_rows > 0 else 0

    effective_min_rows = strict_min_rows
    if adaptive_enabled and strict_min_rows > 0:
        effective_min_rows = min(
            strict_min_rows,
            max(min_floor, baseline_ratio_rows, strict_min_rows - max_shortfall),
        )

    strict_ok = bool(recent_rows >= strict_min_rows)
    effective_ok = bool(recent_rows >= effective_min_rows)
    borderline_eligible = bool(adaptive_enabled and effective_ok and not strict_ok)
    return {
        "adaptive_enabled": bool(adaptive_enabled),
        "recent_selection_rows": recent_rows,
        "baseline_selection_rows": baseline_rows,
        "strict_min_selection_rows": strict_min_rows,
        "effective_min_selection_rows": effective_min_rows,
        "adaptive_min_floor": min_floor,
        "adaptive_baseline_ratio": baseline_ratio,
        "adaptive_baseline_ratio_rows": baseline_ratio_rows,
        "adaptive_max_shortfall": max_shortfall,
        "strict_ok": strict_ok,
        "effective_ok": effective_ok,
        "row_shortfall_vs_strict": max(strict_min_rows - recent_rows, 0),
        "row_shortfall_vs_effective": max(effective_min_rows - recent_rows, 0),
        "borderline_exception_eligible": borderline_eligible,
    }


def _recent_worst_rows(
    df: pd.DataFrame,
    *,
    p_col: str,
    y_col: str,
    ts_col: str,
    regime_col: str | None,
    signal_col: str | None,
    return_col: str | None,
    top_rows: int,
) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    working = df.copy()
    working["_p"] = pd.to_numeric(working[p_col], errors="coerce")
    working["_y"] = pd.to_numeric(working[y_col], errors="coerce")
    working = working.dropna(subset=["_p", "_y"])
    if working.empty:
        return []
    working["calibration_error"] = working["_y"] - working["_p"]
    working["abs_calibration_error"] = working["calibration_error"].abs()
    sort_cols = ["abs_calibration_error"]
    if signal_col and signal_col in working.columns:
        working["_signal"] = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
        sort_cols = ["_signal", "abs_calibration_error"]
    working = working.sort_values(sort_cols, ascending=False).head(max(int(top_rows), 1))
    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        item: Dict[str, Any] = {
            "ts": str(row[ts_col]) if ts_col in working.columns else None,
            "p_up": float(row["_p"]),
            "y_true": int(row["_y"]),
            "calibration_error": float(row["calibration_error"]),
            "abs_calibration_error": float(row["abs_calibration_error"]),
        }
        if regime_col and regime_col in working.columns:
            item[regime_col] = None if pd.isna(row[regime_col]) else str(row[regime_col])
        if signal_col and signal_col in working.columns:
            item[signal_col] = float(pd.to_numeric(pd.Series([row[signal_col]]), errors="coerce").fillna(0.0).iloc[0])
        if return_col and return_col in working.columns:
            item[return_col] = float(pd.to_numeric(pd.Series([row[return_col]]), errors="coerce").fillna(0.0).iloc[0])
        rows.append(item)
    return rows


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
    parser.add_argument("--max-recent-ece", type=float, default=None)
    parser.add_argument("--min-recent-auc", type=float, default=None)
    parser.add_argument("--regime-col", type=str, default="regime_state")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--return-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--selection-scope", choices=("all", "signal_active"), default="all")
    parser.add_argument("--min-selection-rows", type=int, default=0)
    parser.add_argument("--adaptive-selection-rows", action="store_true")
    parser.add_argument("--adaptive-selection-min-floor", type=int, default=0)
    parser.add_argument("--adaptive-selection-baseline-ratio", type=float, default=0.0)
    parser.add_argument("--adaptive-selection-max-shortfall", type=int, default=0)
    parser.add_argument("--probability-bins", type=int, default=10)
    parser.add_argument("--top-rows", type=int, default=20)
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

        selection_scope = str(args.selection_scope or "all").strip().lower()
        selection_recent = _selection_scope_slice(
            recent,
            scope=selection_scope,
            signal_col=args.signal_col if args.signal_col in recent.columns else None,
        )
        selection_baseline = _selection_scope_slice(
            baseline,
            scope=selection_scope,
            signal_col=args.signal_col if args.signal_col in baseline.columns else None,
        )
        selection_recent_m = (
            _metrics(selection_recent, args.p_col, args.y_col)
            if not selection_recent.empty
            else {"rows": 0, "auc": float("nan"), "brier": float("nan"), "ece_10": float("nan")}
        )
        selection_baseline_m = (
            _metrics(selection_baseline, args.p_col, args.y_col)
            if not selection_baseline.empty
            else {"rows": 0, "auc": float("nan"), "brier": float("nan"), "ece_10": float("nan")}
        )

        ece_drift = float(recent_m["ece_10"] - baseline_m["ece_10"]) if baseline_m["rows"] > 0 and recent_m["rows"] > 0 else float("nan")
        selection_ece_drift = (
            float(selection_recent_m["ece_10"] - selection_baseline_m["ece_10"])
            if selection_baseline_m["rows"] > 0 and selection_recent_m["rows"] > 0
            else float("nan")
        )
        recent_gap = _calibration_gap(recent, args.p_col, args.y_col) if not recent.empty else {"mean_probability": None, "observed_rate": None, "gap": None, "abs_gap": None}
        selection_recent_gap = (
            _calibration_gap(selection_recent, args.p_col, args.y_col)
            if not selection_recent.empty
            else {"mean_probability": None, "observed_rate": None, "gap": None, "abs_gap": None}
        )
        selection_row_policy = _resolve_selection_row_policy(
            recent_selection_rows=int(selection_recent_m["rows"]),
            baseline_selection_rows=int(selection_baseline_m["rows"]),
            strict_min_selection_rows=int(args.min_selection_rows),
            adaptive_enabled=bool(args.adaptive_selection_rows),
            adaptive_min_floor=int(args.adaptive_selection_min_floor),
            adaptive_baseline_ratio=float(args.adaptive_selection_baseline_ratio),
            adaptive_max_shortfall=int(args.adaptive_selection_max_shortfall),
        )
        failed_checks: List[str] = []
        selection_rows_ok = bool(selection_row_policy["effective_ok"])
        selection_rows_strict_ok = bool(selection_row_policy["strict_ok"])
        if not selection_rows_ok:
            failed_checks.append("selection_rows_ok")
        if args.min_recent_auc is not None:
            recent_auc_ok = bool(
                np.isfinite(selection_recent_m["auc"]) and float(selection_recent_m["auc"]) >= float(args.min_recent_auc)
            )
            if not recent_auc_ok:
                failed_checks.append("recent_auc_ok")
        else:
            recent_auc_ok = True
        if args.max_recent_ece is not None:
            recent_ece_ok = bool(
                np.isfinite(selection_recent_m["ece_10"])
                and float(selection_recent_m["ece_10"]) <= float(args.max_recent_ece)
            )
            if not recent_ece_ok:
                failed_checks.append("recent_ece_ok")
        else:
            recent_ece_ok = True
        ece_drift_ok = bool(
            np.isfinite(selection_ece_drift) and float(selection_ece_drift) <= float(args.max_ece_drift)
        )
        if not ece_drift_ok:
            failed_checks.append("ece_drift_ok")

        recent_signal_summary = []
        if args.signal_col in recent.columns:
            signal_recent = recent.copy()
            signal_recent["signal_active"] = pd.to_numeric(signal_recent[args.signal_col], errors="coerce").fillna(0.0) > 0.0
            recent_signal_summary = _recent_group_summary(
                signal_recent,
                group_col="signal_active",
                p_col=args.p_col,
                y_col=args.y_col,
                signal_col=args.signal_col if args.signal_col in signal_recent.columns else None,
                return_col=args.return_col if args.return_col in signal_recent.columns else None,
            )

        horizon_reports[str(horizon)] = {
            "overall": overall,
            "baseline": baseline_m,
            "recent": recent_m,
            "ece_drift": ece_drift,
            "ece_drift_alert": bool(np.isfinite(ece_drift) and ece_drift > float(args.max_ece_drift)),
            "promotion_hardening": {
                "selection_ok": len(failed_checks) == 0,
                "failed_checks": failed_checks,
                "checks": {
                    "selection_rows_ok": selection_rows_ok,
                    "selection_rows_strict_ok": selection_rows_strict_ok,
                    "recent_auc_ok": recent_auc_ok,
                    "recent_ece_ok": recent_ece_ok,
                    "ece_drift_ok": ece_drift_ok,
                },
                "thresholds": {
                    "min_recent_auc": args.min_recent_auc,
                    "max_recent_ece": args.max_recent_ece,
                    "max_ece_drift": float(args.max_ece_drift),
                    "selection_scope": selection_scope,
                    "min_selection_rows": int(args.min_selection_rows),
                    "effective_min_selection_rows": int(selection_row_policy["effective_min_selection_rows"]),
                    "adaptive_selection_rows": {
                        "enabled": bool(args.adaptive_selection_rows),
                        "min_floor": int(args.adaptive_selection_min_floor),
                        "baseline_ratio": float(args.adaptive_selection_baseline_ratio),
                        "max_shortfall": int(args.adaptive_selection_max_shortfall),
                    },
                },
                "selection_row_policy": selection_row_policy,
                "borderline_exception": {
                    "eligible": bool(selection_row_policy["borderline_exception_eligible"]),
                    "recent_selection_rows": int(selection_row_policy["recent_selection_rows"]),
                    "baseline_selection_rows": int(selection_row_policy["baseline_selection_rows"]),
                    "strict_min_selection_rows": int(selection_row_policy["strict_min_selection_rows"]),
                    "effective_min_selection_rows": int(selection_row_policy["effective_min_selection_rows"]),
                    "row_shortfall_vs_strict": int(selection_row_policy["row_shortfall_vs_strict"]),
                },
            },
            "recent_diagnostics": {
                "gap": recent_gap,
                "selection_scope": {
                    "scope": selection_scope,
                    "recent": selection_recent_m,
                    "baseline": selection_baseline_m,
                    "ece_drift": selection_ece_drift,
                    "gap": selection_recent_gap,
                },
                "probability_bins": _recent_probability_bin_summary(
                    recent,
                    p_col=args.p_col,
                    y_col=args.y_col,
                    signal_col=args.signal_col if args.signal_col in recent.columns else None,
                    return_col=args.return_col if args.return_col in recent.columns else None,
                    bins=int(args.probability_bins),
                ),
                "by_regime": _recent_group_summary(
                    recent,
                    group_col=args.regime_col,
                    p_col=args.p_col,
                    y_col=args.y_col,
                    signal_col=args.signal_col if args.signal_col in recent.columns else None,
                    return_col=args.return_col if args.return_col in recent.columns else None,
                ),
                "selection_by_regime": _recent_group_summary(
                    selection_recent,
                    group_col=args.regime_col,
                    p_col=args.p_col,
                    y_col=args.y_col,
                    signal_col=args.signal_col if args.signal_col in selection_recent.columns else None,
                    return_col=args.return_col if args.return_col in selection_recent.columns else None,
                ),
                "by_signal_state": recent_signal_summary,
                "worst_rows": _recent_worst_rows(
                    recent,
                    p_col=args.p_col,
                    y_col=args.y_col,
                    ts_col=args.ts_col,
                    regime_col=args.regime_col if args.regime_col in recent.columns else None,
                    signal_col=args.signal_col if args.signal_col in recent.columns else None,
                    return_col=args.return_col if args.return_col in recent.columns else None,
                    top_rows=int(args.top_rows),
                ),
            },
        }

    summary_horizon = str(args.default_horizon)
    if summary_horizon not in horizon_reports and horizon_reports:
        summary_horizon = next(iter(horizon_reports.keys()))
    summary_payload = horizon_reports.get(summary_horizon, {}) if horizon_reports else {}
    payload = {
        "rows": int(len(df)),
        "settings": {
            "baseline_window": int(args.baseline_window),
            "recent_window": int(args.recent_window),
            "max_ece_drift": float(args.max_ece_drift),
            "max_recent_ece": args.max_recent_ece,
            "min_recent_auc": args.min_recent_auc,
            "adaptive_selection_rows": {
                "enabled": bool(args.adaptive_selection_rows),
                "min_floor": int(args.adaptive_selection_min_floor),
                "baseline_ratio": float(args.adaptive_selection_baseline_ratio),
                "max_shortfall": int(args.adaptive_selection_max_shortfall),
            },
        },
        "summary_horizon": summary_horizon,
        "promotion_hardening": summary_payload.get("promotion_hardening") if isinstance(summary_payload, dict) else None,
        "recent_diagnostics": summary_payload.get("recent_diagnostics") if isinstance(summary_payload, dict) else None,
        "horizons": horizon_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
