from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration_metrics(path: Path, horizon: str) -> dict:
    payload = _load_metrics(path)
    horizons = payload.get("horizons", {}) if isinstance(payload, dict) else {}
    metrics = horizons.get(horizon, {}) if isinstance(horizons, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def _extract_calibration_gate_context(calibration_metrics: dict) -> dict:
    if not isinstance(calibration_metrics, dict):
        return {
            "promotion_hardening": {},
            "selection_scope_recent": {},
            "selection_scope_ece_drift": None,
            "borderline_exception": {},
        }
    promotion_hardening = calibration_metrics.get("promotion_hardening", {})
    if not isinstance(promotion_hardening, dict):
        promotion_hardening = {}
    recent_diagnostics = calibration_metrics.get("recent_diagnostics", {})
    if not isinstance(recent_diagnostics, dict):
        recent_diagnostics = {}
    selection_scope = recent_diagnostics.get("selection_scope", {})
    if not isinstance(selection_scope, dict):
        selection_scope = {}
    selection_scope_recent = selection_scope.get("recent", {})
    if not isinstance(selection_scope_recent, dict):
        selection_scope_recent = {}
    borderline_exception = promotion_hardening.get("borderline_exception", {})
    if not isinstance(borderline_exception, dict):
        borderline_exception = {}
    return {
        "promotion_hardening": promotion_hardening,
        "selection_scope_recent": selection_scope_recent,
        "selection_scope_ece_drift": selection_scope.get("ece_drift"),
        "borderline_exception": borderline_exception,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate model promotion using incumbent vs candidate quality metrics.")
    parser.add_argument("--incumbent", type=Path, required=True, help="Incumbent model_quality.json")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate model_quality.json")
    parser.add_argument("--min-auc-delta", type=float, default=0.002)
    parser.add_argument("--max-brier-increase", type=float, default=0.0)
    parser.add_argument("--max-ece-increase", type=float, default=0.01)
    parser.add_argument(
        "--min-trade-count",
        type=int,
        default=10,
        help="Minimum candidate trade_count required for promotion.",
    )
    parser.add_argument(
        "--trade-count-key",
        type=str,
        default="trade_count",
        help="Metric key in quality JSON used as candidate trade count.",
    )
    parser.add_argument(
        "--min-net-return",
        type=float,
        default=0.0,
        help="Minimum candidate net return required for promotion.",
    )
    parser.add_argument(
        "--net-return-key",
        type=str,
        default="net_return_total",
        help="Metric key in quality JSON used as candidate net return.",
    )
    parser.add_argument(
        "--champion-gate",
        type=Path,
        default=None,
        help="Optional champion-challenger gate JSON; when provided it must have promote=true.",
    )
    parser.add_argument(
        "--calibration-robustness",
        type=Path,
        default=None,
        help="Optional calibration_robustness.json; when provided, recent calibration checks must pass.",
    )
    parser.add_argument(
        "--calibration-horizon",
        type=str,
        default="1h",
        help="Horizon key inside calibration_robustness.json to validate.",
    )
    parser.add_argument(
        "--max-ece-drift",
        type=float,
        default=None,
        help="Optional maximum allowed recent-vs-baseline ECE drift.",
    )
    parser.add_argument(
        "--max-recent-ece",
        type=float,
        default=None,
        help="Optional maximum allowed recent ECE.",
    )
    parser.add_argument(
        "--min-recent-auc",
        type=float,
        default=None,
        help="Optional minimum recent AUC.",
    )
    parser.add_argument(
        "--rolling-ab-report",
        type=Path,
        default=None,
        help="Optional rolling_ab_report.json; when provided, rolling window stability checks must pass.",
    )
    parser.add_argument(
        "--max-negative-rolling-windows",
        type=int,
        default=None,
        help="Optional maximum number of rolling windows the candidate may lose.",
    )
    parser.add_argument(
        "--require-nonnegative-rolling-delta",
        action="store_true",
        help="Require overall rolling delta_cum_ret to be non-negative.",
    )
    parser.add_argument(
        "--overlap-triggered-diagnostics",
        type=Path,
        default=None,
        help="Optional overlap_triggered_trade_diagnostics.json; when provided, overlap-triggered execution checks must pass.",
    )
    parser.add_argument(
        "--min-overlap-triggered-trades",
        type=int,
        default=None,
        help="Optional minimum triggered overlap trade count.",
    )
    parser.add_argument(
        "--min-overlap-triggered-net-return",
        type=float,
        default=None,
        help="Optional minimum triggered overlap net return total.",
    )
    parser.add_argument(
        "--min-overlap-triggered-hit-rate",
        type=float,
        default=None,
        help="Optional minimum triggered overlap hit rate.",
    )
    parser.add_argument(
        "--allow-borderline-selection-rows-exception",
        action="store_true",
        help="Allow a borderline selection-row shortfall only when supporting evidence remains strong.",
    )
    parser.add_argument(
        "--require-champion-for-borderline",
        action="store_true",
        help="Require champion significance when applying a borderline selection-row exception.",
    )
    parser.add_argument(
        "--require-rolling-for-borderline",
        action="store_true",
        help="Require rolling stability checks to pass when applying a borderline selection-row exception.",
    )
    parser.add_argument(
        "--require-overlap-for-borderline",
        action="store_true",
        help="Require overlap-triggered execution checks to pass when applying a borderline selection-row exception.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/promotion_gate.json"))
    args = parser.parse_args()

    inc = _load_metrics(args.incumbent)
    cand = _load_metrics(args.candidate)

    auc_delta = float(cand.get("auc", float("nan"))) - float(inc.get("auc", float("nan")))
    brier_delta = float(cand.get("brier", float("nan"))) - float(inc.get("brier", float("nan")))
    ece_delta = float(cand.get("ece_10", float("nan"))) - float(inc.get("ece_10", float("nan")))
    candidate_trade_count = pd.to_numeric(cand.get(args.trade_count_key, float("nan")), errors="coerce")
    candidate_net_return = pd.to_numeric(cand.get(args.net_return_key, float("nan")), errors="coerce")
    champion_promote = None
    if args.champion_gate:
        champion_payload = _load_metrics(args.champion_gate)
        champion_promote = bool(champion_payload.get("promote", False))

    calibration_metrics = None
    if args.calibration_robustness:
        calibration_metrics = _load_calibration_metrics(args.calibration_robustness, args.calibration_horizon)

    rolling_metrics = None
    if args.rolling_ab_report:
        rolling_metrics = _load_metrics(args.rolling_ab_report)

    overlap_triggered_metrics = None
    if args.overlap_triggered_diagnostics:
        overlap_triggered_metrics = _load_metrics(args.overlap_triggered_diagnostics)
    calibration_gate_context = _extract_calibration_gate_context(calibration_metrics)

    checks = {
        "has_auc_delta": bool(pd.notna(auc_delta)),
        "has_brier_delta": bool(pd.notna(brier_delta)),
        "has_ece_delta": bool(pd.notna(ece_delta)),
        "auc_delta_ok": bool(pd.notna(auc_delta) and auc_delta >= args.min_auc_delta),
        "brier_delta_ok": bool(pd.notna(brier_delta) and brier_delta <= args.max_brier_increase),
        "ece_delta_ok": bool(pd.notna(ece_delta) and ece_delta <= args.max_ece_increase),
        "trade_count_ok": bool(pd.notna(candidate_trade_count) and int(candidate_trade_count) >= int(args.min_trade_count)),
        "net_return_ok": bool(pd.notna(candidate_net_return) and float(candidate_net_return) >= float(args.min_net_return)),
    }
    if champion_promote is not None:
        checks["champion_significance_ok"] = bool(champion_promote)
    if calibration_metrics is not None:
        recent = calibration_metrics.get("recent", {}) if isinstance(calibration_metrics, dict) else {}
        if not isinstance(recent, dict):
            recent = {}
        selection_scope_recent = calibration_gate_context.get("selection_scope_recent", {})
        if not isinstance(selection_scope_recent, dict):
            selection_scope_recent = {}
        promotion_hardening = calibration_gate_context.get("promotion_hardening", {})
        if not isinstance(promotion_hardening, dict):
            promotion_hardening = {}
        promotion_checks = promotion_hardening.get("checks", {}) if isinstance(promotion_hardening.get("checks", {}), dict) else {}
        recent_source = selection_scope_recent or recent
        recent_auc = pd.to_numeric(recent_source.get("auc", float("nan")), errors="coerce")
        recent_ece = pd.to_numeric(recent_source.get("ece_10", float("nan")), errors="coerce")
        ece_drift = pd.to_numeric(
            calibration_gate_context.get("selection_scope_ece_drift", calibration_metrics.get("ece_drift", float("nan"))),
            errors="coerce",
        )
        if "selection_rows_ok" in promotion_checks:
            checks["selection_rows_ok"] = bool(promotion_checks.get("selection_rows_ok"))
        if "selection_rows_strict_ok" in promotion_checks:
            checks["selection_rows_strict_ok"] = bool(promotion_checks.get("selection_rows_strict_ok"))
        if args.min_recent_auc is not None:
            checks["recent_auc_ok"] = bool(
                promotion_checks.get("recent_auc_ok")
                if "recent_auc_ok" in promotion_checks
                else (pd.notna(recent_auc) and float(recent_auc) >= float(args.min_recent_auc))
            )
        if args.max_recent_ece is not None:
            checks["recent_ece_ok"] = bool(
                promotion_checks.get("recent_ece_ok")
                if "recent_ece_ok" in promotion_checks
                else (pd.notna(recent_ece) and float(recent_ece) <= float(args.max_recent_ece))
            )
        if args.max_ece_drift is not None:
            checks["ece_drift_ok"] = bool(
                promotion_checks.get("ece_drift_ok")
                if "ece_drift_ok" in promotion_checks
                else (pd.notna(ece_drift) and float(ece_drift) <= float(args.max_ece_drift))
            )
    if rolling_metrics is not None:
        rolling_summary = rolling_metrics.get("rolling_summary", {}) if isinstance(rolling_metrics, dict) else {}
        overall = rolling_metrics.get("overall", {}) if isinstance(rolling_metrics, dict) else {}
        candidate_wins = pd.to_numeric(rolling_summary.get("candidate_wins", float("nan")), errors="coerce")
        baseline_wins = pd.to_numeric(rolling_summary.get("baseline_wins", float("nan")), errors="coerce")
        delta_cum_ret = pd.to_numeric(overall.get("delta_cum_ret", float("nan")), errors="coerce")
        if args.max_negative_rolling_windows is not None:
            checks["rolling_window_losses_ok"] = bool(
                pd.notna(baseline_wins) and int(baseline_wins) <= int(args.max_negative_rolling_windows)
            )
        if args.require_nonnegative_rolling_delta:
            checks["rolling_delta_ok"] = bool(pd.notna(delta_cum_ret) and float(delta_cum_ret) >= 0.0)
        if pd.notna(candidate_wins) and pd.notna(baseline_wins):
            checks["rolling_candidate_not_dominated"] = bool(int(candidate_wins) >= int(baseline_wins))
    if overlap_triggered_metrics is not None:
        overlap_scope = overlap_triggered_metrics.get("overlap_scope", {}) if isinstance(overlap_triggered_metrics, dict) else {}
        triggered_rows = pd.to_numeric(overlap_scope.get("triggered_row_count", float("nan")), errors="coerce")
        triggered_net_return = pd.to_numeric(overlap_scope.get("triggered_net_return_total", float("nan")), errors="coerce")
        triggered_hit_rate = pd.to_numeric(overlap_scope.get("triggered_hit_rate", float("nan")), errors="coerce")
        if args.min_overlap_triggered_trades is not None:
            checks["overlap_triggered_trade_count_ok"] = bool(
                pd.notna(triggered_rows) and int(triggered_rows) >= int(args.min_overlap_triggered_trades)
            )
        if args.min_overlap_triggered_net_return is not None:
            checks["overlap_triggered_net_return_ok"] = bool(
                pd.notna(triggered_net_return) and float(triggered_net_return) >= float(args.min_overlap_triggered_net_return)
            )
        if args.min_overlap_triggered_hit_rate is not None:
            checks["overlap_triggered_hit_rate_ok"] = bool(
                pd.notna(triggered_hit_rate) and float(triggered_hit_rate) >= float(args.min_overlap_triggered_hit_rate)
            )

    borderline_exception = calibration_gate_context.get("borderline_exception", {})
    if not isinstance(borderline_exception, dict):
        borderline_exception = {}
    borderline_gate_applied = False
    borderline_gate_checks = None
    if args.allow_borderline_selection_rows_exception and bool(borderline_exception.get("eligible", False)):
        borderline_gate_applied = True
        champion_ok = True if not args.require_champion_for_borderline else bool(checks.get("champion_significance_ok", False))
        rolling_ok = True
        if args.require_rolling_for_borderline:
            rolling_ok = all(
                bool(checks.get(name, False))
                for name in ("rolling_window_losses_ok", "rolling_delta_ok", "rolling_candidate_not_dominated")
                if name in checks
            ) and any(name in checks for name in ("rolling_window_losses_ok", "rolling_delta_ok", "rolling_candidate_not_dominated"))
        overlap_ok = True
        if args.require_overlap_for_borderline:
            overlap_ok = all(
                bool(checks.get(name, False))
                for name in (
                    "overlap_triggered_trade_count_ok",
                    "overlap_triggered_net_return_ok",
                    "overlap_triggered_hit_rate_ok",
                )
                if name in checks
            ) and any(
                name in checks
                for name in (
                    "overlap_triggered_trade_count_ok",
                    "overlap_triggered_net_return_ok",
                    "overlap_triggered_hit_rate_ok",
                )
            )
        borderline_gate_checks = {
            "champion_ok": champion_ok,
            "rolling_ok": rolling_ok,
            "overlap_ok": overlap_ok,
        }
        checks["borderline_selection_exception_ok"] = bool(champion_ok and rolling_ok and overlap_ok)

    failed_checks = [name for name, ok in checks.items() if not ok]
    if borderline_gate_applied and bool(checks.get("borderline_selection_exception_ok", False)):
        failed_checks = [name for name in failed_checks if name != "selection_rows_strict_ok"]
    warnings = []
    if pd.notna(candidate_trade_count) and int(candidate_trade_count) == 0:
        warnings.append("no_ensemble_trades_in_evaluation_window")
    if borderline_gate_applied and bool(checks.get("borderline_selection_exception_ok", False)):
        warnings.append("borderline_selection_rows_exception_applied")

    promote = len(failed_checks) == 0

    payload = {
        "promote": bool(promote),
        "auc_delta": float(auc_delta),
        "brier_delta": float(brier_delta),
        "ece_delta": float(ece_delta),
        "candidate_trade_count": None if pd.isna(candidate_trade_count) else int(candidate_trade_count),
        "candidate_net_return": None if pd.isna(candidate_net_return) else float(candidate_net_return),
        "champion_promote": champion_promote,
        "calibration_metrics": calibration_metrics,
        "calibration_borderline_exception": borderline_exception,
        "borderline_gate_applied": bool(borderline_gate_applied),
        "borderline_gate_checks": borderline_gate_checks,
        "rolling_ab_metrics": rolling_metrics.get("rolling_summary") if isinstance(rolling_metrics, dict) else None,
        "overlap_triggered_metrics": overlap_triggered_metrics.get("overlap_scope") if isinstance(overlap_triggered_metrics, dict) else None,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "thresholds": {
            "min_auc_delta": args.min_auc_delta,
            "max_brier_increase": args.max_brier_increase,
            "max_ece_increase": args.max_ece_increase,
            "min_trade_count": int(args.min_trade_count),
            "trade_count_key": args.trade_count_key,
            "min_net_return": float(args.min_net_return),
            "net_return_key": args.net_return_key,
            "champion_gate": str(args.champion_gate) if args.champion_gate else None,
            "calibration_robustness": str(args.calibration_robustness) if args.calibration_robustness else None,
            "calibration_horizon": str(args.calibration_horizon),
            "max_ece_drift": args.max_ece_drift,
            "max_recent_ece": args.max_recent_ece,
            "min_recent_auc": args.min_recent_auc,
            "rolling_ab_report": str(args.rolling_ab_report) if args.rolling_ab_report else None,
            "max_negative_rolling_windows": args.max_negative_rolling_windows,
            "require_nonnegative_rolling_delta": bool(args.require_nonnegative_rolling_delta),
            "overlap_triggered_diagnostics": str(args.overlap_triggered_diagnostics) if args.overlap_triggered_diagnostics else None,
            "min_overlap_triggered_trades": args.min_overlap_triggered_trades,
            "min_overlap_triggered_net_return": args.min_overlap_triggered_net_return,
            "min_overlap_triggered_hit_rate": args.min_overlap_triggered_hit_rate,
            "allow_borderline_selection_rows_exception": bool(args.allow_borderline_selection_rows_exception),
            "require_champion_for_borderline": bool(args.require_champion_for_borderline),
            "require_rolling_for_borderline": bool(args.require_rolling_for_borderline),
            "require_overlap_for_borderline": bool(args.require_overlap_for_borderline),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if not promote:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
