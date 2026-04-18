from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_COMPARISON = Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.json")
DEFAULT_DIR_SUMMARY = Path("artifacts/models/featurelift_20260331_rerun/xgb_dir1h/summary.json")
DEFAULT_RET_SUMMARY = Path("artifacts/models/featurelift_20260331_rerun/xgb_ret1h/summary.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/analysis/featurelift_20260331_rerun/diagnostic_1h_walkforward_failure.json")
DEFAULT_OUTPUT_MD = Path("artifacts/analysis/featurelift_20260331_rerun/diagnostic_1h_walkforward_failure.md")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_metric(summary: dict[str, Any], split: str, metric: str) -> float:
    metrics = summary.get("metrics", {})
    split_metrics = metrics.get(split, {}) if isinstance(metrics, dict) else {}
    return float(split_metrics.get(metric, 0.0))


def _build_payload(comparison: dict[str, Any], dir_summary: dict[str, Any], ret_summary: dict[str, Any]) -> dict[str, Any]:
    walkforward = comparison["walkforward"]["1h"]
    dir_cmp = comparison["comparisons"]["1h_direction"]
    ret_cmp = comparison["comparisons"]["1h_regression"]
    walkforward_payload = _load_json(Path(str(walkforward["path"])))
    folds = walkforward_payload.get("folds", []) if isinstance(walkforward_payload.get("folds"), list) else []
    negative_folds = [fold for fold in folds if float(fold.get("cum_ret_net", 0.0)) < 0.0]

    diagnosis_flags = []
    if _summary_metric(dir_summary, "test", "recall") > _summary_metric(dir_summary, "test", "precision"):
        diagnosis_flags.append("holdout_f1_gain_is_recall_led")
    if float(walkforward.get("auc_mean", 0.0)) < 0.53:
        diagnosis_flags.append("walkforward_ranking_near_random")
    if float(walkforward.get("ece_10_mean", 0.0)) > 0.15:
        diagnosis_flags.append("walkforward_calibration_drift_high")
    if float(ret_cmp.get("improvement", 0.0)) < 0.0:
        diagnosis_flags.append("direction_gain_not_supported_by_regression")
    if folds and len(negative_folds) == len(folds):
        diagnosis_flags.append("all_walkforward_folds_negative")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "root_cause": (
            "The 1h rerun improved thresholded holdout F1 mostly through higher positive-class recall, "
            "but ranking power and calibration stayed weak, so rolling walkforward performance remained negative."
        ),
        "diagnosis_flags": diagnosis_flags,
        "holdout": {
            "direction_test_precision": _summary_metric(dir_summary, "test", "precision"),
            "direction_test_recall": _summary_metric(dir_summary, "test", "recall"),
            "direction_test_f1": _summary_metric(dir_summary, "test", "f1"),
            "direction_f1_delta_vs_baseline": float(dir_cmp.get("delta", 0.0)),
            "regression_test_rmse": _summary_metric(ret_summary, "test", "rmse"),
            "regression_rmse_delta_vs_baseline": float(ret_cmp.get("delta", 0.0)),
        },
        "walkforward": {
            "auc_mean": float(walkforward.get("auc_mean", 0.0)),
            "auc_std": float(walkforward.get("auc_std", 0.0)),
            "brier_mean": float(walkforward.get("brier_mean", 0.0)),
            "ece_10_mean": float(walkforward.get("ece_10_mean", 0.0)),
            "cum_ret_net_total": float(walkforward.get("cum_ret_net_total", 0.0)),
            "trade_count_total": int(walkforward.get("trade_count_total", 0)),
            "negative_fold_count": len(negative_folds),
            "fold_count": len(folds),
        },
        "folds": folds,
        "recommended_actions": [
            "Do not promote the 1h candidate into shadow or live until ranking and calibration improve.",
            "Use walkforward AUC, ECE, and net return as the primary 1h promotion gates rather than thresholded holdout F1.",
            "Treat recall-led F1 gains with worse regression RMSE as threshold effects, not as robust directional edge.",
        ],
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    holdout = payload["holdout"]
    walkforward = payload["walkforward"]
    lines = [
        "# 1h Feature-Lift Failure Diagnostic",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Root Cause",
        "",
        payload["root_cause"],
        "",
        "## Evidence",
        "",
        f"- Holdout precision/recall/F1: {holdout['direction_test_precision']:.6f} / {holdout['direction_test_recall']:.6f} / {holdout['direction_test_f1']:.6f}",
        f"- Direction F1 delta vs baseline: {holdout['direction_f1_delta_vs_baseline']:.6f}",
        f"- Regression RMSE delta vs baseline: {holdout['regression_rmse_delta_vs_baseline']:.6f}",
        f"- Walkforward AUC/ECE/net: {walkforward['auc_mean']:.6f} / {walkforward['ece_10_mean']:.6f} / {walkforward['cum_ret_net_total']:.6f}",
        f"- Negative folds: {walkforward['negative_fold_count']} of {walkforward['fold_count']}",
        "",
        "## Flags",
        "",
    ]
    lines.extend(f"- {flag}" for flag in payload.get("diagnosis_flags", []))
    lines.extend(["", "## Recommended Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("recommended_actions", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a concise 1h feature-lift failure diagnostic.")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--direction-summary", type=Path, default=DEFAULT_DIR_SUMMARY)
    parser.add_argument("--regression-summary", type=Path, default=DEFAULT_RET_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = _build_payload(
        _load_json(args.comparison),
        _load_json(args.direction_summary),
        _load_json(args.regression_summary),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote JSON diagnostic to {args.output_json}")
    print(f"Wrote markdown diagnostic to {args.output_markdown}")


if __name__ == "__main__":
    main()