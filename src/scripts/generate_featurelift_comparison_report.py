from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HORIZON_CONFIG: dict[str, dict[str, Any]] = {
    "15m": {
        "baseline_direction": Path("artifacts/models/xgb_dir15m_v1/summary.json"),
        "baseline_regression": Path("artifacts/models/xgb_ret15m_v1/summary.json"),
        "candidate_direction": Path("artifacts/models/featurelift_20260331/xgb_dir15m/summary.json"),
        "candidate_regression": Path("artifacts/models/featurelift_20260331/xgb_ret15m/summary.json"),
        "walkforward": Path("artifacts/analysis/featurelift_20260331/walkforward_15m_xgb.json"),
    },
    "1h": {
        "baseline_direction": Path("artifacts/models/xgb_dir1h_v1/summary.json"),
        "baseline_regression": Path("artifacts/models/xgb_ret1h_v1/summary.json"),
        "candidate_direction": Path("artifacts/models/featurelift_20260331_rerun/xgb_dir1h/summary.json"),
        "candidate_regression": Path("artifacts/models/featurelift_20260331_rerun/xgb_ret1h/summary.json"),
        "walkforward": Path("artifacts/analysis/featurelift_20260331_rerun/walkforward_1h_xgb.json"),
    },
    "4h": {
        "baseline_direction": Path("artifacts/models/xgb_dir4h_v1/summary.json"),
        "baseline_regression": Path("artifacts/models/xgb_ret4h_v1/summary.json"),
        "candidate_direction": Path("artifacts/models/featurelift_20260331_rerun/xgb_dir4h/summary.json"),
        "candidate_regression": Path("artifacts/models/featurelift_20260331_rerun/xgb_ret4h/summary.json"),
        "walkforward": Path("artifacts/analysis/featurelift_20260331_rerun/walkforward_4h_xgb.json"),
    },
    "8h": {
        "baseline_direction": Path("artifacts/models/xgb_dir8h_v1/summary.json"),
        "baseline_regression": Path("artifacts/models/xgb_ret8h_v1/summary.json"),
        "candidate_direction": Path("artifacts/models/featurelift_20260331_rerun/xgb_dir8h/summary.json"),
        "candidate_regression": Path("artifacts/models/featurelift_20260331_rerun/xgb_ret8h/summary.json"),
        "walkforward": Path("artifacts/analysis/featurelift_20260331_rerun/walkforward_8h_xgb.json"),
    },
    "12h": {
        "baseline_direction": Path("artifacts/models/xgb_dir12h_v1/summary.json"),
        "baseline_regression": Path("artifacts/models/xgb_ret12h_v1/summary.json"),
        "candidate_direction": Path("artifacts/models/featurelift_20260331_rerun/xgb_dir12h/summary.json"),
        "candidate_regression": Path("artifacts/models/featurelift_20260331_rerun/xgb_ret12h/summary.json"),
        "walkforward": Path("artifacts/analysis/featurelift_20260331_rerun/walkforward_12h_xgb.json"),
    },
}

ONCHAIN_MANIFEST = Path("data/processed/onchain/source_manifest.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_count(summary: dict[str, Any]) -> int:
    feature_names = summary.get("feature_names")
    if isinstance(feature_names, list):
        return len(feature_names)
    return 0


def _metric_value(summary: dict[str, Any], metric: str) -> float:
    metrics = summary.get("metrics", {})
    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    value = test_metrics.get(metric)
    if value is None:
        raise KeyError(f"Missing test metric '{metric}' in summary.")
    return float(value)


def _compare_metric(metric: str, baseline: float, candidate: float) -> dict[str, float | str]:
    delta = candidate - baseline
    if metric == "rmse":
        improvement = baseline - candidate
    else:
        improvement = candidate - baseline
    return {
        "metric": metric,
        "baseline_test": baseline,
        "candidate_test": candidate,
        "delta": delta,
        "improvement": improvement,
    }


def _build_comparison_report() -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    walkforward: dict[str, Any] = {}

    for horizon, config in HORIZON_CONFIG.items():
        baseline_direction = _load_json(config["baseline_direction"])
        baseline_regression = _load_json(config["baseline_regression"])
        candidate_direction = _load_json(config["candidate_direction"])
        candidate_regression = _load_json(config["candidate_regression"])

        direction_key = f"{horizon}_direction"
        regression_key = f"{horizon}_regression"

        direction_metrics = _compare_metric(
            "f1",
            _metric_value(baseline_direction, "f1"),
            _metric_value(candidate_direction, "f1"),
        )
        direction_metrics["baseline_feature_count"] = _feature_count(baseline_direction)
        direction_metrics["candidate_feature_count"] = _feature_count(candidate_direction)
        direction_metrics["baseline_summary_path"] = str(config["baseline_direction"])
        direction_metrics["candidate_summary_path"] = str(config["candidate_direction"])
        comparisons[direction_key] = direction_metrics

        regression_metrics = _compare_metric(
            "rmse",
            _metric_value(baseline_regression, "rmse"),
            _metric_value(candidate_regression, "rmse"),
        )
        regression_metrics["baseline_feature_count"] = _feature_count(baseline_regression)
        regression_metrics["candidate_feature_count"] = _feature_count(candidate_regression)
        regression_metrics["baseline_summary_path"] = str(config["baseline_regression"])
        regression_metrics["candidate_summary_path"] = str(config["candidate_regression"])
        comparisons[regression_key] = regression_metrics

        walkforward_payload = _load_json(config["walkforward"])
        walkforward[horizon] = {
            "path": str(config["walkforward"]),
            "auc_mean": float(walkforward_payload["auc_mean"]),
            "auc_std": float(walkforward_payload.get("auc_std", 0.0)),
            "brier_mean": float(walkforward_payload["brier_mean"]),
            "ece_10_mean": float(walkforward_payload["ece_10_mean"]),
            "cum_ret_net_total": float(walkforward_payload["cum_ret_net_total"]),
            "trade_count_total": int(walkforward_payload["trade_count_total"]),
        }

    onchain_manifest = _load_json(ONCHAIN_MANIFEST) if ONCHAIN_MANIFEST.exists() else None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "current",
        "supersedes": "artifacts/analysis/featurelift_20260331/comparison_report.json",
        "notes": [
            "This report supersedes the earlier featurelift_20260331 comparison that was contaminated by multi-horizon target leakage.",
            "15m candidate metrics are carried forward from featurelift_20260331 because the leakage fix did not affect the 15m path.",
            "1h, 4h, 8h, and 12h candidate metrics come from featurelift_20260331_rerun after leakage fixes, 1h retune, and on-chain refresh.",
            "Multi-horizon degradations versus the previous feature-lift run are expected and indicate the leaked edge was removed, not that the rerun regressed unexpectedly.",
        ],
        "onchain_refresh": onchain_manifest,
        "comparisons": comparisons,
        "walkforward": walkforward,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Feature-Lift Comparison Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Status",
        "",
        "- Current report after leakage fixes, 1h retune, and on-chain refresh.",
        "- Previous featurelift_20260331 comparison is superseded.",
        "",
        "## Direction And Regression",
        "",
        "| Horizon | Direction F1 Δ | Regression RMSE Δ | Walkforward AUC | Walkforward Net Return |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZON_CONFIG:
        direction = report["comparisons"][f"{horizon}_direction"]
        regression = report["comparisons"][f"{horizon}_regression"]
        wf = report["walkforward"][horizon]
        lines.append(
            f"| {horizon} | {direction['delta']:.6f} | {regression['delta']:.6f} | {wf['auc_mean']:.6f} | {wf['cum_ret_net_total']:.6f} |"
        )

    onchain = report.get("onchain_refresh")
    if isinstance(onchain, dict):
        lines.extend(
            [
                "",
                "## On-Chain Refresh",
                "",
                f"- Provider: {onchain.get('provider', 'unknown')}",
                f"- Coverage: {onchain.get('ts_start', 'unknown')} to {onchain.get('ts_end', 'unknown')}",
                f"- Rows: {onchain.get('row_count', 'unknown')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in report["notes"]],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the corrected feature-lift comparison report.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.md"),
    )
    parser.add_argument(
        "--superseded-output-json",
        type=Path,
        default=Path("artifacts/analysis/featurelift_20260331/comparison_report.json"),
    )
    args = parser.parse_args()

    report = _build_comparison_report()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(report), encoding="utf-8")

    superseded_payload = {
        "status": "superseded",
        "superseded_at": report["generated_at"],
        "superseded_by": str(args.output_json),
        "reason": "The original comparison report included multi-horizon metrics from a leaked feature set.",
    }
    args.superseded_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.superseded_output_json.write_text(json.dumps(superseded_payload, indent=2), encoding="utf-8")

    print(f"Wrote corrected report to {args.output_json}")
    print(f"Wrote markdown summary to {args.output_markdown}")
    print(f"Marked superseded report at {args.superseded_output_json}")


if __name__ == "__main__":
    main()