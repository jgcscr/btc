from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT = "artifacts/reliability/20260515T023232Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_reference_feature_ablation.csv"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/trade_decision_ablation_sweep_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/trade_decision_ablation_sweep_latest.md"


@dataclass(frozen=True)
class AblationVariant:
    name: str
    excluded_features: tuple[str, ...]


DEFAULT_VARIANTS = (
    AblationVariant("baseline", ()),
    AblationVariant("drop_component_agreement_ratio", ("component_agreement_ratio",)),
    AblationVariant("drop_component_entropy", ("component_entropy",)),
    AblationVariant("drop_p_up_ret_mismatch", ("p_up_ret_mismatch",)),
    AblationVariant("drop_regime_is_neutral", ("regime_is_neutral",)),
    AblationVariant(
        "drop_recurring_4h_offenders",
        (
            "component_agreement_ratio",
            "component_entropy",
            "p_up_ret_mismatch",
            "regime_is_neutral",
            "confluence_direction_matches_dominant",
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small set of trade-decision ablations and compare their metrics."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--signal-col", default="signal_ensemble")
    parser.add_argument("--target-col", default="ret_ensemble_net")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--working-dir",
        default="artifacts/analysis/trade_decision_ablation_sweep_runs",
        help="Directory where per-variant model artifacts are written.",
    )
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--feature-meta-path", default=None)
    parser.add_argument(
        "--reference-feature-mode",
        choices=("allow", "disable", "disable_on_source_mismatch"),
        default=None,
    )
    parser.add_argument("--reference-feature-expected-source", default=None)
    parser.add_argument("--reference-feature-max-abs-value", type=float, default=None)
    return parser.parse_args()


def _train_variant(args: argparse.Namespace, variant: AblationVariant, output_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_trade_decision_model",
        "--input",
        str(args.input),
        "--signal-col",
        str(args.signal_col),
        "--target-col",
        str(args.target_col),
        "--threshold",
        str(float(args.threshold)),
        "--output",
        str(output_path),
    ]
    if bool(args.candidate_only):
        cmd.append("--candidate-only")
    if args.feature_meta_path:
        cmd.extend(["--feature-meta-path", str(args.feature_meta_path)])
    if args.reference_feature_mode:
        cmd.extend(["--reference-feature-mode", str(args.reference_feature_mode)])
    if args.reference_feature_expected_source:
        cmd.extend(["--reference-feature-expected-source", str(args.reference_feature_expected_source)])
    if args.reference_feature_max_abs_value is not None:
        cmd.extend(["--reference-feature-max-abs-value", str(float(args.reference_feature_max_abs_value))])
    for feature_name in variant.excluded_features:
        cmd.extend(["--exclude-feature-columns", feature_name])

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "variant": variant.name,
        "excluded_features": list(variant.excluded_features),
        "output_path": str(output_path),
        "stdout_tail": completed.stdout[-4000:],
        "metrics": payload.get("metrics", {}),
        "deploy_ready": bool(payload.get("deploy_ready", False)),
        "deploy_readiness": payload.get("deploy_readiness", {}),
        "threshold": payload.get("threshold"),
        "feature_columns": payload.get("feature_columns", []),
        "excluded_feature_columns": payload.get("excluded_feature_columns", []),
    }


def _score_run(run: Dict[str, Any]) -> tuple[float, float, float]:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    oof_auc = float(metrics.get("oof_auc") if metrics.get("oof_auc") is not None else -1.0)
    auc = float(metrics.get("auc") if metrics.get("auc") is not None else -1.0)
    log_loss = float(metrics.get("log_loss") if metrics.get("log_loss") is not None else 999.0)
    return (oof_auc, auc, -log_loss)


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = ["# Trade Decision Ablation Sweep", ""]
    lines.append("## Ranking")
    for run in payload.get("runs", []):
        metrics = run.get("metrics", {})
        lines.append(
            f"- {run.get('variant')}: deploy_ready={run.get('deploy_ready')}, "
            f"oof_auc={metrics.get('oof_auc')}, auc={metrics.get('auc')}, log_loss={metrics.get('log_loss')}, "
            f"excluded={run.get('excluded_features')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for variant in DEFAULT_VARIANTS:
        output_path = working_dir / f"{variant.name}.json"
        runs.append(_train_variant(args, variant, output_path))

    runs.sort(key=_score_run, reverse=True)
    best_run = runs[0] if runs else None
    baseline = next((run for run in runs if run.get("variant") == "baseline"), None)

    recommendations: List[str] = []
    if best_run and baseline and best_run.get("variant") != "baseline":
        recommendations.append(
            f"The best ablation was `{best_run['variant']}`, which outperformed baseline on the sweep ranking and is the next candidate to inspect more closely for 4h trade-decision retraining."
        )
    else:
        recommendations.append(
            "Baseline remained the best run in this small ablation sweep, so simple feature dropping did not obviously improve the trade-decision model."
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "input": str(args.input),
            "signal_col": str(args.signal_col),
            "target_col": str(args.target_col),
            "threshold": float(args.threshold),
            "candidate_only": bool(args.candidate_only),
        },
        "runs": runs,
        "recommendations": recommendations,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote ablation JSON: {output_json}")
    print(f"Wrote ablation memo: {output_md}")


if __name__ == "__main__":
    main()