from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT_CSV = "artifacts/monitoring/labeled_backtest_4h_regime.csv"
DEFAULT_INCUMBENT_REFERENCE = "artifacts/monitoring/labeled_backtest_1h_incumbent.csv"
DEFAULT_REPLAY_LOG = "artifacts/tmp/shadow_4h_ultra_conservative_12h_beta_confluence075_live_replay.log"
DEFAULT_OUTPUT_DIR = "artifacts/analysis/4h_trade_decision_probe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 4h-specific trade-decision probe model and rescore the fixed 4h replay."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--incumbent-reference-source", default=DEFAULT_INCUMBENT_REFERENCE)
    parser.add_argument("--replay-log", default=DEFAULT_REPLAY_LOG)
    parser.add_argument("--signal-col", default="signal_dir_only")
    parser.add_argument("--target-col", default="ret_realized")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _run_command(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_csv = output_dir / "labeled_backtest_4h_regime_enriched.csv"
    enriched_meta = output_dir / "labeled_backtest_4h_regime_enriched_meta.json"
    model_json = output_dir / "trade_decision_model_4h_probe.json"
    contribution_json = output_dir / "trade_decision_probe_contribution.json"
    contribution_md = output_dir / "trade_decision_probe_contribution.md"
    workflow_json = output_dir / "trade_decision_probe_workflow.json"

    enrich_cmd = [
        sys.executable,
        "-m",
        "src.scripts.enrich_backtest_with_decision_features",
        "--input",
        str(args.input_csv),
        "--output",
        str(enriched_csv),
        "--meta-output",
        str(enriched_meta),
        "--auto-discover-sources",
    ]
    if args.incumbent_reference_source:
        enrich_cmd.extend(["--incumbent-reference-source", str(args.incumbent_reference_source)])

    train_cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_trade_decision_model",
        "--input",
        str(enriched_csv),
        "--signal-col",
        str(args.signal_col),
        "--target-col",
        str(args.target_col),
        "--threshold",
        str(float(args.threshold)),
        "--feature-meta-path",
        str(enriched_meta),
        "--output",
        str(model_json),
    ]
    if bool(args.candidate_only):
        train_cmd.append("--candidate-only")

    contribution_cmd = [
        sys.executable,
        "-m",
        "src.scripts.run_4h_trade_decision_contribution_diagnostic",
        "--log-path",
        str(args.replay_log),
        "--model-path",
        str(model_json),
        "--output-json",
        str(contribution_json),
        "--output-md",
        str(contribution_md),
        "--top-k",
        str(int(args.top_k)),
    ]

    enrich_run = _run_command(enrich_cmd)
    train_run = _run_command(train_cmd)
    contribution_run = _run_command(contribution_cmd)

    enrich_meta_payload = json.loads(enriched_meta.read_text(encoding="utf-8"))
    model_payload = json.loads(model_json.read_text(encoding="utf-8"))
    contribution_payload = json.loads(contribution_json.read_text(encoding="utf-8"))

    result: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "input_csv": str(args.input_csv),
            "incumbent_reference_source": str(args.incumbent_reference_source),
            "replay_log": str(args.replay_log),
            "signal_col": str(args.signal_col),
            "target_col": str(args.target_col),
            "threshold": float(args.threshold),
            "candidate_only": bool(args.candidate_only),
        },
        "artifacts": {
            "enriched_csv": str(enriched_csv),
            "enriched_meta": str(enriched_meta),
            "model_json": str(model_json),
            "contribution_json": str(contribution_json),
            "contribution_md": str(contribution_md),
        },
        "enrichment": {
            "missing_after": enrich_meta_payload.get("missing_after"),
            "backfill_by_column": enrich_meta_payload.get("backfill_by_column"),
            "incumbent_reference": enrich_meta_payload.get("incumbent_reference"),
            "stdout_tail": enrich_run.stdout[-4000:],
        },
        "training": {
            "deploy_ready": model_payload.get("deploy_ready"),
            "metrics": model_payload.get("metrics"),
            "excluded_feature_columns": model_payload.get("excluded_feature_columns"),
            "stdout_tail": train_run.stdout[-4000:],
        },
        "replay_rescore": {
            "trade_probability": contribution_payload.get("trade_probability"),
            "reconstructed_probability": contribution_payload.get("reconstructed_probability"),
            "threshold": contribution_payload.get("threshold"),
            "replay_threshold_gap": contribution_payload.get("replay_threshold_gap"),
            "reconstructed_threshold_gap": contribution_payload.get("reconstructed_threshold_gap"),
            "top_negative_contributions": contribution_payload.get("top_negative_contributions"),
            "stdout_tail": contribution_run.stdout[-4000:],
        },
    }

    workflow_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()