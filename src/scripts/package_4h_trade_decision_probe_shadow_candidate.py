from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG = Path("configs/run_refresh_and_predict.shadow_4h_ultra_conservative_12h_beta_confluence075.yaml")
OUTPUT_CONFIG = Path("configs/run_refresh_and_predict.shadow_4h_trade_decision_probe.yaml")
OUTPUT_JSON = Path("artifacts/analysis/4h_trade_decision_probe/shadow_package.json")
OUTPUT_MD = Path("artifacts/analysis/4h_trade_decision_probe/shadow_package.md")
DEFAULT_PROBE_MODEL = Path("artifacts/analysis/4h_trade_decision_probe/trade_decision_model_4h_probe.json")
DEFAULT_PROBE_WORKFLOW = Path("artifacts/analysis/4h_trade_decision_probe/trade_decision_probe_workflow.json")
DEFAULT_REPLAY_SWEEP = Path("artifacts/analysis/4h_trade_decision_probe/replay_sweep")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_threshold_clears(replay_sweep_dir: Path) -> tuple[int, int]:
    if not replay_sweep_dir.exists():
        return 0, 0
    total = 0
    clears = 0
    for path in sorted(replay_sweep_dir.glob("*.json")):
        payload = _load_json(path)
        total += 1
        gap = payload.get("reconstructed_threshold_gap")
        if gap is not None and float(gap) >= 0.0:
            clears += 1
    return clears, total


def _render_markdown(payload: dict[str, Any]) -> str:
    validation = payload.get("validation", {})
    lines = [
        "# 4h Trade Decision Probe Shadow Package",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Package",
        "",
        f"- Config: {payload['config_path']}",
        f"- Base config: {payload['base_config']}",
        f"- Trade decision model: {payload['trade_decision_model']}",
        "",
        "## Validation",
        "",
        f"- Deploy ready: {validation.get('deploy_ready')}",
        f"- Candidate rows: {validation.get('candidate_rows')}",
        f"- AUC: {validation.get('auc')}",
        f"- OOF AUC: {validation.get('oof_auc')}",
        f"- Log loss: {validation.get('log_loss')}",
        f"- Fixed replay reconstructed probability: {validation.get('fixed_replay_reconstructed_probability')}",
        f"- Fixed replay reconstructed threshold gap: {validation.get('fixed_replay_reconstructed_threshold_gap')}",
        f"- Replay sweep clears: {validation.get('replay_sweep_clears')}/{validation.get('replay_sweep_total')}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the 4h trade-decision probe model into a shadow config.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    parser.add_argument("--trade-decision-model", type=Path, default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--probe-workflow", type=Path, default=DEFAULT_PROBE_WORKFLOW)
    parser.add_argument("--replay-sweep-dir", type=Path, default=DEFAULT_REPLAY_SWEEP)
    args = parser.parse_args()

    config_payload = _load_yaml(args.base_config)
    trade_decision_policy = config_payload.get("trade_decision_policy")
    if not isinstance(trade_decision_policy, dict):
        trade_decision_policy = {"enabled": True}
    trade_decision_policy["model_path"] = str(args.trade_decision_model)
    config_payload["trade_decision_policy"] = trade_decision_policy

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    workflow_payload = _load_json(args.probe_workflow)
    training = workflow_payload.get("training") if isinstance(workflow_payload.get("training"), dict) else {}
    metrics = training.get("metrics") if isinstance(training.get("metrics"), dict) else {}
    replay_rescore = workflow_payload.get("replay_rescore") if isinstance(workflow_payload.get("replay_rescore"), dict) else {}
    clears, total = _count_threshold_clears(args.replay_sweep_dir)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.output_config),
        "base_config": str(args.base_config),
        "trade_decision_model": str(args.trade_decision_model),
        "validation": {
            "deploy_ready": bool(training.get("deploy_ready", False)),
            "candidate_rows": metrics.get("candidate_rows"),
            "auc": metrics.get("auc"),
            "oof_auc": metrics.get("oof_auc"),
            "log_loss": metrics.get("log_loss"),
            "fixed_replay_reconstructed_probability": replay_rescore.get("reconstructed_probability"),
            "fixed_replay_reconstructed_threshold_gap": replay_rescore.get("reconstructed_threshold_gap"),
            "replay_sweep_clears": clears,
            "replay_sweep_total": total,
        },
        "notes": [
            "This package keeps the upstream-cleared 4h candidate config and only overrides the trade-decision model path.",
            "Use this config for shadow validation first; do not treat it as a live promotion config.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote shadow config to {args.output_config}")
    print(f"Wrote package JSON to {args.output_json}")
    print(f"Wrote package markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()