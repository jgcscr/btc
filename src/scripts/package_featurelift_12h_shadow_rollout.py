from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG = Path("configs/run_refresh_and_predict.shadow_simplified.yaml")
OUTPUT_CONFIG = Path("configs/run_refresh_and_predict.shadow_featurelift_12h_candidate.yaml")
OUTPUT_JSON = Path("artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_12h_package.json")
OUTPUT_MD = Path("artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_12h_package.md")
DEFAULT_DIRECTION_DIR = Path("artifacts/models/xgb_dir12h_v1")
DEFAULT_REGRESSION_DIR = Path("artifacts/models/xgb_ret12h_v1")
DEFAULT_WALKFORWARD = Path("artifacts/analysis/intrabar_featurelift_apr2026/walkforward_12h.json")
DEFAULT_TRADE_DECISION_MODEL = Path("artifacts/models/featurelift_20260331_rerun/trade_decision_model_full_history.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render_markdown(payload: dict[str, Any]) -> str:
    validation = payload["validation"]
    lines = [
        "# 12h Shadow Rollout Package",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Recommendation",
        "",
        "Use this package for 12h-only shadow or paper evaluation of intrabar interaction experiments.",
        "",
        "## Package",
        "",
        f"- Config: {payload['config_path']}",
        f"- Direction model dir: {payload['artifacts']['direction_model_dir']}",
        f"- Regression model dir: {payload['artifacts']['regression_model_dir']}",
        f"- Walkforward artifact: {payload['artifacts']['walkforward_path']}",
        f"- Trade decision model: {payload['artifacts']['trade_decision_model']}",
        "",
        "## Validation",
        "",
        f"- Walkforward AUC: {validation['walkforward_auc_mean']:.6f}",
        f"- Walkforward net return: {validation['walkforward_cum_ret_net_total']:.6f}",
        f"- Walkforward trades: {validation['walkforward_trade_count_total']}",
        f"- Direction test F1: {validation['direction_test_f1']:.6f}",
        f"- Regression test RMSE: {validation['regression_test_rmse']:.6f}",
        f"- Trade decision deploy-ready: {validation['trade_decision_deploy_ready']}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reusable 12h shadow rollout package for horizon-specific intrabar interaction evaluation.",
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    parser.add_argument("--direction-dir", type=Path, default=DEFAULT_DIRECTION_DIR)
    parser.add_argument("--regression-dir", type=Path, default=DEFAULT_REGRESSION_DIR)
    parser.add_argument("--walkforward-path", type=Path, default=DEFAULT_WALKFORWARD)
    parser.add_argument("--trade-decision-model", type=Path, default=DEFAULT_TRADE_DECISION_MODEL)
    args = parser.parse_args()

    config_payload = _load_yaml(args.base_config)
    regression_model_dirs = config_payload.get("regression_model_dirs")
    if not isinstance(regression_model_dirs, dict):
        regression_model_dirs = {"enabled": True}
    regression_model_dirs["enabled"] = True
    regression_model_dirs["12h"] = str(args.regression_dir)
    config_payload["regression_model_dirs"] = regression_model_dirs

    regime_model_dirs = config_payload.get("regime_model_dirs")
    if not isinstance(regime_model_dirs, dict):
        regime_model_dirs = {"enabled": True}
    for regime in ("trend_ignition", "neutral", "chop"):
        raw_regime_dirs = regime_model_dirs.get(regime)
        regime_dirs = dict(raw_regime_dirs) if isinstance(raw_regime_dirs, dict) else {}
        regime_dirs["12h"] = str(args.direction_dir)
        regime_model_dirs[regime] = regime_dirs
    regime_model_dirs["enabled"] = True
    config_payload["regime_model_dirs"] = regime_model_dirs

    trade_decision_policy = config_payload.get("trade_decision_policy")
    if not isinstance(trade_decision_policy, dict):
        trade_decision_policy = {}
    trade_decision_policy["model_path"] = str(args.trade_decision_model)
    config_payload["trade_decision_policy"] = trade_decision_policy

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    dir_summary = _load_json(args.direction_dir / "summary.json")
    ret_summary = _load_json(args.regression_dir / "summary.json")
    walkforward = _load_json(args.walkforward_path)
    decision_payload = _load_json(args.trade_decision_model)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.output_config),
        "artifacts": {
            "direction_model_dir": str(args.direction_dir),
            "regression_model_dir": str(args.regression_dir),
            "walkforward_path": str(args.walkforward_path),
            "trade_decision_model": str(args.trade_decision_model),
        },
        "validation": {
            "direction_test_f1": float(dir_summary["metrics"]["test"]["f1"]),
            "regression_test_rmse": float(ret_summary["metrics"]["test"]["rmse"]),
            "walkforward_auc_mean": float(walkforward["auc_mean"]),
            "walkforward_cum_ret_net_total": float(walkforward["cum_ret_net_total"]),
            "walkforward_trade_count_total": int(walkforward["trade_count_total"]),
            "trade_decision_deploy_ready": bool(decision_payload.get("deploy_ready", False)),
        },
        "notes": [
            "This package leaves 1h, 4h, and 8h artifacts unchanged.",
            "Use this config for 12h-specific shadow or paper evaluation only; do not treat it as a live-promotion config.",
            "The current April 27 trimmed-12h ablation underperformed; future 12h work should prefer additive interaction experiments over broad feature removal.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote rollout config to {args.output_config}")
    print(f"Wrote rollout package JSON to {args.output_json}")
    print(f"Wrote rollout package markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()