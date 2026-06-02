from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


BASE_CONFIG = Path("configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
OUTPUT_CONFIG = Path("configs/run_refresh_and_predict.shadow_4h_ultra_conservative.yaml")
OUTPUT_JSON = Path("artifacts/analysis/4h_shadow_ultra_conservative_package.json")
OUTPUT_MD = Path("artifacts/analysis/4h_shadow_ultra_conservative_package.md")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_markdown(payload: dict[str, Any]) -> str:
    validation = payload.get("validation", {})
    lines = [
        "# 4h Direction Shadow Candidate Package",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Package",
        "",
        f"- Config: {payload['config_path']}",
        f"- Direction model dir: {payload['direction_model_dir']}",
        f"- Summary path: {payload['summary_path']}",
        f"- Calibration path: {payload.get('calibration_path')}",
        "",
        "## Validation",
        "",
        f"- Train accuracy: {validation.get('train_accuracy')}",
        f"- Val accuracy: {validation.get('val_accuracy')}",
        f"- Test accuracy: {validation.get('test_accuracy')}",
        f"- Train/val gap: {validation.get('train_val_gap')}",
        f"- Test f1: {validation.get('test_f1')}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a shadow config for a staged 4h direction challenger.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--confluence-min-support-ratio-4h", type=float, default=None)
    parser.add_argument("--confluence-min-aligned-horizons-4h", type=int, default=None)
    parser.add_argument("--trade-threshold-4h-neutral", type=float, default=None)
    args = parser.parse_args()

    config_payload = _load_yaml(args.base_config)

    regime_model_dirs = config_payload.get("regime_model_dirs")
    if not isinstance(regime_model_dirs, dict):
        regime_model_dirs = {"enabled": True}
    regime_model_dirs["enabled"] = True
    for regime in ("trend_ignition", "neutral", "chop"):
        raw_regime_dirs = regime_model_dirs.get(regime)
        regime_dirs = dict(raw_regime_dirs) if isinstance(raw_regime_dirs, dict) else {}
        regime_dirs["4h"] = str(args.direction_dir)
        regime_model_dirs[regime] = regime_dirs
    config_payload["regime_model_dirs"] = regime_model_dirs

    trust_policy = config_payload.get("trust_hardening_policy")
    if not isinstance(trust_policy, dict):
        trust_policy = {"enabled": True}
    summary_paths = trust_policy.get("model_summary_paths_by_horizon")
    if not isinstance(summary_paths, dict):
        summary_paths = {}
    summary_paths["4"] = str(args.summary_path)
    trust_policy["model_summary_paths_by_horizon"] = summary_paths
    config_payload["trust_hardening_policy"] = trust_policy

    if args.calibration_path is not None:
        config_payload["platt_calibration"] = str(args.calibration_path)

    if args.confluence_min_support_ratio_4h is not None or args.confluence_min_aligned_horizons_4h is not None:
        confluence_policy = config_payload.get("confluence_policy")
        if not isinstance(confluence_policy, dict):
            confluence_policy = {"enabled": True}
        if args.confluence_min_support_ratio_4h is not None:
            min_support_ratio_by_horizon = confluence_policy.get("min_support_ratio_by_horizon")
            if not isinstance(min_support_ratio_by_horizon, dict):
                min_support_ratio_by_horizon = {}
            min_support_ratio_by_horizon["4"] = float(args.confluence_min_support_ratio_4h)
            confluence_policy["min_support_ratio_by_horizon"] = min_support_ratio_by_horizon
        if args.confluence_min_aligned_horizons_4h is not None:
            min_aligned_horizons_by_horizon = confluence_policy.get("min_aligned_horizons_by_horizon")
            if not isinstance(min_aligned_horizons_by_horizon, dict):
                min_aligned_horizons_by_horizon = {}
            min_aligned_horizons_by_horizon["4"] = int(args.confluence_min_aligned_horizons_4h)
            confluence_policy["min_aligned_horizons_by_horizon"] = min_aligned_horizons_by_horizon
        config_payload["confluence_policy"] = confluence_policy

    if args.trade_threshold_4h_neutral is not None:
        trade_policy = config_payload.get("trade_decision_policy")
        if not isinstance(trade_policy, dict):
            trade_policy = {"enabled": True}
        thresholds_by_horizon_regime = trade_policy.get("thresholds_by_horizon_regime")
        if not isinstance(thresholds_by_horizon_regime, dict):
            thresholds_by_horizon_regime = {}
        horizon_thresholds = thresholds_by_horizon_regime.get("4")
        if not isinstance(horizon_thresholds, dict):
            horizon_thresholds = {}
        horizon_thresholds["neutral"] = float(args.trade_threshold_4h_neutral)
        thresholds_by_horizon_regime["4"] = horizon_thresholds
        trade_policy["thresholds_by_horizon_regime"] = thresholds_by_horizon_regime
        config_payload["trade_decision_policy"] = trade_policy

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    summary = _load_json(args.summary_path)
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    train = metrics.get("train") if isinstance(metrics.get("train"), dict) else {}
    val = metrics.get("val") if isinstance(metrics.get("val"), dict) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    train_accuracy = float(train.get("accuracy")) if train.get("accuracy") is not None else None
    val_accuracy = float(val.get("accuracy")) if val.get("accuracy") is not None else None
    train_val_gap = abs(train_accuracy - val_accuracy) if train_accuracy is not None and val_accuracy is not None else None

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.output_config),
        "direction_model_dir": str(args.direction_dir),
        "summary_path": str(args.summary_path),
        "calibration_path": str(args.calibration_path) if args.calibration_path else None,
        "validation": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": float(test.get("accuracy")) if test.get("accuracy") is not None else None,
            "train_val_gap": train_val_gap,
            "test_f1": float(test.get("f1")) if test.get("f1") is not None else None,
        },
        "notes": [
            "This package overrides the 4h direction model directory and the 4h trust summary path.",
            "When provided, it also overrides the runtime platt_calibration path for the shadow replay.",
            "When provided, it can also override the 4h confluence thresholds for shadow-only validation.",
            "When provided, it can also override the 4h neutral trade-decision threshold for shadow-only validation.",
            "All non-4h horizons and the rest of the live policy remain unchanged.",
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