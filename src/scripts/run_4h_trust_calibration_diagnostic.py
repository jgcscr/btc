from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from src.runtime.trust_hardening_support import (
    _calibration_divergence_is_suspicious,
    _evaluate_metadata_risk,
    resolve_trust_hardening_policy,
)
from src.scripts.run_live_inference import DEFAULT_LIVE_CONFIG


DEFAULT_TRADE_READY_PATH = "artifacts/monitoring/trade_ready_summary.json"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/trust_calibration_4h_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/trust_calibration_4h_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the current 4h trust and calibration blocker.")
    parser.add_argument("--config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--trade-ready-path", default=DEFAULT_TRADE_READY_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _normalize_horizon_value(value: Any) -> float:
    return float(value)


def _coerce_numeric_horizon(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _direction_from_probability(probability: Any, *, neutral_band: float = 0.0) -> str:
    value = _finite_float_or_none(probability)
    if value is None:
        return "unknown"
    lower = 0.5 - float(neutral_band)
    upper = 0.5 + float(neutral_band)
    if value < lower:
        return "down"
    if value > upper:
        return "up"
    return "neutral"


def _extract_4h_entry(trade_ready_payload: Mapping[str, Any]) -> Dict[str, Any]:
    for entry in trade_ready_payload.get("horizons", []):
        if not isinstance(entry, Mapping):
            continue
        horizon_hours = _finite_float_or_none(entry.get("horizon_hours"))
        if horizon_hours is not None and abs(horizon_hours - 4.0) < 1e-9:
            return dict(entry)
    raise ValueError("4h horizon not found in trade_ready_summary")


def _metadata_summary(path: Path, metadata_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    reasons, missing = _evaluate_metadata_risk(4.0, {4.0: str(path)}, metadata_cfg)
    payload = _read_json(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    train = metrics.get("train") if isinstance(metrics.get("train"), Mapping) else {}
    val = metrics.get("val") if isinstance(metrics.get("val"), Mapping) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), Mapping) else {}
    train_acc = _finite_float_or_none(train.get("accuracy"))
    val_acc = _finite_float_or_none(val.get("accuracy"))
    test_acc = _finite_float_or_none(test.get("accuracy"))
    gap = None
    if train_acc is not None and val_acc is not None:
        gap = abs(train_acc - val_acc)
    return {
        "summary_path": str(path),
        "reasons": reasons,
        "missing": bool(missing),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_val_gap": gap,
        "max_train_val_accuracy_gap": float(metadata_cfg.get("max_train_val_accuracy_gap", 0.03)),
    }


def _calibration_summary(entry: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    suspicious = _calibration_divergence_is_suspicious(
        entry,
        policy=policy,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )
    calibration = entry.get("probability_calibration") if isinstance(entry.get("probability_calibration"), Mapping) else {}
    return {
        "suspicious": bool(suspicious),
        "raw_probability": _finite_float_or_none(entry.get("raw_p_up") or calibration.get("raw_probability")),
        "resolved_probability": _finite_float_or_none(entry.get("p_up") or calibration.get("resolved_probability")),
        "probability_alignment_gap": _finite_float_or_none(entry.get("probability_alignment_gap")),
        "divergence_abs_gap_min": float(policy.get("divergence_abs_gap_min", 0.12)),
        "raw_side": calibration.get("raw_side") or entry.get("raw_p_up_side"),
        "resolved_side": calibration.get("resolved_side") or entry.get("resolved_p_up_side"),
        "divergence_flip_required": bool(policy.get("divergence_flip_required", True)),
    }


def _recommendations(metadata: Mapping[str, Any], calibration: Mapping[str, Any], live_entry: Mapping[str, Any]) -> list[str]:
    recommendations: list[str] = []
    train_val_gap = _finite_float_or_none(metadata.get("train_val_gap"))
    gap_limit = _finite_float_or_none(metadata.get("max_train_val_accuracy_gap"))
    if train_val_gap is not None and gap_limit is not None and train_val_gap > gap_limit:
        recommendations.append(
            "Retrain or replace the 4h model bundle; the train/val accuracy gap materially exceeds the trust policy threshold."
        )
    if bool(calibration.get("suspicious")):
        recommendations.append(
            "Re-fit 4h calibration using current labeled data; the calibrated probability flips side relative to the raw probability with a large gap."
        )
    if str(live_entry.get("trust_status")) != "trusted":
        recommendations.append(
            "Keep the 4h horizon deweighted in live inference until both metadata and calibration blockers clear."
        )
    return recommendations


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h Trust Calibration Diagnostic", ""]
    live = payload.get("live_4h", {})
    metadata = payload.get("metadata", {})
    calibration = payload.get("calibration", {})
    lines.append("## Live State")
    lines.append(f"- trust_status: {live.get('trust_status')}")
    lines.append(f"- trust_reasons: {', '.join(str(item) for item in (live.get('trust_reasons') or [])) or 'none'}")
    lines.append(f"- trust_hardening_action: {live.get('trust_hardening_action')}")
    lines.append(f"- voting_weight_after_trust: {live.get('voting_weight_after_trust')}")
    lines.append("")
    lines.append("## Metadata Risk")
    lines.append(f"- summary_path: {metadata.get('summary_path')}")
    lines.append(f"- train_accuracy: {metadata.get('train_accuracy')}")
    lines.append(f"- val_accuracy: {metadata.get('val_accuracy')}")
    lines.append(f"- test_accuracy: {metadata.get('test_accuracy')}")
    lines.append(f"- train_val_gap: {metadata.get('train_val_gap')}")
    lines.append(f"- allowed_train_val_gap: {metadata.get('max_train_val_accuracy_gap')}")
    lines.append(f"- metadata_reasons: {', '.join(str(item) for item in (metadata.get('reasons') or [])) or 'none'}")
    lines.append("")
    lines.append("## Calibration Risk")
    lines.append(f"- suspicious: {calibration.get('suspicious')}")
    lines.append(f"- raw_probability: {calibration.get('raw_probability')}")
    lines.append(f"- resolved_probability: {calibration.get('resolved_probability')}")
    lines.append(f"- probability_alignment_gap: {calibration.get('probability_alignment_gap')}")
    lines.append(f"- divergence_abs_gap_min: {calibration.get('divergence_abs_gap_min')}")
    lines.append(f"- raw_side: {calibration.get('raw_side')}")
    lines.append(f"- resolved_side: {calibration.get('resolved_side')}")
    lines.append("")
    lines.append("## Recommendations")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    trade_ready_path = Path(args.trade_ready_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    trust_policy = resolve_trust_hardening_policy(
        config_payload.get("trust_hardening_policy"),
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    summary_paths = trust_policy.get("model_summary_paths_by_horizon") if isinstance(trust_policy.get("model_summary_paths_by_horizon"), Mapping) else {}
    summary_path = Path(str(summary_paths.get(4.0, "artifacts/models/lgbm_dir4h_v1/summary.json")))

    trade_ready_payload = _read_json(trade_ready_path)
    live_entry = _extract_4h_entry(trade_ready_payload)
    metadata_cfg = trust_policy.get("metadata_checks") if isinstance(trust_policy.get("metadata_checks"), Mapping) else {}

    metadata = _metadata_summary(summary_path, metadata_cfg)
    calibration = _calibration_summary(live_entry, trust_policy)
    recommendations = _recommendations(metadata, calibration, live_entry)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "trade_ready_path": str(trade_ready_path),
        },
        "live_4h": {
            "trust_status": live_entry.get("trust_status"),
            "trust_reasons": live_entry.get("trust_reasons") or [],
            "trust_hardening_action": live_entry.get("trust_hardening_action"),
            "voting_weight_after_trust": live_entry.get("voting_weight_after_trust"),
            "p_up": live_entry.get("p_up"),
            "raw_p_up": live_entry.get("raw_p_up"),
            "direction_next": live_entry.get("direction_next"),
            "regime_state": live_entry.get("regime_state"),
        },
        "metadata": metadata,
        "calibration": calibration,
        "recommendations": recommendations,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Wrote 4h diagnostic JSON: {output_json}")
    print(f"Wrote 4h diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()