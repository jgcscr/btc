from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


def build_trade_decision_model_shift_guard(
    *,
    model_shift_payload: Dict[str, Any] | None,
    guard_cfg: Dict[str, Any],
    safe_float: Callable[[Any, float], float],
) -> Dict[str, Any]:
    enabled = bool(guard_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "passed": True, "reason": "disabled", "checks": {}, "failed_checks": []}

    payload = model_shift_payload if isinstance(model_shift_payload, dict) else {}
    available = bool(payload.get("available", False))
    fail_when_unavailable = bool(guard_cfg.get("fail_when_unavailable", False))
    if not available:
        passed = not fail_when_unavailable
        failed_checks = [] if passed else ["model_shift_unavailable"]
        return {
            "enabled": True,
            "available": False,
            "passed": passed,
            "reason": "model_shift_unavailable",
            "checks": {"model_shift_available": passed},
            "failed_checks": failed_checks,
        }

    coef_entries = {
        str(item.get("feature")): item
        for item in payload.get("top_coefficient_deltas", [])
        if isinstance(item, dict)
    }
    reference_sources = payload.get("reference_sources", {}) if isinstance(payload.get("reference_sources", {}), dict) else {}
    current_reference = reference_sources.get("current", {}) if isinstance(reference_sources.get("current", {}), dict) else {}
    source_reference = reference_sources.get("source", {}) if isinstance(reference_sources.get("source", {}), dict) else {}
    counterfactual = payload.get("counterfactual_threshold_pass", {}) if isinstance(payload.get("counterfactual_threshold_pass", {}), dict) else {}

    intercept_delta = abs(safe_float((coef_entries.get("__intercept__") or {}).get("coef_delta"), default=0.0))
    reference_feature_names = [
        "incumbent_signal_reference",
        "candidate_only_reference",
        "candidate_incumbent_disagreement",
    ]
    max_reference_coef_delta = max(
        abs(safe_float((coef_entries.get(name) or {}).get("coef_delta"), default=0.0))
        for name in reference_feature_names
    )
    source_not_current_count = int(safe_float(counterfactual.get("source_not_current_count"), default=0.0))
    require_reference_source_stable = bool(guard_cfg.get("require_reference_source_stable", False))
    reference_source_stable = str(current_reference.get("source")) == str(source_reference.get("source"))

    checks = {
        "intercept_delta_ok": intercept_delta <= float(guard_cfg.get("max_abs_intercept_delta", 1e9)),
        "reference_coef_delta_ok": max_reference_coef_delta <= float(guard_cfg.get("max_abs_reference_coef_delta", 1e9)),
        "source_not_current_count_ok": source_not_current_count <= int(guard_cfg.get("max_source_not_current_count", 10**9)),
        "reference_source_stable": True if not require_reference_source_stable else bool(reference_source_stable),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    return {
        "enabled": True,
        "available": True,
        "passed": len(failed_checks) == 0,
        "checks": checks,
        "failed_checks": failed_checks,
        "details": {
            "intercept_delta": intercept_delta,
            "max_reference_coef_delta": max_reference_coef_delta,
            "source_not_current_count": source_not_current_count,
            "current_reference_source": current_reference.get("source"),
            "source_reference_source": source_reference.get("source"),
        },
    }


def apply_trade_decision_model_shift_guard(
    *,
    summary_dir: Path,
    promotion_gate_payload: Dict[str, Any] | None,
    trade_decision_cfg: Dict[str, Any],
    model_shift_payload: Dict[str, Any] | None,
    safe_float: Callable[[Any, float], float],
) -> Dict[str, Any]:
    gate_payload = dict(promotion_gate_payload) if isinstance(promotion_gate_payload, dict) else {}
    model_shift_guard_cfg = (
        trade_decision_cfg.get("model_shift_guard")
        if isinstance(trade_decision_cfg.get("model_shift_guard"), dict)
        else {}
    )
    model_shift_guard_payload = build_trade_decision_model_shift_guard(
        model_shift_payload=model_shift_payload,
        guard_cfg=model_shift_guard_cfg,
        safe_float=safe_float,
    )
    (summary_dir / "trade_decision_model_shift_guard.json").write_text(
        json.dumps(model_shift_guard_payload, indent=2),
        encoding="utf-8",
    )
    gate_payload["trade_decision_model_shift_guard"] = model_shift_guard_payload
    if bool(model_shift_guard_payload.get("enabled", False)) and not bool(model_shift_guard_payload.get("passed", True)):
        existing_failed_checks = gate_payload.get("failed_checks", []) if isinstance(gate_payload.get("failed_checks", []), list) else []
        existing_failed_checks.extend(
            [
                f"trade_decision_model_shift_guard:{name}"
                for name in model_shift_guard_payload.get("failed_checks", [])
            ]
        )
        gate_payload["failed_checks"] = existing_failed_checks
        gate_payload["promote"] = False
        gate_payload["reason"] = "trade_decision_model_shift_guard_failed"
    return gate_payload