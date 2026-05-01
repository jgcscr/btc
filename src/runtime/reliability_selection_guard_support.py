from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from src.runtime.reliability_workflow_common import load_json


def normalize_selection_calibration_guard_rules(rules_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(rules_obj, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in rules_obj:
        if not isinstance(item, dict):
            continue
        regime_state = str(item.get("regime_state", "")).strip().lower()
        if not regime_state:
            continue
        try:
            min_p_up = float(item.get("min_p_up"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(min_p_up):
            continue
        normalized.append({"regime_state": regime_state, "min_p_up": min_p_up})
    return normalized


def dedupe_selection_calibration_guard_rules(
    rules: List[Dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for rule in rules:
        regime_state = str(rule.get("regime_state", "")).strip().lower()
        min_p_up = safe_float(rule.get("min_p_up"), default=float("nan"))
        if not regime_state or not np.isfinite(min_p_up):
            continue
        key = (regime_state, round(float(min_p_up), 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"regime_state": regime_state, "min_p_up": float(min_p_up)})
    return deduped


def load_reusable_selection_calibration_guard_rules(
    *,
    deployed_rule_path: Path,
    deploy_manifest_path: Path | None,
    expected_regime_col: str,
    expected_p_col: str,
) -> Dict[str, Any]:
    manifest_payload: Dict[str, Any] = {}
    if deploy_manifest_path is not None and deploy_manifest_path.exists():
        try:
            manifest_payload = load_json(deploy_manifest_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            manifest_payload = {}
        deployed_variant = str(manifest_payload.get("official_shadow_variant", "none")).strip().lower()
        if deployed_variant != "selection_calibration_guard":
            return {
                "enabled": False,
                "reason": "last_deployed_variant_mismatch",
                "rules": [],
                "source_path": str(deployed_rule_path),
                "source_run_id": manifest_payload.get("run_id"),
                "source_official_shadow_variant": deployed_variant,
            }

    if not deployed_rule_path.exists():
        return {
            "enabled": False,
            "reason": "deployed_rule_not_found",
            "rules": [],
            "source_path": str(deployed_rule_path),
            "source_run_id": manifest_payload.get("run_id"),
            "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
        }

    try:
        deployed_payload = load_json(deployed_rule_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {
            "enabled": False,
            "reason": "deployed_rule_invalid",
            "rules": [],
            "source_path": str(deployed_rule_path),
            "source_run_id": manifest_payload.get("run_id"),
            "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
        }

    if not bool(deployed_payload.get("enabled", False)):
        return {
            "enabled": False,
            "reason": "deployed_rule_disabled",
            "rules": [],
            "source_path": str(deployed_rule_path),
            "source_run_id": manifest_payload.get("run_id"),
            "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
        }

    regime_col = str(deployed_payload.get("regime_col", "")).strip()
    p_col = str(deployed_payload.get("p_col", "")).strip()
    if regime_col != expected_regime_col or p_col != expected_p_col:
        return {
            "enabled": False,
            "reason": "deployed_rule_schema_mismatch",
            "rules": [],
            "source_path": str(deployed_rule_path),
            "source_run_id": manifest_payload.get("run_id"),
            "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
            "expected_regime_col": expected_regime_col,
            "expected_p_col": expected_p_col,
            "actual_regime_col": regime_col,
            "actual_p_col": p_col,
        }

    rules = normalize_selection_calibration_guard_rules(deployed_payload.get("rules", []))
    if not rules:
        return {
            "enabled": False,
            "reason": "deployed_rule_empty",
            "rules": [],
            "source_path": str(deployed_rule_path),
            "source_run_id": manifest_payload.get("run_id"),
            "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
        }

    auto_derive_payload = deployed_payload.get("auto_derive", {}) if isinstance(deployed_payload, dict) else {}
    source_candidate_path = (
        str(auto_derive_payload.get("candidate_path"))
        if isinstance(auto_derive_payload, dict) and auto_derive_payload.get("candidate_path") is not None
        else None
    )

    return {
        "enabled": True,
        "reason": "reused_last_deployed",
        "rules": rules,
        "source_path": str(deployed_rule_path),
        "source_run_id": manifest_payload.get("run_id"),
        "source_official_shadow_variant": str(manifest_payload.get("official_shadow_variant", "none")),
        "source_candidate_path": source_candidate_path,
    }


def augment_selection_guard_candidate_floors(
    *,
    base_floors: List[float],
    reference_rules: List[Dict[str, Any]],
    step: float,
    lower_steps: int,
    upper_steps: int,
    safe_float: Callable[[Any, float], float],
) -> List[float]:
    floors = {round(float(value), 6) for value in base_floors if np.isfinite(float(value))}
    step_value = float(step)
    if not np.isfinite(step_value) or step_value <= 0.0:
        return sorted(floors)

    for rule in reference_rules:
        min_p_up = safe_float(rule.get("min_p_up"), default=float("nan"))
        if not np.isfinite(min_p_up):
            continue
        for offset in range(-max(int(lower_steps), 0), max(int(upper_steps), 0) + 1):
            candidate_floor = float(min_p_up + (float(offset) * step_value))
            if 0.0 <= candidate_floor <= 1.0 and np.isfinite(candidate_floor):
                floors.add(round(candidate_floor, 6))
    return sorted(floors)