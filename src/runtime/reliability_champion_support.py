from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.runtime.reliability_shadow_variant_support import (
    shadow_variant_uses_reference_feature_ablation_model,
)
from src.runtime.reliability_workflow_common import load_json


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def extract_trade_decision_reference_source(feature_meta_path: Path | None) -> str | None:
    if feature_meta_path is None or not feature_meta_path.exists():
        return None
    try:
        payload = load_json(feature_meta_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None
    incumbent_reference = payload.get("incumbent_reference", {}) if isinstance(payload, dict) else {}
    if not isinstance(incumbent_reference, dict):
        return None
    source = incumbent_reference.get("source")
    if source in {None, ""}:
        return None
    return str(source)


def resolve_trade_decision_model_path_for_variant(summary_dir: Path, official_shadow_variant: str) -> Path:
    if shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant):
        ablation_model_path = summary_dir / "trade_decision_model_reference_feature_ablation.json"
        if ablation_model_path.exists():
            return ablation_model_path
    return summary_dir / "trade_decision_model.json"


def resolve_effective_champion_gate(
    *,
    summary_dir: Path,
    champion_gate_payload: Dict[str, Any] | None,
    official_shadow_variant: str,
    champion_gate_source: str,
    policy_aligned_gate_path: Path | None = None,
) -> tuple[Path, Dict[str, Any] | None, Dict[str, Any]]:
    labeled_gate_path = summary_dir / "champion_challenger_gate.json"
    policy_gate_path = policy_aligned_gate_path or (summary_dir / "champion_challenger_policy_aligned_companion.json")
    selected_source = "labeled"
    effective_gate_path = labeled_gate_path
    effective_gate_payload = champion_gate_payload if isinstance(champion_gate_payload, dict) else None

    normalized_source = str(champion_gate_source or "labeled").strip().lower()
    allow_policy_aligned = normalized_source in {"auto", "policy_aligned", "policy_aligned_official_shadow"}
    require_policy_aligned = normalized_source == "policy_aligned"
    auto_policy_aligned = normalized_source in {"auto", "policy_aligned_official_shadow"}
    policy_shadow_active = str(official_shadow_variant or "none").strip().lower() != "none"

    if policy_gate_path.exists() and allow_policy_aligned and (
        require_policy_aligned or (auto_policy_aligned and policy_shadow_active)
    ):
        effective_gate_path = policy_gate_path
        effective_gate_payload = load_json(policy_gate_path)
        selected_source = "policy_aligned"

    resolution = {
        "configured_source": normalized_source,
        "selected_source": selected_source,
        "official_shadow_variant": str(official_shadow_variant or "none"),
        "labeled_gate_path": str(labeled_gate_path),
        "policy_aligned_gate_path": str(policy_gate_path),
        "effective_gate_path": str(effective_gate_path),
        "policy_aligned_available": bool(policy_gate_path.exists()),
    }
    return effective_gate_path, effective_gate_payload, resolution


def build_champion_gate_alignment_check(
    *,
    summary_dir: Path,
    official_shadow_variant: str,
    champion_gate_source: str,
    selection_payload: Dict[str, Any] | None,
    effective_champion_gate_path: Path,
    effective_champion_gate_payload: Dict[str, Any] | None,
    champion_gate_resolution: Dict[str, Any],
    policy_aligned_gate_path: Path | None = None,
) -> Dict[str, Any]:
    normalized_variant = str(official_shadow_variant or "none").strip().lower()
    configured_source = str(champion_gate_source or "labeled").strip().lower()
    labeled_gate_path = summary_dir / "champion_challenger_gate.json"
    policy_gate_path = policy_aligned_gate_path or (summary_dir / "champion_challenger_policy_aligned_companion.json")
    expected_source = "labeled"
    if (
        normalized_variant != "none"
        and configured_source in {"auto", "policy_aligned", "policy_aligned_official_shadow"}
        and policy_gate_path.exists()
    ):
        expected_source = "policy_aligned"

    selection_candidate: Dict[str, Any] | None = None
    if isinstance(selection_payload, dict):
        for candidate in selection_payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("variant", "")).strip().lower() == normalized_variant:
                selection_candidate = candidate
                break

    expected_gate_path = policy_gate_path if expected_source == "policy_aligned" else labeled_gate_path
    selected_source = str(champion_gate_resolution.get("selected_source", "labeled")).strip().lower()
    errors: List[str] = []

    if selected_source != expected_source:
        errors.append(f"selected_source_mismatch expected={expected_source} actual={selected_source}")
    if expected_gate_path != effective_champion_gate_path:
        errors.append(f"effective_gate_path_mismatch expected={expected_gate_path} actual={effective_champion_gate_path}")

    companion_payload = selection_candidate.get("companion", {}) if isinstance(selection_candidate, dict) else {}
    effective_stats = effective_champion_gate_payload.get("stats", {}) if isinstance(effective_champion_gate_payload, dict) else {}
    companion_promote = None
    companion_mean_diff = None
    companion_pvalue = None
    if expected_source == "policy_aligned" and isinstance(companion_payload, dict) and companion_payload:
        companion_promote = bool(companion_payload.get("promote", False))
        companion_mean_diff = _safe_float(companion_payload.get("mean_diff"), default=float("nan"))
        companion_pvalue = _safe_float(companion_payload.get("pvalue_one_sided"), default=float("nan"))
        effective_promote = bool((effective_champion_gate_payload or {}).get("promote", False))
        effective_mean_diff = _safe_float(effective_stats.get("mean_diff"), default=float("nan"))
        effective_pvalue = _safe_float(effective_stats.get("pvalue_one_sided"), default=float("nan"))
        if companion_promote != effective_promote:
            errors.append(f"effective_promote_mismatch expected={companion_promote} actual={effective_promote}")
        if np.isfinite(companion_mean_diff) and np.isfinite(effective_mean_diff) and abs(companion_mean_diff - effective_mean_diff) > 1e-12:
            errors.append(f"effective_mean_diff_mismatch expected={companion_mean_diff} actual={effective_mean_diff}")
        if np.isfinite(companion_pvalue) and np.isfinite(effective_pvalue) and abs(companion_pvalue - effective_pvalue) > 1e-12:
            errors.append(f"effective_pvalue_mismatch expected={companion_pvalue} actual={effective_pvalue}")

    return {
        "passed": not errors,
        "official_shadow_variant": normalized_variant,
        "configured_source": configured_source,
        "expected_source": expected_source,
        "selected_source": selected_source,
        "expected_gate_path": str(expected_gate_path),
        "effective_gate_path": str(effective_champion_gate_path),
        "selection_candidate_found": bool(selection_candidate is not None),
        "selection_candidate_companion": {
            "promote": companion_promote,
            "mean_diff": None if companion_mean_diff is None or not np.isfinite(companion_mean_diff) else companion_mean_diff,
            "pvalue_one_sided": None if companion_pvalue is None or not np.isfinite(companion_pvalue) else companion_pvalue,
        },
        "effective_gate": {
            "promote": bool((effective_champion_gate_payload or {}).get("promote", False)),
            "mean_diff": effective_stats.get("mean_diff") if isinstance(effective_stats, dict) else None,
            "pvalue_one_sided": effective_stats.get("pvalue_one_sided") if isinstance(effective_stats, dict) else None,
        },
        "errors": errors,
    }