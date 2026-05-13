from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from src.runtime.feature_parity_audit import classify_feature_family


DERIVATIVES_FEATURES: tuple[str, ...] = (
    "funding_rate",
    "funding_rate_annualized",
    "funding_rate_zscore_24h",
    "open_interest",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "fut_close_delta_1h",
    "fut_close_pct_change_1h",
    "fut_close_zscore_7h",
    "fut_volume_delta_1h",
    "fut_volume_pct_change_1h",
)

DEFAULT_ANALYSIS_DIR = Path("artifacts/analysis")
DEFAULT_MODELS_ROOT = Path("artifacts/models")
DEFAULT_FUNDING_DIR = Path("data/processed/funding")
DEFAULT_SPOT_DIR = Path("data/spot_klines")
DEFAULT_DATASET_15M_PATH = Path("artifacts/datasets/btc_features_15m_splits.npz")
DEFAULT_DATASET_15M_DIRECTION_PATH = Path("artifacts/datasets/btc_features_15m_direction_splits.npz")
DEFAULT_DATASET_MULTI_PATH = Path("artifacts/datasets/btc_features_multi_horizon_splits.npz")
DEFAULT_DATASET_1H_PATH = Path("artifacts/datasets/btc_features_1h_splits.npz")
DEFAULT_DATASET_1H_DIRECTION_PATH = Path("artifacts/datasets/btc_features_1h_direction_splits.npz")


@dataclass(frozen=True)
class DerivativesPolicySpec:
    name: str
    description: str
    required_features: tuple[str, ...]
    intended_scope: str
    status: str
    rationale: str


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_go_hold(payload: Mapping[str, Any], family: str) -> str | None:
    rankings = payload.get("family_best_rankings")
    if not isinstance(rankings, list):
        return None
    for row in rankings:
        if not isinstance(row, Mapping) or str(row.get("family")) != family:
            continue
        decision = row.get("go_hold") if isinstance(row.get("go_hold"), Mapping) else {}
        return str(decision.get("decision")) if decision.get("decision") is not None else None
    return None


def _extract_family_best(payload: Mapping[str, Any], family: str) -> Mapping[str, Any] | None:
    rankings = payload.get("family_best_rankings")
    if not isinstance(rankings, list):
        return None
    for row in rankings:
        if isinstance(row, Mapping) and str(row.get("family")) == family:
            return row
    return None


def _parse_targets(raw: Any) -> List[float]:
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
        return [float(value) for value in values]
    return [0.25, 1.0, 4.0, 12.0]


def _horizon_label(value: float) -> str:
    if value >= 1.0 and float(value).is_integer():
        return f"{int(value)}h"
    minutes = int(round(value * 60))
    return f"{minutes}m"


def _extract_feature_names(path: Path) -> List[str]:
    payload = _load_json(path) or {}
    names = payload.get("feature_names")
    if not isinstance(names, list):
        return []
    return [str(value) for value in names]


def _collect_training_features(models_root: Path, horizons: Sequence[float]) -> Dict[str, List[str]]:
    by_horizon: Dict[str, List[str]] = {}
    for horizon in horizons:
        label = _horizon_label(float(horizon))
        direction_meta = sorted(models_root.glob(f"xgb_dir{label}_v*/model_metadata_direction.json"))
        regression_meta = sorted(models_root.glob(f"xgb_ret{label}_v*/model_metadata.json"))
        union = set()
        if direction_meta:
            union.update(_extract_feature_names(direction_meta[-1]))
        if regression_meta:
            union.update(_extract_feature_names(regression_meta[-1]))
        by_horizon[label] = sorted(union)
    return by_horizon


def _load_dataset_feature_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=True) as dataset_npz:
            if "feature_names" not in dataset_npz.files:
                return []
            values = dataset_npz["feature_names"].tolist()
    except Exception:
        return []
    return [str(value) for value in values]


def _collect_dataset_features(horizons: Sequence[float]) -> Dict[str, List[str]]:
    by_horizon: Dict[str, List[str]] = {}
    for horizon in horizons:
        label = _horizon_label(float(horizon))
        dataset_paths: List[Path]
        if label == "15m":
            dataset_paths = [DEFAULT_DATASET_15M_PATH, DEFAULT_DATASET_15M_DIRECTION_PATH]
        elif label == "1h":
            dataset_paths = [DEFAULT_DATASET_1H_PATH, DEFAULT_DATASET_1H_DIRECTION_PATH]
        else:
            dataset_paths = [DEFAULT_DATASET_MULTI_PATH]

        feature_union = set()
        for dataset_path in dataset_paths:
            feature_union.update(_load_dataset_feature_names(dataset_path))
        by_horizon[label] = sorted(feature_union)
    return by_horizon


def _family_counts(features: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for feature in features:
        family = classify_feature_family(feature)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _filter_derivatives_features(features: Iterable[str]) -> List[str]:
    return sorted(
        {
            str(feature)
            for feature in features
            if classify_feature_family(str(feature)) == "derivatives"
        }
    )


def _default_derivatives_policy_specs(readiness: str) -> List[Dict[str, Any]]:
    blocked = readiness != "shadow_scaffold_ready"
    status = "blocked" if blocked else "ready_for_shadow_replay"
    rationale_map = {
        "shadow_scaffold_ready": "Core data requirements appear present; policy is ready for first shadow-only replay.",
        "dataset_ready_retrain_required": "Derivatives features exist in rebuilt datasets, but checked model metadata has not been refreshed yet.",
        "not_viable_no_training_usage": "Local derivatives data is now available, but checked training metadata still has zero derivatives-family usage.",
        "needs_data_wiring_first": "Runtime/data coverage is still incomplete for first derivatives shadow replay.",
    }
    rationale = rationale_map.get(
        readiness,
        "Derivatives readiness is still blocked by unresolved data or training dependencies.",
    )
    specs = [
        DerivativesPolicySpec(
            name="funding_conflict_veto_weak",
            description="Block weak trades when funding pressure conflicts with the selected side.",
            required_features=("funding_rate_zscore_24h",),
            intended_scope="1h/4h/8h/12h weak-signal veto",
            status=status,
            rationale=rationale,
        ),
        DerivativesPolicySpec(
            name="funding_extreme_regime_only",
            description="Apply derivatives veto only when funding z-score is extreme in neutral/chop regimes.",
            required_features=("funding_rate_zscore_24h",),
            intended_scope="4h/8h/12h neutral+chop",
            status=status,
            rationale=rationale,
        ),
        DerivativesPolicySpec(
            name="open_interest_expansion_conflict_veto",
            description="Block weak trades when open-interest expansion and futures direction disagree with the selected side.",
            required_features=("open_interest", "fut_close"),
            intended_scope="4h/8h/12h expansion conflicts",
            status=status,
            rationale=rationale,
        ),
        DerivativesPolicySpec(
            name="futures_basis_conflict_veto",
            description="Veto weak trades when futures-price proxy diverges from spot direction.",
            required_features=("fut_close", "close"),
            intended_scope="4h/8h/12h futures-vs-spot disagreement",
            status=status,
            rationale=rationale,
        ),
    ]
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "required_features": list(spec.required_features),
            "intended_scope": spec.intended_scope,
            "status": spec.status,
            "rationale": spec.rationale,
        }
        for spec in specs
    ]


def build_signal_program_dispositions(analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> Dict[str, Any]:
    macro_payload = _load_json(analysis_dir / "macro_shadow_enforcement_latest.json") or {}
    outcome_payload = _load_json(analysis_dir / "state_orderflow_outcome_confirmation_latest.json") or {}
    rolling_payload = _load_json(analysis_dir / "orderflow_rolling_stability_latest.json") or {}
    state_narrow_payload = _load_json(analysis_dir / "state_engineering_narrow_scope_latest.json") or {}
    state_guarded_payload = _load_json(analysis_dir / "state_engineering_guarded_shadow_4h_latest.json") or {}

    macro_sweep = macro_payload.get("sweep") if isinstance(macro_payload.get("sweep"), Mapping) else {}
    macro_recommendation = (
        macro_sweep.get("recommendation")
        if isinstance(macro_sweep.get("recommendation"), Mapping)
        else {}
    )
    macro_disposition = str(macro_recommendation.get("macro_disposition") or "unknown")
    macro_best_assessment = str(macro_recommendation.get("best_assessment") or "unknown")

    state_best = _extract_family_best(outcome_payload, "state_engineering") or {}
    state_decision = _extract_go_hold(outcome_payload, "state_engineering") or "unknown"
    state_narrow_recommendation = (
        state_narrow_payload.get("final_recommendation")
        if isinstance(state_narrow_payload.get("final_recommendation"), Mapping)
        else {}
    )
    state_narrow_best = (
        state_narrow_payload.get("best_candidate")
        if isinstance(state_narrow_payload.get("best_candidate"), Mapping)
        else {}
    )
    state_narrow_decision = str(state_narrow_recommendation.get("decision") or "")
    state_guarded_readiness = (
        state_guarded_payload.get("readiness")
        if isinstance(state_guarded_payload.get("readiness"), Mapping)
        else {}
    )
    state_guarded_summary = (
        state_guarded_payload.get("summary")
        if isinstance(state_guarded_payload.get("summary"), Mapping)
        else {}
    )
    state_guarded_active = str(state_guarded_readiness.get("decision") or "") == "shadow_validation_active"

    rolling_classification = (
        rolling_payload.get("rolling_stability_classification")
        if isinstance(rolling_payload.get("rolling_stability_classification"), Mapping)
        else {}
    )
    orderflow_disposition = str(rolling_classification.get("disposition") or "unknown")
    orderflow_classification = str(rolling_classification.get("classification") or "unknown")

    families = {
        "macro": {
            "status": "closed" if macro_disposition == "deprioritize_for_now" else "active",
            "disposition": macro_disposition,
            "decisive_evidence": {
                "artifact": str(analysis_dir / "macro_shadow_enforcement_latest.json"),
                "best_assessment": macro_best_assessment,
                "advance_to_next_validation_stage": bool(macro_recommendation.get("advance_to_next_validation_stage", False)),
            },
            "recommended_action": "stop_additional_macro_sweeps" if macro_disposition == "deprioritize_for_now" else "reassess_later",
        },
        "order_flow": {
            "status": "closed" if orderflow_disposition == "deprioritize_for_now" else "active",
            "disposition": orderflow_disposition,
            "decisive_evidence": {
                "artifact": str(analysis_dir / "orderflow_rolling_stability_latest.json"),
                "classification": orderflow_classification,
                "pass_count": int(rolling_classification.get("pass_count", 0) or 0),
                "fail_count": int(rolling_classification.get("fail_count", 0) or 0),
            },
            "recommended_action": "stop_additional_orderflow_replay_loops" if orderflow_disposition == "deprioritize_for_now" else "narrow_scope_followup",
        },
        "state_engineering": {
            "status": (
                "closed"
                if state_narrow_decision == "deprioritize_for_now"
                else "active"
                if state_narrow_decision == "continue_narrow_scope_validation"
                else "hold"
            ),
            "disposition": (
                "guarded_shadow_validation_active"
                if state_guarded_active
                else "deprioritize_for_now"
                if state_narrow_decision == "deprioritize_for_now"
                else "continue_narrow_scope_validation"
                if state_narrow_decision == "continue_narrow_scope_validation"
                else "hold"
                if state_decision == "hold"
                else state_decision
            ),
            "decisive_evidence": {
                "artifact": (
                    str(analysis_dir / "state_engineering_guarded_shadow_4h_latest.json")
                    if state_guarded_payload
                    else str(analysis_dir / "state_engineering_narrow_scope_latest.json")
                    if state_narrow_payload
                    else str(analysis_dir / "state_orderflow_outcome_confirmation_latest.json")
                ),
                "best_variant": state_best.get("best_variant"),
                "net_return_proxy_mean_delta": state_best.get("net_return_proxy_mean_delta"),
                "direction_accuracy_proxy_delta": state_best.get("direction_accuracy_proxy_delta"),
                "robustness": state_best.get("robustness"),
                "narrow_scope_decision": state_narrow_recommendation.get("decision") if state_narrow_payload else None,
                "narrow_scope_reason": state_narrow_recommendation.get("reason") if state_narrow_payload else None,
                "narrow_scope_best_slice": state_narrow_best.get("scope") if state_narrow_payload else None,
                "guarded_shadow_readiness": state_guarded_readiness.get("decision") if state_guarded_payload else None,
                "guarded_shadow_changed_snapshot_count": state_guarded_summary.get("changed_snapshot_count") if state_guarded_payload else None,
                "guarded_shadow_assessment": state_guarded_summary.get("assessment") if state_guarded_payload else None,
            },
            "recommended_action": (
                "stop_additional_state_engineering_replays"
                if state_narrow_decision == "deprioritize_for_now"
                else "keep_4h_guarded_shadow_validation_running"
                if state_guarded_active
                else "continue_4h_only_state_engineering_validation"
                if state_narrow_decision == "continue_narrow_scope_validation"
                else "hold_until_stronger_outcome_confirmation"
            ),
        },
    }

    return {
        "generated_from": {
            "macro": str(analysis_dir / "macro_shadow_enforcement_latest.json"),
            "state_orderflow_outcomes": str(analysis_dir / "state_orderflow_outcome_confirmation_latest.json"),
            "orderflow_rolling_stability": str(analysis_dir / "orderflow_rolling_stability_latest.json"),
        },
        "families": families,
        "closed_families": sorted(
            family for family, payload in families.items() if str(payload.get("status")) == "closed"
        ),
        "hold_families": sorted(
            family for family, payload in families.items() if str(payload.get("status")) == "hold"
        ),
        "active_families": sorted(
            family for family, payload in families.items() if str(payload.get("status")) == "active"
        ),
        "next_priority_family": (
            "derivatives"
            if state_guarded_active or state_narrow_decision != "continue_narrow_scope_validation"
            else "state_engineering"
        ),
        "notes": [
            "Macro and order_flow are treated as closed/deprioritized until materially new evidence appears.",
            (
                "State_engineering narrow-scope follow-up closed the family for now because its best slice remained too sparse or weak."
                if state_narrow_decision == "deprioritize_for_now"
                else "State_engineering now has a guarded 4h-only shadow runner active, with fail-close behavior and no non-4h spillover."
                if state_guarded_active
                else "State_engineering stays active only in a narrow validated slice led by the 4h weak-signal veto candidate."
                if state_narrow_decision == "continue_narrow_scope_validation"
                else "State_engineering remains a hold candidate rather than a rollout candidate."
            ),
            "Derivatives data refresh is now wired locally, but checked training metadata still does not reference the family.",
        ],
    }


def build_derivatives_family_audit(
    *,
    config: Mapping[str, Any],
    models_root: Path = DEFAULT_MODELS_ROOT,
    funding_dir: Path = DEFAULT_FUNDING_DIR,
    spot_dir: Path = DEFAULT_SPOT_DIR,
) -> Dict[str, Any]:
    targets = _parse_targets(config.get("targets"))
    by_horizon_features = _collect_training_features(models_root, targets)
    dataset_features_by_horizon = _collect_dataset_features(targets)

    training_derivatives_by_horizon: Dict[str, Any] = {}
    training_union: List[str] = []
    for label, features in by_horizon_features.items():
        deriv_features = _filter_derivatives_features(features)
        training_union.extend(deriv_features)
        training_derivatives_by_horizon[label] = {
            "derivatives_feature_count": len(deriv_features),
            "derivatives_features": deriv_features,
        }

    dataset_derivatives_by_horizon: Dict[str, Any] = {}
    dataset_union: List[str] = []
    for label, features in dataset_features_by_horizon.items():
        deriv_features = _filter_derivatives_features(features)
        dataset_union.extend(deriv_features)
        dataset_derivatives_by_horizon[label] = {
            "derivatives_feature_count": len(deriv_features),
            "derivatives_features": deriv_features,
        }

    coverage = config.get("feature_coverage_policy") if isinstance(config.get("feature_coverage_policy"), Mapping) else {}
    ignored_columns = coverage.get("ignored_columns") if isinstance(coverage.get("ignored_columns"), list) else []
    ignored_sources = coverage.get("ignored_sources") if isinstance(coverage.get("ignored_sources"), list) else []
    ignored_derivatives_columns = _filter_derivatives_features(ignored_columns)

    funding_parquets = sorted(str(path) for path in funding_dir.glob("*.parquet")) if funding_dir.exists() else []
    spot_parquets = sorted(str(path) for path in spot_dir.glob("*.parquet")) if spot_dir.exists() else []

    training_union = sorted(set(training_union))
    dataset_union = sorted(set(dataset_union))
    total_derivatives_training_features = len(training_union)
    total_derivatives_dataset_features = len(dataset_union)
    live_policy_ignores_derivatives = bool(ignored_derivatives_columns)
    funding_available = bool(funding_parquets)
    spot_available = bool(spot_parquets)

    readiness = "needs_data_wiring_first"
    blockers: List[str] = []
    if total_derivatives_training_features == 0:
        if total_derivatives_dataset_features > 0:
            readiness = "dataset_ready_retrain_required"
            blockers.append("checked_model_metadata_not_refreshed_after_derivatives_wiring")
        elif funding_available and spot_available:
            readiness = "not_viable_no_training_usage"
            blockers.append("rebuild_training_datasets_with_derivatives_then_retrain_models")
        else:
            readiness = "not_viable_no_training_usage"
            blockers.append("training_models_do_not_reference_derivatives_family")
    else:
        if not funding_available:
            blockers.append("missing_local_funding_dataset")
        if live_policy_ignores_derivatives:
            blockers.append("live_policy_explicitly_ignores_derivatives_columns")
        if not spot_available:
            blockers.append("missing_spot_klines_for_snapshot_alignment")
        if funding_available and spot_available:
            readiness = "shadow_scaffold_ready"

    if total_derivatives_training_features > 0:
        next_priority_family = "derivatives"
        next_priority_reason = (
            "Training references exist while live conservative runtime still ignores derivatives columns and "
            "has no local funding dataset present."
        )
    elif total_derivatives_dataset_features > 0:
        next_priority_family = "derivatives"
        next_priority_reason = (
            "Derivatives columns are now present in the rebuilt training dataset, but checked model metadata has not yet been refreshed by retraining."
        )
    else:
        next_priority_family = "derivatives"
        next_priority_reason = (
            "Derivatives runtime wiring and local funding data now exist, but checked model metadata still does not "
            "reference derivatives-family features; the next actionable path is dataset rebuild plus retraining."
        )

    audit = {
        "targets": targets,
        "training_derivatives_by_horizon": training_derivatives_by_horizon,
        "training_derivatives_union": training_union,
        "training_derivatives_family_count": total_derivatives_training_features,
        "dataset_derivatives_by_horizon": dataset_derivatives_by_horizon,
        "dataset_derivatives_union": dataset_union,
        "dataset_derivatives_family_count": total_derivatives_dataset_features,
        "training_family_counts": _family_counts(
            feature for features in by_horizon_features.values() for feature in features
        ),
        "runtime_support": {
            "funding_optional_source_supported": True,
            "funding_required_columns": ["funding_rate", "funding_rate_annualized"],
            "open_interest_optional_source_supported": True,
            "futures_columns_zero_imputed_when_missing": True,
            "derivatives_columns_currently_ignored_in_live_policy": ignored_derivatives_columns,
            "ignored_sources": [str(value) for value in ignored_sources],
        },
        "local_data_availability": {
            "funding_dir": str(funding_dir),
            "funding_parquets": funding_parquets,
            "spot_dir": str(spot_dir),
            "spot_parquets_detected": len(spot_parquets),
        },
        "readiness": {
            "decision": readiness,
            "blockers": blockers,
            "next_action": (
                "retrain_checked_models_with_derivatives_dataset"
                if readiness == "dataset_ready_retrain_required"
                else "rebuild_training_datasets_then_retrain_models_with_derivatives"
                if readiness == "not_viable_no_training_usage"
                else "wire_funding_and_futures_inputs_before_shadow_replay"
                if readiness != "shadow_scaffold_ready"
                else "run_first_shadow_derivatives_validation"
            ),
        },
        "shadow_policy_scaffold": _default_derivatives_policy_specs(readiness),
        "next_priority_confirmation": {
            "family": next_priority_family,
            "reason": next_priority_reason,
            "ready_for_shadow_validation": readiness == "shadow_scaffold_ready",
        },
    }
    return audit


def build_derivatives_shadow_scaffold(audit: Mapping[str, Any]) -> Dict[str, Any]:
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), Mapping) else {}
    decision = str(readiness.get("decision") or "needs_data_wiring_first")
    return {
        "family": "derivatives",
        "runner_status": "ready" if decision == "shadow_scaffold_ready" else "blocked",
        "readiness_decision": decision,
        "blockers": list(readiness.get("blockers", [])) if isinstance(readiness.get("blockers"), list) else [],
        "policies": audit.get("shadow_policy_scaffold", []),
        "notes": [
            "This is a scaffold-only artifact; it does not change live inference behavior.",
            "If blocked, resolve local funding/futures data coverage before adding replay logic.",
        ],
    }


def build_derivatives_shadow_candidate_config(
    base_config: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
) -> Dict[str, Any]:
    config = deepcopy(dict(base_config))
    coverage = config.get("feature_coverage_policy")
    if not isinstance(coverage, Mapping):
        coverage = {}
    coverage = dict(coverage)

    ignored_sources = [
        str(source).strip()
        for source in (coverage.get("ignored_sources") or [])
        if str(source).strip()
    ]
    ignored_columns = [
        str(column).strip()
        for column in (coverage.get("ignored_columns") or [])
        if str(column).strip()
    ]

    coverage["enabled"] = True
    coverage["ignored_sources"] = [source for source in ignored_sources if source.lower() != "funding"]
    coverage["ignored_columns"] = [
        column for column in ignored_columns if classify_feature_family(column) != "derivatives"
    ]
    config["feature_coverage_policy"] = coverage

    return config


def build_signal_expansion_rollout_summary(
    *,
    signal_payload: Mapping[str, Any],
    derivatives_audit: Mapping[str, Any],
    derivatives_scaffold: Mapping[str, Any],
    meta_baseline_source_csv: str,
    meta_config_path: str,
    meta_signal_mode: str,
    meta_weight_threshold: float,
    meta_selected_weight_threshold: float | None = None,
    meta_auto_threshold_on_oof: bool | None = None,
    meta_threshold_selection: Mapping[str, Any] | None = None,
    derivatives_config_path: str,
    featurelift_config_path: str,
    featurelift_package_path: str,
    state_guarded_json_path: str,
    state_guarded_md_path: str,
) -> Dict[str, Any]:
    families = signal_payload.get("families") if isinstance(signal_payload.get("families"), Mapping) else {}
    macro_payload = families.get("macro") if isinstance(families, Mapping) and isinstance(families.get("macro"), Mapping) else {}
    state_payload = (
        families.get("state_engineering")
        if isinstance(families, Mapping) and isinstance(families.get("state_engineering"), Mapping)
        else {}
    )
    readiness = derivatives_audit.get("readiness") if isinstance(derivatives_audit.get("readiness"), Mapping) else {}

    return {
        "next_priority_family": "meta_ensemble",
        "follow_on_priority_families": ["derivatives", "featurelift_4h"],
        "program_direction": {
            "meta_ensemble": {
                "status": "priority_eval_lane",
                "recommended_action": "evaluate_before_expanding_base_models",
                "source_csv": meta_baseline_source_csv,
                "config_path": meta_config_path,
                "signal_mode": meta_signal_mode,
                "weight_threshold": meta_weight_threshold,
                "selected_weight_threshold": (
                    float(meta_selected_weight_threshold)
                    if meta_selected_weight_threshold is not None
                    else float(meta_weight_threshold)
                ),
                "auto_threshold_on_oof": bool(meta_auto_threshold_on_oof) if meta_auto_threshold_on_oof is not None else None,
                "threshold_selection": dict(meta_threshold_selection or {}),
            },
            "macro": {
                "status": str(macro_payload.get("status") or "unknown"),
                "disposition": str(macro_payload.get("disposition") or "unknown"),
                "recommended_action": "keep_deprioritized",
            },
            "derivatives": {
                "status": str(readiness.get("decision") or "unknown"),
                "next_action": str(readiness.get("next_action") or "unknown"),
                "candidate_config": derivatives_config_path,
                "scaffold_runner_status": str(derivatives_scaffold.get("runner_status") or "unknown"),
            },
            "featurelift_4h": {
                "status": "shadow_candidate",
                "candidate_config": featurelift_config_path,
                "package_markdown": featurelift_package_path,
            },
            "state_engineering": {
                "status": str(state_payload.get("status") or "unknown"),
                "disposition": str(state_payload.get("disposition") or "unknown"),
                "guarded_shadow_json": state_guarded_json_path,
                "guarded_shadow_markdown": state_guarded_md_path,
            },
        },
        "implementation_notes": [
            "Meta-ensemble evaluation is the first upgrade lane; improve the combiner before expanding the base-model roster.",
            "Derivatives move into a dedicated shadow candidate config rather than remaining ignored by live coverage.",
            "4h feature-lift remains the shadow retrain lane for model-level improvement.",
            "State-engineering stays constrained to the guarded 4h-only runner until stronger evidence appears.",
            "Macro remains contextual and explicitly deprioritized.",
        ],
    }


def render_signal_program_markdown(payload: Mapping[str, Any]) -> str:
    families = payload.get("families", {}) if isinstance(payload.get("families"), Mapping) else {}
    lines: List[str] = []
    lines.append("# Signal Program Dispositions")
    lines.append("")
    lines.append("## Current Status")
    lines.append("| Family | Status | Disposition | Recommended Action |")
    lines.append("| --- | --- | --- | --- |")
    for family in ("macro", "order_flow", "state_engineering"):
        item = families.get(family, {}) if isinstance(families.get(family), Mapping) else {}
        lines.append(
            "| {family} | {status} | {disposition} | {action} |".format(
                family=family,
                status=str(item.get("status", "unknown")),
                disposition=str(item.get("disposition", "unknown")),
                action=str(item.get("recommended_action", "unknown")),
            )
        )
    lines.append("")
    lines.append("## Evidence")
    for family in ("macro", "order_flow", "state_engineering"):
        item = families.get(family, {}) if isinstance(families.get(family), Mapping) else {}
        evidence = item.get("decisive_evidence", {}) if isinstance(item.get("decisive_evidence"), Mapping) else {}
        lines.append(f"### {family}")
        for key, value in evidence.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Program Direction")
    lines.append(f"- Closed families: {', '.join(payload.get('closed_families', []))}")
    lines.append(f"- Hold families: {', '.join(payload.get('hold_families', []))}")
    lines.append(f"- Next priority family: {payload.get('next_priority_family')}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_derivatives_audit_markdown(payload: Mapping[str, Any]) -> str:
    readiness = payload.get("readiness", {}) if isinstance(payload.get("readiness"), Mapping) else {}
    runtime_support = payload.get("runtime_support", {}) if isinstance(payload.get("runtime_support"), Mapping) else {}
    lines: List[str] = []
    lines.append("# Derivatives Family Audit")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Readiness: {readiness.get('decision', 'unknown')}")
    lines.append(f"- Next action: {readiness.get('next_action', 'unknown')}")
    lines.append("")
    lines.append("## Runtime Gap")
    lines.append(f"- Funding optional source supported: {runtime_support.get('funding_optional_source_supported')}")
    lines.append(f"- Open-interest optional source supported: {runtime_support.get('open_interest_optional_source_supported')}")
    lines.append(f"- Futures columns zero-imputed when missing: {runtime_support.get('futures_columns_zero_imputed_when_missing')}")
    lines.append(
        f"- Live policy ignored derivatives columns: {', '.join(runtime_support.get('derivatives_columns_currently_ignored_in_live_policy', []))}"
    )
    lines.append("")
    lines.append("## Training Usage By Horizon")
    lines.append("| Horizon | Derivatives Feature Count | Features |")
    lines.append("| --- | ---: | --- |")
    by_horizon = payload.get("training_derivatives_by_horizon", {}) if isinstance(payload.get("training_derivatives_by_horizon"), Mapping) else {}
    for horizon, item in sorted(by_horizon.items()):
        if not isinstance(item, Mapping):
            continue
        features = item.get("derivatives_features", []) if isinstance(item.get("derivatives_features"), list) else []
        lines.append(
            "| {horizon} | {count} | {features} |".format(
                horizon=horizon,
                count=int(item.get("derivatives_feature_count", 0)),
                features=", ".join(features) if features else "none",
            )
        )
    lines.append("")
    lines.append("## Blockers")
    blockers = readiness.get("blockers", []) if isinstance(readiness.get("blockers"), list) else []
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_derivatives_scaffold_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Derivatives Shadow Validation Scaffold")
    lines.append("")
    lines.append(f"- Runner status: {payload.get('runner_status', 'unknown')}")
    lines.append(f"- Readiness decision: {payload.get('readiness_decision', 'unknown')}")
    blockers = payload.get("blockers", []) if isinstance(payload.get("blockers"), list) else []
    lines.append(f"- Blockers: {', '.join(blockers) if blockers else 'none'}")
    lines.append("")
    lines.append("## Policy Specs")
    lines.append("| Policy | Scope | Required Features | Status |")
    lines.append("| --- | --- | --- | --- |")
    for item in payload.get("policies", []):
        if not isinstance(item, Mapping):
            continue
        features = item.get("required_features", []) if isinstance(item.get("required_features"), list) else []
        lines.append(
            "| {name} | {scope} | {features} | {status} |".format(
                name=str(item.get("name", "unknown")),
                scope=str(item.get("intended_scope", "unknown")),
                features=", ".join(features),
                status=str(item.get("status", "unknown")),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"