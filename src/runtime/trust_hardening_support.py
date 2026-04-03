from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping


SummaryPayload = Dict[str, Dict[str, Any]]


def resolve_trust_hardening_policy(
    config: Mapping[str, Any] | None,
    *,
    normalize_horizon_value: Callable[[Any], float],
    coerce_numeric_horizon: Callable[[Any], float | None],
) -> Dict[str, Any]:
    cfg = config or {}
    horizons = cfg.get("horizons") or [4.0, 8.0]
    high_impact = cfg.get("high_impact_horizons") or horizons

    action_by_horizon: Dict[float, str] = {}
    raw_action_by_horizon = cfg.get("action_by_horizon")
    if isinstance(raw_action_by_horizon, Mapping):
        for key, value in raw_action_by_horizon.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None:
                continue
            action_by_horizon[normalize_horizon_value(horizon)] = str(value).strip().lower()

    deweight_factor_by_horizon: Dict[float, float] = {}
    raw_deweight_by_horizon = cfg.get("deweight_factor_by_horizon")
    if isinstance(raw_deweight_by_horizon, Mapping):
        for key, value in raw_deweight_by_horizon.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                factor = float(value)
            except (TypeError, ValueError):
                continue
            deweight_factor_by_horizon[normalize_horizon_value(horizon)] = max(min(factor, 1.0), 0.0)

    model_summary_paths_by_horizon: Dict[float, str] = {}
    raw_summary_paths = cfg.get("model_summary_paths_by_horizon")
    if isinstance(raw_summary_paths, Mapping):
        for key, value in raw_summary_paths.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None or value is None:
                continue
            model_summary_paths_by_horizon[normalize_horizon_value(horizon)] = str(value)

    metadata_cfg_obj = cfg.get("metadata_checks")
    metadata_cfg = metadata_cfg_obj if isinstance(metadata_cfg_obj, Mapping) else {}
    leakage_signatures = metadata_cfg.get("leakage_signature_features")
    if isinstance(leakage_signatures, list):
        leakage_features = [str(item) for item in leakage_signatures if str(item).strip()]
    else:
        leakage_features = ["ret_1h", "ret_4h", "ret_8h", "ret_12h"]

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "horizons": sorted({normalize_horizon_value(value) for value in horizons}),
        "high_impact_horizons": sorted({normalize_horizon_value(value) for value in high_impact}),
        "fail_closed": bool(cfg.get("fail_closed", False)),
        "default_action": str(cfg.get("default_action") or "exclude").strip().lower(),
        "action_by_horizon": action_by_horizon,
        "deweight_factor": max(min(float(cfg.get("deweight_factor") or 0.35), 1.0), 0.0),
        "deweight_factor_by_horizon": deweight_factor_by_horizon,
        "divergence_abs_gap_min": max(float(cfg.get("divergence_abs_gap_min") or 0.12), 0.0),
        "divergence_flip_required": bool(cfg.get("divergence_flip_required", True)),
        "probability_neutral_band": max(float(cfg.get("probability_neutral_band") or 0.0), 0.0),
        "model_summary_paths_by_horizon": model_summary_paths_by_horizon,
        "metadata_checks": {
            "enabled": bool(metadata_cfg.get("enabled", True)),
            "require_metadata": bool(metadata_cfg.get("require_metadata", True)),
            "leakage_signature_features": leakage_features,
            "max_val_accuracy": min(max(float(metadata_cfg.get("max_val_accuracy") or 0.995), 0.0), 1.0),
            "max_test_accuracy": min(max(float(metadata_cfg.get("max_test_accuracy") or 0.995), 0.0), 1.0),
            "max_train_accuracy": min(max(float(metadata_cfg.get("max_train_accuracy") or 0.999), 0.0), 1.0),
            "max_train_val_accuracy_gap": max(float(metadata_cfg.get("max_train_val_accuracy_gap") or 0.03), 0.0),
        },
    }


def apply_trust_hardening(
    summary: SummaryPayload,
    policy: Mapping[str, Any],
    *,
    coerce_result_horizon: Callable[[Any], float | None],
    direction_vote: Callable[[Mapping[str, Any]], str],
    direction_from_probability: Callable[[Any], str] | Callable[..., str],
    finite_float_or_none: Callable[[Any], float | None],
) -> SummaryPayload:
    if not summary:
        return summary

    enabled = bool(policy.get("enabled", False))
    if not enabled:
        for entry in summary.values():
            entry["trust_status"] = "disabled"
            entry["trust_reasons"] = []
            entry["excluded_from_voting"] = False
            entry["voting_weight_after_trust"] = 1.0
            entry["trust_hardening_action"] = "none"
            entry["trust_hardening_changed_outcome"] = False
        return summary

    covered_horizons = set(policy.get("horizons", []))
    high_impact_horizons = set(policy.get("high_impact_horizons", []))
    fail_closed = bool(policy.get("fail_closed", False))
    metadata_cfg = policy.get("metadata_checks") if isinstance(policy.get("metadata_checks"), Mapping) else {}
    metadata_required = bool(metadata_cfg.get("require_metadata", True))
    metadata_enabled = bool(metadata_cfg.get("enabled", True))
    summary_paths = policy.get("model_summary_paths_by_horizon") if isinstance(policy.get("model_summary_paths_by_horizon"), Mapping) else {}

    dominant_before = _dominant_direction(summary, direction_vote=direction_vote, use_trust_weights=False)

    for entry in summary.values():
        horizon = coerce_result_horizon(entry.get("horizon_hours"))
        trust_reasons: list[str] = []
        action = "none"
        excluded_from_voting = False
        voting_weight_after_trust = 1.0

        if horizon is not None and horizon in covered_horizons:
            if metadata_enabled:
                meta_reasons, metadata_missing = _evaluate_metadata_risk(
                    horizon,
                    summary_paths,
                    metadata_cfg,
                )
                trust_reasons.extend(meta_reasons)
                if metadata_missing and metadata_required and fail_closed:
                    trust_reasons.append("missing_required_trust_metadata")

            if horizon in high_impact_horizons and _calibration_divergence_is_suspicious(
                entry,
                policy=policy,
                direction_from_probability=direction_from_probability,
                finite_float_or_none=finite_float_or_none,
            ):
                trust_reasons.append("calibration_flip_divergence")

            if trust_reasons:
                action = _resolve_action_for_horizon(horizon, policy)
                if action == "exclude":
                    excluded_from_voting = True
                    voting_weight_after_trust = 0.0
                elif action == "deweight":
                    voting_weight_after_trust = _resolve_deweight_factor_for_horizon(horizon, policy)

        entry["trust_status"] = "low_trust" if trust_reasons else "trusted"
        entry["trust_reasons"] = trust_reasons
        entry["excluded_from_voting"] = bool(excluded_from_voting)
        entry["voting_weight_after_trust"] = float(max(voting_weight_after_trust, 0.0))
        entry["trust_hardening_action"] = action if trust_reasons else "none"

    dominant_after = _dominant_direction(summary, direction_vote=direction_vote, use_trust_weights=True)
    changed_outcome = dominant_before != dominant_after
    for entry in summary.values():
        entry["trust_hardening_changed_outcome"] = bool(changed_outcome)
    return summary


def _resolve_action_for_horizon(horizon: float, policy: Mapping[str, Any]) -> str:
    action_map = policy.get("action_by_horizon") if isinstance(policy.get("action_by_horizon"), Mapping) else {}
    if horizon in action_map:
        return str(action_map[horizon]).strip().lower()
    return str(policy.get("default_action") or "exclude").strip().lower()


def _resolve_deweight_factor_for_horizon(horizon: float, policy: Mapping[str, Any]) -> float:
    factor_map = policy.get("deweight_factor_by_horizon") if isinstance(policy.get("deweight_factor_by_horizon"), Mapping) else {}
    if horizon in factor_map:
        try:
            return max(min(float(factor_map[horizon]), 1.0), 0.0)
        except (TypeError, ValueError):
            return max(min(float(policy.get("deweight_factor") or 0.35), 1.0), 0.0)
    return max(min(float(policy.get("deweight_factor") or 0.35), 1.0), 0.0)


def _evaluate_metadata_risk(
    horizon: float,
    summary_paths: Mapping[Any, Any],
    metadata_cfg: Mapping[str, Any],
) -> tuple[list[str], bool]:
    raw_path = summary_paths.get(horizon)
    if raw_path is None:
        raw_path = _default_summary_path(horizon)
    path = Path(str(raw_path))
    if not path.exists():
        return [], True

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ["invalid_trust_metadata"], False

    reasons: list[str] = []
    feature_names = payload.get("feature_names") if isinstance(payload.get("feature_names"), list) else []
    leakage_features = {
        str(item)
        for item in metadata_cfg.get("leakage_signature_features", [])
        if str(item).strip()
    }
    if leakage_features and any(str(name) in leakage_features for name in feature_names):
        reasons.append("metadata_leakage_signature_features")

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    train_acc = _nested_metric(metrics, "train", "accuracy")
    val_acc = _nested_metric(metrics, "val", "accuracy")
    test_acc = _nested_metric(metrics, "test", "accuracy")
    if val_acc is not None and val_acc >= float(metadata_cfg.get("max_val_accuracy", 0.995)):
        reasons.append("metadata_implausible_val_accuracy")
    if test_acc is not None and test_acc >= float(metadata_cfg.get("max_test_accuracy", 0.995)):
        reasons.append("metadata_implausible_test_accuracy")
    if train_acc is not None and train_acc >= float(metadata_cfg.get("max_train_accuracy", 0.999)):
        reasons.append("metadata_implausible_train_accuracy")
    if train_acc is not None and val_acc is not None:
        if abs(train_acc - val_acc) >= float(metadata_cfg.get("max_train_val_accuracy_gap", 0.03)):
            reasons.append("metadata_train_val_gap")

    return reasons, False


def _nested_metric(metrics: Mapping[str, Any], split: str, key: str) -> float | None:
    split_metrics = metrics.get(split)
    if not isinstance(split_metrics, Mapping):
        return None
    value = split_metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calibration_divergence_is_suspicious(
    entry: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    direction_from_probability: Callable[[Any], str] | Callable[..., str],
    finite_float_or_none: Callable[[Any], float | None],
) -> bool:
    divergence_abs_gap_min = float(policy.get("divergence_abs_gap_min") or 0.12)
    divergence_flip_required = bool(policy.get("divergence_flip_required", True))
    neutral_band = float(policy.get("probability_neutral_band") or 0.0)

    calibration = entry.get("probability_calibration") if isinstance(entry.get("probability_calibration"), Mapping) else {}
    raw_probability = finite_float_or_none(entry.get("raw_p_up"))
    if raw_probability is None:
        raw_probability = finite_float_or_none(calibration.get("raw_probability"))
    resolved_probability = finite_float_or_none(entry.get("p_up"))
    if resolved_probability is None:
        resolved_probability = finite_float_or_none(calibration.get("resolved_probability"))
    if raw_probability is None or resolved_probability is None:
        return False

    if abs(resolved_probability - raw_probability) < divergence_abs_gap_min:
        return False

    raw_side = direction_from_probability(raw_probability, neutral_band=neutral_band)
    resolved_side = direction_from_probability(resolved_probability, neutral_band=neutral_band)
    if not divergence_flip_required:
        return True
    return raw_side in {"up", "down"} and resolved_side in {"up", "down"} and raw_side != resolved_side


def _default_summary_path(horizon: float) -> str:
    label = int(round(horizon))
    if abs(horizon - 4.0) <= 1e-6:
        return "artifacts/models/lgbm_dir4h_v1/summary.json"
    if abs(horizon - 8.0) <= 1e-6:
        return "artifacts/models/xgb_dir8h_v1/summary.json"
    if abs(horizon - 12.0) <= 1e-6:
        return "artifacts/models/bilstm_dir12h_v1/summary.json"
    return f"artifacts/models/xgb_dir{label}h_v1/summary.json"


def _dominant_direction(
    summary: SummaryPayload,
    *,
    direction_vote: Callable[[Mapping[str, Any]], str],
    use_trust_weights: bool,
) -> str:
    up_score = 0.0
    down_score = 0.0
    for entry in summary.values():
        coherence_payload = entry.get("forecast_coherence")
        if isinstance(coherence_payload, Mapping) and coherence_payload.get("exclude_from_voting"):
            continue
        if use_trust_weights and bool(entry.get("excluded_from_voting", False)):
            continue
        direction = direction_vote(entry)
        if direction not in {"up", "down"}:
            continue
        weight = 1.0
        if use_trust_weights:
            try:
                weight = max(float(entry.get("voting_weight_after_trust", 1.0) or 0.0), 0.0)
            except (TypeError, ValueError):
                weight = 1.0
        if direction == "up":
            up_score += weight
        else:
            down_score += weight
    if up_score > down_score:
        return "up"
    if down_score > up_score:
        return "down"
    return "neutral"
