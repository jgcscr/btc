from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def normalize_horizon_float_map(
    raw: Any,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
    minimum: float = 0.0,
    maximum: float | None = None,
) -> Dict[float, float]:
    if not isinstance(raw, Mapping):
        return {}
    resolved: Dict[float, float] = {}
    for key, value in raw.items():
        horizon = coerce_numeric_horizon(key)
        if horizon is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        numeric_value = max(numeric_value, minimum)
        if maximum is not None:
            numeric_value = min(numeric_value, maximum)
        resolved[normalize_horizon_value(horizon)] = numeric_value
    return resolved


def normalize_horizon_regime_float_map(
    raw: Any,
    *,
    finite_float_or_none: Callable[[Any], float | None],
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
    minimum: float = 0.0,
    maximum: float | None = None,
) -> Dict[float, Dict[str, float]]:
    if not isinstance(raw, Mapping):
        return {}
    resolved: Dict[float, Dict[str, float]] = {}
    for key, value in raw.items():
        horizon = coerce_numeric_horizon(key)
        if horizon is None or not isinstance(value, Mapping):
            continue
        regime_values: Dict[str, float] = {}
        for regime, raw_number in value.items():
            candidate = raw_number.get("confidence_min") if isinstance(raw_number, Mapping) else raw_number
            numeric_value = finite_float_or_none(candidate)
            if numeric_value is None:
                continue
            numeric_value = max(numeric_value, minimum)
            if maximum is not None:
                numeric_value = min(numeric_value, maximum)
            regime_values[str(regime).strip().lower()] = numeric_value
        if regime_values:
            resolved[normalize_horizon_value(horizon)] = regime_values
    return resolved


def resolve_confidence_min_for_horizon(
    base_confidence_min: float,
    overrides: Mapping[float, Mapping[str, float]] | None,
    *,
    horizon: float,
    regime_state: str,
    normalize_horizon_value: Callable[[float], float],
    format_horizon_label: Callable[[float], str],
) -> tuple[float, str]:
    resolved = max(0.0, min(1.0, float(base_confidence_min)))
    source = "default"
    if not isinstance(overrides, Mapping):
        return resolved, source
    regime_map = overrides.get(normalize_horizon_value(horizon))
    if not isinstance(regime_map, Mapping):
        return resolved, source
    regime_key = str(regime_state).strip().lower()
    override = regime_map.get(regime_key)
    if override is None:
        override = regime_map.get("default")
        if override is None:
            return resolved, source
        source = f"{format_horizon_label(horizon)}@default"
    else:
        source = f"{format_horizon_label(horizon)}@{regime_key}"
    return max(0.0, min(1.0, float(override))), source


def normalize_threshold_overrides(
    overrides: Mapping[int | float | str, Dict[str, float]] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[float, Dict[str, float]]:
    if not isinstance(overrides, Mapping):
        return {}

    normalized: Dict[float, Dict[str, float]] = {}
    for raw_key, raw_entry in overrides.items():
        horizon = coerce_numeric_horizon(raw_key)
        if horizon is None:
            raise ValueError(f"Invalid threshold override horizon: {raw_key!r}")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Threshold override for {raw_key!r} must be a mapping.")
        normalized_horizon = normalize_horizon_value(horizon)
        if normalized_horizon in normalized:
            raise ValueError(f"Duplicate threshold override for normalized horizon {normalized_horizon!r}.")

        entry: Dict[str, float] = {}
        for key in ("p_up_min", "ret_min"):
            if key not in raw_entry:
                raise ValueError(f"Threshold override for {raw_key!r} is missing required key '{key}'.")
            entry[key] = float(raw_entry[key])

        for key in ("max_drawdown", "volatility_ceiling", "volatility_mult", "expected_value_multiplier"):
            if key in raw_entry and raw_entry[key] is not None:
                entry[key] = float(raw_entry[key])

        metric_key = raw_entry.get("volatility_metric")
        if isinstance(metric_key, str) and metric_key.strip():
            entry["volatility_metric"] = metric_key.strip()  # type: ignore[assignment]
        normalized[normalized_horizon] = entry
    return normalized


def resolve_thresholds_for_horizon(
    horizon: float,
    default_p_up: float,
    default_ret: float,
    overrides: Mapping[float, Dict[str, float]] | None,
    *,
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, float]:
    entry: Dict[str, float] | None = None
    if overrides:
        entry = overrides.get(normalize_horizon_value(horizon))

    p_up_value = float((entry or {}).get("p_up_min", default_p_up))
    ret_value = float((entry or {}).get("ret_min", default_ret))
    resolved: Dict[str, float] = {
        "p_up_min": p_up_value,
        "ret_min": ret_value,
    }
    if entry:
        for key in ("max_drawdown", "volatility_ceiling", "volatility_mult", "expected_value_multiplier"):
            if key in entry:
                try:
                    resolved[key] = float(entry[key])
                except (TypeError, ValueError):
                    continue
        metric_key = entry.get("volatility_metric")
        if isinstance(metric_key, str) and metric_key.strip():
            resolved["volatility_metric"] = metric_key.strip()
    return resolved


def resolve_regime_model_weights_policy(
    config: Mapping[str, Any] | None,
    *,
    regimes: Sequence[str],
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
    parse_weight_spec: Callable[[str], Mapping[str, float]],
) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    if not bool(config.get("enabled", False)):
        return {"enabled": False, "weights_by_regime": {}, "weights_by_regime_horizon": {}}

    weights_by_regime: Dict[str, Dict[str, float]] = {}
    weights_by_regime_horizon: Dict[str, Dict[float, Dict[str, float]]] = {}
    for regime in regimes:
        raw = config.get(regime)
        if not raw:
            continue
        if isinstance(raw, Mapping):
            per_horizon: Dict[float, Dict[str, float]] = {}
            for raw_horizon, raw_weights in raw.items():
                horizon = coerce_numeric_horizon(raw_horizon)
                if horizon is None:
                    continue
                parsed = parse_weight_spec(str(raw_weights))
                if parsed:
                    per_horizon[normalize_horizon_value(horizon)] = {str(k): float(v) for k, v in parsed.items()}
            if per_horizon:
                weights_by_regime_horizon[regime] = per_horizon
            continue

        parsed = parse_weight_spec(str(raw))
        if parsed:
            weights_by_regime[regime] = {str(k): float(v) for k, v in parsed.items()}
    return {
        "enabled": True,
        "weights_by_regime": weights_by_regime,
        "weights_by_regime_horizon": weights_by_regime_horizon,
    }


def resolve_direction_ensemble_policy(
    config: Mapping[str, Any] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    cfg = config or {}
    raw_groups = cfg.get("model_groups") if isinstance(cfg.get("model_groups"), Mapping) else {}
    model_groups: Dict[str, str] = {}
    for raw_group, raw_names in raw_groups.items():
        group = str(raw_group).strip().lower()
        if not group:
            continue
        if isinstance(raw_names, str):
            names = [segment.strip().lower() for segment in raw_names.split(",") if segment.strip()]
        elif isinstance(raw_names, Sequence):
            names = [str(item).strip().lower() for item in raw_names if str(item).strip()]
        else:
            continue
        for name in names:
            model_groups[name] = group

    max_active_by_horizon: Dict[float, int] = {}
    raw_active = cfg.get("max_active_by_horizon") if isinstance(cfg.get("max_active_by_horizon"), Mapping) else {}
    for raw_horizon, raw_limit in raw_active.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None or raw_limit is None:
            continue
        max_active_by_horizon[normalize_horizon_value(horizon)] = int(raw_limit)

    max_models_per_group_by_horizon: Dict[float, Dict[str, int]] = {}
    raw_group_caps = (
        cfg.get("max_models_per_group_by_horizon")
        if isinstance(cfg.get("max_models_per_group_by_horizon"), Mapping)
        else {}
    )
    for raw_horizon, raw_caps in raw_group_caps.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None or not isinstance(raw_caps, Mapping):
            continue
        resolved_caps = {
            str(group).strip().lower(): int(limit)
            for group, limit in raw_caps.items()
            if str(group).strip() and limit is not None
        }
        if resolved_caps:
            max_models_per_group_by_horizon[normalize_horizon_value(horizon)] = resolved_caps

    priority_by_horizon: Dict[float, List[str]] = {}
    raw_priorities = cfg.get("priority_by_horizon") if isinstance(cfg.get("priority_by_horizon"), Mapping) else {}
    for raw_horizon, raw_priority in raw_priorities.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None:
            continue
        if isinstance(raw_priority, str):
            values = [segment.strip().lower() for segment in raw_priority.split(",") if segment.strip()]
        elif isinstance(raw_priority, Sequence):
            values = [str(item).strip().lower() for item in raw_priority if str(item).strip()]
        else:
            continue
        if values:
            priority_by_horizon[normalize_horizon_value(horizon)] = values

    preferred_groups_by_horizon: Dict[float, List[str]] = {}
    raw_preferred_groups = (
        cfg.get("preferred_groups_by_horizon")
        if isinstance(cfg.get("preferred_groups_by_horizon"), Mapping)
        else {}
    )
    for raw_horizon, raw_groups in raw_preferred_groups.items():
        horizon = coerce_numeric_horizon(raw_horizon)
        if horizon is None:
            continue
        if isinstance(raw_groups, str):
            groups = [segment.strip().lower() for segment in raw_groups.split(",") if segment.strip()]
        elif isinstance(raw_groups, Sequence):
            groups = [str(item).strip().lower() for item in raw_groups if str(item).strip()]
        else:
            continue
        if groups:
            preferred_groups_by_horizon[normalize_horizon_value(horizon)] = groups

    raw_horizons = cfg.get("horizons")
    scoped_horizons = None
    if isinstance(raw_horizons, Sequence) and not isinstance(raw_horizons, (str, bytes)):
        scoped_horizons = {normalize_horizon_value(item) for item in raw_horizons}

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "horizons": scoped_horizons,
        "lookback_bars": max(int(cfg.get("lookback_bars", 96) or 96), 2),
        "min_history_points": max(int(cfg.get("min_history_points", 24) or 24), 0),
        "max_correlation": float(cfg.get("max_correlation", 0.985) or 0.985),
        "min_mean_abs_probability_gap": float(
            cfg.get("min_mean_abs_probability_gap", 0.02) or 0.02
        ),
        "model_groups": model_groups,
        "max_active_by_horizon": max_active_by_horizon,
        "max_models_per_group_by_horizon": max_models_per_group_by_horizon,
        "priority_by_horizon": priority_by_horizon,
        "preferred_groups_by_horizon": preferred_groups_by_horizon,
    }


def scope_direction_ensemble_policy(
    policy: Mapping[str, Any] | None,
    horizon: float,
    *,
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    if not policy or not bool(policy.get("enabled", False)):
        return {"enabled": False}
    normalized_horizon = normalize_horizon_value(horizon)
    scoped_horizons = policy.get("horizons")
    if isinstance(scoped_horizons, set) and scoped_horizons and normalized_horizon not in scoped_horizons:
        return {"enabled": False}
    priority_by_horizon = policy.get("priority_by_horizon") if isinstance(policy.get("priority_by_horizon"), Mapping) else {}
    max_active_by_horizon = policy.get("max_active_by_horizon") if isinstance(policy.get("max_active_by_horizon"), Mapping) else {}
    max_models_per_group_by_horizon = (
        policy.get("max_models_per_group_by_horizon")
        if isinstance(policy.get("max_models_per_group_by_horizon"), Mapping)
        else {}
    )
    preferred_groups_by_horizon = (
        policy.get("preferred_groups_by_horizon")
        if isinstance(policy.get("preferred_groups_by_horizon"), Mapping)
        else {}
    )
    return {
        "enabled": True,
        "lookback_bars": int(policy.get("lookback_bars", 96) or 96),
        "min_history_points": int(policy.get("min_history_points", 24) or 24),
        "max_correlation": float(policy.get("max_correlation", 0.985) or 0.985),
        "min_mean_abs_probability_gap": float(
            policy.get("min_mean_abs_probability_gap", 0.02) or 0.02
        ),
        "model_groups": dict(policy.get("model_groups") or {}),
        "max_active_models": max_active_by_horizon.get(normalized_horizon),
        "max_models_per_group": dict(max_models_per_group_by_horizon.get(normalized_horizon, {}) or {}),
        "priority_order": list(priority_by_horizon.get(normalized_horizon, []) or []),
        "preferred_groups": list(preferred_groups_by_horizon.get(normalized_horizon, []) or []),
    }


def resolve_regime_model_dirs_policy(
    config: Mapping[str, Any] | None,
    *,
    regimes: Sequence[str],
) -> Dict[str, Any]:
    if not config or not bool(config.get("enabled", False)):
        return {"enabled": False, "paths": {}}

    paths: Dict[str, Dict[str, str]] = {}
    for regime in regimes:
        raw = config.get(regime)
        if isinstance(raw, Mapping):
            paths[regime] = {str(k): str(v) for k, v in raw.items() if v is not None}
    return {"enabled": True, "paths": paths}


def resolve_regime_specific_dir_path(
    default_path: Path,
    *,
    regime_state: str,
    horizon_label: str,
    policy: Mapping[str, Any],
    expected_filename: str,
    version_priority: Sequence[str],
    resolve_best_versioned_model_file: Callable[..., Path],
    stderr_write: Callable[[str], None],
) -> Path:
    if not policy or not bool(policy.get("enabled", False)):
        return default_path
    path_map = policy.get("paths", {})
    regime_map = path_map.get(regime_state) if isinstance(path_map, Mapping) else None
    if not isinstance(regime_map, Mapping):
        return default_path
    override = regime_map.get(horizon_label)
    if not override:
        return default_path
    override_path = Path(str(override)).expanduser()
    override_path = resolve_best_versioned_model_file(
        override_path,
        expected_filename=expected_filename,
        version_priority=version_priority,
    )
    if not override_path.exists():
        stderr_write(
            f"Warning: regime model dir override not found for {horizon_label}@{regime_state}: {override_path}\n"
        )
        return default_path
    return override_path


def apply_regime_weight_overrides(
    base_weights: Mapping[str, float],
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, float]:
    resolved = {str(k): float(v) for k, v in base_weights.items()}
    if not policy or not bool(policy.get("enabled")):
        return resolved
    normalized_horizon = normalize_horizon_value(horizon) if horizon is not None else None
    weights_by_regime_horizon = policy.get("weights_by_regime_horizon") or {}
    if normalized_horizon is not None:
        horizon_overrides = weights_by_regime_horizon.get(regime_state)
        if isinstance(horizon_overrides, Mapping):
            override = horizon_overrides.get(normalized_horizon)
            if isinstance(override, Mapping):
                for key, value in override.items():
                    resolved[str(key)] = float(value)
                return resolved
    weights_by_regime = policy.get("weights_by_regime") or {}
    override = weights_by_regime.get(regime_state)
    if not isinstance(override, Mapping):
        return resolved
    for key, value in override.items():
        resolved[str(key)] = float(value)
    return resolved


def get_active_regime_weight_override(
    *,
    regime_state: str,
    horizon: float | None = None,
    policy: Optional[Mapping[str, Any]],
    normalize_horizon_value: Callable[[float], float],
) -> Optional[Dict[str, float]]:
    if not policy or not bool(policy.get("enabled")):
        return None
    normalized_horizon = normalize_horizon_value(horizon) if horizon is not None else None
    weights_by_regime_horizon = policy.get("weights_by_regime_horizon") or {}
    if normalized_horizon is not None:
        horizon_overrides = weights_by_regime_horizon.get(regime_state)
        if isinstance(horizon_overrides, Mapping):
            override = horizon_overrides.get(normalized_horizon)
            if isinstance(override, Mapping):
                return {str(k): float(v) for k, v in override.items()}
    weights_by_regime = policy.get("weights_by_regime") or {}
    override = weights_by_regime.get(regime_state)
    if isinstance(override, Mapping):
        return {str(k): float(v) for k, v in override.items()}
    return None