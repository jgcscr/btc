from __future__ import annotations

from pathlib import Path

from src.runtime.policy_support import (
    apply_regime_weight_overrides,
    get_active_regime_weight_override,
    normalize_horizon_float_map,
    normalize_horizon_regime_float_map,
    normalize_threshold_overrides,
    resolve_confidence_min_for_horizon,
    resolve_direction_ensemble_policy,
    resolve_regime_model_dirs_policy,
    resolve_regime_model_weights_policy,
    resolve_regime_specific_dir_path,
    resolve_thresholds_for_horizon,
    scope_direction_ensemble_policy,
)


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return round(float(value), 4)


def _finite_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_horizon_label(value):
    numeric = float(value)
    return f"{int(numeric)}h" if numeric.is_integer() else f"{numeric:g}h"


def _parse_weight_spec(spec):
    resolved = {}
    for part in spec.split(","):
        if not part.strip() or ":" not in part:
            continue
        key, value = part.split(":", 1)
        resolved[key.strip()] = float(value)
    return resolved


def test_normalize_horizon_maps_support_clamping_and_nested_confidence_values() -> None:
    numeric = normalize_horizon_float_map(
        {"1": 0.9, "bad": 1.0, "4": 1.4},
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        minimum=0.0,
        maximum=1.0,
    )
    regime = normalize_horizon_regime_float_map(
        {
            "8": {"trend_ignition": {"confidence_min": 0.2}, "default": 1.4},
            "bad": {"neutral": 0.1},
        },
        finite_float_or_none=_finite_float_or_none,
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        minimum=0.0,
        maximum=1.0,
    )

    assert numeric == {1.0: 0.9, 4.0: 1.0}
    assert regime == {8.0: {"trend_ignition": 0.2, "default": 1.0}}


def test_resolve_confidence_min_for_horizon_prefers_regime_then_default() -> None:
    value, source = resolve_confidence_min_for_horizon(
        0.33,
        {8.0: {"trend_ignition": 0.23, "default": 0.44}},
        horizon=8.0,
        regime_state="trend_ignition",
        normalize_horizon_value=_normalize_horizon_value,
        format_horizon_label=_format_horizon_label,
    )
    default_value, default_source = resolve_confidence_min_for_horizon(
        0.33,
        {8.0: {"default": 0.44}},
        horizon=8.0,
        regime_state="neutral",
        normalize_horizon_value=_normalize_horizon_value,
        format_horizon_label=_format_horizon_label,
    )

    assert value == 0.23
    assert source == "8h@trend_ignition"
    assert default_value == 0.44
    assert default_source == "8h@default"


def test_threshold_override_helpers_normalize_duplicates_and_optional_fields() -> None:
    normalized = normalize_threshold_overrides(
        {
            "1": {
                "p_up_min": 0.55,
                "ret_min": 0.003,
                "max_drawdown": 0.01,
                "volatility_metric": "atr",
            }
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )
    resolved = resolve_thresholds_for_horizon(
        1.0,
        0.5,
        0.001,
        normalized,
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert normalized == {
        1.0: {
            "p_up_min": 0.55,
            "ret_min": 0.003,
            "max_drawdown": 0.01,
            "volatility_metric": "atr",
        }
    }
    assert resolved == {
        "p_up_min": 0.55,
        "ret_min": 0.003,
        "max_drawdown": 0.01,
        "volatility_metric": "atr",
    }


def test_normalize_threshold_overrides_rejects_duplicate_normalized_horizons() -> None:
    try:
        normalize_threshold_overrides(
            {
                "1": {"p_up_min": 0.5, "ret_min": 0.0},
                "1.0": {"p_up_min": 0.6, "ret_min": 0.001},
            },
            coerce_numeric_horizon=_coerce_numeric_horizon,
            normalize_horizon_value=_normalize_horizon_value,
        )
    except ValueError as exc:
        assert "Duplicate threshold override" in str(exc)
    else:
        raise AssertionError("Expected duplicate normalized horizon error")


def test_direction_ensemble_policy_resolves_and_scopes_horizon_specific_settings() -> None:
    policy = resolve_direction_ensemble_policy(
        {
            "enabled": True,
            "horizons": [1, "4"],
            "model_groups": {"fast": ["gru", "lstm"], "slow": "xgb, transformer"},
            "max_active_by_horizon": {"1": 2, "4": 3},
            "max_models_per_group_by_horizon": {"1": {"fast": 1}},
            "priority_by_horizon": {"1": "gru, xgb"},
            "preferred_groups_by_horizon": {"1": ["fast"]},
        },
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )

    scoped = scope_direction_ensemble_policy(
        policy,
        1.0,
        normalize_horizon_value=_normalize_horizon_value,
    )
    excluded = scope_direction_ensemble_policy(
        policy,
        8.0,
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert policy["model_groups"] == {
        "gru": "fast",
        "lstm": "fast",
        "xgb": "slow",
        "transformer": "slow",
    }
    assert scoped["enabled"] is True
    assert scoped["max_active_models"] == 2
    assert scoped["max_models_per_group"] == {"fast": 1}
    assert scoped["priority_order"] == ["gru", "xgb"]
    assert scoped["preferred_groups"] == ["fast"]
    assert excluded == {"enabled": False}


def test_regime_weight_policy_prefers_horizon_override_and_reports_active_override() -> None:
    policy = resolve_regime_model_weights_policy(
        {
            "enabled": True,
            "trend_ignition": {"4": "gru:1.2,xgb:0.8"},
            "neutral": "gru:0.5",
        },
        regimes=("trend_ignition", "neutral", "chop"),
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
        parse_weight_spec=_parse_weight_spec,
    )

    resolved = apply_regime_weight_overrides(
        {"gru": 1.0, "lstm": 1.0},
        regime_state="trend_ignition",
        horizon=4.0,
        policy=policy,
        normalize_horizon_value=_normalize_horizon_value,
    )
    active = get_active_regime_weight_override(
        regime_state="trend_ignition",
        horizon=4.0,
        policy=policy,
        normalize_horizon_value=_normalize_horizon_value,
    )

    assert policy["weights_by_regime_horizon"]["trend_ignition"][4.0] == {"gru": 1.2, "xgb": 0.8}
    assert resolved == {"gru": 1.2, "lstm": 1.0, "xgb": 0.8}
    assert active == {"gru": 1.2, "xgb": 0.8}


def test_resolve_regime_specific_dir_path_uses_existing_override(tmp_path: Path) -> None:
    default_path = tmp_path / "default.json"
    default_path.write_text("{}", encoding="utf-8")
    override_dir = tmp_path / "regime"
    override_dir.mkdir()
    override_file = override_dir / "xgb_dir4h_model.json"
    override_file.write_text("{}", encoding="utf-8")

    policy = resolve_regime_model_dirs_policy(
        {
            "enabled": True,
            "trend_ignition": {"4h": str(override_dir)},
        },
        regimes=("trend_ignition", "neutral", "chop"),
    )

    resolved = resolve_regime_specific_dir_path(
        default_path,
        regime_state="trend_ignition",
        horizon_label="4h",
        policy=policy,
        expected_filename="xgb_dir4h_model.json",
        version_priority=("v3", "v2"),
        resolve_best_versioned_model_file=lambda path, **_: path / "xgb_dir4h_model.json",
        stderr_write=lambda _message: None,
    )

    assert resolved == override_file


def test_resolve_regime_specific_dir_path_falls_back_when_override_missing(tmp_path: Path) -> None:
    default_path = tmp_path / "default.json"
    default_path.write_text("{}", encoding="utf-8")
    warnings = []

    resolved = resolve_regime_specific_dir_path(
        default_path,
        regime_state="trend_ignition",
        horizon_label="4h",
        policy={"enabled": True, "paths": {"trend_ignition": {"4h": str(tmp_path / "missing")}}},
        expected_filename="xgb_dir4h_model.json",
        version_priority=("v3",),
        resolve_best_versioned_model_file=lambda path, **_: path / "xgb_dir4h_model.json",
        stderr_write=warnings.append,
    )

    assert resolved == default_path
    assert warnings