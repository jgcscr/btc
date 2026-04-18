from __future__ import annotations

import pytest

from src.runtime.config_normalization_support import normalize_config_value


def _bool_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_targets(value: str):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _normalize_horizon_value(value):
    return round(float(value), 4)


def _normalize_horizon_float_map(raw, *, minimum=0.0, maximum=None):
    resolved = {}
    for key, value in raw.items():
        numeric = max(float(value), minimum)
        if maximum is not None:
            numeric = min(numeric, maximum)
        resolved[_normalize_horizon_value(key)] = numeric
    return resolved


def _normalize_horizon_regime_float_map(raw, *, minimum=0.0, maximum=None):
    resolved = {}
    for key, mapping in raw.items():
        regime_values = {}
        for regime, value in mapping.items():
            numeric = max(float(value), minimum)
            if maximum is not None:
                numeric = min(numeric, maximum)
            regime_values[str(regime)] = numeric
        resolved[_normalize_horizon_value(key)] = regime_values
    return resolved


def _normalize(name, value):
    return normalize_config_value(
        name,
        value,
        default_targets=(0.25, 1.0, 4.0),
        config_int_fields=("hours",),
        config_float_fields=("p_up_min",),
        config_bool_fields=("write_artifacts",),
        config_path_fields=("config",),
        config_allowed_keys=(
            "targets",
            "write_artifacts",
            "hours",
            "p_up_min",
            "feature_coverage_policy",
            "confluence_policy",
            "direction_output_policy",
            "direction_ensemble_policy",
            "execution_policy",
            "regression_model_dirs",
            "trust_hardening_policy",
            "position_size_cap_by_horizon",
            "confidence_min_by_horizon_regime",
            "trade_decision_policy",
        ),
        regimes=("trend_ignition", "neutral", "chop"),
        bool_env=_bool_env,
        parse_targets=_parse_targets,
        normalize_horizon_value=_normalize_horizon_value,
        normalize_horizon_float_map=_normalize_horizon_float_map,
        normalize_horizon_regime_float_map=_normalize_horizon_regime_float_map,
        stderr_write=lambda _message: None,
    )


def test_normalize_config_value_handles_targets_bools_and_horizon_maps() -> None:
    assert _normalize("targets", "1,4") == [1.0, 4.0]
    assert _normalize("write_artifacts", "true") is True
    assert _normalize("position_size_cap_by_horizon", {"1": 0.2, "4": 1.4}) == {1.0: 0.2, 4.0: 1.0}
    assert _normalize("confidence_min_by_horizon_regime", {"4": {"neutral": 0.3}}) == {4.0: {"neutral": 0.3}}


def test_normalize_direction_output_policy_handles_nested_sequences_and_weight_specs() -> None:
    normalized = _normalize(
        "direction_output_policy",
        {
            "enabled": True,
            "horizons": [1, "4"],
            "neutral_band": 0.02,
            "marginal_rerank": {
                "enabled": True,
                "horizons": "1,4",
                "min_component_count": 2,
                "weight_specs": {"default": "gru:1.5,lstm:1.0"},
            },
            "probability_shrinkage": {
                "enabled": True,
                "horizons": [1],
                "regimes": ["neutral", "chop"],
                "strength_by_horizon": {"1": 0.4},
            },
        },
    )

    assert normalized["horizons"] == [1.0, 4.0]
    assert normalized["marginal_rerank"]["horizons"] == [1.0, 4.0]
    assert normalized["marginal_rerank"]["weight_specs"] == {"default": "gru:1.5,lstm:1.0"}
    assert normalized["probability_shrinkage"]["regimes"] == ["neutral", "chop"]
    assert normalized["probability_shrinkage"]["strength_by_horizon"] == {"1": 0.4}


def test_normalize_execution_policy_preserves_horizon_lists_and_nested_mappings() -> None:
    normalized = _normalize(
        "execution_policy",
        {
            "enabled": True,
            "bias_horizons": [4, 8],
            "execution_horizons": [1, "4"],
        },
    )

    assert normalized["enabled"] is True
    assert normalized["bias_horizons"] == [4.0, 8.0]
    assert normalized["execution_horizons"] == [1.0, 4.0]


def test_normalize_execution_policy_rejects_invalid_sequence_type() -> None:
    with pytest.raises(ValueError, match="execution_horizons in execution_policy must be a list/sequence"):
        _normalize(
            "execution_policy",
            {
                "enabled": True,
                "execution_horizons": "1,4",
            },
        )


def test_normalize_confluence_policy_parses_horizon_lists_and_maps() -> None:
    normalized = _normalize(
        "confluence_policy",
        {
            "enabled": True,
            "short_horizons": "1,4",
            "mid_horizons": [8, "12"],
            "min_support_ratio_by_horizon": {"4": 1.0},
        },
    )

    assert normalized["enabled"] is True
    assert normalized["short_horizons"] == [1.0, 4.0]
    assert normalized["mid_horizons"] == [8.0, 12.0]
    assert normalized["min_support_ratio_by_horizon"] == {"4": 1.0}


def test_normalize_trade_decision_policy_preserves_nested_veto_blocks() -> None:
    normalized = _normalize(
        "trade_decision_policy",
        {
            "enabled": True,
            "threshold": 0.55,
            "midband_veto": {
                "enabled": True,
                "p_up_low": 0.56,
                "p_up_high": 0.59,
                "regime_states": ["chop", "neutral"],
            },
        },
    )

    assert normalized["enabled"] is True
    assert normalized["threshold"] == 0.55
    assert normalized["midband_veto"] == {
        "enabled": True,
        "p_up_low": 0.56,
        "p_up_high": 0.59,
        "high_inclusive": False,
        "min_abs_ret_pred": None,
        "max_abs_ret_pred": None,
        "regime_states": ["chop", "neutral"],
    }


def test_normalize_regression_model_dirs_preserves_enabled_and_paths() -> None:
    normalized = _normalize(
        "regression_model_dirs",
        {
            "enabled": True,
            "4h": "artifacts/models/featurelift_20260331_rerun/xgb_ret4h",
        },
    )

    assert normalized == {
        "enabled": True,
        "4h": "artifacts/models/featurelift_20260331_rerun/xgb_ret4h",
    }


def test_normalize_direction_ensemble_policy_parses_horizons_and_group_mappings() -> None:
    normalized = _normalize(
        "direction_ensemble_policy",
        {
            "enabled": True,
            "horizons": "1,4",
            "model_groups": {"tree": ["xgb"]},
            "priority_by_horizon": {"1": ["xgb", "gru"]},
        },
    )

    assert normalized["enabled"] is True
    assert normalized["horizons"] == [1.0, 4.0]
    assert normalized["model_groups"] == {"tree": ["xgb"]}
    assert normalized["priority_by_horizon"] == {"1": ["xgb", "gru"]}


def test_normalize_feature_coverage_policy_parses_ignored_sources() -> None:
    normalized = _normalize(
        "feature_coverage_policy",
        {
            "enabled": True,
            "max_imputed_zero_ratio": 0.2,
            "ignored_sources": ["macro", "onchain"],
        },
    )

    assert normalized["enabled"] is True
    assert normalized["max_imputed_zero_ratio"] == 0.2
    assert normalized["ignored_sources"] == ["macro", "onchain"]


def test_normalize_trust_hardening_policy_parses_horizon_lists_and_mappings() -> None:
    normalized = _normalize(
        "trust_hardening_policy",
        {
            "enabled": True,
            "horizons": "4,8",
            "high_impact_horizons": [4, 8],
            "default_action": "exclude",
            "action_by_horizon": {"4": "exclude", "8": "deweight"},
            "deweight_factor_by_horizon": {"8": 0.3},
            "metadata_checks": {"enabled": True, "require_metadata": True},
        },
    )

    assert normalized["enabled"] is True
    assert normalized["horizons"] == [4.0, 8.0]
    assert normalized["high_impact_horizons"] == [4.0, 8.0]
    assert normalized["default_action"] == "exclude"
    assert normalized["action_by_horizon"] == {"4": "exclude", "8": "deweight"}
    assert normalized["deweight_factor_by_horizon"] == {"8": 0.3}
    assert normalized["metadata_checks"] == {"enabled": True, "require_metadata": True}