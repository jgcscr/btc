from __future__ import annotations

from types import SimpleNamespace

from src.runtime.execution_policy_support import apply_execution_policy, classify_execution_tier, resolve_execution_policy


def _unexpected(*args, **kwargs):  # pragma: no cover - helper for guard tests
    raise AssertionError("unexpected callback invocation")


def test_resolve_execution_policy_normalizes_nested_config_blocks() -> None:
    policy = resolve_execution_policy(
        {
            "enabled": True,
            "bias_horizons": ["8", 4],
            "execution_horizons": [1, "0.25"],
            "horizon_bias_weights": {"4": 0.8, "bad": 1.0},
            "analytics": {
                "enabled": True,
                "min_samples": 5,
                "regime_volatility_buckets": {"enabled": True, "low_vol_quantile": 0.2},
            },
            "target_range_stop_refinement": {"enabled": True, "horizons": ["4", 8], "confidence_min": 1.5},
            "regime_templates": {
                "trend": {"tp_multiplier": 1.2, "entry_mode_by_tier": {"high": "immediate", "": "skip"}}
            },
        },
        normalize_horizon_value=lambda value: float(value),
        coerce_numeric_horizon=lambda value: None if value == "bad" else float(value),
        default_lookback_bars=240,
        default_min_samples=40,
        default_target_range_stop_horizons=(1.0, 4.0),
        default_target_range_stop_confidence_min=0.35,
        default_target_range_stop_buffer_std_mult=0.5,
        default_target_range_stop_min_tighten_fraction=0.2,
    )

    assert policy["enabled"] is True
    assert policy["bias_horizons"] == [4.0, 8.0]
    assert policy["execution_horizons"] == [0.25, 1.0]
    assert policy["horizon_bias_weights"] == {4.0: 0.8}
    assert policy["analytics"]["enabled"] is True
    assert policy["analytics"]["min_samples"] == 10
    assert policy["analytics"]["regime_volatility_buckets"]["enabled"] is True
    assert policy["target_range_stop_refinement"]["horizons"] == [4.0, 8.0]
    assert policy["target_range_stop_refinement"]["confidence_min"] == 1.0
    assert policy["regime_templates"]["trend"]["entry_mode_by_tier"] == {"high": "immediate"}


def test_resolve_execution_policy_preserves_explicit_empty_short_term_strict_horizons() -> None:
    policy = resolve_execution_policy(
        {
            "enabled": True,
            "short_term_strict_horizons": [],
        },
        normalize_horizon_value=lambda value: float(value),
        coerce_numeric_horizon=lambda value: float(value),
        default_lookback_bars=240,
        default_min_samples=40,
        default_target_range_stop_horizons=(1.0, 4.0),
        default_target_range_stop_confidence_min=0.35,
        default_target_range_stop_buffer_std_mult=0.5,
        default_target_range_stop_min_tighten_fraction=0.2,
    )

    assert policy["short_term_strict_horizons"] == []


def test_classify_execution_tier_allows_old_1h_shadow_snapshot_when_strict_short_term_is_disabled() -> None:
    policy = resolve_execution_policy(
        {
            "enabled": True,
            "short_term_strict_horizons": [],
            "short_term_min_support_ratio": 0.8,
            "short_term_min_mid_ratio": 0.8,
            "immediate_entry_min_support_ratio": 0.8,
            "immediate_entry_min_mid_ratio": 1.0,
            "pullback_entry_min_support_ratio": 0.5,
            "pullback_entry_min_mid_ratio": 0.66,
            "high_execution_alignment_ratio": 0.5,
            "medium_execution_alignment_ratio": 0.5,
        },
        normalize_horizon_value=lambda value: float(value),
        coerce_numeric_horizon=lambda value: float(value),
        default_lookback_bars=240,
        default_min_samples=40,
        default_target_range_stop_horizons=(1.0, 4.0),
        default_target_range_stop_confidence_min=0.35,
        default_target_range_stop_buffer_std_mult=0.5,
        default_target_range_stop_min_tighten_fraction=0.2,
    )

    tier = classify_execution_tier(
        {
            "horizon_hours": 1.0,
            "direction_next": "up",
            "confluence_support_ratio": 0.7777777777777778,
            "confluence_mid_term_ratio": 1.0,
        },
        bias_direction="up",
        execution_alignment_ratio=1.0,
        policy=policy,
    )

    assert tier == "medium"


def test_apply_execution_policy_disabled_sets_basic_plan_without_deep_callbacks() -> None:
    summary = {
        "4h": {
            "close": 100000.0,
            "entry_price": 100000.0,
            "trade_action": "long",
        }
    }

    result = apply_execution_policy(
        summary,
        {},
        {"enabled": False},
        regime_neutral="neutral",
        execution_policy_default_lookback_bars=240,
        execution_policy_default_min_samples=40,
        summarize_bias_context=lambda payload, policy: {
            "bias_direction": "up",
            "bias_alignment_ratio": 1.0,
            "execution_entries": [],
            "bias_scores": {"up_score": 1.0, "down_score": 0.0},
            "execution_scores": {"up_score": 1.0, "down_score": 0.0},
            "direction_support_horizons": {"up": ["4h"]},
        },
        execution_side=lambda entry: "long",
        direction_vote=lambda entry: "up",
        execution_alignment_ratio=lambda execution_entries, direction, weights: 1.0,
        classify_execution_tier=lambda entry, bias_direction, execution_alignment_ratio, policy: "high",
        compute_atr_like_price_distance=_unexpected,
        compute_recent_structure=_unexpected,
        build_entry_zone=_unexpected,
        compute_pullback_quality_score=_unexpected,
        compute_disagreement_severity=_unexpected,
        compute_excursion_priors=_unexpected,
        finite_float_or_none=lambda value: None if value is None else float(value),
        finite_float=lambda value, default: default if value is None else float(value),
        resolve_stop_with_guardrails=_unexpected,
        refine_stop_with_target_range=_unexpected,
        resolve_execution_target_reward=_unexpected,
        lookup_horizon_value=lambda mapping, horizon, default: default,
        resolve_execution_upstream_hold_reason=lambda entry: "upstream_model_hold",
    )

    plan = result["4h"]["execution_plan"]
    assert plan["enabled"] is False
    assert plan["status"] == "ready"
    assert plan["entry_mode"] == "disabled"
    assert result["4h"]["bias_support_horizons"] == ["4h"]


def test_apply_execution_policy_rejects_forecast_coherence_before_context_use() -> None:
    summary = {
        "1h": {
            "close": 100000.0,
            "entry_price": 100000.0,
            "trade_action": "short",
            "forecast_coherence": {"triggered": True},
        }
    }

    result = apply_execution_policy(
        summary,
        {},
        {"enabled": True, "require_bias_alignment": True},
        regime_neutral="neutral",
        execution_policy_default_lookback_bars=240,
        execution_policy_default_min_samples=40,
        summarize_bias_context=lambda payload, policy: {
            "bias_direction": "down",
            "bias_alignment_ratio": 0.8,
            "execution_entries": [],
            "bias_scores": {"up_score": 0.1, "down_score": 0.9},
            "execution_scores": {"up_score": 0.2, "down_score": 0.85},
            "direction_support_horizons": {"down": ["1h"]},
        },
        execution_side=lambda entry: "short",
        direction_vote=lambda entry: "down",
        execution_alignment_ratio=lambda execution_entries, direction, weights: 0.8,
        classify_execution_tier=lambda entry, bias_direction, execution_alignment_ratio, policy: "high",
        compute_atr_like_price_distance=_unexpected,
        compute_recent_structure=_unexpected,
        build_entry_zone=_unexpected,
        compute_pullback_quality_score=_unexpected,
        compute_disagreement_severity=_unexpected,
        compute_excursion_priors=_unexpected,
        finite_float_or_none=lambda value: None if value is None else float(value),
        finite_float=lambda value, default: default if value is None else float(value),
        resolve_stop_with_guardrails=_unexpected,
        refine_stop_with_target_range=_unexpected,
        resolve_execution_target_reward=_unexpected,
        lookup_horizon_value=lambda mapping, horizon, default: default,
        resolve_execution_upstream_hold_reason=lambda entry: "upstream_model_hold",
    )

    plan = result["1h"]["execution_plan"]
    assert plan["enabled"] is True
    assert plan["status"] == "rejected"
    assert plan["reason"] == "forecast_coherence_gate"
