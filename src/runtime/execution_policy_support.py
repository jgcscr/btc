from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence


SummaryPayload = Dict[str, Dict[str, Any]]
ExecutionContexts = Mapping[str, Dict[str, Any]]


def resolve_execution_policy(
    config: Mapping[str, Any] | None,
    *,
    normalize_horizon_value: Callable[[Any], float],
    coerce_numeric_horizon: Callable[[Any], float | None],
    default_lookback_bars: int,
    default_min_samples: int,
    default_target_range_stop_horizons: Sequence[float],
    default_target_range_stop_confidence_min: float,
    default_target_range_stop_buffer_std_mult: float,
    default_target_range_stop_min_tighten_fraction: float,
) -> Dict[str, Any]:
    cfg = config or {}

    def normalize_float_map(raw: Any, *, minimum: float = 0.0) -> Dict[float, float]:
        if not isinstance(raw, Mapping):
            return {}
        resolved: Dict[float, float] = {}
        for key, value in raw.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                resolved[horizon] = max(float(value), minimum)
            except (TypeError, ValueError):
                continue
        return resolved

    partial_cfg = cfg.get("partial_take_profit") if isinstance(cfg.get("partial_take_profit"), Mapping) else {}
    trailing_cfg = cfg.get("trailing_stop") if isinstance(cfg.get("trailing_stop"), Mapping) else {}
    analytics_cfg = cfg.get("analytics") if isinstance(cfg.get("analytics"), Mapping) else {}
    analytics_bucket_cfg = (
        analytics_cfg.get("regime_volatility_buckets")
        if isinstance(analytics_cfg.get("regime_volatility_buckets"), Mapping)
        else {}
    )
    guards_cfg = cfg.get("no_trade_guards") if isinstance(cfg.get("no_trade_guards"), Mapping) else {}
    adaptive_tp_cfg = cfg.get("adaptive_take_profit") if isinstance(cfg.get("adaptive_take_profit"), Mapping) else {}
    target_range_stop_cfg = (
        cfg.get("target_range_stop_refinement")
        if isinstance(cfg.get("target_range_stop_refinement"), Mapping)
        else {}
    )
    raw_regime_templates = cfg.get("regime_templates") if isinstance(cfg.get("regime_templates"), Mapping) else {}
    regime_templates: Dict[str, Dict[str, Any]] = {}
    for regime_name, raw_template in raw_regime_templates.items():
        if not isinstance(raw_template, Mapping):
            continue
        entry_mode_by_tier = raw_template.get("entry_mode_by_tier") if isinstance(raw_template.get("entry_mode_by_tier"), Mapping) else {}
        regime_templates[str(regime_name)] = {
            "tp_multiplier": max(float(raw_template.get("tp_multiplier", 1.0) or 1.0), 0.1),
            "time_stop_multiplier": max(float(raw_template.get("time_stop_multiplier", 1.0) or 1.0), 0.1),
            "size_multiplier": max(float(raw_template.get("size_multiplier", 1.0) or 1.0), 0.0),
            "entry_zone_atr_mult": max(float(raw_template.get("entry_zone_atr_mult", 0.0) or 0.0), 0.0),
            "max_chase_atr_mult": max(float(raw_template.get("max_chase_atr_mult", 0.0) or 0.0), 0.0),
            "pullback_quality_floor": max(float(raw_template.get("pullback_quality_floor", 0.0) or 0.0), 0.0),
            "entry_mode_by_tier": {
                str(tier).strip().lower(): str(mode).strip().lower()
                for tier, mode in entry_mode_by_tier.items()
                if str(tier).strip() and str(mode).strip()
            },
        }

    pullback_quality_cfg = cfg.get("pullback_quality") if isinstance(cfg.get("pullback_quality"), Mapping) else {}
    disagreement_cfg = cfg.get("disagreement_severity") if isinstance(cfg.get("disagreement_severity"), Mapping) else {}
    coherence_weighting_cfg = cfg.get("coherence_weighting") if isinstance(cfg.get("coherence_weighting"), Mapping) else {}
    dynamic_rr_floor_cfg = cfg.get("dynamic_rr_floor") if isinstance(cfg.get("dynamic_rr_floor"), Mapping) else {}
    volatility_expansion_stop_cfg = (
        cfg.get("volatility_expansion_stop") if isinstance(cfg.get("volatility_expansion_stop"), Mapping) else {}
    )

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "bias_horizons": sorted({normalize_horizon_value(value) for value in (cfg.get("bias_horizons") or [4.0, 8.0, 12.0])}),
        "execution_horizons": sorted({normalize_horizon_value(value) for value in (cfg.get("execution_horizons") or [0.25, 1.0])}),
        "horizon_bias_weights": normalize_float_map(cfg.get("horizon_bias_weights"), minimum=0.0),
        "short_term_strict_horizons": sorted(
            {normalize_horizon_value(value) for value in (cfg.get("short_term_strict_horizons") or [1.0])}
        ),
        "short_term_min_mid_ratio": max(min(float(cfg.get("short_term_min_mid_ratio") or 0.67), 1.0), 0.0),
        "short_term_min_support_ratio": max(min(float(cfg.get("short_term_min_support_ratio") or 0.75), 1.0), 0.0),
        "short_term_min_mid_ratio_by_horizon": normalize_float_map(
            cfg.get("short_term_min_mid_ratio_by_horizon"),
            minimum=0.0,
        ),
        "min_bias_alignment_ratio": max(min(float(cfg.get("min_bias_alignment_ratio") or 0.0), 1.0), 0.0),
        "short_term_min_support_ratio_by_horizon": normalize_float_map(
            cfg.get("short_term_min_support_ratio_by_horizon"),
            minimum=0.0,
        ),
        "require_bias_alignment": bool(cfg.get("require_bias_alignment", True)),
        "immediate_entry_min_support_ratio": max(min(float(cfg.get("immediate_entry_min_support_ratio") or 0.8), 1.0), 0.0),
        "pullback_entry_min_support_ratio": max(min(float(cfg.get("pullback_entry_min_support_ratio") or 0.6), 1.0), 0.0),
        "immediate_entry_min_mid_ratio": max(min(float(cfg.get("immediate_entry_min_mid_ratio") or 0.67), 1.0), 0.0),
        "pullback_entry_min_mid_ratio": max(min(float(cfg.get("pullback_entry_min_mid_ratio") or 0.5), 1.0), 0.0),
        "high_execution_alignment_ratio": max(min(float(cfg.get("high_execution_alignment_ratio") or 1.0), 1.0), 0.0),
        "medium_execution_alignment_ratio": max(min(float(cfg.get("medium_execution_alignment_ratio") or 0.5), 1.0), 0.0),
        "entry_zone_atr_mult": max(float(cfg.get("entry_zone_atr_mult") or 0.25), 0.01),
        "max_chase_atr_mult": max(float(cfg.get("max_chase_atr_mult") or 0.35), 0.0),
        "session_lookback_bars": max(int(cfg.get("session_lookback_bars") or 8), 2),
        "swing_lookback_bars": max(int(cfg.get("swing_lookback_bars") or 6), 2),
        "structure_buffer_atr_mult": max(float(cfg.get("structure_buffer_atr_mult") or 0.2), 0.0),
        "minimum_rr_by_horizon": normalize_float_map(cfg.get("minimum_rr_by_horizon"), minimum=0.0),
        "time_stop_bars_by_horizon": {
            horizon: max(int(round(value)), 1)
            for horizon, value in normalize_float_map(cfg.get("time_stop_bars_by_horizon"), minimum=1.0).items()
        },
        "partial_take_profit": {
            "enabled": bool(partial_cfg.get("enabled", False)),
            "tp1_r_multiple": max(float(partial_cfg.get("tp1_r_multiple") or 1.0), 0.1),
            "tp1_size_fraction": max(min(float(partial_cfg.get("tp1_size_fraction") or 0.5), 1.0), 0.0),
            "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
        },
        "trailing_stop": {
            "enabled": bool(trailing_cfg.get("enabled", False)),
            "activation_r_multiple": max(float(trailing_cfg.get("activation_r_multiple") or 1.0), 0.1),
            "trail_buffer_atr_mult": max(float(trailing_cfg.get("trail_buffer_atr_mult") or 0.75), 0.0),
        },
        "analytics": {
            "enabled": bool(analytics_cfg.get("enabled", False)),
            "lookback_bars": max(int(analytics_cfg.get("lookback_bars") or default_lookback_bars), 10),
            "mae_quantile": max(min(float(analytics_cfg.get("mae_quantile") or 0.75), 0.99), 0.5),
            "mfe_quantile": max(min(float(analytics_cfg.get("mfe_quantile") or 0.6), 0.99), 0.5),
            "min_samples": max(int(analytics_cfg.get("min_samples") or default_min_samples), 10),
            "regime_volatility_buckets": {
                "enabled": bool(analytics_bucket_cfg.get("enabled", False)),
                "regime_col": str(analytics_bucket_cfg.get("regime_col") or "regime_state"),
                "volatility_col": str(analytics_bucket_cfg.get("volatility_col") or "volatility_realized_24h"),
                "min_bucket_samples": max(int(analytics_bucket_cfg.get("min_bucket_samples") or 12), 1),
                "low_vol_quantile": max(min(float(analytics_bucket_cfg.get("low_vol_quantile") or 0.5), 0.95), 0.05),
                "max_projection_mfe_ratio": max(float(analytics_bucket_cfg.get("max_projection_mfe_ratio") or 1.25), 0.5),
                "breakout_score_threshold": float(analytics_bucket_cfg.get("breakout_score_threshold") or 0.8),
                "chop_score_threshold": float(analytics_bucket_cfg.get("chop_score_threshold") or 0.3),
            },
        },
        "no_trade_guards": {
            "enabled": bool(guards_cfg.get("enabled", False)),
            "min_stop_distance_atr_mult": max(float(guards_cfg.get("min_stop_distance_atr_mult") or 0.35), 0.0),
            "max_stop_distance_atr_mult": max(float(guards_cfg.get("max_stop_distance_atr_mult") or 3.0), 0.0),
            "max_entry_deviation_atr_mult": max(float(guards_cfg.get("max_entry_deviation_atr_mult") or 1.25), 0.0),
            "require_favorable_entry_zone": bool(guards_cfg.get("require_favorable_entry_zone", True)),
        },
        "adaptive_take_profit": {
            "enabled": bool(adaptive_tp_cfg.get("enabled", True)),
            "min_rr_fraction_of_floor": max(min(float(adaptive_tp_cfg.get("min_rr_fraction_of_floor") or 0.85), 1.0), 0.0),
        },
        "target_range_stop_refinement": {
            "enabled": bool(target_range_stop_cfg.get("enabled", False)),
            "horizons": sorted(
                {
                    normalize_horizon_value(value)
                    for value in (target_range_stop_cfg.get("horizons") or default_target_range_stop_horizons)
                }
            ),
            "confidence_min": max(
                min(float(target_range_stop_cfg.get("confidence_min") or default_target_range_stop_confidence_min), 1.0),
                0.0,
            ),
            "buffer_std_mult": max(float(target_range_stop_cfg.get("buffer_std_mult") or default_target_range_stop_buffer_std_mult), 0.0),
            "min_tighten_fraction": max(
                min(float(target_range_stop_cfg.get("min_tighten_fraction") or default_target_range_stop_min_tighten_fraction), 1.0),
                0.0,
            ),
        },
        "pullback_quality": {
            "enabled": bool(pullback_quality_cfg.get("enabled", False)),
            "min_score_by_horizon": normalize_float_map(pullback_quality_cfg.get("min_score_by_horizon"), minimum=0.0),
            "max_vwap_deviation_atr": max(float(pullback_quality_cfg.get("max_vwap_deviation_atr") or 1.5), 0.1),
            "max_candle_expansion_ratio": max(float(pullback_quality_cfg.get("max_candle_expansion_ratio") or 2.0), 0.1),
            "candle_expansion_window": max(int(pullback_quality_cfg.get("candle_expansion_window") or 8), 2),
            "range_expansion_penalty_threshold": max(float(pullback_quality_cfg.get("range_expansion_penalty_threshold") or 1.25), 0.0),
        },
        "disagreement_severity": {
            "enabled": bool(disagreement_cfg.get("enabled", True)),
            "block_threshold": max(min(float(disagreement_cfg.get("block_threshold") or 0.7), 1.0), 0.0),
            "pullback_threshold": max(min(float(disagreement_cfg.get("pullback_threshold") or 0.45), 1.0), 0.0),
            "vwap_extension_penalty_atr": max(float(disagreement_cfg.get("vwap_extension_penalty_atr") or 0.75), 0.0),
            "range_expansion_penalty_threshold": max(float(disagreement_cfg.get("range_expansion_penalty_threshold") or 1.0), 0.0),
        },
        "coherence_weighting": {
            "enabled": bool(coherence_weighting_cfg.get("enabled", False)),
            "low_trust_penalty": max(min(float(coherence_weighting_cfg.get("low_trust_penalty") or 0.35), 1.0), 0.0),
            "blocked_penalty": max(min(float(coherence_weighting_cfg.get("blocked_penalty") or 1.0), 1.0), 0.0),
            "p_up_conflict_penalty": max(min(float(coherence_weighting_cfg.get("p_up_conflict_penalty") or 0.2), 1.0), 0.0),
            "consensus_bonus": max(float(coherence_weighting_cfg.get("consensus_bonus") or 0.1), 0.0),
            "neutral_band": max(float(coherence_weighting_cfg.get("neutral_band") or 0.02), 0.0),
            "min_multiplier": max(min(float(coherence_weighting_cfg.get("min_multiplier") or 0.1), 1.0), 0.0),
            "by_horizon": normalize_float_map(coherence_weighting_cfg.get("by_horizon"), minimum=0.0),
        },
        "dynamic_rr_floor": {
            "enabled": bool(dynamic_rr_floor_cfg.get("enabled", False)),
            "mfe_mae_scale": max(float(dynamic_rr_floor_cfg.get("mfe_mae_scale") or 0.9), 0.0),
            "max_adjustment": max(min(float(dynamic_rr_floor_cfg.get("max_adjustment") or 0.35), 1.0), 0.0),
            "min_samples": max(int(dynamic_rr_floor_cfg.get("min_samples") or 40), 1),
            "default_floor": max(float(dynamic_rr_floor_cfg.get("default_floor") or 0.0), 0.0),
            "min_floor_by_horizon": normalize_float_map(dynamic_rr_floor_cfg.get("min_floor_by_horizon"), minimum=0.0),
            "max_floor_by_horizon": normalize_float_map(dynamic_rr_floor_cfg.get("max_floor_by_horizon"), minimum=0.0),
            "regime_multiplier": {
                str(key).strip().lower(): max(float(value), 0.0)
                for key, value in (dynamic_rr_floor_cfg.get("regime_multiplier") or {}).items()
                if str(key).strip()
            }
            if isinstance(dynamic_rr_floor_cfg.get("regime_multiplier"), Mapping)
            else {},
        },
        "volatility_expansion_stop": {
            "enabled": bool(volatility_expansion_stop_cfg.get("enabled", False)),
            "expansion_threshold": max(float(volatility_expansion_stop_cfg.get("expansion_threshold") or 1.15), 0.0),
            "stop_multiplier": max(float(volatility_expansion_stop_cfg.get("stop_multiplier") or 1.1), 0.1),
            "max_multiplier": max(float(volatility_expansion_stop_cfg.get("max_multiplier") or 1.5), 0.1),
            "regimes": [
                str(value).strip().lower()
                for value in (volatility_expansion_stop_cfg.get("regimes") or [])
                if str(value).strip()
            ],
        },
        "regime_templates": regime_templates,
    }


def apply_execution_policy(
    summary: SummaryPayload,
    contexts: ExecutionContexts,
    policy: Mapping[str, Any],
    *,
    regime_neutral: str,
    execution_policy_default_lookback_bars: int,
    execution_policy_default_min_samples: int,
    summarize_bias_context: Callable[[SummaryPayload, Mapping[str, Any]], Mapping[str, Any]],
    execution_side: Callable[[Mapping[str, Any]], str],
    direction_vote: Callable[[Mapping[str, Any]], str],
    execution_alignment_ratio: Callable[[Any, str, Mapping[str, Any]], float],
    classify_execution_tier: Callable[[Mapping[str, Any], str, float, Mapping[str, Any]], str],
    compute_atr_like_price_distance: Callable[..., float],
    compute_recent_structure: Callable[..., Mapping[str, Any]],
    build_entry_zone: Callable[..., Mapping[str, Any]],
    compute_pullback_quality_score: Callable[..., Mapping[str, Any]],
    compute_disagreement_severity: Callable[..., Mapping[str, Any]],
    compute_excursion_priors: Callable[..., Mapping[str, Any]],
    finite_float_or_none: Callable[[Any], float | None],
    finite_float: Callable[[Any, float], float],
    resolve_stop_with_guardrails: Callable[..., Mapping[str, Any]],
    refine_stop_with_target_range: Callable[..., Mapping[str, Any]],
    resolve_execution_target_reward: Callable[..., Mapping[str, Any]],
    lookup_horizon_value: Callable[[Mapping[str, Any], float, Any], Any],
    resolve_execution_upstream_hold_reason: Callable[[Mapping[str, Any]], str],
) -> SummaryPayload:
    if not summary:
        return summary

    bias_context = summarize_bias_context(summary, policy)
    bias_direction = str(bias_context.get("bias_direction", "neutral"))
    bias_alignment_ratio = float(bias_context.get("bias_alignment_ratio", 0.0))
    execution_entries = bias_context.get("execution_entries", [])
    weights = policy.get("horizon_bias_weights") if isinstance(policy.get("horizon_bias_weights"), Mapping) else {}

    for label, entry in summary.items():
        market_price = float(entry.get("close", entry.get("entry_price", 0.0)) or 0.0)
        entry["market_price"] = market_price
        entry["execution_prior_provenance"] = {
            "analytics_source": "unavailable",
            "matched_regime": None,
            "volatility_bucket": None,
            "bucket_threshold": None,
            "sample_count": 0,
            "stop_source": None,
            "stop_adjustment_type": None,
            "target_source": "existing_or_projection",
        }
        side = execution_side(entry)
        direction = direction_vote(entry)
        upstream_hold = str(entry.get("trade_action", "hold")) == "hold"
        alignment_ratio = execution_alignment_ratio(execution_entries, direction=direction, weights=weights)
        tier = classify_execution_tier(
            entry,
            bias_direction=bias_direction,
            execution_alignment_ratio=alignment_ratio,
            policy=policy,
        )
        bias_scores = bias_context.get("bias_scores") if isinstance(bias_context.get("bias_scores"), Mapping) else {}
        execution_scores = bias_context.get("execution_scores") if isinstance(bias_context.get("execution_scores"), Mapping) else {}
        bias_score_value = float((bias_scores.get("up_score") if direction == "up" else bias_scores.get("down_score")) or 0.0)
        execution_score_value = float(
            (execution_scores.get("up_score") if direction == "up" else execution_scores.get("down_score")) or 0.0
        )
        support_horizons = list((bias_context.get("direction_support_horizons") or {}).get(direction, []))
        entry["bias_score"] = bias_score_value
        entry["execution_score"] = execution_score_value
        entry["bias_support_horizons"] = support_horizons
        entry["bias_support_is_8h_standalone"] = support_horizons == ["8h"]
        plan: Dict[str, Any] = {
            "enabled": bool(policy.get("enabled", False)),
            "bias_direction": bias_direction,
            "bias_alignment_ratio": bias_alignment_ratio,
            "execution_alignment_ratio": float(alignment_ratio),
            "bias_score": float(bias_score_value),
            "execution_score": float(execution_score_value),
            "confluence_tier": tier,
            "status": "ready",
            "reason": "pass",
            "side": side,
            "entry_mode": "disabled",
            "pending_trade_action": side,
            "partial_take_profit": None,
            "time_stop": None,
            "trailing_stop": None,
            "analytics": {"available": False},
            "structure": None,
            "stop_management": None,
        }
        if not bool(policy.get("enabled", False)):
            entry["execution_plan"] = plan
            continue

        forecast_coherence = entry.get("forecast_coherence")
        if isinstance(forecast_coherence, Mapping) and forecast_coherence.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "forecast_coherence_gate"
            entry["execution_plan"] = plan
            continue

        if bool(policy.get("require_bias_alignment", True)) and bias_direction != "neutral" and direction != bias_direction:
            plan["status"] = "rejected"
            plan["reason"] = "bias_direction_conflict"
            entry["execution_plan"] = plan
            continue

        context = contexts.get(label)
        if not context:
            plan["status"] = "rejected"
            plan["reason"] = "missing_execution_context"
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["execution_plan"] = plan
            continue

        prepared = context["prepared"]
        index = int(context["index"])
        horizon = float(context["horizon"])
        residual_std = float(context["residual_std"])
        regime_state = str(entry.get("regime_state", regime_neutral))
        regime_template = (policy.get("regime_templates") or {}).get(regime_state, {})
        horizon_steps = max(int(round(horizon)), 1)
        atr_distance = compute_atr_like_price_distance(
            prepared.df_all,
            index=index,
            fallback_close=market_price,
            fallback_return_std=residual_std,
        )
        structure = compute_recent_structure(
            prepared.df_all,
            index=index,
            session_lookback_bars=int(policy.get("session_lookback_bars", 8)),
            swing_lookback_bars=int(policy.get("swing_lookback_bars", 6)),
            atr_distance=atr_distance,
            fallback_price=market_price,
        )
        plan["structure"] = structure
        entry_zone = build_entry_zone(
            market_price=market_price,
            side=side,
            structure=structure,
            policy=policy,
            regime_template=regime_template,
        )
        preferred_entry = float(entry_zone["preferred_entry_price"])
        plan.update(entry_zone)

        pullback_quality = compute_pullback_quality_score(
            entry=entry,
            frame=prepared.df_all,
            index=index,
            market_price=market_price,
            side=side,
            structure=structure,
            atr_distance=atr_distance,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
        )
        disagreement_severity = compute_disagreement_severity(
            entry,
            bias_context=bias_context,
            policy=policy,
            atr_distance=atr_distance,
            structure=structure,
        )
        plan["pullback_quality"] = pullback_quality
        plan["disagreement_severity"] = disagreement_severity
        entry["disagreement_severity"] = disagreement_severity

        template_max_chase = float(regime_template.get("max_chase_atr_mult", 0.0) or 0.0)
        max_chase = (template_max_chase if template_max_chase > 0.0 else float(policy.get("max_chase_atr_mult", 0.35))) * atr_distance
        market_deviation = abs(market_price - preferred_entry)
        if tier == "high" and (bool(entry_zone["entry_ready"]) or market_deviation <= max_chase):
            entry_mode = "immediate"
            planned_entry = market_price
        elif tier in {"high", "medium"}:
            entry_mode = "pullback"
            planned_entry = preferred_entry
        else:
            entry_mode = "blocked"
            planned_entry = preferred_entry

        template_entry_modes = regime_template.get("entry_mode_by_tier") if isinstance(regime_template.get("entry_mode_by_tier"), Mapping) else {}
        template_entry_mode = str(template_entry_modes.get(tier) or "").strip().lower()
        if template_entry_mode in {"immediate", "pullback", "blocked"}:
            if template_entry_mode == "blocked":
                entry_mode = "blocked"
            elif template_entry_mode == "pullback" and entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif template_entry_mode == "immediate" and entry_mode == "pullback" and bool(entry_zone["entry_ready"]):
                entry_mode = "immediate"
                planned_entry = market_price

        if disagreement_severity.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "short_term_disagreement"
        elif disagreement_severity.get("pullback_only") and entry_mode == "immediate":
            entry_mode = "pullback"
            planned_entry = preferred_entry

        if pullback_quality.get("triggered"):
            if entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif entry_mode == "pullback":
                plan["status"] = "rejected"
                plan["reason"] = "pullback_quality_insufficient"
        plan["entry_mode"] = entry_mode

        analytics_cfg = policy.get("analytics", {}) if isinstance(policy.get("analytics"), Mapping) else {}
        analytics_payload: Mapping[str, Any] = {"available": False}
        if analytics_cfg.get("enabled"):
            analytics_payload = compute_excursion_priors(
                prepared.df_all,
                index=index,
                horizon_steps=horizon_steps,
                side=side,
                lookback_bars=int(analytics_cfg.get("lookback_bars", execution_policy_default_lookback_bars)),
                min_samples=int(analytics_cfg.get("min_samples", execution_policy_default_min_samples)),
                mae_quantile=float(analytics_cfg.get("mae_quantile", 0.75)),
                mfe_quantile=float(analytics_cfg.get("mfe_quantile", 0.6)),
                current_regime=regime_state,
                current_volatility=finite_float_or_none((entry.get("volatility") or {}).get("current")),
                bucket_policy=analytics_cfg.get("regime_volatility_buckets"),
            )
        plan["analytics"] = analytics_payload

        existing_stop = float(entry.get("stop_loss", planned_entry))
        existing_take = float(entry.get("take_profit", planned_entry))
        structure_buffer = atr_distance * float(policy.get("structure_buffer_atr_mult", 0.2))
        if side == "long":
            structure_stop = min(float(structure["session_low"]), float(structure["swing_low"])) - structure_buffer
            analytic_stop = planned_entry * (1.0 - float(analytics_payload.get("mae_distance") or 0.0))
        else:
            structure_stop = max(float(structure["session_high"]), float(structure["swing_high"])) + structure_buffer
            analytic_stop = planned_entry * (1.0 + float(analytics_payload.get("mae_distance") or 0.0))
        analytic_stop_value = analytic_stop if analytics_payload.get("available") else None

        guards_cfg = policy.get("no_trade_guards", {}) if isinstance(policy.get("no_trade_guards"), Mapping) else {}
        stop_resolution = resolve_stop_with_guardrails(
            side=side,
            planned_entry=planned_entry,
            existing_stop=existing_stop,
            structure_stop=structure_stop,
            analytic_stop=analytic_stop_value,
            atr_distance=atr_distance,
            guards_cfg=guards_cfg,
            analytic_stop_preferred=bool(analytics_payload.get("available")) and str(analytics_payload.get("source")) != "global",
        )
        selected_stop = float(stop_resolution["stop_loss"])
        risk_unit = float(stop_resolution["risk_unit"])
        stop_refinement = refine_stop_with_target_range(
            side=side,
            planned_entry=planned_entry,
            selected_stop=selected_stop,
            risk_unit=risk_unit,
            atr_distance=atr_distance,
            horizon=horizon,
            projected_high=finite_float_or_none(entry.get("projected_high")),
            projected_low=finite_float_or_none(entry.get("projected_low")),
            projected_high_confidence=finite_float_or_none(entry.get("projected_high_confidence")),
            projected_low_confidence=finite_float_or_none(entry.get("projected_low_confidence")),
            projected_high_residual_std=finite_float_or_none(entry.get("projected_high_residual_std")),
            projected_low_residual_std=finite_float_or_none(entry.get("projected_low_residual_std")),
            policy=policy,
            guards_cfg=guards_cfg,
        )
        if stop_refinement.get("applied"):
            selected_stop = float(stop_refinement["stop_loss"])
            risk_unit = float(stop_refinement["risk_unit"])
        stop_scaling_payload = {
            "applied": False,
            "reason": "not_triggered",
            "multiplier": 1.0,
            "risk_unit_before": float(risk_unit),
            "risk_unit_after": float(risk_unit),
        }
        regime_stop_multiplier = max(float(regime_template.get("stop_multiplier", 1.0) or 1.0), 0.1)
        if regime_stop_multiplier > 1.0:
            scaled_risk = risk_unit * regime_stop_multiplier
            selected_stop = planned_entry - scaled_risk if side == "long" else planned_entry + scaled_risk
            risk_unit = float(max(scaled_risk, 1e-8))
            stop_scaling_payload = {
                "applied": True,
                "reason": "regime_stop_multiplier",
                "multiplier": float(regime_stop_multiplier),
                "risk_unit_before": float(stop_scaling_payload["risk_unit_before"]),
                "risk_unit_after": float(risk_unit),
            }

        vol_stop_cfg = policy.get("volatility_expansion_stop") if isinstance(policy.get("volatility_expansion_stop"), Mapping) else {}
        if bool(vol_stop_cfg.get("enabled", False)):
            expansion_value = abs(finite_float(entry.get("range_expansion_1h"), 0.0))
            expansion_threshold = float(vol_stop_cfg.get("expansion_threshold", 1.15) or 1.15)
            scoped_regimes = {str(v).strip().lower() for v in (vol_stop_cfg.get("regimes") or []) if str(v).strip()}
            regime_allowed = (not scoped_regimes) or (regime_state in scoped_regimes)
            if regime_allowed and expansion_value >= expansion_threshold:
                stop_multiplier = max(float(vol_stop_cfg.get("stop_multiplier", 1.1) or 1.1), 0.1)
                max_multiplier = max(float(vol_stop_cfg.get("max_multiplier", 1.5) or 1.5), 0.1)
                stop_multiplier = min(stop_multiplier, max_multiplier)
                scaled_risk = risk_unit * stop_multiplier
                selected_stop = planned_entry - scaled_risk if side == "long" else planned_entry + scaled_risk
                risk_unit = float(max(scaled_risk, 1e-8))
                stop_scaling_payload = {
                    "applied": True,
                    "reason": "volatility_expansion_stop",
                    "multiplier": float(stop_multiplier),
                    "expansion_value": float(expansion_value),
                    "expansion_threshold": float(expansion_threshold),
                    "risk_unit_before": float(stop_scaling_payload.get("risk_unit_after", stop_scaling_payload["risk_unit_before"])),
                    "risk_unit_after": float(risk_unit),
                }
        plan["stop_management"] = {
            "source": stop_resolution.get("source"),
            "adjustment": stop_resolution.get("adjustment"),
            "target_range_refinement": stop_refinement.get("details"),
            "stop_scaling": stop_scaling_payload,
        }

        if guards_cfg.get("enabled"):
            max_entry_dev = float(guards_cfg.get("max_entry_deviation_atr_mult", 1.25)) * atr_distance
            if bool(guards_cfg.get("require_favorable_entry_zone", True)) and market_deviation > max_entry_dev and entry_mode == "immediate":
                plan["status"] = "rejected"
                plan["reason"] = "entry_too_extended"

        target_resolution = resolve_execution_target_reward(
            side=side,
            planned_entry=planned_entry,
            existing_take=existing_take,
            projected_high=finite_float_or_none(entry.get("projected_high")),
            projected_low=finite_float_or_none(entry.get("projected_low")),
            analytics_payload=analytics_payload,
            risk_unit=risk_unit,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
            regime_state=regime_state,
        )
        selected_take = float(target_resolution["selected_take"])
        risk_reward_ratio = float(target_resolution["risk_reward_ratio"])
        plan["target_management"] = dict(target_resolution["target_management"])
        if target_resolution["status"] != "pass":
            plan["status"] = "rejected"
            plan["reason"] = str(target_resolution["reason"])

        partial_cfg = policy.get("partial_take_profit", {}) if isinstance(policy.get("partial_take_profit"), Mapping) else {}
        partial_take_profit = None
        if partial_cfg.get("enabled"):
            tp1_distance = risk_unit * float(partial_cfg.get("tp1_r_multiple", 1.0))
            tp1_price = planned_entry + tp1_distance if side == "long" else planned_entry - tp1_distance
            partial_take_profit = {
                "enabled": True,
                "tp1_price": tp1_price,
                "tp1_size_fraction": float(partial_cfg.get("tp1_size_fraction", 0.5)),
                "tp2_price": selected_take,
                "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
            }

        trailing_cfg = policy.get("trailing_stop", {}) if isinstance(policy.get("trailing_stop"), Mapping) else {}
        trailing_stop = None
        if trailing_cfg.get("enabled"):
            activation_distance = risk_unit * float(trailing_cfg.get("activation_r_multiple", 1.0))
            trailing_stop = {
                "enabled": True,
                "activation_price": planned_entry + activation_distance if side == "long" else planned_entry - activation_distance,
                "trail_buffer": atr_distance * float(trailing_cfg.get("trail_buffer_atr_mult", 0.75)),
            }

        time_stop_map = policy.get("time_stop_bars_by_horizon", {}) if isinstance(policy.get("time_stop_bars_by_horizon"), Mapping) else {}
        base_time_stop = max(int(round(lookup_horizon_value(time_stop_map, horizon, max(horizon_steps, 1)))), 1)
        time_stop_mult = float(regime_template.get("time_stop_multiplier", 1.0) or 1.0)
        recommended_time_stop = max(int(round(base_time_stop * time_stop_mult)), 1)
        if analytics_payload.get("available") and analytics_payload.get("peak_step_p50"):
            recommended_time_stop = min(recommended_time_stop, max(int(analytics_payload["peak_step_p50"] * 1.25), 1))
        time_stop_payload = {
            "enabled": True,
            "bars": recommended_time_stop,
            "reason": "stagnation_exit",
        }

        if plan["status"] == "ready" and entry_mode == "pullback":
            if bool(entry_zone["entry_ready"]):
                plan["status"] = "ready"
                plan["reason"] = "pass"
            else:
                plan["status"] = "waiting_pullback"
                plan["reason"] = "await_pullback_entry_zone"
        elif plan["status"] == "ready" and entry_mode == "blocked":
            plan["status"] = "rejected"
            plan["reason"] = "low_execution_confluence"

        position_size = float(entry.get("position_size", 0.0))
        position_size *= float(regime_template.get("size_multiplier", 1.0) or 1.0)
        if tier == "medium":
            position_size *= 0.85
        elif tier == "low":
            position_size = 0.0

        plan["partial_take_profit"] = partial_take_profit
        plan["time_stop"] = time_stop_payload
        plan["trailing_stop"] = trailing_stop

        entry["entry_price"] = float(planned_entry)
        entry["stop_loss"] = float(selected_stop)
        entry["take_profit"] = float(selected_take)
        entry["risk_reward_ratio"] = float(risk_reward_ratio)
        entry["position_size"] = float(max(position_size, 0.0))
        analytics_payload_final = plan.get("analytics") if isinstance(plan.get("analytics"), Mapping) else {}
        stop_management = plan.get("stop_management") if isinstance(plan.get("stop_management"), Mapping) else {}
        entry["execution_prior_provenance"] = {
            "analytics_source": analytics_payload_final.get("source", "unavailable") if analytics_payload_final else "unavailable",
            "matched_regime": analytics_payload_final.get("matched_regime"),
            "volatility_bucket": analytics_payload_final.get("volatility_bucket"),
            "bucket_threshold": analytics_payload_final.get("bucket_threshold"),
            "sample_count": analytics_payload_final.get("sample_count"),
            "stop_source": stop_management.get("source"),
            "stop_adjustment_type": (stop_management.get("adjustment") or {}).get("type") if stop_management else None,
            "target_source": str((plan.get("target_management") or {}).get("source") or "existing_or_projection"),
        }
        entry["execution_plan"] = plan

        if upstream_hold and plan["status"] == "ready":
            plan["status"] = "bias_only_ready"
            plan["reason"] = resolve_execution_upstream_hold_reason(entry)
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        elif plan["status"] != "ready":
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        else:
            entry["trade_action"] = side
    return summary
