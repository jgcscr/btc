from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


SummaryPayload = Dict[str, Dict[str, Any]]


def resolve_forecast_coherence_policy(
    config: Mapping[str, Any] | None,
    *,
    normalize_horizon_value: Callable[[Any], float],
) -> Dict[str, Any]:
    cfg = config or {}
    horizons = cfg.get("horizons") or [1.0, 4.0, 8.0, 12.0]
    consensus_relief_horizons = cfg.get("consensus_relief_horizons") or [1.0, 4.0]
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "horizons": sorted({normalize_horizon_value(value) for value in horizons}),
        "block_on_direction_ret_mismatch": bool(cfg.get("block_on_direction_ret_mismatch", True)),
        "block_on_direction_projected_price_mismatch": bool(cfg.get("block_on_direction_projected_price_mismatch", True)),
        "block_on_p_up_ret_mismatch": bool(cfg.get("block_on_p_up_ret_mismatch", True)),
        "p_up_neutral_band": max(float(cfg.get("p_up_neutral_band") or 0.02), 0.0),
        "min_p_up_edge": max(float(cfg.get("min_p_up_edge") or 0.05), 0.0),
        "min_abs_ret_pred": max(float(cfg.get("min_abs_ret_pred") or 0.0), 0.0),
        "allow_consensus_p_up_ret_relief": bool(cfg.get("allow_consensus_p_up_ret_relief", False)),
        "consensus_relief_horizons": sorted({normalize_horizon_value(value) for value in consensus_relief_horizons}),
        "consensus_relief_max_p_up_edge": max(float(cfg.get("consensus_relief_max_p_up_edge") or 0.12), 0.0),
        "consensus_relief_exclude_from_voting": bool(cfg.get("consensus_relief_exclude_from_voting", False)),
        "exclude_blocked_horizons_from_voting": bool(cfg.get("exclude_blocked_horizons_from_voting", True)),
    }


def forecast_coherence_excluded(entry: Mapping[str, Any]) -> bool:
    payload = entry.get("forecast_coherence")
    return bool(
        (isinstance(payload, Mapping) and payload.get("exclude_from_voting"))
        or entry.get("excluded_from_voting")
    )


def coherence_weight_multiplier(
    entry: Mapping[str, Any],
    *,
    horizon: float,
    policy: Mapping[str, Any],
    lookup_horizon_value: Callable[[Mapping[str, Any], float, Any], Any],
    direction_from_ret_pred: Callable[[Any], str],
    direction_from_projected_price: Callable[[Any, Any], str],
    direction_from_probability: Callable[[Any], str] | Callable[..., str],
) -> float:
    weighting_cfg = policy.get("coherence_weighting") if isinstance(policy.get("coherence_weighting"), Mapping) else {}
    base_multiplier = lookup_horizon_value(
        weighting_cfg.get("by_horizon", {}) if isinstance(weighting_cfg.get("by_horizon"), Mapping) else {},
        horizon,
        1.0,
    )
    base_multiplier = max(float(base_multiplier), 0.0)
    if not bool(weighting_cfg.get("enabled", False)):
        return base_multiplier

    multiplier = base_multiplier
    min_multiplier = max(min(float(weighting_cfg.get("min_multiplier", 0.1) or 0.1), 1.5), 0.0)
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    low_trust_penalty = max(min(float(weighting_cfg.get("low_trust_penalty", 0.35) or 0.35), 1.0), 0.0)
    blocked_penalty = max(min(float(weighting_cfg.get("blocked_penalty", 1.0) or 1.0), 1.0), 0.0)
    p_up_conflict_penalty = max(min(float(weighting_cfg.get("p_up_conflict_penalty", 0.2) or 0.2), 1.0), 0.0)
    consensus_bonus = max(float(weighting_cfg.get("consensus_bonus", 0.1) or 0.1), 0.0)

    if bool(coherence.get("triggered")):
        multiplier *= max(0.0, 1.0 - blocked_penalty)
    elif bool(coherence.get("low_trust")):
        multiplier *= max(0.0, 1.0 - low_trust_penalty)

    ret_side = str(coherence.get("ret_pred_side") or direction_from_ret_pred(entry.get("ret_pred")))
    projected_side = str(
        coherence.get("projected_price_side")
        or direction_from_projected_price(entry.get("close"), entry.get("projected_price"))
    )
    neutral_band = float(weighting_cfg.get("neutral_band", 0.02) or 0.02)
    p_up_side = str(
        coherence.get("p_up_side")
        or direction_from_probability(entry.get("p_up"), neutral_band=neutral_band)
    )
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else None
    if consensus_side is not None and p_up_side in {"up", "down"}:
        if p_up_side != consensus_side:
            multiplier *= max(0.0, 1.0 - p_up_conflict_penalty)
        else:
            multiplier *= 1.0 + consensus_bonus

    return max(float(multiplier), min_multiplier)


def apply_forecast_coherence_policy(
    summary: SummaryPayload,
    policy: Mapping[str, Any],
    *,
    coerce_result_horizon: Callable[[Any], float | None],
    direction_vote: Callable[[Mapping[str, Any]], str],
    direction_from_ret_pred: Callable[[Any], str],
    direction_from_projected_price: Callable[[Any, Any], str],
    direction_from_probability: Callable[[Any], str] | Callable[..., str],
    finite_float_or_none: Callable[[Any], float | None],
    append_gate_trace: Callable[..., None],
) -> SummaryPayload:
    if not summary:
        return summary

    enabled = bool(policy.get("enabled", False))
    scoped_horizons = set(policy.get("horizons", []))
    neutral_band = float(policy.get("p_up_neutral_band", 0.02) or 0.0)
    min_p_up_edge = float(policy.get("min_p_up_edge", 0.05) or 0.0)
    min_abs_ret_pred = float(policy.get("min_abs_ret_pred", 0.0) or 0.0)
    exclude_from_voting = bool(policy.get("exclude_blocked_horizons_from_voting", True))
    allow_consensus_relief = bool(policy.get("allow_consensus_p_up_ret_relief", False))
    consensus_relief_horizons = set(policy.get("consensus_relief_horizons", []))
    consensus_relief_max_p_up_edge = float(policy.get("consensus_relief_max_p_up_edge", 0.12) or 0.0)
    consensus_relief_exclude_from_voting = bool(policy.get("consensus_relief_exclude_from_voting", False))

    for entry in summary.values():
        horizon = coerce_result_horizon(entry.get("horizon_hours"))
        direction = direction_vote(entry)
        ret_side = direction_from_ret_pred(entry.get("ret_pred"))
        projected_side = direction_from_projected_price(entry.get("close"), entry.get("projected_price"))
        p_up_side = direction_from_probability(entry.get("p_up"), neutral_band=neutral_band)
        p_up_value = finite_float_or_none(entry.get("p_up"))
        ret_pred_value = abs(float(entry.get("ret_pred", 0.0)))
        p_up_edge = abs(p_up_value - 0.5) if p_up_value is not None else None
        consensus_relief_applied = False

        payload = {
            "enabled": enabled,
            "evaluated": bool(enabled and horizon in scoped_horizons),
            "exclude_from_voting": False,
            "direction_side": direction,
            "ret_pred_side": ret_side,
            "projected_price_side": projected_side,
            "p_up_side": p_up_side,
            "triggered": False,
            "reasons": [],
            "advisory_reasons": [],
            "low_trust": False,
            "consensus_relief_applied": False,
        }

        if not enabled or horizon not in scoped_horizons:
            entry["forecast_coherence"] = payload
            continue

        reasons: list[str] = []
        if bool(policy.get("block_on_direction_ret_mismatch", True)) and ret_side != "neutral" and direction != ret_side:
            reasons.append("direction_ret_mismatch")
        if (
            bool(policy.get("block_on_direction_projected_price_mismatch", True))
            and projected_side != "neutral"
            and direction != projected_side
        ):
            reasons.append("direction_projected_price_mismatch")
        if (
            bool(policy.get("block_on_p_up_ret_mismatch", True))
            and p_up_side != "neutral"
            and ret_side != "neutral"
            and p_up_side != ret_side
            and p_up_edge is not None
            and p_up_edge >= min_p_up_edge
            and ret_pred_value >= min_abs_ret_pred
        ):
            consensus_relief_applied = bool(
                allow_consensus_relief
                and horizon in consensus_relief_horizons
                and direction in {"up", "down"}
                and direction == ret_side == projected_side
                and p_up_edge <= consensus_relief_max_p_up_edge
            )
            if not consensus_relief_applied:
                reasons.append("p_up_ret_mismatch")

        advisory_reasons: list[str] = []
        consensus_side = direction if direction == ret_side == projected_side and direction in {"up", "down"} else None
        if consensus_relief_applied:
            advisory_reasons.append("consensus_p_up_ret_mismatch_relief")
        if (
            bool(policy.get("block_on_p_up_ret_mismatch", True))
            and consensus_side is not None
            and p_up_side != "neutral"
            and p_up_side != consensus_side
            and p_up_edge is not None
            and p_up_edge < min_p_up_edge
            and ret_pred_value >= min_abs_ret_pred
            and not reasons
            and not consensus_relief_applied
            and (str(entry.get("trade_action", "hold")) == "hold" or not bool(entry.get("signal_ensemble", 0)))
        ):
            advisory_reasons.append("low_edge_p_up_ret_mismatch")

        payload["reasons"] = reasons
        payload["advisory_reasons"] = advisory_reasons
        payload["triggered"] = bool(reasons)
        payload["low_trust"] = bool(advisory_reasons)
        payload["consensus_relief_applied"] = bool(consensus_relief_applied)
        payload["exclude_from_voting"] = bool(reasons) and exclude_from_voting
        if advisory_reasons and exclude_from_voting:
            payload["exclude_from_voting"] = bool(
                not consensus_relief_applied or consensus_relief_exclude_from_voting
            )
        entry["forecast_coherence"] = payload
        if reasons:
            append_gate_trace(
                entry,
                stage="forecast_coherence",
                reason="|".join(reasons),
                triggered=True,
                blocking=True,
            )
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["direction_next_display"] = "neutral"
            direction_output = entry.get("direction_output")
            if isinstance(direction_output, dict):
                direction_output["coherence_override"] = {
                    "applied": True,
                    "reason": "forecast_coherence_gate",
                    "raw_direction": direction_output.get("direction"),
                }
                direction_output["direction"] = "neutral"
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["pre_forecast_coherence_triggered"] = bool(trade_decision.get("triggered", False))
                trade_decision["triggered"] = False
                trade_decision["blocked"] = True
                trade_decision["blocking_reason"] = "forecast_coherence_gate"
                trade_decision["forecast_coherence_gate_triggered"] = True
                trade_decision["forecast_coherence_gate_reasons"] = reasons
        elif advisory_reasons:
            append_gate_trace(
                entry,
                stage="forecast_coherence",
                reason="|".join(advisory_reasons),
                triggered=True,
                blocking=False,
            )
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["forecast_coherence_low_trust"] = True
                trade_decision["forecast_coherence_low_trust_reasons"] = advisory_reasons
    return summary
