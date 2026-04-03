from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


def build_execution_prior_summary(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    analytics_source_counts: Dict[str, int] = {}
    stop_source_counts: Dict[str, int] = {}
    target_source_counts: Dict[str, int] = {}
    for entry in summary.values():
        provenance = entry.get("execution_prior_provenance") if isinstance(entry, Mapping) else None
        if not isinstance(provenance, Mapping):
            continue
        analytics_source = str(provenance.get("analytics_source") or "unavailable")
        stop_source = str(provenance.get("stop_source") or "unknown")
        target_source = str(provenance.get("target_source") or "unknown")
        analytics_source_counts[analytics_source] = analytics_source_counts.get(analytics_source, 0) + 1
        stop_source_counts[stop_source] = stop_source_counts.get(stop_source, 0) + 1
        target_source_counts[target_source] = target_source_counts.get(target_source, 0) + 1
    return {
        "analytics_source_counts": analytics_source_counts,
        "stop_source_counts": stop_source_counts,
        "target_source_counts": target_source_counts,
    }


def finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def coerce_result_horizon(value: Any) -> float | None:
    numeric = finite_float_or_none(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def build_stub_summary(
    targets: Iterable[float],
    p_up_min: float,
    ret_min: float,
    *,
    close: float = 0.0,
    ts_iso: str | None = None,
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None = None,
    normalize_horizon_value: Callable[[Any], float],
    format_horizon_label: Callable[[float], str],
    resolve_thresholds_for_horizon: Callable[
        [float, float, float, Mapping[int | float | str, Dict[str, float]] | None],
        Dict[str, float],
    ],
    confidence_min_default: float,
    regime_neutral: str,
) -> Dict[str, Dict[str, float | str | int | bool | dict[str, Any] | None]]:
    generated_ts = ts_iso or datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Dict[str, float | str | int | bool | dict[str, Any] | None]] = {}
    normalized_targets = sorted({normalize_horizon_value(horizon) for horizon in targets})
    for horizon in normalized_targets:
        label = format_horizon_label(horizon)
        horizon_thresholds = resolve_thresholds_for_horizon(
            horizon,
            p_up_min,
            ret_min,
            thresholds_by_horizon,
        )
        summary[label] = {
            "timestamp": generated_ts,
            "horizon_hours": horizon,
            "close": close,
            "p_up": 0.5,
            "p_trend_ignition": 0.0,
            "ignition_state": 0,
            "ignition_cooldown_active": False,
            "ret_pred": 0.0,
            "projected_price": close,
            "signal_ensemble": 0,
            "signal_dir_only": 0,
            "confidence_score": 0.0,
            "position_size": 0.0,
            "confidence_min": confidence_min_default,
            "confidence_filter_triggered": False,
            "p_up_components": {},
            "stop_loss": close,
            "take_profit": close,
            "expected_value": 0.0,
            "thresholds": horizon_thresholds,
            "regime_state": regime_neutral,
            "regime_score": 0.0,
            "projected_high": close,
            "projected_low": close,
            "projected_high_confidence": 0.0,
            "projected_low_confidence": 0.0,
            "volatility": {
                "snapshot": {},
                "ceiling": horizon_thresholds.get("volatility_ceiling"),
                "triggered": False,
            },
            "volatility_flag": False,
            "target_range_overrides": {
                "stop_loss": None,
                "take_profit": None,
            },
            "execution_plan": {
                "enabled": False,
                "status": "dry_run",
                "reason": "dry_run",
            },
            "direction_only_fallback": {
                "active": False,
                "side": None,
                "size_factor": 0.0,
                "stop_loss_fallback": None,
                "take_profit_fallback": None,
                "reason": "dry_run",
                "cooldown_active": False,
            },
        }
        summary[label]["thresholds"]["p_up_min_effective"] = horizon_thresholds["p_up_min"]
        summary[label]["thresholds"]["ret_min_effective"] = horizon_thresholds["ret_min"]
        summary[label]["thresholds"]["adaptive_scale"] = 1.0
    return summary


def confidence_level_from_score(value: Any, *, finite_float_or_none: Callable[[Any], float | None]) -> str:
    score = finite_float_or_none(value)
    if score is None:
        return "Low"
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def prompt_direction_label(direction: str) -> str:
    normalized = str(direction).strip().lower()
    if normalized == "up":
        return "Long"
    if normalized == "down":
        return "Short"
    return "Neutral"


def format_usd_value(value: Any, *, finite_float_or_none: Callable[[Any], float | None]) -> str | None:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return None
    return f"${numeric:,.2f}"


def prompt_effective_direction(entry: Mapping[str, Any]) -> str:
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    direction_display = str(entry.get("direction_next_display") or "neutral").lower()
    if bool(coherence.get("triggered")) and direction_display == "neutral":
        internal_direction = str(entry.get("direction_next") or "neutral").lower()
        if internal_direction in {"up", "down"}:
            return internal_direction
    return direction_display


def build_prompt_forecast_clause(
    label: str,
    entry: Mapping[str, Any],
    *,
    finite_float_or_none: Callable[[Any], float | None],
) -> str:
    direction_display = prompt_effective_direction(entry)
    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    projected_high = format_usd_value(entry.get("projected_high"), finite_float_or_none=finite_float_or_none)
    projected_low = format_usd_value(entry.get("projected_low"), finite_float_or_none=finite_float_or_none)
    projected_price = format_usd_value(entry.get("projected_price"), finite_float_or_none=finite_float_or_none)

    clause = f"{label}: {direction_display}"
    if projected_high and projected_low:
        clause += f", projected range {projected_low} to {projected_high}"
    elif projected_price:
        clause += f", projected price {projected_price}"

    if coherence.get("triggered"):
        clause += " (coherence blocked)"
    elif plan.get("reason") not in {None, "pass", "upstream_model_hold", "confluence_gate", "await_pullback_entry_zone"}:
        clause += f" ({plan.get('reason')})"
    elif plan.get("status") == "bias_only_ready":
        hold_reason = "confluence gate" if plan.get("reason") == "confluence_gate" else "upstream hold"
        clause += f" (bias ready, {hold_reason})"
    return clause


def prompt_status_rank(status: str) -> int:
    return {
        "ready": 0,
        "waiting_pullback": 1,
        "bias_only_ready": 2,
        "analysis_only": 3,
        "rejected": 4,
        "no_trade": 5,
    }.get(str(status or "rejected"), 6)


def prompt_reason_rank(reason: str | None) -> int:
    return {
        "pass": 0,
        "await_pullback_entry_zone": 1,
        "upstream_model_hold": 2,
        "confluence_gate": 2,
        "low_execution_confluence": 3,
        "insufficient_mfe_headroom": 4,
        "bias_direction_conflict": 5,
        "forecast_coherence_gate": 6,
    }.get(str(reason or "pass"), 7)


def prompt_confluence_rank(tier: str | None) -> int:
    return {
        "high": 0,
        "medium": 1,
        "low": 2,
    }.get(str(tier or "low"), 3)


def prompt_entry_rank(
    label: str,
    entry: Mapping[str, Any],
    *,
    coerce_result_horizon: Callable[[Any], float | None],
    finite_float_or_none: Callable[[Any], float | None],
) -> tuple[int, int, int, float, float, float, float, float, float, float]:
    horizon = coerce_result_horizon(entry.get("horizon_hours"))
    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
    status_rank = prompt_status_rank(str(plan.get("status") or "rejected"))
    reason_rank = prompt_reason_rank(plan.get("reason"))
    confluence_rank = prompt_confluence_rank(plan.get("confluence_tier"))
    execution_alignment = float(plan.get("execution_alignment_ratio") or 0.0)
    bias_alignment = float(plan.get("bias_alignment_ratio") or 0.0)
    execution_score = float(plan.get("execution_score") or 0.0)
    bias_score = float(plan.get("bias_score") or 0.0)
    confidence_score = float(finite_float_or_none(entry.get("confidence_score")) or 0.0)
    horizon_preference = {4.0: 0, 12.0: 1, 8.0: 2, 1.0: 3, 0.25: 4}
    preference_rank = float(horizon_preference.get(horizon, 9.0))
    if bool(entry.get("bias_support_is_8h_standalone")) and horizon == 8.0:
        preference_rank += 2.0
    return (
        status_rank,
        reason_rank,
        confluence_rank,
        -execution_alignment,
        -bias_alignment,
        -execution_score,
        -bias_score,
        -confidence_score,
        preference_rank,
        -(float(horizon) if horizon is not None else 0.0),
    )


def select_prompt_candidate_entries(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    coerce_result_horizon: Callable[[Any], float | None],
    finite_float_or_none: Callable[[Any], float | None],
) -> list[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]]:
    ranked_entries: list[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]] = []
    directional_hourly_or_higher_present = False
    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        horizon = coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        direction_display = prompt_effective_direction(entry)
        rank = prompt_entry_rank(
            label,
            entry,
            coerce_result_horizon=coerce_result_horizon,
            finite_float_or_none=finite_float_or_none,
        )
        ranked_entries.append((rank, label, entry))
        if direction_display in {"up", "down"} and horizon >= 1.0:
            directional_hourly_or_higher_present = True

    if directional_hourly_or_higher_present:
        filtered_entries = []
        for rank, label, entry in ranked_entries:
            horizon = coerce_result_horizon(entry.get("horizon_hours"))
            direction_display = prompt_effective_direction(entry)
            if direction_display in {"up", "down"} and horizon is not None and horizon < 1.0:
                continue
            filtered_entries.append((rank, label, entry))
        return filtered_entries
    return ranked_entries


def select_prompt_preferred_entry(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    coerce_result_horizon: Callable[[Any], float | None],
    finite_float_or_none: Callable[[Any], float | None],
) -> tuple[str | None, Mapping[str, Any] | None, Dict[str, Any] | None]:
    ranked_entries = select_prompt_candidate_entries(
        summary,
        coerce_result_horizon=coerce_result_horizon,
        finite_float_or_none=finite_float_or_none,
    )
    side_entries: Dict[str, list[tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]]]] = {
        "up": [],
        "down": [],
    }
    for rank, label, entry in ranked_entries:
        direction_display = prompt_effective_direction(entry)
        if direction_display in side_entries:
            side_entries[direction_display].append((rank, label, entry))

    side_profiles: list[tuple[tuple[int, int, int, int, int, int, float, float, float, float], str, tuple[tuple[int, int, int, float, float, float, float, float, float, float], str, Mapping[str, Any]], Dict[str, Any]]] = []
    for side, entries in side_entries.items():
        if not entries:
            continue
        ordered_entries = sorted(entries, key=lambda item: item[0])
        ready_like_count = 0
        high_timeframe_count = 0
        avg_execution_alignment = 0.0
        avg_bias_alignment = 0.0
        support_horizons: list[str] = []
        for rank, label, entry in ordered_entries:
            del rank
            plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
            status = str(plan.get("status") or "rejected")
            if status in {"ready", "waiting_pullback", "bias_only_ready"}:
                ready_like_count += 1
            horizon = coerce_result_horizon(entry.get("horizon_hours"))
            if horizon is not None and horizon >= 8.0:
                high_timeframe_count += 1
            avg_execution_alignment += float(plan.get("execution_alignment_ratio") or 0.0)
            avg_bias_alignment += float(plan.get("bias_alignment_ratio") or 0.0)
            support_horizons.append(label)
        avg_execution_alignment /= max(len(ordered_entries), 1)
        avg_bias_alignment /= max(len(ordered_entries), 1)
        best_rank, best_label, best_entry = ordered_entries[0]
        side_rank = (
            best_rank[0],
            -ready_like_count,
            -high_timeframe_count,
            -len(ordered_entries),
            best_rank[1],
            best_rank[2],
            -avg_execution_alignment,
            -avg_bias_alignment,
            best_rank[5],
            best_rank[6],
        )
        side_profiles.append(
            (
                side_rank,
                side,
                (best_rank, best_label, best_entry),
                {
                    "side": side,
                    "support_horizons": support_horizons,
                    "support_count": len(ordered_entries),
                    "high_timeframe_count": high_timeframe_count,
                    "ready_like_count": ready_like_count,
                    "avg_execution_alignment": float(avg_execution_alignment),
                    "avg_bias_alignment": float(avg_bias_alignment),
                    "conflict_present": sum(1 for side_bucket in side_entries.values() if side_bucket) > 1,
                },
            )
        )

    if side_profiles:
        side_profiles.sort(key=lambda item: item[0])
        _side_rank, _side, best_entry_tuple, side_profile = side_profiles[0]
        _best_rank, best_label, best_entry = best_entry_tuple
        return best_label, best_entry, side_profile

    ranked_entries.sort(key=lambda item: item[0])
    if ranked_entries:
        _rank, preferred_label, preferred_entry = ranked_entries[0]
        return preferred_label, preferred_entry, None
    return None, None, None


def build_operator_summary_compact(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    preferred_label: str | None,
    preferred_entry: Mapping[str, Any] | None,
    market_direction: str,
    execution_state: str,
    blocking_factors: Sequence[str],
    prompt_effective_direction: Callable[[Mapping[str, Any]], str],
) -> Dict[str, Any]:
    normalized_market_direction = str(market_direction).strip().lower()
    if normalized_market_direction == "long":
        normalized_market_direction = "up"
    elif normalized_market_direction == "short":
        normalized_market_direction = "down"
    elif normalized_market_direction not in {"up", "down"}:
        normalized_market_direction = "neutral"

    primary_blocker = None
    if blocking_factors:
        primary_blocker = str(blocking_factors[0])
    elif preferred_entry is not None:
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        if plan.get("reason") not in {None, "pass"}:
            primary_blocker = str(plan.get("reason"))

    action = "stand_aside"
    if execution_state == "ready":
        action = "enter_now"
    elif execution_state == "waiting_pullback":
        action = "wait_for_pullback"
    elif execution_state == "bias_only_ready":
        action = "bias_only"

    support_horizons = []
    max_disagreement_score = 0.0
    caution_flags: list[str] = []
    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        if prompt_effective_direction(entry) == normalized_market_direction:
            support_horizons.append(label)
        disagreement = plan.get("disagreement_severity") if isinstance(plan.get("disagreement_severity"), Mapping) else {}
        max_disagreement_score = max(max_disagreement_score, float(disagreement.get("score") or 0.0))
        if bool(entry.get("bias_support_is_8h_standalone")):
            caution_flags.append("8h_standalone_bias")
        if disagreement.get("triggered"):
            caution_flags.append("short_term_disagreement")

    if preferred_entry is not None:
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        pullback_quality = plan.get("pullback_quality") if isinstance(plan.get("pullback_quality"), Mapping) else {}
        if pullback_quality.get("triggered"):
            caution_flags.append("pullback_quality_insufficient")

    return {
        "market_bias": str(market_direction),
        "preferred_horizon": preferred_label,
        "recommended_operator_action": action,
        "primary_blocker": primary_blocker,
        "support_horizons": support_horizons,
        "max_disagreement_score": float(max_disagreement_score),
        "caution_flags": sorted(set(caution_flags)),
    }


def build_prompt_ready_summary(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    select_prompt_preferred_entry: Callable[[Mapping[str, Mapping[str, Any]]], tuple[str | None, Mapping[str, Any] | None, Dict[str, Any] | None]],
    horizon_sort_key: Callable[[str], Any],
    finite_float_or_none: Callable[[Any], float | None],
) -> Dict[str, Any]:
    preferred_label, preferred_entry, side_profile = select_prompt_preferred_entry(summary)

    trend_parts: list[str] = []
    blocking_factors: list[str] = []
    for label in sorted(summary.keys(), key=horizon_sort_key):
        entry = summary[label]
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
        clause = build_prompt_forecast_clause(label, entry, finite_float_or_none=finite_float_or_none)
        if coherence.get("triggered"):
            blocking_factors.extend(str(reason) for reason in coherence.get("reasons", []))
        elif plan.get("reason") not in {None, "pass", "upstream_model_hold", "confluence_gate", "await_pullback_entry_zone"}:
            blocking_factors.append(str(plan.get("reason")))
        trend_parts.append(clause)

    selected_direction = "Neutral"
    preferred_horizon = None
    confidence_level = "Low"
    tradeable = False
    execution_state = "no_trade"
    pending_trade_action = None
    entry_point = None
    stop_loss = None
    take_profit = None
    risk_reward_ratio = None
    rationale = "No horizon produced a coherent executable trade setup."

    if preferred_entry is not None and preferred_label is not None:
        direction_display = prompt_effective_direction(preferred_entry)
        plan = preferred_entry.get("execution_plan") if isinstance(preferred_entry.get("execution_plan"), Mapping) else {}
        target_management = plan.get("target_management") if isinstance(plan.get("target_management"), Mapping) else {}
        selected_direction = prompt_direction_label(direction_display)
        preferred_horizon = preferred_label
        confidence_level = confidence_level_from_score(
            preferred_entry.get("confidence_score"),
            finite_float_or_none=finite_float_or_none,
        )
        execution_state = str(plan.get("status") or "no_trade")
        pending_trade_action = str(plan.get("pending_trade_action") or "").lower() or None
        tradeable = execution_state in {"ready", "waiting_pullback"} and selected_direction != "Neutral"
        if execution_state in {"ready", "waiting_pullback", "bias_only_ready"} and selected_direction != "Neutral":
            entry_point = finite_float_or_none(preferred_entry.get("entry_price"))
            stop_loss = finite_float_or_none(preferred_entry.get("stop_loss"))
            take_profit = finite_float_or_none(preferred_entry.get("take_profit"))
            risk_reward_ratio = finite_float_or_none(preferred_entry.get("risk_reward_ratio"))
        rationale_parts = [f"Preferred horizon {preferred_label} carries the strongest post-policy bias."]
        if side_profile and side_profile.get("conflict_present"):
            support_horizons = side_profile.get("support_horizons") or []
            support_text = ", ".join(str(value) for value in support_horizons)
            rationale_parts[0] = (
                f"Preferred horizon {preferred_label} wins side arbitration for {selected_direction.lower()} bias "
                f"across {support_text}."
            )
        if plan.get("reason") == "forecast_coherence_gate":
            rationale_parts.append("The forecast remains directional, but forecast coherence blocks it from execution.")
        elif plan.get("status") == "bias_only_ready":
            rationale_parts.append("The horizon is structurally aligned, but the upstream action remains hold.")
        elif plan.get("status") == "waiting_pullback":
            rationale_parts.append("Bias is valid, but price is outside the preferred entry zone.")
        if target_management.get("adapted_to_mfe_headroom"):
            rationale_parts.append("Take-profit was resized to empirical MFE headroom instead of rejecting the setup.")
        elif plan.get("reason") not in {None, "pass"}:
            rationale_parts.append(f"Current blocker: {plan.get('reason')}.")
        rationale = " ".join(rationale_parts)

    formatted_response = "\n".join(
        [
            "Market Outlook & Strategy",
            f"Selected Direction: {selected_direction}",
            f"Preferred Horizon: {preferred_horizon or 'None'}",
            f"Confidence Level: {confidence_level}",
            f"Pending Trade Action: {(pending_trade_action or 'hold').title() if selected_direction != 'Neutral' else 'Hold'}",
            "",
            "Trade Execution Plan (USD)",
            f"Entry Point: {format_usd_value(entry_point, finite_float_or_none=finite_float_or_none) or 'No trade'}",
            f"Stop Loss: {format_usd_value(stop_loss, finite_float_or_none=finite_float_or_none) or 'No trade'}",
            f"Take Profit: {format_usd_value(take_profit, finite_float_or_none=finite_float_or_none) or 'No trade'}",
            f"Risk/Reward Ratio: {f'{risk_reward_ratio:.2f}' if risk_reward_ratio is not None else 'Not applicable'}",
            "",
            "Analysis Summary",
            f"Trend Forecast: {'; '.join(trend_parts)}",
            f"Rationale: {rationale}",
        ]
    )

    blocking_factors = sorted({factor for factor in blocking_factors if factor})
    operator_compact = build_operator_summary_compact(
        summary,
        preferred_label=preferred_label,
        preferred_entry=preferred_entry,
        market_direction=selected_direction,
        execution_state=execution_state,
        blocking_factors=blocking_factors,
        prompt_effective_direction=prompt_effective_direction,
    )
    return {
        "market_outlook_strategy": {
            "selected_direction": selected_direction,
            "preferred_horizon": preferred_horizon,
            "confidence_level": confidence_level,
            "pending_trade_action": pending_trade_action,
            "tradeable": tradeable,
            "execution_state": execution_state,
        },
        "trade_execution_plan_usd": {
            "entry_point": entry_point,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
        },
        "analysis_summary": {
            "trend_forecast": trend_parts,
            "rationale": rationale,
            "blocking_factors": blocking_factors,
        },
        "operator_summary_compact": operator_compact,
        "formatted_response": formatted_response,
    }


def build_blocked_trade_analytics(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    gate_stage_counts: Dict[str, int] = {}
    gate_reason_counts: Dict[str, int] = {}
    by_horizon: Dict[str, Dict[str, Any]] = {}
    blocked_total = 0
    ready_total = 0
    waiting_total = 0
    bias_only_total = 0

    for label, entry in summary.items():
        if not isinstance(entry, Mapping):
            continue
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        status = str(plan.get("status") or "unknown")
        reason = str(plan.get("reason") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if status == "ready":
            ready_total += 1
        elif status == "waiting_pullback":
            waiting_total += 1
        elif status == "bias_only_ready":
            bias_only_total += 1
            blocked_total += 1
        elif status == "rejected":
            blocked_total += 1

        horizon_payload = by_horizon.setdefault(
            label,
            {"status_counts": {}, "reason_counts": {}, "trade_action": str(entry.get("trade_action") or "hold")},
        )
        horizon_payload["status_counts"][status] = horizon_payload["status_counts"].get(status, 0) + 1
        horizon_payload["reason_counts"][reason] = horizon_payload["reason_counts"].get(reason, 0) + 1

        gate_trace = entry.get("gate_trace") if isinstance(entry.get("gate_trace"), list) else []
        for gate_entry in gate_trace:
            if not isinstance(gate_entry, Mapping) or not bool(gate_entry.get("triggered", False)):
                continue
            stage = str(gate_entry.get("stage") or "unknown")
            gate_reason = str(gate_entry.get("reason") or "unknown")
            gate_stage_counts[stage] = gate_stage_counts.get(stage, 0) + 1
            gate_reason_counts[gate_reason] = gate_reason_counts.get(gate_reason, 0) + 1
            horizon_gate_counts = horizon_payload.setdefault("gate_stage_counts", {})
            horizon_gate_reasons = horizon_payload.setdefault("gate_reason_counts", {})
            horizon_gate_counts[stage] = horizon_gate_counts.get(stage, 0) + 1
            horizon_gate_reasons[gate_reason] = horizon_gate_reasons.get(gate_reason, 0) + 1

    return {
        "total_horizons": len(summary),
        "ready_total": ready_total,
        "waiting_pullback_total": waiting_total,
        "bias_only_total": bias_only_total,
        "blocked_total": blocked_total,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "gate_stage_counts": gate_stage_counts,
        "gate_reason_counts": gate_reason_counts,
        "by_horizon": by_horizon,
    }


def build_degradation_monitoring(
    history: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
    resolve_degradation_monitoring_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]],
    horizon_sort_key: Callable[[str], Any],
    finite_float_or_none: Callable[[Any], float | None],
) -> Dict[str, Any]:
    resolved_policy = resolve_degradation_monitoring_policy(policy)
    if not resolved_policy.get("enabled", False):
        return {"enabled": False, "basis": "proxy_history"}

    lookback = int(resolved_policy.get("lookback_snapshots", 30))
    min_snapshots = int(resolved_policy.get("min_snapshots", 10))
    recent_history = list(history[-lookback:])
    horizon_labels: set[str] = set()
    for item in recent_history:
        predictions = item.get("predictions") if isinstance(item, Mapping) else None
        if isinstance(predictions, Mapping):
            horizon_labels.update(str(label) for label in predictions.keys())

    by_horizon: Dict[str, Any] = {}
    alarms: list[Dict[str, Any]] = []
    for horizon_label in sorted(horizon_labels, key=horizon_sort_key):
        rows: list[Mapping[str, Any]] = []
        for item in recent_history:
            predictions = item.get("predictions") if isinstance(item, Mapping) else None
            entry = predictions.get(horizon_label) if isinstance(predictions, Mapping) else None
            if isinstance(entry, Mapping):
                rows.append(entry)
        if len(rows) < min_snapshots:
            by_horizon[horizon_label] = {
                "samples": len(rows),
                "alarm": False,
                "reasons": ["insufficient_history"],
            }
            continue

        ready_like = 0
        blocked = 0
        confidence_values: list[float] = []
        expected_net_values: list[float] = []
        for entry in rows:
            plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
            status = str(plan.get("status") or "unknown")
            if status in {"ready", "waiting_pullback", "bias_only_ready"}:
                ready_like += 1
            if status in {"rejected", "bias_only_ready"}:
                blocked += 1
            confidence = finite_float_or_none(entry.get("confidence_score"))
            if confidence is not None:
                confidence_values.append(confidence)
            trade_decision = entry.get("trade_decision") if isinstance(entry.get("trade_decision"), Mapping) else {}
            expected_net = finite_float_or_none(trade_decision.get("expected_net"))
            if expected_net is not None:
                expected_net_values.append(expected_net)

        sample_count = len(rows)
        ready_ratio = ready_like / max(sample_count, 1)
        blocked_ratio = blocked / max(sample_count, 1)
        avg_confidence = float(sum(confidence_values) / max(len(confidence_values), 1)) if confidence_values else None
        avg_expected_net = float(sum(expected_net_values) / max(len(expected_net_values), 1)) if expected_net_values else None
        reasons: list[str] = []
        if ready_ratio < float(resolved_policy.get("min_ready_ratio", 0.1)):
            reasons.append("ready_ratio_below_floor")
        if blocked_ratio > float(resolved_policy.get("max_blocked_ratio", 0.85)):
            reasons.append("blocked_ratio_above_ceiling")
        if avg_confidence is not None and avg_confidence < float(resolved_policy.get("min_confidence", 0.0)):
            reasons.append("confidence_below_floor")
        if avg_expected_net is not None and avg_expected_net < float(resolved_policy.get("min_expected_net", 0.0)):
            reasons.append("expected_net_below_floor")

        alarm = bool(reasons)
        by_horizon[horizon_label] = {
            "samples": sample_count,
            "ready_ratio": float(ready_ratio),
            "blocked_ratio": float(blocked_ratio),
            "avg_confidence": avg_confidence,
            "avg_expected_net": avg_expected_net,
            "alarm": alarm,
            "reasons": reasons,
        }
        if alarm:
            alarms.append({"horizon": horizon_label, "reasons": reasons})

    return {
        "enabled": True,
        "basis": "proxy_history",
        "lookback_snapshots": lookback,
        "min_snapshots": min_snapshots,
        "alarms": alarms,
        "by_horizon": by_horizon,
    }


def resolve_degradation_monitoring_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lookback_snapshots": max(int(cfg.get("lookback_snapshots") or 30), 3),
        "min_snapshots": max(int(cfg.get("min_snapshots") or 10), 1),
        "min_ready_ratio": max(min(float(cfg.get("min_ready_ratio") or 0.1), 1.0), 0.0),
        "max_blocked_ratio": max(min(float(cfg.get("max_blocked_ratio") or 0.85), 1.0), 0.0),
        "min_expected_net": float(cfg.get("min_expected_net") or 0.0),
        "min_confidence": max(min(float(cfg.get("min_confidence") or 0.0), 1.0), 0.0),
    }


def build_runtime_prompt_ready_summary(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    horizon_sort_key: Callable[[str], Any],
) -> Dict[str, Any]:
    return build_prompt_ready_summary(
        summary,
        select_prompt_preferred_entry=lambda payload: select_prompt_preferred_entry(
            payload,
            coerce_result_horizon=coerce_result_horizon,
            finite_float_or_none=finite_float_or_none,
        ),
        horizon_sort_key=horizon_sort_key,
        finite_float_or_none=finite_float_or_none,
    )


def build_runtime_degradation_monitoring(
    history: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None,
    *,
    horizon_sort_key: Callable[[str], Any],
) -> Dict[str, Any]:
    return build_degradation_monitoring(
        history,
        policy=policy,
        resolve_degradation_monitoring_policy=resolve_degradation_monitoring_policy,
        horizon_sort_key=horizon_sort_key,
        finite_float_or_none=finite_float_or_none,
    )


def write_prediction_summary(
    summary: Dict[str, Dict[str, Any]],
    *,
    degradation_policy: Mapping[str, Any] | None,
    latest_prediction_path: Path,
    history_prediction_path: Path,
    build_prompt_ready_summary_fn: Callable[[Mapping[str, Mapping[str, Any]]], Dict[str, Any]],
    build_blocked_trade_analytics_fn: Callable[[Mapping[str, Mapping[str, Any]]], Dict[str, Any]],
    build_degradation_monitoring_fn: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any] | None], Dict[str, Any]],
    print_fn: Callable[[str], None],
) -> dict[str, Any]:
    latest_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    execution_prior_summary = build_execution_prior_summary(summary)
    prompt_ready_summary = build_prompt_ready_summary_fn(summary)
    blocked_trade_analytics = build_blocked_trade_analytics_fn(summary)
    json_payload = {
        "generated_at": generated_at,
        "predictions": summary,
        "execution_prior_summary": execution_prior_summary,
        "blocked_trade_analytics": blocked_trade_analytics,
        "prompt_ready_summary": prompt_ready_summary,
    }
    latest_prediction_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print_fn(json.dumps(json_payload, indent=2))

    history_entry = dict(json_payload)
    history: list[Dict[str, object]] = []
    if history_prediction_path.exists():
        try:
            history = json.loads(history_prediction_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    history.append(history_entry)
    history_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    history_prediction_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    json_payload["degradation_monitoring"] = build_degradation_monitoring_fn(history, degradation_policy)
    history[-1]["degradation_monitoring"] = json_payload["degradation_monitoring"]
    history_prediction_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    latest_prediction_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return json_payload
