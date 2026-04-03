from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


SummaryPayload = Dict[str, Dict[str, Any]]


def resolve_confluence_policy(
    config: Mapping[str, Any] | None,
    *,
    coerce_numeric_horizon: Callable[[Any], float | None],
    normalize_horizon_value: Callable[[float], float],
) -> Dict[str, Any]:
    cfg = config or {}
    short_horizons = cfg.get("short_horizons") or [0.25, 1.0]
    mid_horizons = cfg.get("mid_horizons") or [4.0, 8.0, 12.0]

    def _normalize_horizon_map(raw: Any, *, minimum: float = 0.0, maximum: float | None = None) -> Dict[float, float]:
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
            resolved[horizon] = numeric_value
        return resolved

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "short_horizons": sorted({normalize_horizon_value(v) for v in short_horizons}),
        "mid_horizons": sorted({normalize_horizon_value(v) for v in mid_horizons}),
        "min_support_ratio": max(min(float(cfg.get("min_support_ratio") or 0.6), 1.0), 0.0),
        "min_support_ratio_by_horizon": _normalize_horizon_map(
            cfg.get("min_support_ratio_by_horizon"),
            minimum=0.0,
            maximum=1.0,
        ),
        "min_mid_term_ratio": max(min(float(cfg.get("min_mid_term_ratio") or 0.5), 1.0), 0.0),
        "min_short_term_ratio": max(min(float(cfg.get("min_short_term_ratio") or 0.5), 1.0), 0.0),
        "dominant_ratio_floor": max(min(float(cfg.get("dominant_ratio_floor") or 0.55), 1.0), 0.0),
        "min_aligned_horizons": max(int(cfg.get("min_aligned_horizons") or 2), 1),
        "min_aligned_horizons_by_horizon": _normalize_horizon_map(
            cfg.get("min_aligned_horizons_by_horizon"),
            minimum=1.0,
        ),
        "require_mid_term_alignment": bool(cfg.get("require_mid_term_alignment", True)),
        "require_short_term_alignment": bool(cfg.get("require_short_term_alignment", False)),
    }


def apply_confluence_policy(
    summary: SummaryPayload,
    policy: Mapping[str, Any],
    *,
    forecast_coherence_excluded: Callable[[Mapping[str, Any]], bool],
    coerce_result_horizon: Callable[[Any], float | None],
    direction_vote: Callable[[Mapping[str, Any]], str],
    lookup_horizon_value: Callable[[Mapping[float, float], float, float], float],
    append_gate_trace: Callable[..., None],
) -> SummaryPayload:
    if not summary:
        return summary

    labeled_entries: list[tuple[str, Dict[str, Any], float]] = []
    for label, entry in summary.items():
        if forecast_coherence_excluded(entry):
            continue
        horizon = coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        labeled_entries.append((label, entry, horizon))

    if not labeled_entries:
        return summary

    short_horizons = set(policy.get("short_horizons", []))
    mid_horizons = set(policy.get("mid_horizons", []))
    up_count = sum(1 for _label, entry, _h in labeled_entries if direction_vote(entry) == "up")
    down_count = len(labeled_entries) - up_count
    dominant_direction = "neutral"
    dominant_ratio = 0.5
    if up_count > down_count:
        dominant_direction = "up"
        dominant_ratio = up_count / len(labeled_entries)
    elif down_count > up_count:
        dominant_direction = "down"
        dominant_ratio = down_count / len(labeled_entries)

    for _label, entry, horizon in labeled_entries:
        current_direction = direction_vote(entry)
        aligned = [item for item in labeled_entries if direction_vote(item[1]) == current_direction]
        aligned_count = len(aligned)
        support_ratio = aligned_count / len(labeled_entries)
        min_aligned_horizons = int(
            round(
                lookup_horizon_value(
                    policy.get("min_aligned_horizons_by_horizon", {}) if isinstance(policy.get("min_aligned_horizons_by_horizon"), Mapping) else {},
                    horizon,
                    float(policy.get("min_aligned_horizons", 2)),
                )
            )
        )
        min_support_ratio = lookup_horizon_value(
            policy.get("min_support_ratio_by_horizon", {}) if isinstance(policy.get("min_support_ratio_by_horizon"), Mapping) else {},
            horizon,
            float(policy.get("min_support_ratio", 0.6)),
        )

        short_entries = [item for item in labeled_entries if item[2] in short_horizons]
        mid_entries = [item for item in labeled_entries if item[2] in mid_horizons]
        short_ratio = (
            sum(1 for _other_label, other_entry, _other_h in short_entries if direction_vote(other_entry) == current_direction) / len(short_entries)
            if short_entries else None
        )
        mid_ratio = (
            sum(1 for _other_label, other_entry, _other_h in mid_entries if direction_vote(other_entry) == current_direction) / len(mid_entries)
            if mid_entries else None
        )

        confluence_triggered = False
        reasons: list[str] = []
        if str(entry.get("trade_action", "hold")) != "hold":
            if aligned_count < min_aligned_horizons:
                confluence_triggered = True
                reasons.append("aligned_horizons_below_min")
            if support_ratio < min_support_ratio:
                confluence_triggered = True
                reasons.append("support_ratio_below_min")
            if bool(policy.get("require_mid_term_alignment", True)) and mid_ratio is not None and mid_ratio < float(policy.get("min_mid_term_ratio", 0.5)):
                confluence_triggered = True
                reasons.append("mid_term_ratio_below_min")
            if bool(policy.get("require_short_term_alignment", False)) and short_ratio is not None and short_ratio < float(policy.get("min_short_term_ratio", 0.5)):
                confluence_triggered = True
                reasons.append("short_term_ratio_below_min")
            if dominant_direction != "neutral" and dominant_direction != current_direction and dominant_ratio >= float(policy.get("dominant_ratio_floor", 0.55)):
                confluence_triggered = True
                reasons.append("dominant_direction_conflict")

        entry["confluence"] = {
            "enabled": bool(policy.get("enabled", False)),
            "dominant_direction": dominant_direction,
            "dominant_ratio": float(dominant_ratio),
            "aligned_horizons": int(aligned_count),
            "total_horizons": int(len(labeled_entries)),
            "support_ratio": float(support_ratio),
            "short_term_ratio": None if short_ratio is None else float(short_ratio),
            "mid_term_ratio": None if mid_ratio is None else float(mid_ratio),
            "triggered": bool(confluence_triggered),
            "reasons": reasons,
        }
        entry["confluence_support_ratio"] = float(support_ratio)
        entry["confluence_short_term_ratio"] = None if short_ratio is None else float(short_ratio)
        entry["confluence_mid_term_ratio"] = None if mid_ratio is None else float(mid_ratio)
        entry["confluence_direction_matches_dominant"] = 0.0 if dominant_direction == "neutral" else float(dominant_direction == current_direction)
        if confluence_triggered:
            append_gate_trace(
                entry,
                stage="confluence",
                reason="|".join(reasons),
                triggered=True,
                blocking=True,
            )
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            trade_decision = entry.get("trade_decision")
            if isinstance(trade_decision, dict):
                trade_decision["confluence_gate_triggered"] = True
                trade_decision["confluence_gate_reasons"] = reasons
    return summary