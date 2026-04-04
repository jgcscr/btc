from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import pandas as pd

from src.runtime.macro_shadow_simulator import (
    CONF_BUCKETS,
    READY_STATES,
    _confidence_bucket,
    _deepcopy_jsonable,
    _direction_from_trade_action,
    _horizon_to_float,
    _normalize_regime,
    _to_float_or_none,
)
from src.runtime.macro_shadow_simulator import load_prediction_history
from src.trading.feature_engineering import augment_hourly_price_features


STATE_ENGINEERING_FEATURES: tuple[str, ...] = (
    "trend_path_efficiency_4h",
    "trend_path_efficiency_8h",
    "trend_directional_persistence_4h",
    "trend_directional_persistence_8h",
    "range_compression_ratio_4h_24h",
    "range_compression_transition_8h",
    "price_distance_to_high_atr_24h",
    "price_distance_to_low_atr_24h",
    "price_distance_to_high_pct_rank_24h",
    "price_distance_to_low_pct_rank_24h",
    "volume_regime_zscore_24h",
)

ORDER_FLOW_FEATURES: tuple[str, ...] = (
    "cvd_ratio_6h",
    "cvd_zscore_6h",
    "trades_taker_imbalance_acceleration_3h",
    "trades_taker_imbalance_persistence_6h",
    "interaction_imbalance_trend_6h",
    "interaction_breakout_volume_8h",
    "vwap_deviation_8h",
)

ALL_FAMILY_FEATURES: tuple[str, ...] = tuple(sorted(set(STATE_ENGINEERING_FEATURES + ORDER_FLOW_FEATURES)))


@dataclass(frozen=True)
class FamilySignal:
    family: str
    bias: str
    score: float
    strength: float
    available_feature_count: int


@dataclass(frozen=True)
class FamilyShadowPolicy:
    name: str
    description: str
    enforcement_mode: str = "weak_signal_veto"
    min_horizon_hours: float = 1.0
    enabled_horizons: tuple[float, ...] = ()
    enabled_regimes: tuple[str, ...] = ()
    confidence_band_min: float | None = None
    confidence_band_max: float | None = None
    weak_trade_confidence_max: float = 0.7
    weak_trade_expected_value_max: float = 0.001
    strong_trade_expected_value_min: float = 0.003
    strong_trade_confidence_min: float = 0.75
    min_signal_strength: float = 0.08
    confluence_disagreement_threshold: float = 0.2


@dataclass(frozen=True)
class FamilySnapshotState:
    status: str
    asof_ts: str | None
    lag_hours: float | None
    values: Dict[str, float]
    state_signal: FamilySignal
    order_flow_signal: FamilySignal


@dataclass(frozen=True)
class FamilySnapshotReplayResult:
    baseline_strategy: Dict[str, Any]
    shadow_strategy: Dict[str, Any]
    baseline_predictions: Dict[str, Dict[str, Any]]
    shadow_predictions: Dict[str, Dict[str, Any]]
    state: FamilySnapshotState
    changed_horizons: List[str]
    changed_regimes: List[str]
    beneficial_blocks: int
    harmful_blocks: int
    diagnostics: Dict[str, int]


def _to_horizon_label(value: float) -> str:
    if value >= 1.0 and float(value).is_integer():
        return f"{int(value)}h"
    if value >= 1.0:
        return f"{value:g}h"
    minutes = int(round(value * 60))
    return f"{minutes}m"


def load_spot_feature_frame(spot_dir: Path) -> pd.DataFrame:
    files = sorted(spot_dir.glob("*.parquet"))
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if "ts" not in frame.columns:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["ts", *ALL_FAMILY_FEATURES])

    merged = pd.concat(frames, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset="ts", keep="last")

    enriched = augment_hourly_price_features(merged, strict_missing=False)
    for col in ALL_FAMILY_FEATURES:
        if col not in enriched.columns:
            enriched[col] = 0.0
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)

    return enriched[["ts", *ALL_FAMILY_FEATURES]].reset_index(drop=True)


def _score_state_engineering(values: Mapping[str, float]) -> FamilySignal:
    directional_components = [
        (_to_float_or_none(values.get("price_distance_to_low_atr_24h")) or 0.0)
        - (_to_float_or_none(values.get("price_distance_to_high_atr_24h")) or 0.0),
        (_to_float_or_none(values.get("price_distance_to_low_pct_rank_24h")) or 0.0)
        - (_to_float_or_none(values.get("price_distance_to_high_pct_rank_24h")) or 0.0),
        _to_float_or_none(values.get("range_compression_transition_8h")) or 0.0,
    ]
    strength_components = [
        _to_float_or_none(values.get("trend_path_efficiency_4h")) or 0.0,
        _to_float_or_none(values.get("trend_path_efficiency_8h")) or 0.0,
        _to_float_or_none(values.get("trend_directional_persistence_4h")) or 0.0,
        _to_float_or_none(values.get("trend_directional_persistence_8h")) or 0.0,
        abs(_to_float_or_none(values.get("range_compression_transition_8h")) or 0.0),
        abs((_to_float_or_none(values.get("volume_regime_zscore_24h")) or 0.0) / 3.0),
    ]
    directional = sum(directional_components) / max(len(directional_components), 1)
    strength = min(1.0, max(0.0, sum(strength_components) / max(len(strength_components), 1)))
    score = directional * (0.5 + strength)
    score = max(-1.0, min(1.0, score))

    bias = "neutral"
    if score > 0.06:
        bias = "long"
    elif score < -0.06:
        bias = "short"

    available_count = sum(1 for key in STATE_ENGINEERING_FEATURES if key in values)
    return FamilySignal(
        family="state_engineering",
        bias=bias,
        score=float(score),
        strength=float(abs(score)),
        available_feature_count=available_count,
    )


def _score_order_flow(values: Mapping[str, float]) -> FamilySignal:
    components = [
        _to_float_or_none(values.get("cvd_ratio_6h")) or 0.0,
        (_to_float_or_none(values.get("cvd_zscore_6h")) or 0.0) / 3.0,
        _to_float_or_none(values.get("trades_taker_imbalance_acceleration_3h")) or 0.0,
        _to_float_or_none(values.get("interaction_imbalance_trend_6h")) or 0.0,
        (_to_float_or_none(values.get("interaction_breakout_volume_8h")) or 0.0) / 5.0,
        _to_float_or_none(values.get("vwap_deviation_8h")) or 0.0,
    ]
    base_score = sum(components) / max(len(components), 1)
    persistence = _to_float_or_none(values.get("trades_taker_imbalance_persistence_6h")) or 0.0
    persistence_boost = 0.8 + min(max(persistence, 0.0), 1.0) * 0.4
    score = max(-1.0, min(1.0, base_score * persistence_boost))

    bias = "neutral"
    if score > 0.05:
        bias = "long"
    elif score < -0.05:
        bias = "short"

    available_count = sum(1 for key in ORDER_FLOW_FEATURES if key in values)
    return FamilySignal(
        family="order_flow",
        bias=bias,
        score=float(score),
        strength=float(abs(score)),
        available_feature_count=available_count,
    )


def resolve_family_snapshot_state(
    *,
    snapshot_ts: pd.Timestamp,
    feature_frame: pd.DataFrame,
    max_staleness_hours: float,
) -> FamilySnapshotState:
    if feature_frame.empty:
        neutral = FamilySignal("state_engineering", "neutral", 0.0, 0.0, 0)
        neutral_of = FamilySignal("order_flow", "neutral", 0.0, 0.0, 0)
        return FamilySnapshotState(
            status="unavailable",
            asof_ts=None,
            lag_hours=None,
            values={},
            state_signal=neutral,
            order_flow_signal=neutral_of,
        )

    row = feature_frame[feature_frame["ts"] <= snapshot_ts].tail(1)
    if row.empty:
        neutral = FamilySignal("state_engineering", "neutral", 0.0, 0.0, 0)
        neutral_of = FamilySignal("order_flow", "neutral", 0.0, 0.0, 0)
        return FamilySnapshotState(
            status="unavailable",
            asof_ts=None,
            lag_hours=None,
            values={},
            state_signal=neutral,
            order_flow_signal=neutral_of,
        )

    asof_ts = pd.Timestamp(row.iloc[0]["ts"])
    lag_hours = max((snapshot_ts - asof_ts).total_seconds() / 3600.0, 0.0)
    status = "available" if lag_hours <= max_staleness_hours else "stale"

    values = {
        key: float(_to_float_or_none(row.iloc[0][key]) or 0.0)
        for key in ALL_FAMILY_FEATURES
        if key in row.columns
    }
    state_signal = _score_state_engineering(values)
    order_flow_signal = _score_order_flow(values)

    return FamilySnapshotState(
        status=status,
        asof_ts=asof_ts.isoformat(),
        lag_hours=float(lag_hours),
        values=values,
        state_signal=state_signal,
        order_flow_signal=order_flow_signal,
    )


def _pick_strategy_from_predictions(predictions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    candidates: list[tuple[str, float, Mapping[str, Any]]] = []
    for label, entry in predictions.items():
        plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
        status = str(plan.get("status") or "no_trade")
        action = str(entry.get("trade_action") or "hold").lower()
        if status in READY_STATES and action in {"long", "short"}:
            score = _to_float_or_none(entry.get("execution_score"))
            if score is None:
                score = 0.0
            candidates.append((label, float(score), entry))

    if not candidates:
        return {
            "preferred_horizon": None,
            "selected_direction": "Neutral",
            "execution_state": "no_trade",
            "pending_trade_action": "hold",
            "tradeable": False,
            "reason": "no_tradeable_horizon",
        }

    candidates.sort(key=lambda item: item[1], reverse=True)
    label, _score, entry = candidates[0]
    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), Mapping) else {}
    action = str(entry.get("trade_action") or "hold").lower()
    selected_direction = "Long" if action == "long" else "Short" if action == "short" else "Neutral"
    execution_state = str(plan.get("status") or "no_trade")
    return {
        "preferred_horizon": label,
        "selected_direction": selected_direction,
        "execution_state": execution_state,
        "pending_trade_action": action,
        "tradeable": execution_state in {"ready", "waiting_pullback"} and selected_direction != "Neutral",
        "reason": str(plan.get("reason") or "pass"),
    }


def _baseline_strategy(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    prs = snapshot.get("prompt_ready_summary") if isinstance(snapshot.get("prompt_ready_summary"), Mapping) else {}
    outlook = prs.get("market_outlook_strategy") if isinstance(prs.get("market_outlook_strategy"), Mapping) else {}
    preferred = outlook.get("preferred_horizon")
    predictions = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), Mapping) else {}
    reason = None
    if isinstance(preferred, str) and preferred in predictions:
        entry = predictions.get(preferred)
        plan = entry.get("execution_plan") if isinstance(entry, Mapping) and isinstance(entry.get("execution_plan"), Mapping) else {}
        reason = plan.get("reason")
    return {
        "preferred_horizon": preferred,
        "selected_direction": outlook.get("selected_direction"),
        "execution_state": outlook.get("execution_state"),
        "pending_trade_action": outlook.get("pending_trade_action"),
        "tradeable": bool(outlook.get("tradeable", False)),
        "reason": str(reason or "unknown"),
    }


def _is_weak_trade(entry: Mapping[str, Any], policy: FamilyShadowPolicy) -> bool:
    expected_value = _to_float_or_none(entry.get("expected_value"))
    confidence = _to_float_or_none(entry.get("confidence_score"))
    if expected_value is None:
        expected_value = 0.0
    if confidence is None:
        confidence = 0.0
    return confidence <= policy.weak_trade_confidence_max or expected_value <= policy.weak_trade_expected_value_max


def _is_strong_trade(entry: Mapping[str, Any], policy: FamilyShadowPolicy) -> bool:
    expected_value = _to_float_or_none(entry.get("expected_value"))
    confidence = _to_float_or_none(entry.get("confidence_score"))
    if expected_value is None or confidence is None:
        return False
    return expected_value >= policy.strong_trade_expected_value_min and confidence >= policy.strong_trade_confidence_min


def _policy_applies(
    *,
    entry: Mapping[str, Any],
    horizon: float,
    policy: FamilyShadowPolicy,
) -> tuple[bool, str]:
    if horizon < policy.min_horizon_hours:
        return False, "below_min_horizon"

    if policy.enabled_horizons:
        in_scope = any(abs(horizon - h) <= 1e-6 for h in policy.enabled_horizons)
        if not in_scope:
            return False, "horizon_out_of_scope"

    regime = _normalize_regime(entry.get("regime_state"))
    if policy.enabled_regimes and regime not in {r.lower() for r in policy.enabled_regimes}:
        return False, "regime_out_of_scope"

    confidence = _to_float_or_none(entry.get("confidence_score"))
    if policy.confidence_band_min is not None and (confidence is None or confidence < policy.confidence_band_min):
        return False, "confidence_out_of_scope"
    if policy.confidence_band_max is not None and (confidence is None or confidence > policy.confidence_band_max):
        return False, "confidence_out_of_scope"

    return True, "in_scope"


def default_family_policy_variants() -> List[FamilyShadowPolicy]:
    return [
        FamilyShadowPolicy(
            name="weak_signal_veto_only",
            description="Block weak conflicting trades when family signal opposes trade direction",
            enforcement_mode="weak_signal_veto",
        ),
        FamilyShadowPolicy(
            name="mid_band_confidence_gating",
            description="Apply weak veto only for mid-confidence trades",
            enforcement_mode="weak_signal_veto",
            confidence_band_min=0.45,
            confidence_band_max=0.65,
        ),
        FamilyShadowPolicy(
            name="chop_only_enforcement",
            description="Apply weak veto only in chop regime",
            enforcement_mode="weak_signal_veto",
            enabled_regimes=("chop",),
        ),
        FamilyShadowPolicy(
            name="neutral_only_enforcement",
            description="Apply weak veto only in neutral regime",
            enforcement_mode="weak_signal_veto",
            enabled_regimes=("neutral",),
        ),
        FamilyShadowPolicy(
            name="horizon_4h_8h_only",
            description="Apply weak veto only for 4h/8h horizons",
            enforcement_mode="weak_signal_veto",
            enabled_horizons=(4.0, 8.0),
        ),
        FamilyShadowPolicy(
            name="horizon_4h_8h_12h_only",
            description="Apply weak veto only for 4h/8h/12h horizons",
            enforcement_mode="weak_signal_veto",
            enabled_horizons=(4.0, 8.0, 12.0),
        ),
        FamilyShadowPolicy(
            name="confluence_veto_disagreement",
            description="Veto weak trades only when state/order-flow signals strongly disagree",
            enforcement_mode="confluence_veto_disagreement",
        ),
        FamilyShadowPolicy(
            name="confluence_relief_disagreement",
            description="Weak veto with relief when state/order-flow strongly disagree",
            enforcement_mode="confluence_relief_disagreement",
        ),
    ]


def replay_snapshot_with_family_shadow(
    snapshot: Mapping[str, Any],
    *,
    state: FamilySnapshotState,
    family: str,
    policy: FamilyShadowPolicy,
) -> FamilySnapshotReplayResult:
    predictions = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), Mapping) else {}
    baseline_predictions = {
        str(label): _deepcopy_jsonable(entry)
        for label, entry in predictions.items()
        if isinstance(entry, Mapping)
    }
    shadow_predictions: Dict[str, Dict[str, Any]] = _deepcopy_jsonable(baseline_predictions)

    changed_horizons: List[str] = []
    changed_regimes: List[str] = []
    beneficial_blocks = 0
    harmful_blocks = 0
    diagnostics: Dict[str, int] = {
        "feature_unavailable": 0,
        "feature_stale": 0,
        "signal_neutral": 0,
        "conflict_total": 0,
        "conflict_weak": 0,
        "conflict_strong": 0,
        "scope_horizon_out": 0,
        "scope_regime_out": 0,
        "scope_confidence_out": 0,
        "confluence_disagreement": 0,
    }

    primary_signal = state.state_signal if family == "state_engineering" else state.order_flow_signal
    secondary_signal = state.order_flow_signal if family == "state_engineering" else state.state_signal

    for label, entry in list(shadow_predictions.items()):
        horizon = _horizon_to_float(label)
        if horizon is None:
            continue

        applies, scope_reason = _policy_applies(entry=entry, horizon=horizon, policy=policy)
        if not applies:
            if scope_reason in {"horizon_out_of_scope", "below_min_horizon"}:
                diagnostics["scope_horizon_out"] += 1
            elif scope_reason == "regime_out_of_scope":
                diagnostics["scope_regime_out"] += 1
            elif scope_reason == "confidence_out_of_scope":
                diagnostics["scope_confidence_out"] += 1
            continue

        baseline_entry = baseline_predictions.get(label, {})
        side = _direction_from_trade_action(str(entry.get("trade_action") or "hold").lower())
        if side == "neutral":
            continue

        shadow_note: Dict[str, Any] = {
            "applied": False,
            "reason": "no_change",
            "family": family,
            "feature_status": state.status,
            "family_bias": primary_signal.bias,
            "family_score": primary_signal.score,
            "secondary_bias": secondary_signal.bias,
            "secondary_score": secondary_signal.score,
        }

        if state.status in {"unavailable", "stale"}:
            diagnostics[f"feature_{state.status}"] += 1
            shadow_note["reason"] = f"feature_{state.status}_fail_open"
            entry["family_shadow"] = shadow_note
            continue

        if primary_signal.bias == "neutral":
            diagnostics["signal_neutral"] += 1
            shadow_note["reason"] = "family_signal_neutral"
            entry["family_shadow"] = shadow_note
            continue

        if side != primary_signal.bias:
            diagnostics["conflict_total"] += 1
            weak = _is_weak_trade(entry, policy)
            strong = _is_strong_trade(baseline_entry, policy)
            if weak:
                diagnostics["conflict_weak"] += 1
            else:
                diagnostics["conflict_strong"] += 1

            strong_disagreement = (
                primary_signal.bias in {"long", "short"}
                and secondary_signal.bias in {"long", "short"}
                and primary_signal.bias != secondary_signal.bias
                and abs(primary_signal.score) >= policy.confluence_disagreement_threshold
                and abs(secondary_signal.score) >= policy.confluence_disagreement_threshold
            )
            if strong_disagreement:
                diagnostics["confluence_disagreement"] += 1

            should_block = False
            if policy.enforcement_mode == "weak_signal_veto":
                should_block = weak and abs(primary_signal.score) >= policy.min_signal_strength
            elif policy.enforcement_mode == "confluence_veto_disagreement":
                should_block = weak and strong_disagreement
            elif policy.enforcement_mode == "confluence_relief_disagreement":
                should_block = weak and abs(primary_signal.score) >= policy.min_signal_strength and not strong_disagreement

            if should_block:
                entry["trade_action"] = "hold"
                entry["signal_ensemble"] = 0
                plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), MutableMapping) else {}
                plan["status"] = "rejected"
                plan["reason"] = f"family_shadow_{family}_veto"
                entry["execution_plan"] = plan
                shadow_note["applied"] = True
                shadow_note["reason"] = "family_conflict_blocked"
                changed_horizons.append(label)
                changed_regimes.append(_normalize_regime(entry.get("regime_state")))
                beneficial_blocks += 1
                if strong:
                    harmful_blocks += 1
            else:
                shadow_note["reason"] = "family_conflict_not_blocked"
        else:
            shadow_note["reason"] = "family_aligned"

        entry["family_shadow"] = shadow_note

    baseline_strategy = _baseline_strategy(snapshot)
    if changed_horizons:
        shadow_strategy = _pick_strategy_from_predictions(shadow_predictions)
    else:
        shadow_strategy = dict(baseline_strategy)

    return FamilySnapshotReplayResult(
        baseline_strategy=baseline_strategy,
        shadow_strategy=shadow_strategy,
        baseline_predictions=baseline_predictions,
        shadow_predictions=shadow_predictions,
        state=state,
        changed_horizons=sorted(set(changed_horizons)),
        changed_regimes=sorted(set(changed_regimes)),
        beneficial_blocks=beneficial_blocks,
        harmful_blocks=harmful_blocks,
        diagnostics=diagnostics,
    )


def _increment(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def summarize_family_replay(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    replay_results: Sequence[FamilySnapshotReplayResult],
) -> Dict[str, Any]:
    selected_direction_delta: Dict[str, int] = {}
    execution_state_reason_delta: Dict[str, int] = {}
    preferred_horizon_delta: Dict[str, int] = {}
    entry_outcome_delta: Dict[str, int] = {}
    per_horizon_deltas: Dict[str, Dict[str, int]] = {}
    per_regime_deltas: Dict[str, Dict[str, int]] = {}
    per_confidence_bucket_deltas: Dict[str, Dict[str, int]] = {}
    diagnostics_totals: Dict[str, int] = {}

    changed_snapshots = 0
    changed_selected_direction_count = 0
    changed_tradeable_count = 0
    beneficial_blocks = 0
    harmful_blocks = 0
    state_status_counts: Dict[str, int] = {}

    for snap, result in zip(snapshots, replay_results):
        beneficial_blocks += result.beneficial_blocks
        harmful_blocks += result.harmful_blocks
        _increment(state_status_counts, result.state.status)
        for key, value in result.diagnostics.items():
            diagnostics_totals[key] = diagnostics_totals.get(key, 0) + int(value)

        base = result.baseline_strategy
        shadow = result.shadow_strategy

        base_dir = str(base.get("selected_direction") or "Neutral")
        shadow_dir = str(shadow.get("selected_direction") or "Neutral")
        if base_dir != shadow_dir:
            changed_selected_direction_count += 1
            changed_snapshots += 1
            _increment(selected_direction_delta, f"{base_dir}->{shadow_dir}")

            preferred_horizon = str(base.get("preferred_horizon") or "")
            horizon_entry = None
            predictions = snap.get("predictions") if isinstance(snap.get("predictions"), Mapping) else {}
            if preferred_horizon and preferred_horizon in predictions:
                horizon_entry = predictions.get(preferred_horizon)

            regime = _normalize_regime(horizon_entry.get("regime_state") if isinstance(horizon_entry, Mapping) else "unknown")
            reg_counter = per_regime_deltas.setdefault(regime, {"changed_snapshot": 0, "beneficial_blocks": 0, "harmful_blocks": 0})
            reg_counter["changed_snapshot"] += 1
            reg_counter["beneficial_blocks"] += int(result.beneficial_blocks)
            reg_counter["harmful_blocks"] += int(result.harmful_blocks)

            confidence = _to_float_or_none(
                horizon_entry.get("confidence_score") if isinstance(horizon_entry, Mapping) else None
            )
            conf_bucket = _confidence_bucket(confidence)
            conf_counter = per_confidence_bucket_deltas.setdefault(conf_bucket, {"changed_snapshot": 0, "beneficial_blocks": 0, "harmful_blocks": 0})
            conf_counter["changed_snapshot"] += 1
            conf_counter["beneficial_blocks"] += int(result.beneficial_blocks)
            conf_counter["harmful_blocks"] += int(result.harmful_blocks)

        base_tradeable = bool(base.get("tradeable", False))
        shadow_tradeable = bool(shadow.get("tradeable", False))
        if base_tradeable != shadow_tradeable:
            changed_tradeable_count += 1
            _increment(entry_outcome_delta, f"{base_tradeable}->{shadow_tradeable}")

        base_h = str(base.get("preferred_horizon") or "None")
        shadow_h = str(shadow.get("preferred_horizon") or "None")
        if base_h != shadow_h:
            _increment(preferred_horizon_delta, f"{base_h}->{shadow_h}")

        base_state = str(base.get("execution_state") or "unknown")
        shadow_state = str(shadow.get("execution_state") or "unknown")
        base_reason = str(base.get("reason") or "unknown")
        shadow_reason = str(shadow.get("reason") or "unknown")
        if base_state != shadow_state or base_reason != shadow_reason:
            _increment(execution_state_reason_delta, f"{base_state}:{base_reason}->{shadow_state}:{shadow_reason}")

        for horizon_label in result.changed_horizons:
            per_h = per_horizon_deltas.setdefault(horizon_label, {"changed_trade_action": 0, "blocked": 0})
            per_h["changed_trade_action"] += 1
            per_h["blocked"] += 1

    total = max(len(replay_results), 1)
    changed_ratio = changed_snapshots / total
    quality_score = beneficial_blocks - harmful_blocks

    assessment = "neutral"
    if quality_score >= 20 and changed_ratio >= 0.03:
        assessment = "beneficial"
    elif harmful_blocks > beneficial_blocks:
        assessment = "harmful"

    informative_but_inert = bool(diagnostics_totals.get("conflict_total", 0) > 0 and changed_snapshots == 0)
    informative_but_destructive = bool(harmful_blocks > beneficial_blocks and diagnostics_totals.get("conflict_total", 0) > 0)
    concentrated_narrow_context = bool(
        changed_snapshots > 0
        and (
            len(per_horizon_deltas) <= 1
            or len(per_regime_deltas) <= 1
            or len(per_confidence_bucket_deltas) <= 1
        )
    )
    not_useful_in_window = bool(changed_snapshots == 0 and beneficial_blocks == 0)

    concentration = {
        "top_horizon": max(per_horizon_deltas.items(), key=lambda item: item[1].get("changed_trade_action", 0))[0]
        if per_horizon_deltas
        else None,
        "top_regime": max(per_regime_deltas.items(), key=lambda item: item[1].get("changed_snapshot", 0))[0]
        if per_regime_deltas
        else None,
        "top_confidence_bucket": max(
            per_confidence_bucket_deltas.items(), key=lambda item: item[1].get("changed_snapshot", 0)
        )[0]
        if per_confidence_bucket_deltas
        else None,
    }

    return {
        "snapshot_count": len(replay_results),
        "changed_snapshot_count": changed_snapshots,
        "changed_snapshot_ratio": round(changed_ratio, 6),
        "changed_selected_direction_count": changed_selected_direction_count,
        "changed_tradeable_count": changed_tradeable_count,
        "state_status_counts": state_status_counts,
        "selected_direction_delta": dict(sorted(selected_direction_delta.items())),
        "preferred_horizon_delta": dict(sorted(preferred_horizon_delta.items())),
        "entry_outcome_delta": dict(sorted(entry_outcome_delta.items())),
        "execution_state_reason_delta": dict(sorted(execution_state_reason_delta.items())),
        "per_horizon_deltas": dict(sorted(per_horizon_deltas.items())),
        "per_regime_deltas": dict(sorted(per_regime_deltas.items())),
        "per_confidence_bucket_deltas": dict(sorted(per_confidence_bucket_deltas.items())),
        "concentration": concentration,
        "diagnostics": diagnostics_totals,
        "beneficial_blocks": int(beneficial_blocks),
        "harmful_blocks": int(harmful_blocks),
        "quality_score": int(quality_score),
        "assessment": assessment,
        "diagnostic_interpretation": {
            "informative_but_too_inert": informative_but_inert,
            "informative_but_too_destructive_under_veto": informative_but_destructive,
            "concentrated_in_narrow_contexts": concentrated_narrow_context,
            "not_adding_useful_signal_in_window": not_useful_in_window,
        },
    }


def build_family_snapshot_delta_rows(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    replay_results: Sequence[FamilySnapshotReplayResult],
    family: str,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snap, result in zip(snapshots, replay_results):
        base = result.baseline_strategy
        shadow = result.shadow_strategy
        if base == shadow and not result.changed_horizons:
            continue
        signal = result.state.state_signal if family == "state_engineering" else result.state.order_flow_signal
        rows.append(
            {
                "generated_at": str(snap.get("generated_at") or ""),
                "feature_status": result.state.status,
                "feature_lag_hours": result.state.lag_hours,
                "family_bias": signal.bias,
                "family_score": signal.score,
                "baseline_preferred_horizon": base.get("preferred_horizon"),
                "shadow_preferred_horizon": shadow.get("preferred_horizon"),
                "baseline_selected_direction": base.get("selected_direction"),
                "shadow_selected_direction": shadow.get("selected_direction"),
                "baseline_execution_state": base.get("execution_state"),
                "shadow_execution_state": shadow.get("execution_state"),
                "baseline_reason": base.get("reason"),
                "shadow_reason": shadow.get("reason"),
                "changed_horizons": result.changed_horizons,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def run_family_policy_sweep(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    feature_frame: pd.DataFrame,
    family: str,
    policies: Sequence[FamilyShadowPolicy],
    max_staleness_hours: float,
) -> Dict[str, Any]:
    variants: List[Dict[str, Any]] = []

    for policy in policies:
        replay_results: List[FamilySnapshotReplayResult] = []
        replay_snapshots: List[Mapping[str, Any]] = []
        for snapshot in snapshots:
            snap_ts = pd.to_datetime(snapshot.get("generated_at"), utc=True, errors="coerce")
            if pd.isna(snap_ts):
                continue
            state = resolve_family_snapshot_state(
                snapshot_ts=snap_ts,
                feature_frame=feature_frame,
                max_staleness_hours=max_staleness_hours,
            )
            replay = replay_snapshot_with_family_shadow(
                snapshot,
                state=state,
                family=family,
                policy=policy,
            )
            replay_results.append(replay)
            replay_snapshots.append(snapshot)

        summary = summarize_family_replay(
            snapshots=replay_snapshots,
            replay_results=replay_results,
        )
        delta_rows = build_family_snapshot_delta_rows(
            snapshots=replay_snapshots,
            replay_results=replay_results,
            family=family,
            limit=300,
        )
        rank_score = (
            2.0 * float(summary.get("beneficial_blocks", 0))
            - 2.5 * float(summary.get("harmful_blocks", 0))
            + 0.5 * float(summary.get("changed_snapshot_count", 0))
        )
        variants.append(
            {
                "policy": {
                    "name": policy.name,
                    "description": policy.description,
                    "enforcement_mode": policy.enforcement_mode,
                    "enabled_horizons": [_to_horizon_label(v) for v in policy.enabled_horizons],
                    "enabled_regimes": list(policy.enabled_regimes),
                    "confidence_band_min": policy.confidence_band_min,
                    "confidence_band_max": policy.confidence_band_max,
                    "min_horizon_hours": policy.min_horizon_hours,
                    "min_signal_strength": policy.min_signal_strength,
                    "confluence_disagreement_threshold": policy.confluence_disagreement_threshold,
                },
                "summary": summary,
                "snapshot_deltas": delta_rows,
                "rank_score": rank_score,
            }
        )

    variants.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
    best = variants[0] if variants else None
    best_assessment = str(best.get("summary", {}).get("assessment", "neutral")) if isinstance(best, Mapping) else "neutral"
    advance = bool(
        isinstance(best, Mapping)
        and best_assessment == "beneficial"
        and int(best.get("summary", {}).get("beneficial_blocks", 0)) > int(best.get("summary", {}).get("harmful_blocks", 0))
    )
    disposition = "hold_for_targeted_research"
    if advance:
        disposition = "go_to_deeper_validation"
    elif isinstance(best, Mapping) and int(best.get("summary", {}).get("changed_snapshot_count", 0)) == 0:
        disposition = "deprioritize_for_now"

    return {
        "family": family,
        "variant_count": len(variants),
        "variant_rankings": [
            {
                "rank": idx + 1,
                "policy": row.get("policy", {}).get("name"),
                "rank_score": row.get("rank_score"),
                "assessment": row.get("summary", {}).get("assessment"),
                "changed_snapshot_count": row.get("summary", {}).get("changed_snapshot_count"),
                "beneficial_blocks": row.get("summary", {}).get("beneficial_blocks"),
                "harmful_blocks": row.get("summary", {}).get("harmful_blocks"),
            }
            for idx, row in enumerate(variants)
        ],
        "variants": variants,
        "recommendation": {
            "best_policy": best.get("policy", {}).get("name") if isinstance(best, Mapping) else None,
            "best_assessment": best_assessment,
            "advance_to_next_validation_stage": advance,
            "family_disposition": disposition,
        },
    }


def run_state_order_flow_shadow_validation(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    feature_frame: pd.DataFrame,
    policies: Sequence[FamilyShadowPolicy],
    max_staleness_hours: float,
) -> Dict[str, Any]:
    state_result = run_family_policy_sweep(
        snapshots=snapshots,
        feature_frame=feature_frame,
        family="state_engineering",
        policies=policies,
        max_staleness_hours=max_staleness_hours,
    )
    order_result = run_family_policy_sweep(
        snapshots=snapshots,
        feature_frame=feature_frame,
        family="order_flow",
        policies=policies,
        max_staleness_hours=max_staleness_hours,
    )

    family_rankings = []
    for item in [state_result, order_result]:
        rec = item.get("recommendation", {}) if isinstance(item, Mapping) else {}
        top = item.get("variant_rankings", [None])[0] if isinstance(item.get("variant_rankings"), list) else None
        family_rankings.append(
            {
                "family": item.get("family"),
                "best_policy": rec.get("best_policy"),
                "best_assessment": rec.get("best_assessment"),
                "family_disposition": rec.get("family_disposition"),
                "rank_score": top.get("rank_score") if isinstance(top, Mapping) else 0.0,
            }
        )

    family_rankings.sort(key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)
    best_family = family_rankings[0] if family_rankings else None

    return {
        "family_rankings": family_rankings,
        "families": {
            "state_engineering": state_result,
            "order_flow": order_result,
        },
        "overall_recommendation": {
            "best_family": best_family.get("family") if isinstance(best_family, Mapping) else None,
            "best_policy": best_family.get("best_policy") if isinstance(best_family, Mapping) else None,
            "advance_to_deeper_validation": bool(
                isinstance(best_family, Mapping)
                and any(
                    item.get("recommendation", {}).get("advance_to_next_validation_stage")
                    for item in [state_result, order_result]
                    if isinstance(item, Mapping)
                )
            ),
            "macro_disposition": "remain_deprioritized",
        },
    }


def render_family_shadow_markdown_report(payload: Mapping[str, Any]) -> str:
    sweep = payload.get("sweep", {}) if isinstance(payload.get("sweep"), Mapping) else {}
    families = sweep.get("families", {}) if isinstance(sweep.get("families"), Mapping) else {}
    rankings = sweep.get("family_rankings", []) if isinstance(sweep.get("family_rankings"), list) else []
    recommendation = sweep.get("overall_recommendation", {}) if isinstance(sweep.get("overall_recommendation"), Mapping) else {}

    lines: List[str] = []
    lines.append("# State/Order-Flow Shadow Policy Sweep")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Shadow-only replay for state_engineering and order_flow families.")
    lines.append("- No production inference-path changes or live config promotion.")
    lines.append("")

    lines.append("## Family Ranking")
    lines.append("| Rank | Family | Best Policy | Assessment | Disposition | Score |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    for idx, row in enumerate(rankings, start=1):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {rank} | {family} | {policy} | {assessment} | {disp} | {score:.2f} |".format(
                rank=idx,
                family=str(row.get("family", "unknown")),
                policy=str(row.get("best_policy", "n/a")),
                assessment=str(row.get("best_assessment", "unknown")),
                disp=str(row.get("family_disposition", "n/a")),
                score=float(row.get("rank_score", 0.0)),
            )
        )
    lines.append("")

    for family_name in ["state_engineering", "order_flow"]:
        family_payload = families.get(family_name, {}) if isinstance(families.get(family_name), Mapping) else {}
        variant_rankings = family_payload.get("variant_rankings", []) if isinstance(family_payload.get("variant_rankings"), list) else []
        best_variant = family_payload.get("variants", [None])[0] if isinstance(family_payload.get("variants"), list) and family_payload.get("variants") else None
        best_summary = best_variant.get("summary", {}) if isinstance(best_variant, Mapping) and isinstance(best_variant.get("summary"), Mapping) else {}
        interpretation = best_summary.get("diagnostic_interpretation", {}) if isinstance(best_summary.get("diagnostic_interpretation"), Mapping) else {}

        lines.append(f"## {family_name} Details")
        lines.append("| Rank | Policy | Assessment | Changed | Beneficial | Harmful | Score |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in variant_rankings[:6]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {rank} | {policy} | {assessment} | {changed} | {beneficial} | {harmful} | {score:.2f} |".format(
                    rank=int(row.get("rank", 0)),
                    policy=str(row.get("policy", "unknown")),
                    assessment=str(row.get("assessment", "unknown")),
                    changed=int(row.get("changed_snapshot_count", 0)),
                    beneficial=int(row.get("beneficial_blocks", 0)),
                    harmful=int(row.get("harmful_blocks", 0)),
                    score=float(row.get("rank_score", 0.0)),
                )
            )
        lines.append("")
        lines.append("- Informative but too inert: {0}".format(bool(interpretation.get("informative_but_too_inert", False))))
        lines.append("- Informative but too destructive under veto logic: {0}".format(bool(interpretation.get("informative_but_too_destructive_under_veto", False))))
        lines.append("- Concentrated in narrow contexts: {0}".format(bool(interpretation.get("concentrated_in_narrow_contexts", False))))
        lines.append("- Not adding useful signal in tested window: {0}".format(bool(interpretation.get("not_adding_useful_signal_in_window", False))))
        lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- Best family: {recommendation.get('best_family')}")
    lines.append(f"- Best policy: {recommendation.get('best_policy')}")
    lines.append(f"- Advance to deeper validation: {recommendation.get('advance_to_deeper_validation')}")
    lines.append(f"- Macro status: {recommendation.get('macro_disposition')}")
    lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "ALL_FAMILY_FEATURES",
    "FamilyShadowPolicy",
    "STATE_ENGINEERING_FEATURES",
    "ORDER_FLOW_FEATURES",
    "default_family_policy_variants",
    "load_prediction_history",
    "load_spot_feature_frame",
    "render_family_shadow_markdown_report",
    "run_state_order_flow_shadow_validation",
]
