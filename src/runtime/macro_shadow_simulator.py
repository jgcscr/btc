from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import pandas as pd


MACRO_FEATURES_USED: tuple[str, ...] = (
    "macro_dollar_proxy_change_1d",
    "macro_us10y_change_1d",
    "macro_eurusd_change_1d",
)

READY_STATES: tuple[str, ...] = ("ready", "waiting_pullback", "bias_only_ready")
CONF_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.45),
    ("mid", 0.45, 0.65),
    ("high", 0.65, 1.01),
)


@dataclass(frozen=True)
class MacroShadowPolicy:
    name: str = "baseline_existing_rule"
    description: str = "Block weak macro-conflicting trades on horizons >=1h"
    min_horizon_hours: float = 1.0
    max_staleness_hours: float = 24.0
    enforcement_mode: str = "block_weak_conflict"
    enabled_horizons: tuple[float, ...] = ()
    enabled_regimes: tuple[str, ...] = ()
    confidence_band_min: float | None = None
    confidence_band_max: float | None = None
    block_conflict_confidence_max: float = 0.7
    weak_trade_expected_value_max: float = 0.001
    strong_trade_expected_value_min: float = 0.003
    strong_trade_confidence_min: float = 0.75


@dataclass(frozen=True)
class MacroSnapshotState:
    status: str
    asof_ts: str | None
    lag_hours: float | None
    values: Dict[str, float]
    macro_bias: str
    score: int


@dataclass(frozen=True)
class SnapshotReplayResult:
    baseline_strategy: Dict[str, Any]
    shadow_strategy: Dict[str, Any]
    baseline_predictions: Dict[str, Dict[str, Any]]
    shadow_predictions: Dict[str, Dict[str, Any]]
    macro_state: MacroSnapshotState
    changed_horizons: List[str]
    beneficial_blocks: int
    harmful_blocks: int
    changed_regimes: List[str]
    diagnostics: Dict[str, int]


def _to_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _direction_from_trade_action(action: str) -> str:
    normalized = str(action).strip().lower()
    if normalized == "long":
        return "long"
    if normalized == "short":
        return "short"
    return "neutral"


def _horizon_to_float(label: str) -> float | None:
    text = str(label).strip().lower()
    try:
        if text.endswith("h"):
            return float(text[:-1])
        if text.endswith("m"):
            return float(text[:-1]) / 60.0
        return float(text)
    except ValueError:
        return None


def _to_horizon_label(value: float) -> str:
    if value >= 1.0 and float(value).is_integer():
        return f"{int(value)}h"
    if value >= 1.0:
        return f"{value:g}h"
    minutes = int(round(value * 60))
    return f"{minutes}m"


def _normalize_regime(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    for label, low, high in CONF_BUCKETS:
        if low <= value < high:
            return label
    return "unknown"


def _deepcopy_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def default_policy_variants(max_staleness_hours: float) -> List[MacroShadowPolicy]:
    return [
        MacroShadowPolicy(
            name="baseline_existing_rule",
            description="Block weak macro-conflicting trades on horizons >=1h",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="block_weak_conflict",
        ),
        MacroShadowPolicy(
            name="strict_macro_veto_only",
            description="Always veto macro-conflicting trades in scope",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="strict_veto_conflict",
        ),
        MacroShadowPolicy(
            name="macro_bias_override_weak",
            description="Override weak conflicting trades to macro bias",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="bias_override_weak",
        ),
        MacroShadowPolicy(
            name="macro_mid_long_only",
            description="Apply macro rule only on 4h/8h/12h",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="block_weak_conflict",
            enabled_horizons=(4.0, 8.0, 12.0),
        ),
        MacroShadowPolicy(
            name="macro_neutral_chop_only",
            description="Apply macro rule only in neutral or chop regimes",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="block_weak_conflict",
            enabled_regimes=("neutral", "chop"),
        ),
        MacroShadowPolicy(
            name="macro_low_confidence_band",
            description="Apply macro rule only for low-confidence trades",
            max_staleness_hours=max_staleness_hours,
            enforcement_mode="block_weak_conflict",
            confidence_band_max=0.58,
        ),
    ]


def load_prediction_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def load_macro_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts", *MACRO_FEATURES_USED])
    frame = pd.read_parquet(path)
    if "ts" not in frame.columns:
        return pd.DataFrame(columns=["ts", *MACRO_FEATURES_USED])
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset="ts", keep="last")
    for column in MACRO_FEATURES_USED:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["ts", *MACRO_FEATURES_USED]].reset_index(drop=True)


def resolve_macro_state(
    *,
    snapshot_ts: pd.Timestamp,
    macro_frame: pd.DataFrame,
    policy: MacroShadowPolicy,
) -> MacroSnapshotState:
    if macro_frame.empty:
        return MacroSnapshotState(
            status="unavailable",
            asof_ts=None,
            lag_hours=None,
            values={},
            macro_bias="neutral",
            score=0,
        )

    row = macro_frame[macro_frame["ts"] <= snapshot_ts].tail(1)
    if row.empty:
        return MacroSnapshotState(
            status="unavailable",
            asof_ts=None,
            lag_hours=None,
            values={},
            macro_bias="neutral",
            score=0,
        )

    macro_ts = pd.Timestamp(row.iloc[0]["ts"])
    lag_hours = max((snapshot_ts - macro_ts).total_seconds() / 3600.0, 0.0)
    values = {
        column: _to_float_or_none(row.iloc[0][column])
        for column in MACRO_FEATURES_USED
    }

    status = "available"
    if lag_hours > policy.max_staleness_hours:
        status = "stale"

    score = 0
    # Risk-on proxy: USD down, UST10Y down, EURUSD up.
    dollar_change = values.get("macro_dollar_proxy_change_1d")
    if dollar_change is not None:
        score += 1 if dollar_change < 0 else -1 if dollar_change > 0 else 0
    us10y_change = values.get("macro_us10y_change_1d")
    if us10y_change is not None:
        score += 1 if us10y_change < 0 else -1 if us10y_change > 0 else 0
    eurusd_change = values.get("macro_eurusd_change_1d")
    if eurusd_change is not None:
        score += 1 if eurusd_change > 0 else -1 if eurusd_change < 0 else 0

    if score >= 2:
        bias = "long"
    elif score <= -2:
        bias = "short"
    else:
        bias = "neutral"

    return MacroSnapshotState(
        status=status,
        asof_ts=macro_ts.isoformat(),
        lag_hours=float(lag_hours),
        values={k: float(v) for k, v in values.items() if v is not None},
        macro_bias=bias,
        score=int(score),
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


def _is_weak_trade(entry: Mapping[str, Any], policy: MacroShadowPolicy) -> bool:
    expected_value = _to_float_or_none(entry.get("expected_value"))
    confidence = _to_float_or_none(entry.get("confidence_score"))
    if expected_value is None:
        expected_value = 0.0
    if confidence is None:
        confidence = 0.0
    return confidence <= policy.block_conflict_confidence_max or expected_value <= policy.weak_trade_expected_value_max


def _is_strong_trade(entry: Mapping[str, Any], policy: MacroShadowPolicy) -> bool:
    expected_value = _to_float_or_none(entry.get("expected_value"))
    confidence = _to_float_or_none(entry.get("confidence_score"))
    if expected_value is None or confidence is None:
        return False
    return expected_value >= policy.strong_trade_expected_value_min and confidence >= policy.strong_trade_confidence_min


def _policy_applies_to_entry(
    *,
    entry: Mapping[str, Any],
    horizon: float,
    policy: MacroShadowPolicy,
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


def replay_snapshot_with_macro_shadow(
    snapshot: Mapping[str, Any],
    *,
    macro_state: MacroSnapshotState,
    policy: MacroShadowPolicy,
) -> SnapshotReplayResult:
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
        "macro_neutral": 0,
        "macro_stale": 0,
        "macro_unavailable": 0,
        "conflict_total": 0,
        "conflict_weak": 0,
        "conflict_strong": 0,
        "scope_horizon_out": 0,
        "scope_regime_out": 0,
        "scope_confidence_out": 0,
    }

    for label, entry in list(shadow_predictions.items()):
        horizon = _horizon_to_float(label)
        if horizon is None:
            continue

        applies, scope_reason = _policy_applies_to_entry(entry=entry, horizon=horizon, policy=policy)
        if not applies:
            if scope_reason == "horizon_out_of_scope" or scope_reason == "below_min_horizon":
                diagnostics["scope_horizon_out"] += 1
            elif scope_reason == "regime_out_of_scope":
                diagnostics["scope_regime_out"] += 1
            elif scope_reason == "confidence_out_of_scope":
                diagnostics["scope_confidence_out"] += 1
            continue

        baseline_entry = baseline_predictions.get(label, {})
        action = str(entry.get("trade_action") or "hold").lower()
        side = _direction_from_trade_action(action)
        if side == "neutral":
            continue

        shadow_note: Dict[str, Any] = {
            "applied": False,
            "reason": "no_change",
            "macro_status": macro_state.status,
            "macro_bias": macro_state.macro_bias,
            "macro_score": macro_state.score,
        }

        if macro_state.status in {"unavailable", "stale"}:
            diagnostics[f"macro_{macro_state.status}"] += 1
            shadow_note["reason"] = f"macro_{macro_state.status}_fail_open"
            entry["macro_shadow"] = shadow_note
            continue

        if macro_state.macro_bias == "neutral":
            diagnostics["macro_neutral"] += 1
            shadow_note["reason"] = "macro_neutral"
            entry["macro_shadow"] = shadow_note
            continue

        if side != macro_state.macro_bias:
            diagnostics["conflict_total"] += 1
            if _is_weak_trade(entry, policy):
                diagnostics["conflict_weak"] += 1
                if policy.enforcement_mode == "bias_override_weak":
                    entry["trade_action"] = macro_state.macro_bias
                    entry["direction_next"] = "up" if macro_state.macro_bias == "long" else "down"
                    entry["signal_ensemble"] = 1
                    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), MutableMapping) else {}
                    if str(plan.get("status") or "") == "rejected":
                        plan["status"] = "ready"
                    plan["reason"] = "macro_shadow_bias_override_weak"
                    entry["execution_plan"] = plan
                    shadow_note["applied"] = True
                    shadow_note["reason"] = "macro_conflict_overridden"
                    changed_horizons.append(label)
                    changed_regimes.append(_normalize_regime(entry.get("regime_state")))
                else:
                    entry["trade_action"] = "hold"
                    entry["signal_ensemble"] = 0
                    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), MutableMapping) else {}
                    plan["status"] = "rejected"
                    plan["reason"] = "macro_shadow_conflict_block"
                    entry["execution_plan"] = plan
                    shadow_note["applied"] = True
                    shadow_note["reason"] = "macro_conflict_blocked"
                    changed_horizons.append(label)
                    changed_regimes.append(_normalize_regime(entry.get("regime_state")))
                    beneficial_blocks += 1
            else:
                diagnostics["conflict_strong"] += 1
                if policy.enforcement_mode == "strict_veto_conflict":
                    entry["trade_action"] = "hold"
                    entry["signal_ensemble"] = 0
                    plan = entry.get("execution_plan") if isinstance(entry.get("execution_plan"), MutableMapping) else {}
                    plan["status"] = "rejected"
                    plan["reason"] = "macro_shadow_strict_veto"
                    entry["execution_plan"] = plan
                    shadow_note["applied"] = True
                    shadow_note["reason"] = "macro_conflict_strict_blocked"
                    changed_horizons.append(label)
                    changed_regimes.append(_normalize_regime(entry.get("regime_state")))
                    if _is_strong_trade(baseline_entry, policy):
                        harmful_blocks += 1
                else:
                    shadow_note["reason"] = "macro_conflict_no_block_strong_trade"
        else:
            shadow_note["reason"] = "macro_aligned"

        entry["macro_shadow"] = shadow_note

    baseline_strategy = _baseline_strategy(snapshot)
    if changed_horizons:
        shadow_strategy = _pick_strategy_from_predictions(shadow_predictions)
    else:
        shadow_strategy = dict(baseline_strategy)

    return SnapshotReplayResult(
        baseline_strategy=baseline_strategy,
        shadow_strategy=shadow_strategy,
        baseline_predictions=baseline_predictions,
        shadow_predictions=shadow_predictions,
        macro_state=macro_state,
        changed_horizons=sorted(set(changed_horizons)),
        beneficial_blocks=beneficial_blocks,
        harmful_blocks=harmful_blocks,
        changed_regimes=sorted(set(changed_regimes)),
        diagnostics=diagnostics,
    )


def _increment(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def summarize_replay_results(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    replay_results: Sequence[SnapshotReplayResult],
) -> Dict[str, Any]:
    selected_direction_delta: Dict[str, int] = {}
    execution_state_reason_delta: Dict[str, int] = {}
    preferred_horizon_delta: Dict[str, int] = {}
    entry_outcome_delta: Dict[str, int] = {}
    per_horizon_deltas: Dict[str, Dict[str, int]] = {}
    per_regime_deltas: Dict[str, Dict[str, int]] = {}
    per_confidence_bucket_deltas: Dict[str, Dict[str, int]] = {}
    baseline_execution_state_deltas: Dict[str, int] = {}
    bias_shift_delta: Dict[str, int] = {"long_to_short": 0, "short_to_long": 0, "to_neutral": 0, "from_neutral": 0}
    diagnostics_totals: Dict[str, int] = {}

    changed_snapshots = 0
    beneficial_blocks = 0
    harmful_blocks = 0
    macro_status_counts: Dict[str, int] = {}

    for snap, result in zip(snapshots, replay_results):
        beneficial_blocks += result.beneficial_blocks
        harmful_blocks += result.harmful_blocks
        _increment(macro_status_counts, result.macro_state.status)
        for key, value in result.diagnostics.items():
            diagnostics_totals[key] = diagnostics_totals.get(key, 0) + int(value)

        base = result.baseline_strategy
        shadow = result.shadow_strategy

        base_dir = str(base.get("selected_direction") or "Neutral")
        shadow_dir = str(shadow.get("selected_direction") or "Neutral")
        if base_dir != shadow_dir:
            changed_snapshots += 1
            _increment(selected_direction_delta, f"{base_dir}->{shadow_dir}")
            _increment(baseline_execution_state_deltas, str(base.get("execution_state") or "unknown"))

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

            b = base_dir.lower()
            s = shadow_dir.lower()
            if b == "long" and s == "short":
                bias_shift_delta["long_to_short"] += 1
            elif b == "short" and s == "long":
                bias_shift_delta["short_to_long"] += 1
            elif s == "neutral" and b in {"long", "short"}:
                bias_shift_delta["to_neutral"] += 1
            elif b == "neutral" and s in {"long", "short"}:
                bias_shift_delta["from_neutral"] += 1

        base_tradeable = bool(base.get("tradeable", False))
        shadow_tradeable = bool(shadow.get("tradeable", False))
        if base_tradeable != shadow_tradeable:
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
            _increment(
                execution_state_reason_delta,
                f"{base_state}:{base_reason}->{shadow_state}:{shadow_reason}",
            )

        for horizon_label in result.changed_horizons:
            per_h = per_horizon_deltas.setdefault(horizon_label, {"changed_trade_action": 0, "blocked": 0})
            per_h["changed_trade_action"] += 1
            per_h["blocked"] += 1

    total = max(len(replay_results), 1)
    changed_ratio = changed_snapshots / total

    quality_score = beneficial_blocks - harmful_blocks
    classification = "neutral"
    if quality_score >= 20 and changed_ratio >= 0.05:
        classification = "beneficial"
    elif harmful_blocks > beneficial_blocks:
        classification = "harmful"

    return {
        "snapshot_count": len(replay_results),
        "changed_snapshot_count": changed_snapshots,
        "changed_snapshot_ratio": round(changed_ratio, 6),
        "macro_status_counts": macro_status_counts,
        "selected_direction_delta": dict(sorted(selected_direction_delta.items())),
        "preferred_horizon_delta": dict(sorted(preferred_horizon_delta.items())),
        "entry_outcome_delta": dict(sorted(entry_outcome_delta.items())),
        "execution_state_reason_delta": dict(sorted(execution_state_reason_delta.items())),
        "per_horizon_deltas": dict(sorted(per_horizon_deltas.items())),
        "per_regime_deltas": dict(sorted(per_regime_deltas.items())),
        "per_confidence_bucket_deltas": dict(sorted(per_confidence_bucket_deltas.items())),
        "baseline_execution_state_deltas": dict(sorted(baseline_execution_state_deltas.items())),
        "bias_shift_delta": bias_shift_delta,
        "diagnostics": diagnostics_totals,
        "beneficial_blocks": int(beneficial_blocks),
        "harmful_blocks": int(harmful_blocks),
        "quality_score": int(quality_score),
        "assessment": classification,
    }


def run_policy_sweep(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    macro_frame: pd.DataFrame,
    policies: Sequence[MacroShadowPolicy],
) -> Dict[str, Any]:
    variant_results: List[Dict[str, Any]] = []

    for policy in policies:
        replay_results: List[SnapshotReplayResult] = []
        replay_snapshots: List[Mapping[str, Any]] = []
        for snapshot in snapshots:
            snap_ts = pd.to_datetime(snapshot.get("generated_at"), utc=True, errors="coerce")
            if pd.isna(snap_ts):
                continue
            macro_state = resolve_macro_state(
                snapshot_ts=snap_ts,
                macro_frame=macro_frame,
                policy=policy,
            )
            replay = replay_snapshot_with_macro_shadow(
                snapshot,
                macro_state=macro_state,
                policy=policy,
            )
            replay_results.append(replay)
            replay_snapshots.append(snapshot)

        summary = summarize_replay_results(
            snapshots=replay_snapshots,
            replay_results=replay_results,
        )
        rows = build_snapshot_delta_rows(
            snapshots=replay_snapshots,
            replay_results=replay_results,
            limit=300,
        )
        rank_score = (
            2.0 * float(summary.get("beneficial_blocks", 0))
            - 2.5 * float(summary.get("harmful_blocks", 0))
            + 0.5 * float(summary.get("changed_snapshot_count", 0))
        )
        variant_results.append(
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
                    "max_staleness_hours": policy.max_staleness_hours,
                },
                "summary": summary,
                "snapshot_deltas": rows,
                "rank_score": rank_score,
            }
        )

    variant_results.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)

    first_pass = next((row for row in variant_results if row.get("policy", {}).get("name") == "baseline_existing_rule"), None)
    first_pass_diag = first_pass.get("summary", {}).get("diagnostics", {}) if isinstance(first_pass, Mapping) else {}
    first_pass_reasoning = {
        "macro_bias_rarely_conflicting": bool(int(first_pass_diag.get("conflict_total", 0)) == 0),
        "enforcement_too_conservative": bool(
            int(first_pass_diag.get("conflict_total", 0)) > 0
            and int(first_pass_diag.get("conflict_weak", 0)) == 0
        ),
        "horizons_or_regimes_out_of_scope": bool(
            int(first_pass_diag.get("scope_horizon_out", 0)) + int(first_pass_diag.get("scope_regime_out", 0)) > 0
        ),
        "recent_slice_lacked_macro_conditions": bool(
            int(first_pass_diag.get("macro_neutral", 0)) >= int(first_pass.get("summary", {}).get("snapshot_count", 0) * 0.6)
            if isinstance(first_pass, Mapping)
            else False
        ),
    }

    best = variant_results[0] if variant_results else None
    best_assessment = str(best.get("summary", {}).get("assessment", "neutral")) if isinstance(best, Mapping) else "neutral"
    next_stage = bool(
        isinstance(best, Mapping)
        and best_assessment == "beneficial"
        and int(best.get("summary", {}).get("beneficial_blocks", 0)) > int(best.get("summary", {}).get("harmful_blocks", 0))
    )

    recommendation = "held_for_more_research"
    if next_stage:
        recommendation = "advance_to_next_validation_stage"
    elif isinstance(best, Mapping) and int(best.get("summary", {}).get("changed_snapshot_count", 0)) == 0:
        recommendation = "deprioritize_for_now"

    return {
        "variant_count": len(variant_results),
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
            for idx, row in enumerate(variant_results)
        ],
        "variants": variant_results,
        "first_pass_diagnosis": first_pass_reasoning,
        "recommendation": {
            "best_policy": best.get("policy", {}).get("name") if isinstance(best, Mapping) else None,
            "best_assessment": best_assessment,
            "advance_to_next_validation_stage": next_stage,
            "macro_disposition": recommendation,
        },
    }


def build_snapshot_delta_rows(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    replay_results: Sequence[SnapshotReplayResult],
    limit: int = 200,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snap, result in zip(snapshots, replay_results):
        base = result.baseline_strategy
        shadow = result.shadow_strategy
        if base == shadow and not result.changed_horizons:
            continue
        rows.append(
            {
                "generated_at": str(snap.get("generated_at") or ""),
                "macro_status": result.macro_state.status,
                "macro_bias": result.macro_state.macro_bias,
                "macro_score": result.macro_state.score,
                "macro_lag_hours": result.macro_state.lag_hours,
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


def render_shadow_markdown_report(payload: Mapping[str, Any]) -> str:
    sweep = payload.get("sweep", {}) if isinstance(payload.get("sweep"), Mapping) else None
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}

    if sweep and isinstance(sweep, Mapping):
        rankings = sweep.get("variant_rankings", []) if isinstance(sweep.get("variant_rankings"), list) else []
        recommendation = sweep.get("recommendation", {}) if isinstance(sweep.get("recommendation"), Mapping) else {}
        first_pass_diag = sweep.get("first_pass_diagnosis", {}) if isinstance(sweep.get("first_pass_diagnosis"), Mapping) else {}

        lines: List[str] = []
        lines.append("# Macro Shadow Policy Sweep")
        lines.append("")
        lines.append("## Scope")
        lines.append("- Shadow-only replay with explicit macro policy variants.")
        lines.append("- Production inference path/config remains unchanged.")
        lines.append("")
        lines.append("## Why First Pass Was Neutral")
        lines.append(f"- Macro bias rarely conflicting: {first_pass_diag.get('macro_bias_rarely_conflicting', False)}")
        lines.append(f"- Enforcement too conservative: {first_pass_diag.get('enforcement_too_conservative', False)}")
        lines.append(f"- Horizons/regimes out of scope: {first_pass_diag.get('horizons_or_regimes_out_of_scope', False)}")
        lines.append(f"- Recent slice lacked macro conditions: {first_pass_diag.get('recent_slice_lacked_macro_conditions', False)}")
        lines.append("")
        lines.append("## Variant Ranking")
        lines.append("| Rank | Policy | Assessment | Changed | Beneficial | Harmful | Score |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in rankings:
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
        lines.append("## Recommendation")
        lines.append(f"- Best policy: {recommendation.get('best_policy')}")
        lines.append(f"- Best assessment: {recommendation.get('best_assessment')}")
        lines.append(f"- Advance to next validation stage: {recommendation.get('advance_to_next_validation_stage')}")
        lines.append(f"- Macro disposition: {recommendation.get('macro_disposition')}")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines: List[str] = []
    lines.append("# Macro Shadow Enforcement Replay")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Shadow-only replay; production decision path unchanged.")
    lines.append("- Baseline truth standard references leakage-safe rerun context.")
    lines.append("")

    lines.append("## Policy")
    lines.append("- Macro features used: macro_dollar_proxy_change_1d, macro_us10y_change_1d, macro_eurusd_change_1d")
    lines.append("- Macro bias rule: >=2 aligned risk-on signals => long bias, <=-2 => short bias, else neutral")
    lines.append("- Enforcement rule: for horizons >= 1h, block weak trades that conflict with macro bias")
    lines.append("- Fail-open: unavailable/stale macro data does not block; it is counted and reported")
    lines.append("")

    lines.append("## Outcome")
    lines.append(f"- Assessment: {summary.get('assessment', 'unknown')}")
    lines.append(f"- Snapshots replayed: {summary.get('snapshot_count', 0)}")
    lines.append(f"- Changed snapshots: {summary.get('changed_snapshot_count', 0)}")
    lines.append(f"- Beneficial blocks: {summary.get('beneficial_blocks', 0)}")
    lines.append(f"- Harmful blocks: {summary.get('harmful_blocks', 0)}")
    lines.append("")

    lines.append("## Key Deltas")
    lines.append(f"- Selected direction deltas: {summary.get('selected_direction_delta', {})}")
    lines.append(f"- Preferred horizon deltas: {summary.get('preferred_horizon_delta', {})}")
    lines.append(f"- Entry outcome deltas: {summary.get('entry_outcome_delta', {})}")
    lines.append(f"- Bias shift deltas: {summary.get('bias_shift_delta', {})}")
    lines.append("")

    recommendation = "reject_for_now"
    assessment = str(summary.get("assessment") or "neutral")
    if assessment == "beneficial":
        recommendation = "advance_to_next_validation_stage"
    elif assessment == "neutral":
        recommendation = "more_targeted_testing"

    lines.append("## Recommendation")
    lines.append(f"- Recommended next action: {recommendation}")
    lines.append("- Conservative interpretation: treat this as screening evidence, not proof of alpha.")
    lines.append("")

    return "\n".join(lines) + "\n"
