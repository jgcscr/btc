from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.runtime.family_shadow_simulator import (
    FamilyShadowPolicy,
    _confidence_bucket,
    _normalize_regime,
    _to_float_or_none,
    default_family_policy_variants,
    load_prediction_history,
    replay_snapshot_with_family_shadow,
    resolve_family_snapshot_state,
)

TARGET_HORIZONS: tuple[str, ...] = ("4h", "8h", "12h")


@dataclass(frozen=True)
class DecisionOutcome:
    has_trade: bool
    direction: str
    horizon: str | None
    signed_return: float | None
    favorable_excursion: float | None
    adverse_move: float | None
    direction_accuracy: float | None


def _format_horizon_label(horizon: float) -> str:
    if float(horizon).is_integer() and horizon >= 1.0:
        return f"{int(round(horizon))}h"
    if horizon < 1.0:
        return f"{int(round(horizon * 60))}m"
    return f"{horizon:g}h"


def _parse_horizon_hours(horizon: str) -> float:
    value = str(horizon).strip().lower()
    if value.endswith("h"):
        return float(value[:-1])
    if value.endswith("m"):
        return float(value[:-1]) / 60.0
    return float(value)


def _to_trade_direction(selected_direction: Any) -> str:
    text = str(selected_direction or "").strip().lower()
    if text == "long":
        return "long"
    if text == "short":
        return "short"
    return "neutral"


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.nanmean(arr))


def load_spot_ohlcv_with_outcomes(spot_dir: Path, horizons: Sequence[str]) -> pd.DataFrame:
    files = sorted(spot_dir.glob("*.parquet"))
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frame = pd.read_parquet(file)
        except Exception:
            continue
        if {"ts", "close", "high", "low"}.issubset(frame.columns):
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No spot OHLCV parquet with ts/close/high/low found under {spot_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out = out.dropna(subset=["ts", "close", "high", "low"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    out = out.reset_index(drop=True)
    out["ts_hour"] = out["ts"].dt.floor("h")

    for horizon in horizons:
        steps = int(round(_parse_horizon_hours(horizon)))
        if steps <= 0:
            continue
        label = _format_horizon_label(_parse_horizon_hours(horizon))
        close_next_col = f"close_next_{label}"
        ret_col = f"ret_{label}_realized"
        high_fwd_col = f"high_fwd_{label}"
        low_fwd_col = f"low_fwd_{label}"

        out[close_next_col] = out["close"].shift(-steps)
        out[ret_col] = (out[close_next_col] / out["close"]) - 1.0

        high_windows = []
        low_windows = []
        highs = out["high"].to_numpy(dtype=float)
        lows = out["low"].to_numpy(dtype=float)
        n = len(out)
        for idx in range(n):
            start = idx + 1
            end = idx + 1 + steps
            if end <= n and start < end:
                high_windows.append(float(np.nanmax(highs[start:end])))
                low_windows.append(float(np.nanmin(lows[start:end])))
            else:
                high_windows.append(np.nan)
                low_windows.append(np.nan)
        out[high_fwd_col] = high_windows
        out[low_fwd_col] = low_windows

    return out


def _build_outcome_lookup(ohlcv_with_outcomes: pd.DataFrame, horizons: Sequence[str]) -> Dict[tuple[pd.Timestamp, str], Dict[str, float]]:
    lookup: Dict[tuple[pd.Timestamp, str], Dict[str, float]] = {}
    for _, row in ohlcv_with_outcomes.iterrows():
        ts_hour = pd.Timestamp(row["ts_hour"])
        entry_close = _to_float_or_none(row.get("close"))
        if entry_close is None or entry_close <= 0:
            continue
        for horizon in horizons:
            label = _format_horizon_label(_parse_horizon_hours(horizon))
            close_next = _to_float_or_none(row.get(f"close_next_{label}"))
            ret_realized = _to_float_or_none(row.get(f"ret_{label}_realized"))
            high_fwd = _to_float_or_none(row.get(f"high_fwd_{label}"))
            low_fwd = _to_float_or_none(row.get(f"low_fwd_{label}"))
            if close_next is None or ret_realized is None or high_fwd is None or low_fwd is None:
                continue
            lookup[(ts_hour, label)] = {
                "entry_close": float(entry_close),
                "close_next": float(close_next),
                "ret_realized": float(ret_realized),
                "high_fwd": float(high_fwd),
                "low_fwd": float(low_fwd),
            }
    return lookup


def _decision_outcome(
    *,
    strategy: Mapping[str, Any],
    generated_at: Any,
    outcome_lookup: Mapping[tuple[pd.Timestamp, str], Mapping[str, float]],
) -> DecisionOutcome:
    tradeable = bool(strategy.get("tradeable", False))
    direction = _to_trade_direction(strategy.get("selected_direction"))
    horizon = str(strategy.get("preferred_horizon") or "") or None
    if not tradeable or direction == "neutral" or not horizon:
        return DecisionOutcome(False, direction, horizon, None, None, None, None)

    ts = pd.to_datetime(generated_at, utc=True, errors="coerce")
    if pd.isna(ts):
        return DecisionOutcome(False, direction, horizon, None, None, None, None)
    ts_hour = pd.Timestamp(ts.floor("h"))
    key = (ts_hour, horizon)
    payload = outcome_lookup.get(key)
    if not payload:
        return DecisionOutcome(False, direction, horizon, None, None, None, None)

    ret = float(payload["ret_realized"])
    entry = float(payload["entry_close"])
    high_fwd = float(payload["high_fwd"])
    low_fwd = float(payload["low_fwd"])

    if direction == "long":
        signed_return = ret
        favorable = max((high_fwd / entry) - 1.0, 0.0)
        adverse = max(1.0 - (low_fwd / entry), 0.0)
    else:
        signed_return = -ret
        favorable = max(1.0 - (low_fwd / entry), 0.0)
        adverse = max((high_fwd / entry) - 1.0, 0.0)

    accuracy = 1.0 if signed_return > 0 else 0.0
    return DecisionOutcome(True, direction, horizon, float(signed_return), float(favorable), float(adverse), float(accuracy))


def _aggregate_decision_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    trade_rows = [row for row in rows if bool(row.get("has_trade", False))]
    returns = [float(row["signed_return"]) for row in trade_rows if row.get("signed_return") is not None]
    favors = [float(row["favorable_excursion"]) for row in trade_rows if row.get("favorable_excursion") is not None]
    adverse = [float(row["adverse_move"]) for row in trade_rows if row.get("adverse_move") is not None]
    accuracy = [float(row["direction_accuracy"]) for row in trade_rows if row.get("direction_accuracy") is not None]

    return {
        "snapshot_count": int(len(rows)),
        "trade_count": int(len(trade_rows)),
        "direction_accuracy_proxy": _safe_mean(accuracy),
        "net_return_proxy_mean": _safe_mean(returns),
        "favorable_excursion_proxy_mean": _safe_mean(favors),
        "adverse_move_proxy_mean": _safe_mean(adverse),
    }


def _segment_rows(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        bucket = str(row.get(key, "unknown") or "unknown")
        buckets.setdefault(bucket, []).append(row)
    return {bucket: _aggregate_decision_rows(bucket_rows) for bucket, bucket_rows in sorted(buckets.items())}


def _diff_metrics(baseline: Mapping[str, Any], shadow: Mapping[str, Any]) -> Dict[str, Any]:
    def _delta(name: str) -> float | None:
        a = baseline.get(name)
        b = shadow.get(name)
        if a is None or b is None:
            return None
        return float(b) - float(a)

    return {
        "trade_count_delta": int(shadow.get("trade_count", 0)) - int(baseline.get("trade_count", 0)),
        "direction_accuracy_proxy_delta": _delta("direction_accuracy_proxy"),
        "net_return_proxy_mean_delta": _delta("net_return_proxy_mean"),
        "favorable_excursion_proxy_mean_delta": _delta("favorable_excursion_proxy_mean"),
        "adverse_move_proxy_mean_delta": _delta("adverse_move_proxy_mean"),
    }


def _promotion_guardrails() -> Dict[str, Any]:
    return {
        "minimum_trade_samples_per_family_variant": 120,
        "minimum_net_return_proxy_improvement": 0.0003,
        "minimum_direction_accuracy_improvement": 0.01,
        "maximum_harmful_veto_rate": 0.40,
        "minimum_veto_precision": 0.55,
        "minimum_target_horizon_coverage": 2,
        "minimum_target_regime_coverage": 2,
        "stability_requirement": "Must remain positive on at least 2 non-overlapping recent windows.",
    }


def _evaluate_go_hold(summary: Mapping[str, Any], guardrails: Mapping[str, Any]) -> Dict[str, Any]:
    veto_count = int(summary.get("veto_count", 0))
    harmful = int(summary.get("removed_good_trade_count", 0))
    harmful_rate = (harmful / veto_count) if veto_count > 0 else 0.0

    deltas = summary.get("overall_delta", {}) if isinstance(summary.get("overall_delta"), Mapping) else {}
    net_delta = _to_float_or_none(deltas.get("net_return_proxy_mean_delta"))
    acc_delta = _to_float_or_none(deltas.get("direction_accuracy_proxy_delta"))

    horizon_positive = int(summary.get("positive_target_horizon_count", 0))
    regime_positive = int(summary.get("positive_target_regime_count", 0))
    trade_count = int(summary.get("shadow", {}).get("trade_count", 0))

    pass_checks = {
        "sample_size": trade_count >= int(guardrails["minimum_trade_samples_per_family_variant"]),
        "net_improvement": (net_delta is not None and net_delta >= float(guardrails["minimum_net_return_proxy_improvement"])),
        "accuracy_improvement": (acc_delta is not None and acc_delta >= float(guardrails["minimum_direction_accuracy_improvement"])),
        "harmful_veto_rate": harmful_rate <= float(guardrails["maximum_harmful_veto_rate"]),
        "veto_precision": float(summary.get("veto_precision", 0.0) or 0.0) >= float(guardrails["minimum_veto_precision"]),
        "horizon_coverage": horizon_positive >= int(guardrails["minimum_target_horizon_coverage"]),
        "regime_coverage": regime_positive >= int(guardrails["minimum_target_regime_coverage"]),
    }

    go = all(pass_checks.values())
    return {
        "decision": "go" if go else "hold",
        "checks": pass_checks,
        "harmful_veto_rate": harmful_rate,
    }


def evaluate_family_variant(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    family: str,
    policy: FamilyShadowPolicy,
    feature_frame: pd.DataFrame,
    outcome_lookup: Mapping[tuple[pd.Timestamp, str], Mapping[str, float]],
    max_staleness_hours: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    vetoed_bad = 0
    vetoed_good = 0
    vetoed_good_returns: List[float] = []
    retained_shadow_rows: List[Dict[str, Any]] = []

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

        baseline = replay.baseline_strategy
        shadow = replay.shadow_strategy

        preds = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), Mapping) else {}
        base_h = str(baseline.get("preferred_horizon") or "")
        base_pred = preds.get(base_h) if isinstance(preds, Mapping) else {}
        regime = _normalize_regime(base_pred.get("regime_state") if isinstance(base_pred, Mapping) else "unknown")
        confidence = _to_float_or_none(base_pred.get("confidence_score") if isinstance(base_pred, Mapping) else None)
        confidence_bucket = _confidence_bucket(confidence)

        baseline_outcome = _decision_outcome(
            strategy=baseline,
            generated_at=snapshot.get("generated_at"),
            outcome_lookup=outcome_lookup,
        )
        shadow_outcome = _decision_outcome(
            strategy=shadow,
            generated_at=snapshot.get("generated_at"),
            outcome_lookup=outcome_lookup,
        )

        row = {
            "generated_at": str(snapshot.get("generated_at") or ""),
            "family": family,
            "variant": policy.name,
            "baseline_tradeable": bool(baseline.get("tradeable", False)),
            "shadow_tradeable": bool(shadow.get("tradeable", False)),
            "baseline_direction": _to_trade_direction(baseline.get("selected_direction")),
            "shadow_direction": _to_trade_direction(shadow.get("selected_direction")),
            "baseline_horizon": baseline_outcome.horizon,
            "shadow_horizon": shadow_outcome.horizon,
            "regime": regime,
            "confidence_bucket": confidence_bucket,
            "has_trade": shadow_outcome.has_trade,
            "signed_return": shadow_outcome.signed_return,
            "favorable_excursion": shadow_outcome.favorable_excursion,
            "adverse_move": shadow_outcome.adverse_move,
            "direction_accuracy": shadow_outcome.direction_accuracy,
            "changed_selected_direction": bool(baseline.get("selected_direction") != shadow.get("selected_direction")),
            "changed_tradeable": bool(baseline.get("tradeable", False) != shadow.get("tradeable", False)),
            "changed_execution_reason": bool(
                (baseline.get("execution_state"), baseline.get("reason"))
                != (shadow.get("execution_state"), shadow.get("reason"))
            ),
        }
        rows.append(row)

        vetoed = bool(baseline_outcome.has_trade and not shadow_outcome.has_trade)
        retained = bool(baseline_outcome.has_trade and shadow_outcome.has_trade)
        if vetoed:
            if baseline_outcome.signed_return is not None and baseline_outcome.signed_return <= 0:
                vetoed_bad += 1
            if baseline_outcome.signed_return is not None and baseline_outcome.signed_return > 0:
                vetoed_good += 1
                vetoed_good_returns.append(float(baseline_outcome.signed_return))
        if retained:
            retained_shadow_rows.append(
                {
                    "has_trade": True,
                    "signed_return": shadow_outcome.signed_return,
                    "favorable_excursion": shadow_outcome.favorable_excursion,
                    "adverse_move": shadow_outcome.adverse_move,
                    "direction_accuracy": shadow_outcome.direction_accuracy,
                }
            )

    baseline_projection_rows = []
    shadow_projection_rows = []
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
        preds = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), Mapping) else {}
        base_h = str(replay.baseline_strategy.get("preferred_horizon") or "")
        base_pred = preds.get(base_h) if isinstance(preds, Mapping) else {}
        regime = _normalize_regime(base_pred.get("regime_state") if isinstance(base_pred, Mapping) else "unknown")
        confidence = _to_float_or_none(base_pred.get("confidence_score") if isinstance(base_pred, Mapping) else None)
        confidence_bucket = _confidence_bucket(confidence)

        b = _decision_outcome(
            strategy=replay.baseline_strategy,
            generated_at=snapshot.get("generated_at"),
            outcome_lookup=outcome_lookup,
        )
        s = _decision_outcome(
            strategy=replay.shadow_strategy,
            generated_at=snapshot.get("generated_at"),
            outcome_lookup=outcome_lookup,
        )
        baseline_projection_rows.append(
            {
                "has_trade": b.has_trade,
                "signed_return": b.signed_return,
                "favorable_excursion": b.favorable_excursion,
                "adverse_move": b.adverse_move,
                "direction_accuracy": b.direction_accuracy,
                "horizon": b.horizon or "unknown",
                "regime": regime,
                "confidence_bucket": confidence_bucket,
            }
        )
        shadow_projection_rows.append(
            {
                "has_trade": s.has_trade,
                "signed_return": s.signed_return,
                "favorable_excursion": s.favorable_excursion,
                "adverse_move": s.adverse_move,
                "direction_accuracy": s.direction_accuracy,
                "horizon": s.horizon or "unknown",
                "regime": regime,
                "confidence_bucket": confidence_bucket,
            }
        )

    baseline_summary = _aggregate_decision_rows(baseline_projection_rows)
    shadow_summary = _aggregate_decision_rows(shadow_projection_rows)
    overall_delta = _diff_metrics(baseline_summary, shadow_summary)

    by_horizon = {}
    base_h = _segment_rows(baseline_projection_rows, "horizon")
    sh_h = _segment_rows(shadow_projection_rows, "horizon")
    for key in sorted(set(base_h.keys()) | set(sh_h.keys())):
        by_horizon[key] = {
            "baseline": base_h.get(key, _aggregate_decision_rows([])),
            "shadow": sh_h.get(key, _aggregate_decision_rows([])),
            "delta": _diff_metrics(base_h.get(key, _aggregate_decision_rows([])), sh_h.get(key, _aggregate_decision_rows([]))),
        }

    by_regime = {}
    base_r = _segment_rows(baseline_projection_rows, "regime")
    sh_r = _segment_rows(shadow_projection_rows, "regime")
    for key in sorted(set(base_r.keys()) | set(sh_r.keys())):
        by_regime[key] = {
            "baseline": base_r.get(key, _aggregate_decision_rows([])),
            "shadow": sh_r.get(key, _aggregate_decision_rows([])),
            "delta": _diff_metrics(base_r.get(key, _aggregate_decision_rows([])), sh_r.get(key, _aggregate_decision_rows([]))),
        }

    by_confidence_bucket = {}
    base_c = _segment_rows(baseline_projection_rows, "confidence_bucket")
    sh_c = _segment_rows(shadow_projection_rows, "confidence_bucket")
    for key in sorted(set(base_c.keys()) | set(sh_c.keys())):
        by_confidence_bucket[key] = {
            "baseline": base_c.get(key, _aggregate_decision_rows([])),
            "shadow": sh_c.get(key, _aggregate_decision_rows([])),
            "delta": _diff_metrics(base_c.get(key, _aggregate_decision_rows([])), sh_c.get(key, _aggregate_decision_rows([]))),
        }

    veto_count = vetoed_bad + vetoed_good
    veto_precision = (vetoed_bad / veto_count) if veto_count > 0 else 0.0
    opportunity_cost = _safe_mean(vetoed_good_returns)
    retained_quality = _aggregate_decision_rows(retained_shadow_rows)

    positive_target_horizon_count = sum(
        1
        for h in TARGET_HORIZONS
        if h in by_horizon
        and _to_float_or_none(by_horizon[h].get("delta", {}).get("net_return_proxy_mean_delta")) is not None
        and float(by_horizon[h]["delta"]["net_return_proxy_mean_delta"]) > 0.0
    )
    positive_target_regime_count = sum(
        1
        for r in ("neutral", "chop")
        if r in by_regime
        and _to_float_or_none(by_regime[r].get("delta", {}).get("net_return_proxy_mean_delta")) is not None
        and float(by_regime[r]["delta"]["net_return_proxy_mean_delta"]) > 0.0
    )

    robust_or_narrow = "robust" if (positive_target_horizon_count >= 2 and positive_target_regime_count >= 2) else "narrow"

    summary = {
        "family": family,
        "variant": policy.name,
        "policy": {
            "name": policy.name,
            "description": policy.description,
            "enforcement_mode": policy.enforcement_mode,
        },
        "baseline": baseline_summary,
        "shadow": shadow_summary,
        "overall_delta": overall_delta,
        "veto_count": veto_count,
        "prevented_bad_trade_count": vetoed_bad,
        "removed_good_trade_count": vetoed_good,
        "veto_precision": veto_precision,
        "opportunity_cost_proxy_mean": opportunity_cost,
        "retained_trade_quality": retained_quality,
        "changed_selected_direction_count": int(sum(1 for row in rows if row.get("changed_selected_direction"))),
        "changed_tradeable_count": int(sum(1 for row in rows if row.get("changed_tradeable"))),
        "changed_execution_reason_count": int(sum(1 for row in rows if row.get("changed_execution_reason"))),
        "changed_snapshot_count": int(
            sum(
                1
                for row in rows
                if row.get("changed_selected_direction")
                or row.get("changed_tradeable")
                or row.get("changed_execution_reason")
            )
        ),
        "by_horizon": by_horizon,
        "by_regime": by_regime,
        "by_confidence_bucket": by_confidence_bucket,
        "positive_target_horizon_count": positive_target_horizon_count,
        "positive_target_regime_count": positive_target_regime_count,
        "robustness": robust_or_narrow,
        "sample_sparse": bool(int(shadow_summary.get("trade_count", 0)) < 60),
    }

    return summary


def _top_variants_from_shadow_artifact(path: Path, top_n: int) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sweep = payload.get("sweep", {}) if isinstance(payload.get("sweep"), Mapping) else {}
    families = sweep.get("families", {}) if isinstance(sweep.get("families"), Mapping) else {}
    out: Dict[str, List[str]] = {}
    for family in ("order_flow", "state_engineering"):
        family_payload = families.get(family, {}) if isinstance(families.get(family), Mapping) else {}
        rankings = family_payload.get("variant_rankings", []) if isinstance(family_payload.get("variant_rankings"), list) else []
        names = [str(item.get("policy")) for item in rankings if isinstance(item, Mapping) and item.get("policy")]
        out[family] = names[:top_n]
    return out


def run_confirmation_pass(
    *,
    shadow_artifact_path: Path,
    history_path: Path,
    spot_dir: Path,
    recent_window: int,
    max_staleness_hours: float,
    top_n: int = 2,
) -> Dict[str, Any]:
    snapshots = load_prediction_history(history_path)
    if recent_window > 0:
        snapshots = snapshots[-recent_window:]

    target_policies = _top_variants_from_shadow_artifact(shadow_artifact_path, top_n=top_n)
    all_policies = {policy.name: policy for policy in default_family_policy_variants()}

    from src.runtime.family_shadow_simulator import load_spot_feature_frame

    feature_frame = load_spot_feature_frame(spot_dir)
    ohlcv_outcomes = load_spot_ohlcv_with_outcomes(spot_dir, horizons=TARGET_HORIZONS)
    outcome_lookup = _build_outcome_lookup(ohlcv_outcomes, horizons=TARGET_HORIZONS)

    results: Dict[str, List[Dict[str, Any]]] = {"order_flow": [], "state_engineering": []}
    for family, policy_names in target_policies.items():
        for name in policy_names:
            policy = all_policies.get(name)
            if policy is None:
                continue
            summary = evaluate_family_variant(
                snapshots=snapshots,
                family=family,
                policy=policy,
                feature_frame=feature_frame,
                outcome_lookup=outcome_lookup,
                max_staleness_hours=max_staleness_hours,
            )
            results[family].append(summary)

    for family in results:
        results[family].sort(
            key=lambda item: (
                _to_float_or_none(item.get("overall_delta", {}).get("net_return_proxy_mean_delta")) or -999.0,
                _to_float_or_none(item.get("overall_delta", {}).get("direction_accuracy_proxy_delta")) or -999.0,
            ),
            reverse=True,
        )

    guardrails = _promotion_guardrails()
    decisions = {
        family: [
            {
                "variant": item.get("variant"),
                "go_hold": _evaluate_go_hold(item, guardrails),
            }
            for item in summaries
        ]
        for family, summaries in results.items()
    }

    family_best = []
    for family, summaries in results.items():
        if not summaries:
            continue
        best = summaries[0]
        family_best.append(
            {
                "family": family,
                "best_variant": best.get("variant"),
                "net_return_proxy_mean_delta": best.get("overall_delta", {}).get("net_return_proxy_mean_delta"),
                "direction_accuracy_proxy_delta": best.get("overall_delta", {}).get("direction_accuracy_proxy_delta"),
                "robustness": best.get("robustness"),
                "go_hold": _evaluate_go_hold(best, guardrails),
            }
        )

    family_best.sort(
        key=lambda item: (
            _to_float_or_none(item.get("net_return_proxy_mean_delta")) or -999.0,
            _to_float_or_none(item.get("direction_accuracy_proxy_delta")) or -999.0,
        ),
        reverse=True,
    )

    return {
        "recent_window": int(recent_window),
        "top_n_variants_per_family": int(top_n),
        "families": results,
        "family_best_rankings": family_best,
        "decisions": decisions,
        "promotion_guardrails": guardrails,
        "macro_disposition": "remain_deprioritized",
    }


def render_confirmation_markdown(payload: Mapping[str, Any]) -> str:
    rankings = payload.get("family_best_rankings", []) if isinstance(payload.get("family_best_rankings"), list) else []
    families = payload.get("families", {}) if isinstance(payload.get("families"), Mapping) else {}
    guardrails = payload.get("promotion_guardrails", {}) if isinstance(payload.get("promotion_guardrails"), Mapping) else {}

    lines: List[str] = []
    lines.append("# State/Order-Flow Outcome Confirmation")
    lines.append("")
    lines.append("## Headline Recommendation")
    lines.append("| Rank | Family | Best Variant | Net Return Delta | Accuracy Delta | Robustness | Decision |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for idx, row in enumerate(rankings, start=1):
        if not isinstance(row, Mapping):
            continue
        decision = row.get("go_hold", {}).get("decision") if isinstance(row.get("go_hold"), Mapping) else "hold"
        lines.append(
            "| {rank} | {family} | {variant} | {ret} | {acc} | {robust} | {decision} |".format(
                rank=idx,
                family=str(row.get("family", "unknown")),
                variant=str(row.get("best_variant", "n/a")),
                ret=(f"{float(row.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(row.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                acc=(f"{float(row.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(row.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                robust=str(row.get("robustness", "unknown")),
                decision=str(decision),
            )
        )
    lines.append("")

    for family in ("order_flow", "state_engineering"):
        variants = families.get(family, []) if isinstance(families.get(family), list) else []
        lines.append(f"## {family} Variants")
        lines.append("| Variant | Decision | Trade Delta | Return Delta | Accuracy Delta | Veto Precision | Harmful Veto Rate | Robustness |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for item in variants:
            if not isinstance(item, Mapping):
                continue
            delta = item.get("overall_delta", {}) if isinstance(item.get("overall_delta"), Mapping) else {}
            decision = _evaluate_go_hold(item, guardrails)
            lines.append(
                "| {variant} | {decision} | {trade_delta} | {ret_delta} | {acc_delta} | {veto_precision:.3f} | {harm_rate:.3f} | {robust} |".format(
                    variant=str(item.get("variant", "unknown")),
                    decision=str(decision.get("decision", "hold")),
                    trade_delta=int(delta.get("trade_count_delta", 0)),
                    ret_delta=(f"{float(delta.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(delta.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                    acc_delta=(f"{float(delta.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(delta.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                    veto_precision=float(item.get("veto_precision", 0.0) or 0.0),
                    harm_rate=float(decision.get("harmful_veto_rate", 0.0) or 0.0),
                    robust=str(item.get("robustness", "unknown")),
                )
            )
        lines.append("")

    lines.append("## Promotion Guardrails")
    lines.append(f"- Minimum trade samples per family+variant: {guardrails.get('minimum_trade_samples_per_family_variant')}")
    lines.append(f"- Minimum net return proxy improvement: {guardrails.get('minimum_net_return_proxy_improvement')}")
    lines.append(f"- Minimum direction accuracy improvement: {guardrails.get('minimum_direction_accuracy_improvement')}")
    lines.append(f"- Maximum harmful veto rate: {guardrails.get('maximum_harmful_veto_rate')}")
    lines.append(f"- Minimum veto precision: {guardrails.get('minimum_veto_precision')}")
    lines.append(f"- Minimum target horizon coverage: {guardrails.get('minimum_target_horizon_coverage')}")
    lines.append(f"- Minimum target regime coverage: {guardrails.get('minimum_target_regime_coverage')}")
    lines.append(f"- Stability requirement: {guardrails.get('stability_requirement')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This pass confirms realized outcome proxies, distinct from prior veto-diagnostic sweep counts.")
    lines.append("- Macro remains deprioritized in this workflow.")
    lines.append("")

    return "\n".join(lines) + "\n"


def _two_window_thresholds() -> Dict[str, Any]:
    return {
        "minimum_per_window_sample_size": 120,
        "minimum_per_window_net_return_proxy_delta": 0.0003,
        "minimum_per_window_accuracy_delta": 0.01,
        "maximum_per_window_harmful_veto_rate": 0.40,
        "minimum_per_window_veto_precision": 0.55,
        "minimum_positive_target_horizons": 2,
        "minimum_positive_target_regimes": 2,
        "maximum_window_gain_concentration_share": 0.75,
        "maximum_horizon_gain_concentration_share": 0.80,
        "maximum_regime_gain_concentration_share": 0.80,
        "immediate_disable_on_any_window_net_delta_below": -0.0005,
        "immediate_disable_on_any_window_accuracy_delta_below": -0.02,
        "immediate_disable_on_any_window_harmful_veto_rate_above": 0.50,
        "immediate_disable_on_any_window_veto_precision_below": 0.50,
    }


def _gain_share(values: Sequence[float]) -> float | None:
    positives = [float(v) for v in values if v is not None and float(v) > 0.0]
    if not positives:
        return None
    total = float(sum(positives))
    if total <= 0.0:
        return None
    return float(max(positives) / total)


def _target_bucket_positive_deltas(by_bucket: Mapping[str, Any], targets: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target in targets:
        payload = by_bucket.get(target, {}) if isinstance(by_bucket.get(target), Mapping) else {}
        delta = payload.get("delta", {}) if isinstance(payload.get("delta"), Mapping) else {}
        val = _to_float_or_none(delta.get("net_return_proxy_mean_delta"))
        if val is not None and val > 0.0:
            out[target] = float(val)
    return out


def split_two_non_overlapping_recent_windows(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    window_size: int,
) -> Dict[str, List[Mapping[str, Any]]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    required = window_size * 2
    if len(snapshots) < required:
        raise ValueError(
            f"Need at least {required} snapshots for two non-overlapping windows; found {len(snapshots)}"
        )

    tail = list(snapshots[-required:])
    return {
        "window_older": tail[:window_size],
        "window_newer": tail[window_size:],
    }


def _resolve_effective_window_size(*, snapshot_count: int, requested_window_size: int) -> int:
    max_non_overlapping = snapshot_count // 2
    if max_non_overlapping <= 0:
        raise ValueError("Need at least 2 snapshots for non-overlapping two-window stability check")
    if requested_window_size <= 0:
        return int(max_non_overlapping)
    return int(min(requested_window_size, max_non_overlapping))


def _window_summary_row(window_name: str, summary: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    overall_delta = summary.get("overall_delta", {}) if isinstance(summary.get("overall_delta"), Mapping) else {}
    return {
        "window": window_name,
        "sample_count": int(summary.get("shadow", {}).get("snapshot_count", 0)),
        "changed_snapshot_count": int(summary.get("changed_snapshot_count", 0)),
        "net_return_proxy_mean_delta": _to_float_or_none(overall_delta.get("net_return_proxy_mean_delta")),
        "direction_accuracy_proxy_delta": _to_float_or_none(overall_delta.get("direction_accuracy_proxy_delta")),
        "harmful_veto_rate": float(decision.get("harmful_veto_rate", 0.0) or 0.0),
        "veto_precision": float(summary.get("veto_precision", 0.0) or 0.0),
        "blocked_trade_quality": _to_float_or_none(summary.get("opportunity_cost_proxy_mean")),
        "retained_trade_quality": _to_float_or_none(
            summary.get("retained_trade_quality", {}).get("net_return_proxy_mean")
            if isinstance(summary.get("retained_trade_quality"), Mapping)
            else None
        ),
        "go_hold": str(decision.get("decision", "hold")),
    }


def _assess_two_window_stability(
    *,
    window_summaries: Mapping[str, Mapping[str, Any]],
    window_decisions: Mapping[str, Mapping[str, Any]],
    aggregate_summary: Mapping[str, Any],
    aggregate_decision: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    older = window_summaries.get("window_older", {}) if isinstance(window_summaries.get("window_older"), Mapping) else {}
    newer = window_summaries.get("window_newer", {}) if isinstance(window_summaries.get("window_newer"), Mapping) else {}

    older_delta = _to_float_or_none(older.get("overall_delta", {}).get("net_return_proxy_mean_delta"))
    newer_delta = _to_float_or_none(newer.get("overall_delta", {}).get("net_return_proxy_mean_delta"))
    older_acc = _to_float_or_none(older.get("overall_delta", {}).get("direction_accuracy_proxy_delta"))
    newer_acc = _to_float_or_none(newer.get("overall_delta", {}).get("direction_accuracy_proxy_delta"))

    positive_window_deltas = [v for v in (older_delta, newer_delta) if v is not None and v > 0.0]
    window_gain_concentration_share = _gain_share(positive_window_deltas)

    by_horizon = aggregate_summary.get("by_horizon", {}) if isinstance(aggregate_summary.get("by_horizon"), Mapping) else {}
    by_regime = aggregate_summary.get("by_regime", {}) if isinstance(aggregate_summary.get("by_regime"), Mapping) else {}
    horizon_positive = _target_bucket_positive_deltas(by_horizon, TARGET_HORIZONS)
    regime_positive = _target_bucket_positive_deltas(by_regime, ("neutral", "chop"))

    horizon_gain_concentration_share = _gain_share(list(horizon_positive.values()))
    regime_gain_concentration_share = _gain_share(list(regime_positive.values()))

    both_windows_pass = all(
        str(window_decisions.get(name, {}).get("decision", "hold")) == "go"
        for name in ("window_older", "window_newer")
    )

    concentrated_in_one_window = (
        window_gain_concentration_share is not None
        and window_gain_concentration_share > float(thresholds["maximum_window_gain_concentration_share"])
    )
    concentrated_in_one_horizon = (
        horizon_gain_concentration_share is not None
        and horizon_gain_concentration_share > float(thresholds["maximum_horizon_gain_concentration_share"])
    )
    concentrated_in_one_regime = (
        regime_gain_concentration_share is not None
        and regime_gain_concentration_share > float(thresholds["maximum_regime_gain_concentration_share"])
    )

    aggregate_go = str(aggregate_decision.get("decision", "hold")) == "go"
    robust_enough = bool(
        both_windows_pass
        and aggregate_go
        and not concentrated_in_one_window
        and not concentrated_in_one_horizon
        and not concentrated_in_one_regime
    )

    immediate_disable = bool(
        any(
            val is not None and val <= float(thresholds["immediate_disable_on_any_window_net_delta_below"])
            for val in (older_delta, newer_delta)
        )
        or any(
            val is not None and val <= float(thresholds["immediate_disable_on_any_window_accuracy_delta_below"])
            for val in (older_acc, newer_acc)
        )
        or any(
            float(window_decisions.get(name, {}).get("harmful_veto_rate", 0.0) or 0.0)
            >= float(thresholds["immediate_disable_on_any_window_harmful_veto_rate_above"])
            for name in ("window_older", "window_newer")
        )
        or any(
            float(window_summaries.get(name, {}).get("veto_precision", 0.0) or 0.0)
            <= float(thresholds["immediate_disable_on_any_window_veto_precision_below"])
            for name in ("window_older", "window_newer")
        )
    )

    readiness = "ready_for_shadow_production" if robust_enough else "not_ready_more_validation_needed"

    return {
        "both_windows_pass_guardrails": both_windows_pass,
        "aggregate_pass_guardrails": aggregate_go,
        "concentrated_in_one_window": concentrated_in_one_window,
        "concentrated_in_one_horizon": concentrated_in_one_horizon,
        "concentrated_in_one_regime": concentrated_in_one_regime,
        "window_gain_concentration_share": window_gain_concentration_share,
        "horizon_gain_concentration_share": horizon_gain_concentration_share,
        "regime_gain_concentration_share": regime_gain_concentration_share,
        "robust_enough_for_guarded_shadow": robust_enough,
        "readiness_recommendation": readiness,
        "immediate_disable_triggered": immediate_disable,
    }


def run_order_flow_two_window_stability(
    *,
    history_path: Path,
    spot_dir: Path,
    window_size: int,
    max_staleness_hours: float,
    family: str = "order_flow",
    variant: str = "weak_signal_veto_only",
) -> Dict[str, Any]:
    if family != "order_flow":
        raise ValueError("This workflow is scoped to family='order_flow'")

    snapshots = load_prediction_history(history_path)
    effective_window_size = _resolve_effective_window_size(
        snapshot_count=len(snapshots),
        requested_window_size=window_size,
    )
    windows = split_two_non_overlapping_recent_windows(snapshots, window_size=effective_window_size)

    from src.runtime.family_shadow_simulator import load_spot_feature_frame

    feature_frame = load_spot_feature_frame(spot_dir)
    ohlcv_outcomes = load_spot_ohlcv_with_outcomes(spot_dir, horizons=TARGET_HORIZONS)
    outcome_lookup = _build_outcome_lookup(ohlcv_outcomes, horizons=TARGET_HORIZONS)

    policies = {policy.name: policy for policy in default_family_policy_variants()}
    policy = policies.get(variant)
    if policy is None:
        raise ValueError(f"Unknown policy variant: {variant}")

    guardrails = _promotion_guardrails()
    thresholds = _two_window_thresholds()

    window_summaries: Dict[str, Dict[str, Any]] = {}
    window_decisions: Dict[str, Dict[str, Any]] = {}
    for window_name, window_snaps in windows.items():
        summary = evaluate_family_variant(
            snapshots=window_snaps,
            family=family,
            policy=policy,
            feature_frame=feature_frame,
            outcome_lookup=outcome_lookup,
            max_staleness_hours=max_staleness_hours,
        )
        decision = _evaluate_go_hold(summary, guardrails)
        window_summaries[window_name] = summary
        window_decisions[window_name] = decision

    aggregate_snaps = list(windows["window_older"]) + list(windows["window_newer"])
    aggregate_summary = evaluate_family_variant(
        snapshots=aggregate_snaps,
        family=family,
        policy=policy,
        feature_frame=feature_frame,
        outcome_lookup=outcome_lookup,
        max_staleness_hours=max_staleness_hours,
    )
    aggregate_decision = _evaluate_go_hold(aggregate_summary, guardrails)

    stability = _assess_two_window_stability(
        window_summaries=window_summaries,
        window_decisions=window_decisions,
        aggregate_summary=aggregate_summary,
        aggregate_decision=aggregate_decision,
        thresholds=thresholds,
    )

    return {
        "family": family,
        "variant": variant,
        "window_size_requested": int(window_size),
        "window_size_effective": int(effective_window_size),
        "snapshot_count_available": int(len(snapshots)),
        "windows": {
            name: {
                "summary": summary,
                "go_hold": window_decisions.get(name, {}),
                "headline": _window_summary_row(name, summary, window_decisions.get(name, {})),
            }
            for name, summary in window_summaries.items()
        },
        "aggregate": {
            "summary": aggregate_summary,
            "go_hold": aggregate_decision,
            "headline": _window_summary_row("aggregate", aggregate_summary, aggregate_decision),
        },
        "stability_assessment": stability,
        "promotion_guardrails": guardrails,
        "shadow_readiness_thresholds": thresholds,
        "macro_disposition": "remain_deprioritized",
        "state_engineering_disposition": "hold",
    }


def render_order_flow_two_window_readiness_memo(payload: Mapping[str, Any]) -> str:
    windows = payload.get("windows", {}) if isinstance(payload.get("windows"), Mapping) else {}
    older = windows.get("window_older", {}) if isinstance(windows.get("window_older"), Mapping) else {}
    newer = windows.get("window_newer", {}) if isinstance(windows.get("window_newer"), Mapping) else {}
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), Mapping) else {}
    stability = (
        payload.get("stability_assessment", {})
        if isinstance(payload.get("stability_assessment"), Mapping)
        else {}
    )
    thresholds = (
        payload.get("shadow_readiness_thresholds", {})
        if isinstance(payload.get("shadow_readiness_thresholds"), Mapping)
        else {}
    )

    older_headline = older.get("headline", {}) if isinstance(older.get("headline"), Mapping) else {}
    newer_headline = newer.get("headline", {}) if isinstance(newer.get("headline"), Mapping) else {}
    aggregate_headline = aggregate.get("headline", {}) if isinstance(aggregate.get("headline"), Mapping) else {}

    older_summary = older.get("summary", {}) if isinstance(older.get("summary"), Mapping) else {}
    newer_summary = newer.get("summary", {}) if isinstance(newer.get("summary"), Mapping) else {}
    aggregate_summary = aggregate.get("summary", {}) if isinstance(aggregate.get("summary"), Mapping) else {}

    lines: List[str] = []
    lines.append("# Order-Flow Two-Window Stability and Shadow-Production Readiness")
    lines.append("")
    lines.append("## Headline Recommendation")
    lines.append(f"- Recommendation: **{stability.get('readiness_recommendation', 'not_ready_more_validation_needed')}**")
    lines.append(f"- Family: {payload.get('family', 'order_flow')}")
    lines.append(f"- Variant: {payload.get('variant', 'weak_signal_veto_only')}")
    lines.append("")

    lines.append("## Two-Window Evidence")
    lines.append("| Window | Sample Count | Changed Snapshots | Net Return Delta | Accuracy Delta | Harmful Veto Rate | Veto Precision | Blocked-Trade Quality | Retained-Trade Quality | Guardrail Decision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in (older_headline, newer_headline, aggregate_headline):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {window} | {sample} | {changed} | {ret} | {acc} | {harm:.3f} | {vp:.3f} | {blocked} | {retained} | {decision} |".format(
                window=str(row.get("window", "unknown")),
                sample=int(row.get("sample_count", 0)),
                changed=int(row.get("changed_snapshot_count", 0)),
                ret=(f"{float(row.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(row.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                acc=(f"{float(row.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(row.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                harm=float(row.get("harmful_veto_rate", 0.0) or 0.0),
                vp=float(row.get("veto_precision", 0.0) or 0.0),
                blocked=(f"{float(row.get('blocked_trade_quality')):.6f}" if _to_float_or_none(row.get("blocked_trade_quality")) is not None else "n/a"),
                retained=(f"{float(row.get('retained_trade_quality')):.6f}" if _to_float_or_none(row.get("retained_trade_quality")) is not None else "n/a"),
                decision=str(row.get("go_hold", "hold")),
            )
        )
    lines.append("")

    lines.append("## Per-Horizon Breakdown (4h, 8h, 12h)")
    lines.append("| Scope | Horizon | Net Return Delta | Accuracy Delta | Trade Delta |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for scope, summary in (
        ("window_older", older_summary),
        ("window_newer", newer_summary),
        ("aggregate", aggregate_summary),
    ):
        by_h = summary.get("by_horizon", {}) if isinstance(summary.get("by_horizon"), Mapping) else {}
        for horizon in TARGET_HORIZONS:
            row = by_h.get(horizon, {}) if isinstance(by_h.get(horizon), Mapping) else {}
            delta = row.get("delta", {}) if isinstance(row.get("delta"), Mapping) else {}
            lines.append(
                "| {scope} | {horizon} | {ret} | {acc} | {trade_delta} |".format(
                    scope=scope,
                    horizon=horizon,
                    ret=(f"{float(delta.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(delta.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                    acc=(f"{float(delta.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(delta.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                    trade_delta=int(delta.get("trade_count_delta", 0)),
                )
            )
    lines.append("")

    lines.append("## Per-Regime Breakdown (neutral, chop)")
    lines.append("| Scope | Regime | Net Return Delta | Accuracy Delta | Trade Delta |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for scope, summary in (
        ("window_older", older_summary),
        ("window_newer", newer_summary),
        ("aggregate", aggregate_summary),
    ):
        by_r = summary.get("by_regime", {}) if isinstance(summary.get("by_regime"), Mapping) else {}
        for regime in ("neutral", "chop"):
            row = by_r.get(regime, {}) if isinstance(by_r.get(regime), Mapping) else {}
            delta = row.get("delta", {}) if isinstance(row.get("delta"), Mapping) else {}
            lines.append(
                "| {scope} | {regime} | {ret} | {acc} | {trade_delta} |".format(
                    scope=scope,
                    regime=regime,
                    ret=(f"{float(delta.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(delta.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                    acc=(f"{float(delta.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(delta.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                    trade_delta=int(delta.get("trade_count_delta", 0)),
                )
            )
    lines.append("")

    lines.append("## Stability Assessment")
    lines.append(f"- Passed guardrails in both windows: {stability.get('both_windows_pass_guardrails')}")
    lines.append(f"- Gains concentrated in one window only: {stability.get('concentrated_in_one_window')}")
    lines.append(f"- Gains concentrated in one horizon only: {stability.get('concentrated_in_one_horizon')}")
    lines.append(f"- Gains concentrated in one regime only: {stability.get('concentrated_in_one_regime')}")
    lines.append(f"- Robust enough for guarded shadow-production stage: {stability.get('robust_enough_for_guarded_shadow')}")
    lines.append("")

    lines.append("## Fail-Close Conditions (Future Shadow-Production)")
    lines.append(f"- Minimum per-window sample size: {thresholds.get('minimum_per_window_sample_size')}")
    lines.append(f"- Minimum per-window net-return-proxy delta: {thresholds.get('minimum_per_window_net_return_proxy_delta')}")
    lines.append(f"- Minimum per-window accuracy delta: {thresholds.get('minimum_per_window_accuracy_delta')}")
    lines.append(f"- Maximum per-window harmful veto rate: {thresholds.get('maximum_per_window_harmful_veto_rate')}")
    lines.append(f"- Minimum per-window veto precision: {thresholds.get('minimum_per_window_veto_precision')}")
    lines.append(f"- Max window gain concentration share: {thresholds.get('maximum_window_gain_concentration_share')}")
    lines.append(f"- Max horizon gain concentration share: {thresholds.get('maximum_horizon_gain_concentration_share')}")
    lines.append(f"- Max regime gain concentration share: {thresholds.get('maximum_regime_gain_concentration_share')}")
    lines.append("")

    lines.append("## Rollback Triggers")
    lines.append(
        f"- Immediately disable shadow policy if any window net delta <= {thresholds.get('immediate_disable_on_any_window_net_delta_below')}"
    )
    lines.append(
        f"- Immediately disable shadow policy if any window accuracy delta <= {thresholds.get('immediate_disable_on_any_window_accuracy_delta_below')}"
    )
    lines.append(
        f"- Immediately disable shadow policy if any window harmful veto rate >= {thresholds.get('immediate_disable_on_any_window_harmful_veto_rate_above')}"
    )
    lines.append(
        f"- Immediately disable shadow policy if any window veto precision <= {thresholds.get('immediate_disable_on_any_window_veto_precision_below')}"
    )
    lines.append("- Disable if either non-overlapping stability window fails guardrails in the next confirmation cycle.")
    lines.append("")

    lines.append("## Residual Risks")
    lines.append("- Realized proxy outcomes may drift from live execution quality under microstructure changes.")
    lines.append("- Stability remains sensitive to snapshot volume contraction in target horizons and regimes.")
    lines.append("- Veto-only benefit can decay if baseline model confidence calibration shifts.")
    lines.append("")

    lines.append("## Recommended Rollout Shape If Ready")
    lines.append("- Eligible horizons: 4h, 8h, 12h only.")
    lines.append("- Eligible regimes: neutral and chop only.")
    lines.append("- Confidence scope: keep current weak-signal slice from weak_signal_veto_only policy.")
    lines.append("- Policy mode: veto-only.")
    lines.append("- Shadow fail mode: fail-closed (disable policy effects when required diagnostics are missing).")
    lines.append("")

    return "\n".join(lines) + "\n"


def split_rolling_non_overlapping_windows(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    window_size: int,
    max_windows: int = 0,
) -> List[Dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    count = len(snapshots)
    full_window_count = count // window_size
    if full_window_count <= 0:
        raise ValueError(f"Need at least {window_size} snapshots; found {count}")

    usable_count = full_window_count * window_size
    tail = list(snapshots[-usable_count:])

    windows: List[Dict[str, Any]] = []
    for idx in range(full_window_count):
        start = idx * window_size
        end = start + window_size
        chunk = tail[start:end]
        windows.append(
            {
                "window_index": idx,
                "window_label": f"window_{idx + 1}",
                "start_offset": start,
                "end_offset_exclusive": end,
                "snapshots": chunk,
            }
        )

    if max_windows > 0 and len(windows) > max_windows:
        windows = windows[-max_windows:]
        for idx, item in enumerate(windows):
            item["window_index"] = idx
            item["window_label"] = f"window_{idx + 1}"
    return windows


def _rolling_window_headline(
    *,
    window_label: str,
    summary: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    base = _window_summary_row(window_label, summary, decision)
    base["target_horizon_coverage"] = int(summary.get("positive_target_horizon_count", 0))
    base["target_regime_coverage"] = int(summary.get("positive_target_regime_count", 0))
    return base


def _delta_for_bucket(summary: Mapping[str, Any], section: str, bucket: str) -> float | None:
    group = summary.get(section, {}) if isinstance(summary.get(section), Mapping) else {}
    item = group.get(bucket, {}) if isinstance(group.get(bucket), Mapping) else {}
    delta = item.get("delta", {}) if isinstance(item.get("delta"), Mapping) else {}
    return _to_float_or_none(delta.get("net_return_proxy_mean_delta"))


def _analyze_failure_clusters(failed_windows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    horizon_fail_counts: Dict[str, int] = {h: 0 for h in TARGET_HORIZONS}
    regime_fail_counts: Dict[str, int] = {r: 0 for r in ("neutral", "chop")}
    confidence_fail_counts: Dict[str, int] = {}

    for window in failed_windows:
        summary = window.get("summary", {}) if isinstance(window.get("summary"), Mapping) else {}
        for horizon in TARGET_HORIZONS:
            val = _delta_for_bucket(summary, "by_horizon", horizon)
            if val is None or val <= 0.0:
                horizon_fail_counts[horizon] += 1
        for regime in ("neutral", "chop"):
            val = _delta_for_bucket(summary, "by_regime", regime)
            if val is None or val <= 0.0:
                regime_fail_counts[regime] += 1

        by_conf = summary.get("by_confidence_bucket", {}) if isinstance(summary.get("by_confidence_bucket"), Mapping) else {}
        for bucket, payload in by_conf.items():
            if not isinstance(payload, Mapping):
                continue
            delta = payload.get("delta", {}) if isinstance(payload.get("delta"), Mapping) else {}
            net = _to_float_or_none(delta.get("net_return_proxy_mean_delta"))
            if net is None or net <= 0.0:
                confidence_fail_counts[str(bucket)] = confidence_fail_counts.get(str(bucket), 0) + 1

    failure_count = max(len(failed_windows), 1)
    horizon_cluster = {
        key: {"count": value, "share": float(value / failure_count)}
        for key, value in horizon_fail_counts.items()
    }
    regime_cluster = {
        key: {"count": value, "share": float(value / failure_count)}
        for key, value in regime_fail_counts.items()
    }
    confidence_cluster = {
        key: {"count": value, "share": float(value / failure_count)}
        for key, value in sorted(confidence_fail_counts.items())
    }

    dominant_horizon = max(horizon_cluster.items(), key=lambda kv: kv[1]["share"])[0] if horizon_cluster else None
    dominant_regime = max(regime_cluster.items(), key=lambda kv: kv[1]["share"])[0] if regime_cluster else None
    dominant_confidence = max(confidence_cluster.items(), key=lambda kv: kv[1]["share"])[0] if confidence_cluster else None

    return {
        "horizon_failure_cluster": horizon_cluster,
        "regime_failure_cluster": regime_cluster,
        "confidence_failure_cluster": confidence_cluster,
        "dominant_failure_horizon": dominant_horizon,
        "dominant_failure_regime": dominant_regime,
        "dominant_failure_confidence_bucket": dominant_confidence,
    }


def _analyze_positive_slice_dependency(window_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def _collect(section: str, buckets: Sequence[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {bucket: [] for bucket in buckets}
        for item in window_results:
            summary = item.get("summary", {}) if isinstance(item.get("summary"), Mapping) else {}
            for bucket in buckets:
                val = _delta_for_bucket(summary, section, bucket)
                if val is not None:
                    out[bucket].append(float(val))
        return out

    horizon = _collect("by_horizon", TARGET_HORIZONS)
    regime = _collect("by_regime", ("neutral", "chop"))

    confidence: Dict[str, List[float]] = {}
    for item in window_results:
        summary = item.get("summary", {}) if isinstance(item.get("summary"), Mapping) else {}
        by_conf = summary.get("by_confidence_bucket", {}) if isinstance(summary.get("by_confidence_bucket"), Mapping) else {}
        for bucket, payload in by_conf.items():
            if not isinstance(payload, Mapping):
                continue
            delta = payload.get("delta", {}) if isinstance(payload.get("delta"), Mapping) else {}
            net = _to_float_or_none(delta.get("net_return_proxy_mean_delta"))
            if net is None:
                continue
            confidence.setdefault(str(bucket), []).append(float(net))

    def _summarize(values: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        positive_strength: Dict[str, float] = {}
        for key, series in values.items():
            arr = [float(v) for v in series]
            if not arr:
                summary[key] = {"positive_windows": 0, "coverage": 0.0, "avg_delta": None}
                continue
            positive = sum(1 for v in arr if v > 0.0)
            avg = float(np.nanmean(np.asarray(arr, dtype=float)))
            coverage = float(positive / len(arr))
            summary[key] = {"positive_windows": positive, "coverage": coverage, "avg_delta": avg}
            if avg > 0.0:
                positive_strength[key] = avg

        concentration_share = _gain_share(list(positive_strength.values()))
        dominant_key = max(positive_strength.items(), key=lambda kv: kv[1])[0] if positive_strength else None
        return {
            "buckets": summary,
            "dominant_positive_bucket": dominant_key,
            "dominant_positive_share": concentration_share,
        }

    return {
        "horizon": _summarize(horizon),
        "regime": _summarize(regime),
        "confidence": _summarize(confidence),
    }


def _classify_rolling_stability(
    *,
    window_results: Sequence[Mapping[str, Any]],
    failure_clusters: Mapping[str, Any],
    positive_dependency: Mapping[str, Any],
) -> Dict[str, Any]:
    total = len(window_results)
    pass_count = sum(1 for item in window_results if str(item.get("go_hold", {}).get("decision", "hold")) == "go")
    fail_count = total - pass_count

    pass_rate = float(pass_count / total) if total > 0 else 0.0
    classification = "not_enough_evidence"
    if total >= 3:
        if pass_rate >= 0.70:
            classification = "persistently_stable"
        elif pass_rate >= 0.35:
            classification = "conditionally_stable"
        else:
            classification = "unstable"

    recent_span = min(2, total)
    recent_windows = list(window_results[-recent_span:]) if recent_span > 0 else []
    older_windows = list(window_results[:-recent_span]) if total > recent_span else []
    recent_fail_rate = (
        float(sum(1 for item in recent_windows if str(item.get("go_hold", {}).get("decision", "hold")) != "go") / len(recent_windows))
        if recent_windows
        else 0.0
    )
    older_fail_rate = (
        float(sum(1 for item in older_windows if str(item.get("go_hold", {}).get("decision", "hold")) != "go") / len(older_windows))
        if older_windows
        else 0.0
    )

    concentrated_recent = bool(recent_windows and older_windows and (recent_fail_rate - older_fail_rate) >= 0.30)

    horizon_dom_share = _to_float_or_none(
        positive_dependency.get("horizon", {}).get("dominant_positive_share")
        if isinstance(positive_dependency.get("horizon"), Mapping)
        else None
    )
    regime_dom_share = _to_float_or_none(
        positive_dependency.get("regime", {}).get("dominant_positive_share")
        if isinstance(positive_dependency.get("regime"), Mapping)
        else None
    )
    narrow_slice = bool(
        (horizon_dom_share is not None and horizon_dom_share >= 0.80)
        or (regime_dom_share is not None and regime_dom_share >= 0.80)
    )

    dominant_failure_regime_share = _to_float_or_none(
        failure_clusters.get("regime_failure_cluster", {})
        .get(str(failure_clusters.get("dominant_failure_regime")), {})
        .get("share")
        if isinstance(failure_clusters.get("regime_failure_cluster"), Mapping)
        else None
    )
    dominant_failure_horizon_share = _to_float_or_none(
        failure_clusters.get("horizon_failure_cluster", {})
        .get(str(failure_clusters.get("dominant_failure_horizon")), {})
        .get("share")
        if isinstance(failure_clusters.get("horizon_failure_cluster"), Mapping)
        else None
    )

    disposition = "deprioritize_for_now"
    if classification == "persistently_stable":
        disposition = "continue_deeper_validation"
    elif classification == "conditionally_stable" and narrow_slice:
        disposition = "narrow_scope_followup_validation"
    elif classification == "conditionally_stable":
        disposition = "deprioritize_for_now"
    elif classification == "not_enough_evidence":
        disposition = "collect_more_evidence"

    return {
        "classification": classification,
        "disposition": disposition,
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
        "total_windows": int(total),
        "pass_rate": pass_rate,
        "recent_fail_rate": recent_fail_rate,
        "older_fail_rate": older_fail_rate,
        "instability_concentrated_in_recent_period": concentrated_recent,
        "persistent_across_history": bool(classification == "unstable" and not concentrated_recent),
        "dominant_failure_regime_share": dominant_failure_regime_share,
        "dominant_failure_horizon_share": dominant_failure_horizon_share,
        "narrow_positive_slice_dependency": narrow_slice,
        "narrowest_credible_scope_if_conditional": {
            "horizon": positive_dependency.get("horizon", {}).get("dominant_positive_bucket")
            if isinstance(positive_dependency.get("horizon"), Mapping)
            else None,
            "regime": positive_dependency.get("regime", {}).get("dominant_positive_bucket")
            if isinstance(positive_dependency.get("regime"), Mapping)
            else None,
            "confidence_bucket": positive_dependency.get("confidence", {}).get("dominant_positive_bucket")
            if isinstance(positive_dependency.get("confidence"), Mapping)
            else None,
        },
    }


def run_order_flow_rolling_stability(
    *,
    history_path: Path,
    spot_dir: Path,
    window_size: int,
    max_staleness_hours: float,
    max_windows: int = 0,
    family: str = "order_flow",
    variant: str = "weak_signal_veto_only",
) -> Dict[str, Any]:
    if family != "order_flow":
        raise ValueError("This workflow is scoped to family='order_flow'")

    snapshots = load_prediction_history(history_path)
    windows = split_rolling_non_overlapping_windows(
        snapshots,
        window_size=window_size,
        max_windows=max_windows,
    )

    from src.runtime.family_shadow_simulator import load_spot_feature_frame

    feature_frame = load_spot_feature_frame(spot_dir)
    ohlcv_outcomes = load_spot_ohlcv_with_outcomes(spot_dir, horizons=TARGET_HORIZONS)
    outcome_lookup = _build_outcome_lookup(ohlcv_outcomes, horizons=TARGET_HORIZONS)

    policies = {policy.name: policy for policy in default_family_policy_variants()}
    policy = policies.get(variant)
    if policy is None:
        raise ValueError(f"Unknown policy variant: {variant}")

    guardrails = _promotion_guardrails()
    window_results: List[Dict[str, Any]] = []
    for window in windows:
        summary = evaluate_family_variant(
            snapshots=window["snapshots"],
            family=family,
            policy=policy,
            feature_frame=feature_frame,
            outcome_lookup=outcome_lookup,
            max_staleness_hours=max_staleness_hours,
        )
        decision = _evaluate_go_hold(summary, guardrails)
        window_results.append(
            {
                "window_index": int(window["window_index"]),
                "window_label": str(window["window_label"]),
                "summary": summary,
                "go_hold": decision,
                "headline": _rolling_window_headline(
                    window_label=str(window["window_label"]),
                    summary=summary,
                    decision=decision,
                ),
            }
        )

    failed_windows = [item for item in window_results if str(item.get("go_hold", {}).get("decision", "hold")) != "go"]
    failure_clusters = _analyze_failure_clusters(failed_windows)
    positive_dependency = _analyze_positive_slice_dependency(window_results)
    classification = _classify_rolling_stability(
        window_results=window_results,
        failure_clusters=failure_clusters,
        positive_dependency=positive_dependency,
    )

    aggregate_snaps: List[Mapping[str, Any]] = []
    for window in windows:
        aggregate_snaps.extend(window["snapshots"])
    aggregate_summary = evaluate_family_variant(
        snapshots=aggregate_snaps,
        family=family,
        policy=policy,
        feature_frame=feature_frame,
        outcome_lookup=outcome_lookup,
        max_staleness_hours=max_staleness_hours,
    )
    aggregate_decision = _evaluate_go_hold(aggregate_summary, guardrails)

    return {
        "family": family,
        "variant": variant,
        "window_size": int(window_size),
        "max_windows": int(max_windows),
        "snapshot_count_available": int(len(snapshots)),
        "window_count_evaluated": int(len(window_results)),
        "window_results": window_results,
        "aggregate": {
            "summary": aggregate_summary,
            "go_hold": aggregate_decision,
            "headline": _rolling_window_headline(
                window_label="aggregate",
                summary=aggregate_summary,
                decision=aggregate_decision,
            ),
        },
        "failure_cluster_analysis": failure_clusters,
        "positive_slice_dependency": positive_dependency,
        "rolling_stability_classification": classification,
        "promotion_guardrails": guardrails,
        "macro_disposition": "remain_deprioritized",
        "state_engineering_disposition": "hold",
    }


def render_order_flow_rolling_stability_memo(payload: Mapping[str, Any]) -> str:
    window_results = payload.get("window_results", []) if isinstance(payload.get("window_results"), list) else []
    classification = (
        payload.get("rolling_stability_classification", {})
        if isinstance(payload.get("rolling_stability_classification"), Mapping)
        else {}
    )
    clusters = (
        payload.get("failure_cluster_analysis", {})
        if isinstance(payload.get("failure_cluster_analysis"), Mapping)
        else {}
    )
    dependency = (
        payload.get("positive_slice_dependency", {})
        if isinstance(payload.get("positive_slice_dependency"), Mapping)
        else {}
    )

    lines: List[str] = []
    lines.append("# Order-Flow Rolling Stability Diagnosis")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Classification: **{classification.get('classification', 'not_enough_evidence')}**")
    lines.append(f"- Recommended disposition: **{classification.get('disposition', 'deprioritize_for_now')}**")
    lines.append("")

    lines.append("## Pass/Fail Summary")
    lines.append(f"- Windows evaluated: {classification.get('total_windows', 0)}")
    lines.append(f"- Pass count: {classification.get('pass_count', 0)}")
    lines.append(f"- Fail count: {classification.get('fail_count', 0)}")
    lines.append(f"- Pass rate: {classification.get('pass_rate', 0.0):.3f}")
    lines.append(f"- Instability concentrated in recent period: {classification.get('instability_concentrated_in_recent_period')}")
    lines.append(f"- Persistent across history: {classification.get('persistent_across_history')}")
    lines.append("")

    lines.append("## Per-Window Results")
    lines.append("| Window | Sample Count | Net Return Delta | Accuracy Delta | Harmful Veto Rate | Veto Precision | Horizon Coverage | Regime Coverage | Decision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in window_results:
        if not isinstance(item, Mapping):
            continue
        row = item.get("headline", {}) if isinstance(item.get("headline"), Mapping) else {}
        lines.append(
            "| {window} | {sample} | {ret} | {acc} | {harm:.3f} | {vp:.3f} | {h_cov} | {r_cov} | {decision} |".format(
                window=str(row.get("window", "unknown")),
                sample=int(row.get("sample_count", 0)),
                ret=(f"{float(row.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(row.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                acc=(f"{float(row.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(row.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                harm=float(row.get("harmful_veto_rate", 0.0) or 0.0),
                vp=float(row.get("veto_precision", 0.0) or 0.0),
                h_cov=int(row.get("target_horizon_coverage", 0)),
                r_cov=int(row.get("target_regime_coverage", 0)),
                decision=str(row.get("go_hold", "hold")),
            )
        )
    lines.append("")

    lines.append("## Failure Root-Cause Clustering")
    lines.append(f"- Dominant failing horizon: {clusters.get('dominant_failure_horizon')}")
    lines.append(f"- Dominant failing regime: {clusters.get('dominant_failure_regime')}")
    lines.append(f"- Dominant failing confidence bucket: {clusters.get('dominant_failure_confidence_bucket')}")
    lines.append("")

    lines.append("## Narrow-Slice Dependency")
    horizon_dep = dependency.get("horizon", {}) if isinstance(dependency.get("horizon"), Mapping) else {}
    regime_dep = dependency.get("regime", {}) if isinstance(dependency.get("regime"), Mapping) else {}
    conf_dep = dependency.get("confidence", {}) if isinstance(dependency.get("confidence"), Mapping) else {}
    lines.append(f"- Dominant positive horizon: {horizon_dep.get('dominant_positive_bucket')}")
    lines.append(f"- Dominant positive horizon share: {horizon_dep.get('dominant_positive_share')}")
    lines.append(f"- Dominant positive regime: {regime_dep.get('dominant_positive_bucket')}")
    lines.append(f"- Dominant positive regime share: {regime_dep.get('dominant_positive_share')}")
    lines.append(f"- Dominant positive confidence bucket: {conf_dep.get('dominant_positive_bucket')}")
    lines.append(f"- Dominant positive confidence share: {conf_dep.get('dominant_positive_share')}")
    lines.append("")

    scope = (
        classification.get("narrowest_credible_scope_if_conditional", {})
        if isinstance(classification.get("narrowest_credible_scope_if_conditional"), Mapping)
        else {}
    )
    lines.append("## Conditional Scope (If Any)")
    lines.append(f"- Horizon: {scope.get('horizon')}")
    lines.append(f"- Regime: {scope.get('regime')}")
    lines.append(f"- Confidence bucket: {scope.get('confidence_bucket')}")
    lines.append("")

    lines.append("## Diagnostic Conclusion")
    lines.append("- This workflow is diagnostic-only and does not change live or shadow-production behavior.")
    if str(classification.get("disposition")) == "deprioritize_for_now":
        lines.append("- Recommendation: deprioritize_for_now unless materially new evidence appears.")
    elif str(classification.get("disposition")) == "narrow_scope_followup_validation":
        lines.append("- Recommendation: narrow_scope_followup_validation within the listed conditional slice.")
    elif str(classification.get("disposition")) == "continue_deeper_validation":
        lines.append("- Recommendation: continue_deeper_validation with strict guardrail checks.")
    else:
        lines.append("- Recommendation: collect more evidence before additional experimentation.")
    lines.append("")

    return "\n".join(lines) + "\n"


STATE_NARROW_CONFIDENCE_BUCKETS: tuple[str, ...] = ("high", "mid", "low")


def _state_narrow_scope_thresholds() -> Dict[str, Any]:
    return {
        "minimum_snapshot_count": 100,
        "minimum_shadow_trade_count": 25,
        "minimum_changed_snapshot_count": 10,
        "minimum_net_return_proxy_improvement": 0.0003,
        "minimum_direction_accuracy_improvement": 0.01,
        "maximum_harmful_veto_rate": 0.40,
        "minimum_veto_precision": 0.55,
    }


def _baseline_slice_context(snapshot: Mapping[str, Any]) -> Dict[str, str]:
    prompt_ready = snapshot.get("prompt_ready_summary") if isinstance(snapshot.get("prompt_ready_summary"), Mapping) else {}
    outlook = prompt_ready.get("market_outlook_strategy") if isinstance(prompt_ready.get("market_outlook_strategy"), Mapping) else {}
    preferred_horizon = str(outlook.get("preferred_horizon") or "unknown")
    predictions = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), Mapping) else {}
    entry = predictions.get(preferred_horizon, {}) if isinstance(predictions, Mapping) else {}
    regime = _normalize_regime(entry.get("regime_state") if isinstance(entry, Mapping) else "unknown")
    confidence = _to_float_or_none(entry.get("confidence_score") if isinstance(entry, Mapping) else None)
    return {
        "horizon": preferred_horizon,
        "regime": regime,
        "confidence_bucket": _confidence_bucket(confidence),
    }


def _state_slice_scope_label(filters: Mapping[str, str]) -> str:
    ordered = [
        ("horizon", filters.get("horizon")),
        ("regime", filters.get("regime")),
        ("confidence_bucket", filters.get("confidence_bucket")),
    ]
    parts = [f"{key}={value}" for key, value in ordered if value]
    return "all" if not parts else " | ".join(parts)


def _state_slice_candidate_filters() -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for horizon in TARGET_HORIZONS:
        candidates.append({"horizon": horizon})
    for regime in ("neutral", "chop"):
        candidates.append({"regime": regime})
    for confidence_bucket in STATE_NARROW_CONFIDENCE_BUCKETS:
        candidates.append({"confidence_bucket": confidence_bucket})
    for horizon in TARGET_HORIZONS:
        for regime in ("neutral", "chop"):
            candidates.append({"horizon": horizon, "regime": regime})
    for horizon in TARGET_HORIZONS:
        for confidence_bucket in STATE_NARROW_CONFIDENCE_BUCKETS:
            candidates.append({"horizon": horizon, "confidence_bucket": confidence_bucket})
    for regime in ("neutral", "chop"):
        for confidence_bucket in STATE_NARROW_CONFIDENCE_BUCKETS:
            candidates.append({"regime": regime, "confidence_bucket": confidence_bucket})
    for horizon in TARGET_HORIZONS:
        for regime in ("neutral", "chop"):
            for confidence_bucket in STATE_NARROW_CONFIDENCE_BUCKETS:
                candidates.append(
                    {
                        "horizon": horizon,
                        "regime": regime,
                        "confidence_bucket": confidence_bucket,
                    }
                )
    return candidates


def _matches_state_slice(context: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    for key, value in filters.items():
        if str(context.get(key) or "") != str(value):
            return False
    return True


def _assess_state_narrow_scope_candidate(
    summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    delta = summary.get("overall_delta", {}) if isinstance(summary.get("overall_delta"), Mapping) else {}
    shadow = summary.get("shadow", {}) if isinstance(summary.get("shadow"), Mapping) else {}
    snapshot_count = int(shadow.get("snapshot_count", 0) or 0)
    shadow_trade_count = int(shadow.get("trade_count", 0) or 0)
    changed_snapshot_count = int(summary.get("changed_snapshot_count", 0) or 0)
    net_delta = _to_float_or_none(delta.get("net_return_proxy_mean_delta"))
    accuracy_delta = _to_float_or_none(delta.get("direction_accuracy_proxy_delta"))
    veto_count = int(summary.get("veto_count", 0) or 0)
    removed_good = int(summary.get("removed_good_trade_count", 0) or 0)
    harmful_veto_rate = (removed_good / veto_count) if veto_count > 0 else 0.0
    veto_precision = float(summary.get("veto_precision", 0.0) or 0.0)

    checks = {
        "snapshot_count": snapshot_count >= int(thresholds["minimum_snapshot_count"]),
        "shadow_trade_count": shadow_trade_count >= int(thresholds["minimum_shadow_trade_count"]),
        "changed_snapshot_count": changed_snapshot_count >= int(thresholds["minimum_changed_snapshot_count"]),
        "net_return_proxy_mean_delta": (
            net_delta is not None and net_delta >= float(thresholds["minimum_net_return_proxy_improvement"])
        ),
        "direction_accuracy_proxy_delta": (
            accuracy_delta is not None and accuracy_delta >= float(thresholds["minimum_direction_accuracy_improvement"])
        ),
        "harmful_veto_rate": harmful_veto_rate <= float(thresholds["maximum_harmful_veto_rate"]),
        "veto_precision": veto_precision >= float(thresholds["minimum_veto_precision"]),
    }

    viable = all(checks.values())
    classification = "negative"
    if viable:
        classification = "viable"
    elif (net_delta is not None and net_delta > 0.0) and (accuracy_delta is not None and accuracy_delta > 0.0):
        classification = "positive_but_too_sparse"
    elif changed_snapshot_count == 0:
        classification = "inert"

    return {
        "classification": classification,
        "viable": viable,
        "checks": checks,
        "snapshot_count": snapshot_count,
        "shadow_trade_count": shadow_trade_count,
        "changed_snapshot_count": changed_snapshot_count,
        "net_return_proxy_mean_delta": net_delta,
        "direction_accuracy_proxy_delta": accuracy_delta,
        "harmful_veto_rate": harmful_veto_rate,
        "veto_precision": veto_precision,
    }


def _state_candidate_rank_tuple(candidate: Mapping[str, Any]) -> tuple[float, float, int, int, int]:
    assessment = candidate.get("assessment", {}) if isinstance(candidate.get("assessment"), Mapping) else {}
    classification = str(assessment.get("classification") or "negative")
    priority = {
        "viable": 3,
        "positive_but_too_sparse": 2,
        "inert": 1,
        "negative": 0,
    }.get(classification, 0)
    return (
        float(priority),
        float(_to_float_or_none(assessment.get("net_return_proxy_mean_delta")) or -999.0),
        int(assessment.get("shadow_trade_count", 0) or 0),
        int(assessment.get("snapshot_count", 0) or 0),
        int(assessment.get("changed_snapshot_count", 0) or 0),
    )


def run_state_engineering_narrow_scope_followup(
    *,
    history_path: Path,
    spot_dir: Path,
    max_staleness_hours: float,
    recent_window: int = 0,
    family: str = "state_engineering",
    variant: str = "weak_signal_veto_only",
) -> Dict[str, Any]:
    if family != "state_engineering":
        raise ValueError("This workflow is scoped to family='state_engineering'")

    snapshots = load_prediction_history(history_path)
    if recent_window > 0:
        snapshots = snapshots[-recent_window:]

    from src.runtime.family_shadow_simulator import load_spot_feature_frame

    feature_frame = load_spot_feature_frame(spot_dir)
    ohlcv_outcomes = load_spot_ohlcv_with_outcomes(spot_dir, horizons=TARGET_HORIZONS)
    outcome_lookup = _build_outcome_lookup(ohlcv_outcomes, horizons=TARGET_HORIZONS)
    policies = {policy.name: policy for policy in default_family_policy_variants()}
    policy = policies.get(variant)
    if policy is None:
        raise ValueError(f"Unknown policy variant: {variant}")

    thresholds = _state_narrow_scope_thresholds()
    annotated = [
        {
            "snapshot": snapshot,
            "context": _baseline_slice_context(snapshot),
        }
        for snapshot in snapshots
    ]

    candidate_rankings: List[Dict[str, Any]] = []
    skipped_due_to_snapshot_count: List[Dict[str, Any]] = []
    for filters in _state_slice_candidate_filters():
        subset = [item["snapshot"] for item in annotated if _matches_state_slice(item["context"], filters)]
        if len(subset) < int(thresholds["minimum_snapshot_count"]):
            skipped_due_to_snapshot_count.append(
                {
                    "scope": _state_slice_scope_label(filters),
                    "filters": dict(filters),
                    "snapshot_count": int(len(subset)),
                }
            )
            continue

        summary = evaluate_family_variant(
            snapshots=subset,
            family=family,
            policy=policy,
            feature_frame=feature_frame,
            outcome_lookup=outcome_lookup,
            max_staleness_hours=max_staleness_hours,
        )
        assessment = _assess_state_narrow_scope_candidate(summary, thresholds)
        candidate_rankings.append(
            {
                "scope": _state_slice_scope_label(filters),
                "filters": dict(filters),
                "summary": summary,
                "assessment": assessment,
            }
        )

    candidate_rankings.sort(key=_state_candidate_rank_tuple, reverse=True)
    best_candidate = candidate_rankings[0] if candidate_rankings else None
    best_assessment = best_candidate.get("assessment", {}) if isinstance(best_candidate, Mapping) else {}

    final_decision = "deprioritize_for_now"
    reason = "No state_engineering slice met minimum snapshot-count requirements for narrow-scope follow-up."
    if isinstance(best_candidate, Mapping):
        classification = str(best_assessment.get("classification") or "negative")
        if classification == "viable":
            final_decision = "continue_narrow_scope_validation"
            reason = "A narrow state_engineering slice cleared minimum outcome and sample thresholds."
        elif classification == "positive_but_too_sparse":
            reason = "The best positive state_engineering slice remained too sparse after vetoes to justify further replay promotion."
        elif classification == "inert":
            reason = "The strongest narrow slice produced no meaningful state_engineering effect on baseline decisions."
        else:
            reason = "The strongest narrow slice remained net-negative or failed basic quality checks."

    return {
        "family": family,
        "variant": variant,
        "recent_window": int(recent_window),
        "snapshot_count_available": int(len(snapshots)),
        "thresholds": thresholds,
        "candidate_count_evaluated": int(len(candidate_rankings)),
        "candidate_rankings": candidate_rankings,
        "skipped_due_to_snapshot_count": skipped_due_to_snapshot_count,
        "best_candidate": best_candidate,
        "final_recommendation": {
            "decision": final_decision,
            "reason": reason,
            "close_state_engineering_for_now": final_decision == "deprioritize_for_now",
        },
    }


def render_state_engineering_narrow_scope_memo(payload: Mapping[str, Any]) -> str:
    best_candidate = payload.get("best_candidate", {}) if isinstance(payload.get("best_candidate"), Mapping) else {}
    best_assessment = best_candidate.get("assessment", {}) if isinstance(best_candidate.get("assessment"), Mapping) else {}
    best_summary = best_candidate.get("summary", {}) if isinstance(best_candidate.get("summary"), Mapping) else {}
    final_recommendation = (
        payload.get("final_recommendation", {})
        if isinstance(payload.get("final_recommendation"), Mapping)
        else {}
    )

    lines: List[str] = []
    lines.append("# State-Engineering Narrow-Scope Follow-Up")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Decision: **{final_recommendation.get('decision', 'deprioritize_for_now')}**")
    lines.append(f"- Reason: {final_recommendation.get('reason', 'n/a')}")
    lines.append(f"- Candidate slices evaluated: {payload.get('candidate_count_evaluated', 0)}")
    lines.append("")
    lines.append("## Best Slice")
    if best_candidate:
        delta = best_summary.get("overall_delta", {}) if isinstance(best_summary.get("overall_delta"), Mapping) else {}
        lines.append(f"- Scope: {best_candidate.get('scope')}")
        lines.append(f"- Classification: {best_assessment.get('classification')}")
        lines.append(f"- Snapshot count: {best_assessment.get('snapshot_count')}")
        lines.append(f"- Shadow trade count: {best_assessment.get('shadow_trade_count')}")
        lines.append(f"- Changed snapshot count: {best_assessment.get('changed_snapshot_count')}")
        lines.append(f"- Net return delta: {delta.get('net_return_proxy_mean_delta')}")
        lines.append(f"- Accuracy delta: {delta.get('direction_accuracy_proxy_delta')}")
        lines.append(f"- Veto precision: {best_assessment.get('veto_precision')}")
        lines.append(f"- Harmful veto rate: {best_assessment.get('harmful_veto_rate')}")
    else:
        lines.append("- No slice cleared minimum snapshot-count requirements.")
    lines.append("")
    lines.append("## Top Candidates")
    lines.append("| Scope | Class | Snapshots | Shadow Trades | Changed | Net Return Delta | Accuracy Delta | Veto Precision | Harmful Veto Rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for candidate in (payload.get("candidate_rankings", []) if isinstance(payload.get("candidate_rankings"), list) else [])[:10]:
        if not isinstance(candidate, Mapping):
            continue
        assessment = candidate.get("assessment", {}) if isinstance(candidate.get("assessment"), Mapping) else {}
        lines.append(
            "| {scope} | {classification} | {snapshots} | {trades} | {changed} | {ret} | {acc} | {vp:.3f} | {harm:.3f} |".format(
                scope=str(candidate.get("scope", "unknown")),
                classification=str(assessment.get("classification", "unknown")),
                snapshots=int(assessment.get("snapshot_count", 0) or 0),
                trades=int(assessment.get("shadow_trade_count", 0) or 0),
                changed=int(assessment.get("changed_snapshot_count", 0) or 0),
                ret=(f"{float(assessment.get('net_return_proxy_mean_delta')):.6f}" if _to_float_or_none(assessment.get("net_return_proxy_mean_delta")) is not None else "n/a"),
                acc=(f"{float(assessment.get('direction_accuracy_proxy_delta')):.4f}" if _to_float_or_none(assessment.get("direction_accuracy_proxy_delta")) is not None else "n/a"),
                vp=float(assessment.get("veto_precision", 0.0) or 0.0),
                harm=float(assessment.get("harmful_veto_rate", 0.0) or 0.0),
            )
        )
    lines.append("")
    lines.append("## Thresholds")
    for key, value in (payload.get("thresholds", {}) if isinstance(payload.get("thresholds"), Mapping) else {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines) + "\n"
