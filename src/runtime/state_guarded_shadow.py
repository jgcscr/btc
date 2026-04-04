from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from src.runtime.family_shadow_simulator import (
    FamilyShadowPolicy,
    load_prediction_history,
    load_spot_feature_frame,
    replay_snapshot_with_family_shadow,
    resolve_family_snapshot_state,
    summarize_family_replay,
)


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _validate_guarded_scope(narrow_scope_payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(narrow_scope_payload, Mapping):
        return {
            "ready_for_guarded_shadow": False,
            "decision": None,
            "best_scope": None,
            "blockers": ["missing_state_narrow_scope_artifact"],
        }

    recommendation = (
        narrow_scope_payload.get("final_recommendation")
        if isinstance(narrow_scope_payload.get("final_recommendation"), Mapping)
        else {}
    )
    best_candidate = (
        narrow_scope_payload.get("best_candidate")
        if isinstance(narrow_scope_payload.get("best_candidate"), Mapping)
        else {}
    )
    decision = str(recommendation.get("decision") or "") or None
    best_scope = str(best_candidate.get("scope") or "") or None

    blockers: List[str] = []
    if decision != "continue_narrow_scope_validation":
        blockers.append("state_family_not_currently_validated_for_narrow_followup")
    if best_scope != "horizon=4h":
        blockers.append("best_narrow_scope_is_not_exactly_horizon_4h")

    return {
        "ready_for_guarded_shadow": not blockers,
        "decision": decision,
        "best_scope": best_scope,
        "blockers": blockers,
    }


def _build_fail_close_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    status_counts = summary.get("state_status_counts", {}) if isinstance(summary.get("state_status_counts"), Mapping) else {}
    unavailable = int(status_counts.get("unavailable", 0) or 0)
    stale = int(status_counts.get("stale", 0) or 0)
    return {
        "mode": "fail_closed",
        "disable_when_feature_state_unavailable": True,
        "disable_when_feature_state_stale": True,
        "disabled_unavailable_snapshot_count": unavailable,
        "disabled_stale_snapshot_count": stale,
        "disabled_snapshot_count": unavailable + stale,
    }


def run_state_engineering_guarded_shadow(
    *,
    history_path: Path,
    spot_dir: Path,
    narrow_scope_artifact_path: Path,
    max_staleness_hours: float,
    recent_window: int = 0,
) -> Dict[str, Any]:
    narrow_scope_payload = _load_json(narrow_scope_artifact_path)
    scope_validation = _validate_guarded_scope(narrow_scope_payload)

    snapshots = load_prediction_history(history_path)
    if recent_window > 0:
        snapshots = snapshots[-recent_window:]

    payload: Dict[str, Any] = {
        "family": "state_engineering",
        "variant": "state_engineering_guarded_4h_only",
        "policy": {
            "name": "state_engineering_guarded_4h_only",
            "description": "Guarded 4h-only weak-signal veto backed by narrow-scope validation.",
            "enforcement_mode": "weak_signal_veto",
            "enabled_horizons": ["4h"],
        },
        "scope_validation": scope_validation,
        "guardrails": {
            "required_narrow_scope_decision": "continue_narrow_scope_validation",
            "required_best_scope": "horizon=4h",
            "max_staleness_hours": float(max_staleness_hours),
            "fail_close_mode": "disable_policy_effects_when_state_diagnostics_missing_or_stale",
        },
        "snapshot_count_available": int(len(snapshots)),
    }
    if not scope_validation.get("ready_for_guarded_shadow"):
        payload["summary"] = None
        payload["fail_close_summary"] = {
            "mode": "fail_closed",
            "disabled_snapshot_count": 0,
        }
        payload["readiness"] = {
            "decision": "blocked",
            "blockers": list(scope_validation.get("blockers", [])),
        }
        return payload

    feature_frame = load_spot_feature_frame(spot_dir)
    policy = FamilyShadowPolicy(
        name="state_engineering_guarded_4h_only",
        description="Guarded 4h-only weak-signal veto backed by narrow-scope validation.",
        enforcement_mode="weak_signal_veto",
        enabled_horizons=(4.0,),
    )

    replay_results = []
    replay_snapshots: List[Mapping[str, Any]] = []
    for snapshot in snapshots:
        snapshot_ts = pd.to_datetime(snapshot.get("generated_at"), utc=True, errors="coerce")
        if pd.isna(snapshot_ts):
            continue
        state = resolve_family_snapshot_state(
            snapshot_ts=snapshot_ts,
            feature_frame=feature_frame,
            max_staleness_hours=max_staleness_hours,
        )
        replay = replay_snapshot_with_family_shadow(
            snapshot,
            state=state,
            family="state_engineering",
            policy=policy,
        )
        replay_results.append(replay)
        replay_snapshots.append(snapshot)

    summary = summarize_family_replay(
        snapshots=replay_snapshots,
        replay_results=replay_results,
    )
    fail_close_summary = _build_fail_close_summary(summary)
    non_4h_changes = sum(
        1
        for replay in replay_results
        for horizon_label in replay.changed_horizons
        if str(horizon_label) != "4h"
    )
    readiness_blockers: List[str] = []
    if non_4h_changes > 0:
        readiness_blockers.append("guarded_runner_changed_non_4h_horizons")

    payload["summary"] = summary
    payload["fail_close_summary"] = fail_close_summary
    payload["readiness"] = {
        "decision": "shadow_validation_active" if not readiness_blockers else "blocked",
        "blockers": readiness_blockers,
        "non_4h_changed_horizon_count": int(non_4h_changes),
    }
    return payload


def render_state_engineering_guarded_shadow_markdown(payload: Mapping[str, Any]) -> str:
    scope_validation = payload.get("scope_validation", {}) if isinstance(payload.get("scope_validation"), Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    fail_close = payload.get("fail_close_summary", {}) if isinstance(payload.get("fail_close_summary"), Mapping) else {}
    readiness = payload.get("readiness", {}) if isinstance(payload.get("readiness"), Mapping) else {}

    lines: List[str] = []
    lines.append("# State-Engineering Guarded 4h Shadow Runner")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Readiness: **{readiness.get('decision', 'blocked')}**")
    lines.append(f"- Narrow-scope decision: {scope_validation.get('decision')}")
    lines.append(f"- Best validated scope: {scope_validation.get('best_scope')}")
    lines.append("")
    lines.append("## Policy")
    lines.append("- Family: state_engineering")
    lines.append("- Enforcement mode: weak_signal_veto")
    lines.append("- Eligible horizon: 4h only")
    lines.append("- Production behavior: unchanged; this is shadow-only validation")
    lines.append("")
    lines.append("## Fail-Close Rules")
    lines.append(f"- Disable policy effects when feature state is unavailable: {fail_close.get('disable_when_feature_state_unavailable')}")
    lines.append(f"- Disable policy effects when feature state is stale: {fail_close.get('disable_when_feature_state_stale')}")
    lines.append(f"- Disabled snapshots (unavailable): {fail_close.get('disabled_unavailable_snapshot_count', 0)}")
    lines.append(f"- Disabled snapshots (stale): {fail_close.get('disabled_stale_snapshot_count', 0)}")
    lines.append(f"- Disabled snapshots (total): {fail_close.get('disabled_snapshot_count', 0)}")
    lines.append("")
    if summary:
        lines.append("## Replay Summary")
        lines.append(f"- Snapshots replayed: {summary.get('snapshot_count', 0)}")
        lines.append(f"- Changed snapshots: {summary.get('changed_snapshot_count', 0)}")
        lines.append(f"- Assessment: {summary.get('assessment')}")
        lines.append(f"- Beneficial blocks: {summary.get('beneficial_blocks', 0)}")
        lines.append(f"- Harmful blocks: {summary.get('harmful_blocks', 0)}")
        lines.append(f"- 4h changed trade actions: {summary.get('per_horizon_deltas', {}).get('4h', {}).get('changed_trade_action', 0)}")
        lines.append("")
    blockers = readiness.get("blockers", []) if isinstance(readiness.get("blockers"), list) else []
    lines.append("## Blockers")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "_build_fail_close_summary",
    "_validate_guarded_scope",
    "render_state_engineering_guarded_shadow_markdown",
    "run_state_engineering_guarded_shadow",
]