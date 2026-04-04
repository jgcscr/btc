from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.runtime.macro_shadow_simulator import (
    MacroShadowPolicy,
    build_snapshot_delta_rows,
    default_policy_variants,
    load_macro_features,
    load_prediction_history,
    render_shadow_markdown_report,
    replay_snapshot_with_macro_shadow,
    resolve_macro_state,
    run_policy_sweep,
    summarize_replay_results,
)


DEFAULT_CONFIG = "configs/run_refresh_and_predict.live_conservative_binance_only.yaml"
DEFAULT_HISTORY = "artifacts/predictions/history.json"
DEFAULT_MACRO = "data/processed/macro/daily_features.parquet"
DEFAULT_PARITY_AUDIT = "artifacts/analysis/train_live_feature_parity_latest.json"
DEFAULT_JSON_OUT = "artifacts/analysis/macro_shadow_enforcement_latest.json"
DEFAULT_MD_OUT = "artifacts/analysis/macro_shadow_enforcement_latest.md"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _load_parity_context(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay prediction history with shadow-only macro enforcement policy sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Live profile config path.")
    parser.add_argument("--history-path", default=DEFAULT_HISTORY, help="Prediction history JSON path.")
    parser.add_argument("--macro-path", default=DEFAULT_MACRO, help="Macro feature parquet path.")
    parser.add_argument("--parity-audit-path", default=DEFAULT_PARITY_AUDIT, help="Train/live feature parity audit JSON.")
    parser.add_argument("--recent-window", type=int, default=2000, help="Number of most recent snapshots to replay.")
    parser.add_argument("--output-json", default=DEFAULT_JSON_OUT, help="JSON output artifact path.")
    parser.add_argument("--output-md", default=DEFAULT_MD_OUT, help="Markdown output artifact path.")
    parser.add_argument("--max-staleness-hours", type=float, default=24.0, help="Macro freshness threshold for shadow gating.")
    parser.add_argument("--disable-sweep", action="store_true", help="Run only baseline policy instead of multi-variant sweep.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = _load_config(Path(args.config))
    parity = _load_parity_context(Path(args.parity_audit_path))

    history = load_prediction_history(Path(args.history_path))
    if not history:
        raise SystemExit("No prediction history snapshots found.")

    if args.recent_window > 0:
        history = history[-args.recent_window :]

    macro_frame = load_macro_features(Path(args.macro_path))
    baseline_policy = MacroShadowPolicy(max_staleness_hours=float(args.max_staleness_hours))

    replay_results = []
    replay_snapshots = []
    for snapshot in history:
        snap_ts = pd.to_datetime(snapshot.get("generated_at"), utc=True, errors="coerce")
        if pd.isna(snap_ts):
            continue
        macro_state = resolve_macro_state(
            snapshot_ts=snap_ts,
            macro_frame=macro_frame,
            policy=baseline_policy,
        )
        replay = replay_snapshot_with_macro_shadow(
            snapshot,
            macro_state=macro_state,
            policy=baseline_policy,
        )
        replay_results.append(replay)
        replay_snapshots.append(snapshot)

    summary = summarize_replay_results(
        snapshots=replay_snapshots,
        replay_results=replay_results,
    )
    delta_rows = build_snapshot_delta_rows(
        snapshots=replay_snapshots,
        replay_results=replay_results,
        limit=300,
    )

    sweep_payload = None
    if not args.disable_sweep:
        sweep_payload = run_policy_sweep(
            snapshots=replay_snapshots,
            macro_frame=macro_frame,
            policies=default_policy_variants(max_staleness_hours=float(args.max_staleness_hours)),
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "history_path": str(args.history_path),
        "macro_path": str(args.macro_path),
        "recent_window": int(args.recent_window),
        "policy": {
            "name": baseline_policy.name,
            "description": baseline_policy.description,
            "enforcement_mode": baseline_policy.enforcement_mode,
            "min_horizon_hours": baseline_policy.min_horizon_hours,
            "max_staleness_hours": baseline_policy.max_staleness_hours,
            "block_conflict_confidence_max": baseline_policy.block_conflict_confidence_max,
            "weak_trade_expected_value_max": baseline_policy.weak_trade_expected_value_max,
            "strong_trade_expected_value_min": baseline_policy.strong_trade_expected_value_min,
            "strong_trade_confidence_min": baseline_policy.strong_trade_confidence_min,
            "macro_features_used": [
                "macro_dollar_proxy_change_1d",
                "macro_us10y_change_1d",
                "macro_eurusd_change_1d",
            ],
            "macro_bias_logic": "risk_on_votes>=2=>long, <=-2=>short, else neutral",
        },
        "parity_audit_context": {
            "ignored_families": parity.get("ignored_families", []),
            "stale_tolerated_families": parity.get("stale_tolerated_families", []),
            "top_candidates": parity.get("likely_untapped_candidates", [])[:3],
            "leakage_safe_evidence": parity.get("leakage_safe_evidence", {}),
        },
        "live_profile_context": {
            "targets": cfg.get("targets", []),
            "feature_coverage_policy": cfg.get("feature_coverage_policy", {}),
        },
        "summary": summary,
        "snapshot_deltas": delta_rows,
        "sweep": sweep_payload,
        "conclusion": {
            "assessment": (
                sweep_payload.get("recommendation", {}).get("best_assessment")
                if isinstance(sweep_payload, dict)
                else summary.get("assessment", "neutral")
            ),
            "move_to_next_validation_stage": (
                bool(sweep_payload.get("recommendation", {}).get("advance_to_next_validation_stage"))
                if isinstance(sweep_payload, dict)
                else bool(summary.get("assessment") == "beneficial")
            ),
            "note": "Shadow replay only; no live behavior change applied.",
        },
    }

    json_out = Path(args.output_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_out = Path(args.output_md)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_shadow_markdown_report(payload), encoding="utf-8")

    print(f"Wrote macro shadow JSON: {json_out}")
    print(f"Wrote macro shadow report: {md_out}")


if __name__ == "__main__":
    main()
