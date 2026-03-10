from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _horizon_sort_key(label: str) -> Tuple[float, str]:
    if label.endswith("h"):
        try:
            return (float(label[:-1]), label)
        except ValueError:
            return (float("inf"), label)
    return (float("inf"), label)


def _extract_horizon_row(snapshot: Dict[str, Any], horizon: str) -> Dict[str, Any]:
    horizons_obj = snapshot.get("horizons") if isinstance(snapshot.get("horizons"), dict) else {}
    row = horizons_obj.get(horizon) if isinstance(horizons_obj.get(horizon), dict) else {}
    trade_decision = row.get("trade_decision") if isinstance(row.get("trade_decision"), dict) else {}
    abstention = row.get("abstention") if isinstance(row.get("abstention"), dict) else {}
    volatility = row.get("volatility") if isinstance(row.get("volatility"), dict) else {}

    return {
        "signal_ensemble": row.get("signal_ensemble"),
        "trade_action": row.get("trade_action"),
        "trade_decision_triggered": trade_decision.get("triggered"),
        "abstention_triggered": abstention.get("triggered"),
        "p_up": _num(row.get("p_up")),
        "expected_value": _num(row.get("expected_value")),
        "expected_net": _num(trade_decision.get("expected_net")),
        "regime_state": row.get("regime_state"),
        "volatility_realized_24h": _num(volatility.get("volatility_realized_24h")),
        "volatility_ewm_24h": _num(volatility.get("volatility_ewm_24h")),
        "volatility_garch_like": _num(volatility.get("volatility_garch_like")),
    }


def _float_delta(midband_value: Optional[float], default_value: Optional[float]) -> Optional[float]:
    if midband_value is None or default_value is None:
        return None
    return float(midband_value - default_value)


def _build_horizon_diff(default_row: Dict[str, Any], midband_row: Dict[str, Any]) -> Dict[str, Any]:
    signal_delta = None
    try:
        if default_row.get("signal_ensemble") is not None and midband_row.get("signal_ensemble") is not None:
            signal_delta = int(midband_row.get("signal_ensemble")) - int(default_row.get("signal_ensemble"))
    except Exception:
        signal_delta = None

    operational_diff = (
        default_row.get("trade_action") != midband_row.get("trade_action")
        or default_row.get("signal_ensemble") != midband_row.get("signal_ensemble")
    )
    decision_state_diff = (
        default_row.get("trade_decision_triggered") != midband_row.get("trade_decision_triggered")
        or default_row.get("abstention_triggered") != midband_row.get("abstention_triggered")
    )
    score_diff = any(
        [
            default_row.get("p_up") != midband_row.get("p_up"),
            default_row.get("expected_value") != midband_row.get("expected_value"),
            default_row.get("expected_net") != midband_row.get("expected_net"),
            default_row.get("regime_state") != midband_row.get("regime_state"),
            default_row.get("volatility_realized_24h") != midband_row.get("volatility_realized_24h"),
            default_row.get("volatility_ewm_24h") != midband_row.get("volatility_ewm_24h"),
            default_row.get("volatility_garch_like") != midband_row.get("volatility_garch_like"),
        ]
    )

    if operational_diff:
        diff_class = "operational"
    elif decision_state_diff:
        diff_class = "decision-state-only"
    elif score_diff:
        diff_class = "score-only"
    else:
        diff_class = "none"

    expected_delta_basis = "expected_net" if (
        default_row.get("expected_net") is not None and midband_row.get("expected_net") is not None
    ) else "expected_value"

    return {
        "default": default_row,
        "midband": midband_row,
        "deltas_midband_minus_default": {
            "signal_ensemble": signal_delta,
            "p_up": _float_delta(midband_row.get("p_up"), default_row.get("p_up")),
            "expected_value": _float_delta(midband_row.get("expected_value"), default_row.get("expected_value")),
            "expected_net": _float_delta(midband_row.get("expected_net"), default_row.get("expected_net")),
        },
        "differences": {
            "trade_action_changed": bool(default_row.get("trade_action") != midband_row.get("trade_action")),
            "trade_decision_triggered_changed": bool(
                default_row.get("trade_decision_triggered") != midband_row.get("trade_decision_triggered")
            ),
            "abstention_triggered_changed": bool(
                default_row.get("abstention_triggered") != midband_row.get("abstention_triggered")
            ),
            "regime_state_changed": bool(default_row.get("regime_state") != midband_row.get("regime_state")),
            "volatility_changed": bool(
                default_row.get("volatility_realized_24h") != midband_row.get("volatility_realized_24h")
                or default_row.get("volatility_ewm_24h") != midband_row.get("volatility_ewm_24h")
                or default_row.get("volatility_garch_like") != midband_row.get("volatility_garch_like")
            ),
        },
        "difference_class": diff_class,
        "expected_delta_basis": expected_delta_basis,
    }


def _is_actionable(horizon_row: Dict[str, Any]) -> bool:
    action = str(horizon_row.get("trade_action") or "").lower()
    return action not in {"", "hold"}


def _collect_snapshots(run_root: Path) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    if not run_root.exists():
        return snapshots

    for run_path in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        summary_dir = run_path / "summary"
        manifest_path = summary_dir / "workflow_manifest.json"
        snapshot_path = summary_dir / "live_predictions_snapshot.json"
        if not manifest_path.exists() or not snapshot_path.exists():
            continue

        try:
            manifest = _load_json(manifest_path)
            snapshot = _load_json(snapshot_path)
        except Exception:
            continue

        profile = manifest.get("profile") if isinstance(manifest.get("profile"), dict) else {}
        profile_id = str(profile.get("id", "")).strip()
        if profile_id not in {"default_runtime", "midband_paper_evaluation"}:
            continue

        snapshots.append(
            {
                "run_id": run_path.name,
                "profile_id": profile_id,
                "profile_name": str(profile.get("name", profile_id)),
                "snapshot_path": snapshot_path,
                "snapshot": snapshot,
            }
        )
    return snapshots


def _match_pairs(default_runs: List[Dict[str, Any]], midband_runs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    default_idx = 0
    last_used_default = -1

    for midband in midband_runs:
        while default_idx < len(default_runs) and default_runs[default_idx]["run_id"] <= midband["run_id"]:
            default_idx += 1
        candidate_idx = default_idx - 1
        if candidate_idx <= last_used_default:
            continue
        if candidate_idx < 0:
            continue
        pairs.append((default_runs[candidate_idx], midband))
        last_used_default = candidate_idx

    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build longitudinal default-vs-midband report from per-run live snapshot artifacts."
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/reliability"))
    parser.add_argument("--include-run-id", type=str, default=None)
    parser.add_argument("--include-profile-id", type=str, default=None)
    parser.add_argument("--include-profile-name", type=str, default=None)
    parser.add_argument("--include-snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshots = _collect_snapshots(args.run_root)

    if args.include_snapshot is not None and args.include_run_id and args.include_profile_id:
        include_path = Path(args.include_snapshot)
        if include_path.exists():
            include_payload = _load_json(include_path)
            snapshots = [
                row
                for row in snapshots
                if not (
                    str(row.get("run_id")) == str(args.include_run_id)
                    and str(row.get("profile_id")) == str(args.include_profile_id)
                )
            ]
            snapshots.append(
                {
                    "run_id": str(args.include_run_id),
                    "profile_id": str(args.include_profile_id),
                    "profile_name": str(args.include_profile_name or args.include_profile_id),
                    "snapshot_path": include_path,
                    "snapshot": include_payload,
                }
            )

    snapshots = sorted(snapshots, key=lambda row: str(row.get("run_id", "")))
    default_runs = [row for row in snapshots if row["profile_id"] == "default_runtime"]
    midband_runs = [row for row in snapshots if row["profile_id"] == "midband_paper_evaluation"]
    pairs = _match_pairs(default_runs, midband_runs)

    horizon_trade_action_diff_counts: Dict[str, int] = defaultdict(int)
    horizon_abs_pup_deltas: Dict[str, List[float]] = defaultdict(list)
    horizon_abs_expected_deltas: Dict[str, List[float]] = defaultdict(list)

    pair_rows: List[Dict[str, Any]] = []
    pairs_with_score_diff = 0
    pairs_with_decision_state_diff = 0
    pairs_with_operational_diff = 0
    pairs_midband_actionable_only = 0
    pairs_default_actionable_only = 0

    for default_run, midband_run in pairs:
        default_snapshot = default_run["snapshot"]
        midband_snapshot = midband_run["snapshot"]

        default_horizons = default_snapshot.get("horizons") if isinstance(default_snapshot.get("horizons"), dict) else {}
        midband_horizons = midband_snapshot.get("horizons") if isinstance(midband_snapshot.get("horizons"), dict) else {}
        labels = sorted(set(default_horizons.keys()) | set(midband_horizons.keys()), key=_horizon_sort_key)

        pair_has_score = False
        pair_has_decision_state = False
        pair_has_operational = False
        pair_default_actionable = False
        pair_midband_actionable = False
        horizons_out: Dict[str, Any] = {}

        for label in labels:
            drow = _extract_horizon_row(default_snapshot, label)
            mrow = _extract_horizon_row(midband_snapshot, label)
            diff_row = _build_horizon_diff(drow, mrow)
            horizons_out[label] = diff_row

            diff_class = diff_row.get("difference_class")
            if diff_class == "score-only":
                pair_has_score = True
            elif diff_class == "decision-state-only":
                pair_has_decision_state = True
            elif diff_class == "operational":
                pair_has_operational = True

            if bool(diff_row["differences"]["trade_action_changed"]):
                horizon_trade_action_diff_counts[label] += 1

            pup_delta = diff_row["deltas_midband_minus_default"].get("p_up")
            if pup_delta is not None:
                horizon_abs_pup_deltas[label].append(abs(float(pup_delta)))

            expected_basis = str(diff_row.get("expected_delta_basis", "expected_value"))
            expected_delta = diff_row["deltas_midband_minus_default"].get(expected_basis)
            if expected_delta is not None:
                horizon_abs_expected_deltas[label].append(abs(float(expected_delta)))

            pair_default_actionable = pair_default_actionable or _is_actionable(drow)
            pair_midband_actionable = pair_midband_actionable or _is_actionable(mrow)

        if pair_has_score:
            pairs_with_score_diff += 1
        if pair_has_decision_state:
            pairs_with_decision_state_diff += 1
        if pair_has_operational:
            pairs_with_operational_diff += 1
        if pair_midband_actionable and not pair_default_actionable:
            pairs_midband_actionable_only += 1
        if pair_default_actionable and not pair_midband_actionable:
            pairs_default_actionable_only += 1

        pair_rows.append(
            {
                "default_run_id": default_run["run_id"],
                "midband_run_id": midband_run["run_id"],
                "default_snapshot": str(default_run["snapshot_path"]),
                "midband_snapshot": str(midband_run["snapshot_path"]),
                "horizons": horizons_out,
                "pair_summary": {
                    "has_score_difference": pair_has_score,
                    "has_decision_state_difference": pair_has_decision_state,
                    "has_operational_difference": pair_has_operational,
                    "midband_actionable_default_not": pair_midband_actionable and (not pair_default_actionable),
                    "default_actionable_midband_not": pair_default_actionable and (not pair_midband_actionable),
                },
            }
        )

    total_pairs = len(pair_rows)
    operational_ratio = (pairs_with_operational_diff / total_pairs) if total_pairs > 0 else 0.0
    actionable_asymmetry = pairs_midband_actionable_only + pairs_default_actionable_only

    if total_pairs == 0:
        verdict_label = "no meaningful live divergence yet"
        verdict_reason = "no matched default/midband snapshot pairs found"
    elif pairs_with_operational_diff == 0 and actionable_asymmetry == 0:
        verdict_label = "no meaningful live divergence yet"
        verdict_reason = "differences are mostly score-level or decision-state-level without action divergence"
    elif operational_ratio >= 0.30 or pairs_with_operational_diff >= 5:
        verdict_label = "operationally different often enough to justify extended paper tracking focus"
        verdict_reason = "operational trade-action divergence appears frequently across matched run pairs"
    else:
        verdict_label = "early evidence of meaningful divergence"
        verdict_reason = "some operational/actionable divergence exists but not yet frequent"

    output_payload = {
        "matching": {
            "method": "greedy_one_to_one_latest_prior_default",
            "details": "Pairs each midband run to the latest default run at-or-before it, using each default run at most once.",
            "default_runs_found": len(default_runs),
            "midband_runs_found": len(midband_runs),
            "matched_pairs": total_pairs,
        },
        "pairs": pair_rows,
        "aggregate_summary": {
            "total_matched_pairs": total_pairs,
            "pairs_with_any_score_difference": pairs_with_score_diff,
            "pairs_with_any_decision_state_difference": pairs_with_decision_state_diff,
            "pairs_with_any_operational_trade_action_difference": pairs_with_operational_diff,
            "pairs_midband_actionable_default_not": pairs_midband_actionable_only,
            "pairs_default_actionable_midband_not": pairs_default_actionable_only,
            "horizon_trade_action_difference_counts": dict(sorted(horizon_trade_action_diff_counts.items(), key=lambda kv: _horizon_sort_key(kv[0]))),
            "horizon_avg_abs_p_up_delta": {
                key: (sum(values) / len(values) if values else None)
                for key, values in sorted(horizon_abs_pup_deltas.items(), key=lambda kv: _horizon_sort_key(kv[0]))
            },
            "horizon_avg_abs_expected_delta": {
                key: (sum(values) / len(values) if values else None)
                for key, values in sorted(horizon_abs_expected_deltas.items(), key=lambda kv: _horizon_sort_key(kv[0]))
            },
        },
        "verdict": {
            "label": verdict_label,
            "reason": verdict_reason,
            "operational_divergence_ratio": operational_ratio,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
