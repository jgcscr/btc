from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _bool(value: Any) -> bool:
    return bool(value)


def _pair_has_class(pair: Dict[str, Any], klass: str) -> bool:
    horizons = pair.get("horizons", {})
    if not isinstance(horizons, dict):
        return False
    for row in horizons.values():
        if isinstance(row, dict) and str(row.get("difference_class", "")) == klass:
            return True
    return False


def _pair_has_trade_action_diff(pair: Dict[str, Any]) -> bool:
    horizons = pair.get("horizons", {})
    if not isinstance(horizons, dict):
        return False
    for row in horizons.values():
        if not isinstance(row, dict):
            continue
        diffs = row.get("differences", {})
        if isinstance(diffs, dict) and _bool(diffs.get("trade_action_changed", False)):
            return True
    return False


def _max_consecutive_true(flags: List[bool]) -> int:
    best = 0
    current = 0
    for value in flags:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _latest_consecutive_true(flags: List[bool]) -> int:
    current = 0
    for value in reversed(flags):
        if value:
            current += 1
        else:
            break
    return int(current)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an observational watchlist status from default-vs-midband longitudinal artifact.",
    )
    parser.add_argument("--longitudinal-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-matched-pairs", type=int, default=8)
    parser.add_argument("--early-operational-streak", type=int, default=2)
    parser.add_argument("--early-actionable-asymmetry-streak", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.longitudinal_input.exists():
        raise FileNotFoundError(args.longitudinal_input)

    longitudinal = _load_json(args.longitudinal_input)
    pairs_obj = longitudinal.get("pairs", [])
    pairs: List[Dict[str, Any]] = [row for row in pairs_obj if isinstance(row, dict)]

    total_pairs = int(len(pairs))

    score_only_count = sum(1 for pair in pairs if _pair_has_class(pair, "score-only"))
    decision_state_only_count = sum(1 for pair in pairs if _pair_has_class(pair, "decision-state-only"))
    operational_trade_action_count = sum(1 for pair in pairs if _pair_has_trade_action_diff(pair))

    midband_actionable_default_not_count = sum(
        1
        for pair in pairs
        if _bool(pair.get("pair_summary", {}).get("midband_actionable_default_not", False))
    )
    default_actionable_midband_not_count = sum(
        1
        for pair in pairs
        if _bool(pair.get("pair_summary", {}).get("default_actionable_midband_not", False))
    )

    operational_flags = [_pair_has_trade_action_diff(pair) for pair in pairs]
    actionable_asymmetry_flags = [
        _bool(pair.get("pair_summary", {}).get("midband_actionable_default_not", False))
        or _bool(pair.get("pair_summary", {}).get("default_actionable_midband_not", False))
        for pair in pairs
    ]

    max_operational_streak = _max_consecutive_true(operational_flags)
    latest_operational_streak = _latest_consecutive_true(operational_flags)
    max_actionable_asymmetry_streak = _max_consecutive_true(actionable_asymmetry_flags)
    latest_actionable_asymmetry_streak = _latest_consecutive_true(actionable_asymmetry_flags)

    reached_target_pairs = total_pairs >= int(args.target_matched_pairs)
    early_operational_trigger = latest_operational_streak >= int(args.early_operational_streak)
    early_actionable_asymmetry_trigger = (
        latest_actionable_asymmetry_streak >= int(args.early_actionable_asymmetry_streak)
    )

    if reached_target_pairs:
        recommendation_status = "ready for formal reassessment"
        verdict = (
            f"Matched pairs reached target ({total_pairs}/{int(args.target_matched_pairs)}), "
            "so formal reassessment is now due."
        )
    elif early_operational_trigger or early_actionable_asymmetry_trigger:
        recommendation_status = "early re-evaluation triggered"
        trigger_reasons: List[str] = []
        if early_operational_trigger:
            trigger_reasons.append(
                "consecutive operational trade_action divergence"
            )
        if early_actionable_asymmetry_trigger:
            trigger_reasons.append(
                "consecutive actionable asymmetry"
            )
        verdict = (
            "Early re-evaluation is triggered due to "
            + " and ".join(trigger_reasons)
            + "."
        )
    else:
        recommendation_status = "watchlist only"
        verdict = (
            "Operational/actionable divergence is not repeating consecutively yet and "
            f"matched pairs are below target ({total_pairs}/{int(args.target_matched_pairs)})."
        )

    additional_pairs_needed = max(0, int(args.target_matched_pairs) - total_pairs)

    payload = {
        "source_longitudinal": str(args.longitudinal_input),
        "summary": {
            "total_matched_pairs": total_pairs,
            "target_matched_pair_threshold": int(args.target_matched_pairs),
            "pairs_with_any_score_only_difference": int(score_only_count),
            "pairs_with_any_decision_state_only_difference": int(decision_state_only_count),
            "pairs_with_any_operational_trade_action_difference": int(operational_trade_action_count),
            "pairs_midband_actionable_default_not": int(midband_actionable_default_not_count),
            "pairs_default_actionable_midband_not": int(default_actionable_midband_not_count),
            "additional_pairs_needed_for_formal_reassessment": int(additional_pairs_needed),
        },
        "streaks": {
            "operational_divergence": {
                "latest_consecutive_count": int(latest_operational_streak),
                "max_consecutive_count": int(max_operational_streak),
            },
            "actionable_asymmetry": {
                "latest_consecutive_count": int(latest_actionable_asymmetry_streak),
                "max_consecutive_count": int(max_actionable_asymmetry_streak),
            },
        },
        "triggers": {
            "reached_target_matched_pairs": bool(reached_target_pairs),
            "early_operational_streak_triggered": bool(early_operational_trigger),
            "early_actionable_asymmetry_streak_triggered": bool(early_actionable_asymmetry_trigger),
            "thresholds": {
                "target_matched_pairs": int(args.target_matched_pairs),
                "early_operational_streak": int(args.early_operational_streak),
                "early_actionable_asymmetry_streak": int(args.early_actionable_asymmetry_streak),
            },
        },
        "recommendation_status": str(recommendation_status),
        "verdict": str(verdict),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()