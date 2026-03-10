from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Dict[str, Any]:
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


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _horizon_sort_key(label: str) -> tuple[float, str]:
    if label.endswith("h"):
        try:
            return (float(label[:-1]), label)
        except ValueError:
            return (float("inf"), label)
    return (float("inf"), label)


def _extract_horizon(snapshot: Dict[str, Any], label: str) -> Dict[str, Any]:
    horizons = snapshot.get("horizons") if isinstance(snapshot.get("horizons"), dict) else {}
    row = horizons.get(label) if isinstance(horizons.get(label), dict) else {}
    trade_decision = row.get("trade_decision") if isinstance(row.get("trade_decision"), dict) else {}
    abstention = row.get("abstention") if isinstance(row.get("abstention"), dict) else {}
    volatility = row.get("volatility") if isinstance(row.get("volatility"), dict) else {}

    return {
        "timestamp": row.get("timestamp"),
        "age_seconds": _num(row.get("age_seconds")),
        "fresh": _bool(row.get("fresh")),
        "signal_ensemble": row.get("signal_ensemble"),
        "trade_action": row.get("trade_action"),
        "trade_decision_triggered": _bool(trade_decision.get("triggered")),
        "abstention_triggered": _bool(abstention.get("triggered")),
        "p_up": _num(row.get("p_up")),
        "expected_value": _num(row.get("expected_value")),
        "expected_net": _num(trade_decision.get("expected_net")),
        "trade_probability": _num(trade_decision.get("trade_probability")),
        "regime_state": row.get("regime_state"),
        "volatility_realized_24h": _num(volatility.get("volatility_realized_24h")),
        "volatility_ewm_24h": _num(volatility.get("volatility_ewm_24h")),
        "volatility_garch_like": _num(volatility.get("volatility_garch_like")),
        "volatility_triggered": _bool(volatility.get("volatility_triggered")),
    }


def _float_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def _has_diff(default_row: Dict[str, Any], midband_row: Dict[str, Any], keys: List[str]) -> bool:
    for key in keys:
        if default_row.get(key) != midband_row.get(key):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare default vs midband paper live snapshots.")
    parser.add_argument("--default-snapshot", type=Path, required=True)
    parser.add_argument("--midband-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.default_snapshot.exists():
        raise FileNotFoundError(args.default_snapshot)
    if not args.midband_snapshot.exists():
        raise FileNotFoundError(args.midband_snapshot)

    default_payload = _load(args.default_snapshot)
    midband_payload = _load(args.midband_snapshot)

    default_horizons = default_payload.get("horizons") if isinstance(default_payload.get("horizons"), dict) else {}
    midband_horizons = midband_payload.get("horizons") if isinstance(midband_payload.get("horizons"), dict) else {}

    labels = sorted(set(default_horizons.keys()) | set(midband_horizons.keys()), key=_horizon_sort_key)

    per_horizon: Dict[str, Any] = {}
    operational_diff_horizons: List[str] = []
    score_only_diff_horizons: List[str] = []
    any_actionable = False
    default_all_hold = True
    midband_all_hold = True

    operational_keys = [
        "signal_ensemble",
        "trade_action",
    ]
    decision_state_keys = [
        "trade_decision_triggered",
        "abstention_triggered",
    ]
    score_keys = [
        "p_up",
        "expected_value",
        "expected_net",
        "trade_probability",
        "volatility_realized_24h",
        "volatility_ewm_24h",
        "volatility_garch_like",
        "age_seconds",
        "fresh",
        "timestamp",
    ]

    for label in labels:
        drow = _extract_horizon(default_payload, label)
        mrow = _extract_horizon(midband_payload, label)

        d_actionable = str(drow.get("trade_action") or "").lower() not in {"", "hold"}
        m_actionable = str(mrow.get("trade_action") or "").lower() not in {"", "hold"}

        any_actionable = any_actionable or d_actionable or m_actionable
        default_all_hold = default_all_hold and (not d_actionable)
        midband_all_hold = midband_all_hold and (not m_actionable)

        op_diff = _has_diff(drow, mrow, operational_keys)
        decision_state_diff = _has_diff(drow, mrow, decision_state_keys)
        score_diff = _has_diff(drow, mrow, score_keys)

        if op_diff:
            operational_diff_horizons.append(label)
        elif score_diff:
            score_only_diff_horizons.append(label)

        per_horizon[label] = {
            "default": drow,
            "midband_paper": mrow,
            "deltas_midband_minus_default": {
                "p_up": _float_delta(mrow.get("p_up"), drow.get("p_up")),
                "expected_value": _float_delta(mrow.get("expected_value"), drow.get("expected_value")),
                "expected_net": _float_delta(mrow.get("expected_net"), drow.get("expected_net")),
                "trade_probability": _float_delta(mrow.get("trade_probability"), drow.get("trade_probability")),
                "volatility_realized_24h": _float_delta(
                    mrow.get("volatility_realized_24h"), drow.get("volatility_realized_24h")
                ),
                "volatility_ewm_24h": _float_delta(mrow.get("volatility_ewm_24h"), drow.get("volatility_ewm_24h")),
                "volatility_garch_like": _float_delta(
                    mrow.get("volatility_garch_like"), drow.get("volatility_garch_like")
                ),
            },
            "flags": {
                "differs_operationally": op_diff,
                "differs_decision_state": decision_state_diff,
                "differs_score_level": score_diff,
                "default_actionable": d_actionable,
                "midband_actionable": m_actionable,
            },
        }

    decision_state_only_diff_horizons = [
        label
        for label, row in per_horizon.items()
        if bool(row.get("flags", {}).get("differs_decision_state"))
        and not bool(row.get("flags", {}).get("differs_operationally"))
    ]
    profiles_differ = (
        len(operational_diff_horizons) > 0
        or len(decision_state_only_diff_horizons) > 0
        or len(score_only_diff_horizons) > 0
    )
    only_score_differences = (
        len(operational_diff_horizons) == 0
        and len(decision_state_only_diff_horizons) == 0
        and len(score_only_diff_horizons) > 0
    )

    comparison = {
        "default_profile": {
            "profile_id": default_payload.get("profile", {}).get("id"),
            "profile_name": default_payload.get("profile", {}).get("name"),
            "run_id": default_payload.get("run_id"),
            "snapshot": str(args.default_snapshot),
        },
        "midband_paper_profile": {
            "profile_id": midband_payload.get("profile", {}).get("id"),
            "profile_name": midband_payload.get("profile", {}).get("name"),
            "run_id": midband_payload.get("run_id"),
            "snapshot": str(args.midband_snapshot),
        },
        "per_horizon": per_horizon,
        "overall_summary": {
            "profiles_differ": profiles_differ,
            "difference_only_probabilities_or_scores": only_score_differences,
            "either_profile_actionable": any_actionable,
            "both_resolve_to_hold": default_all_hold and midband_all_hold,
            "operationally_meaningful_difference": len(operational_diff_horizons) > 0,
            "operational_diff_horizons": operational_diff_horizons,
            "decision_state_only_diff_horizons": decision_state_only_diff_horizons,
            "score_only_diff_horizons": score_only_diff_horizons,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
