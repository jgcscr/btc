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
    if label.endswith("m"):
        try:
            return (float(label[:-1]) / 60.0, label)
        except ValueError:
            return (float("inf"), label)
    return (float("inf"), label)


def _extract_horizon(snapshot: Dict[str, Any], label: str) -> Dict[str, Any]:
    horizons = snapshot.get("predictions") if isinstance(snapshot.get("predictions"), dict) else {}
    row = horizons.get(label) if isinstance(horizons.get(label), dict) else {}
    trade_decision = row.get("trade_decision") if isinstance(row.get("trade_decision"), dict) else {}
    abstention = row.get("abstention") if isinstance(row.get("abstention"), dict) else {}
    volatility = row.get("volatility") if isinstance(row.get("volatility"), dict) else {}
    execution_plan = row.get("execution_plan") if isinstance(row.get("execution_plan"), dict) else {}
    forecast = row.get("forecast_coherence") if isinstance(row.get("forecast_coherence"), dict) else {}

    return {
        "timestamp": row.get("timestamp"),
        "signal_ensemble": row.get("signal_ensemble"),
        "trade_action": row.get("trade_action"),
        "trade_decision_triggered": _bool(trade_decision.get("triggered")),
        "abstention_triggered": _bool(abstention.get("triggered")),
        "p_up": _num(row.get("p_up")),
        "ret_pred": _num(row.get("ret_pred")),
        "expected_value": _num(row.get("expected_value")),
        "expected_net": _num(trade_decision.get("expected_net")),
        "trade_probability": _num(trade_decision.get("trade_probability")),
        "regime_state": row.get("regime_state"),
        "midband_veto_triggered": _bool((trade_decision.get("midband_veto") or {}).get("triggered")),
        "execution_status": execution_plan.get("status"),
        "execution_reason": execution_plan.get("reason"),
        "forecast_triggered": _bool(forecast.get("triggered")),
        "forecast_reasons": forecast.get("reasons") if isinstance(forecast.get("reasons"), list) else [],
        "volatility_realized_24h": _num((volatility.get("snapshot") or {}).get("volatility_realized_24h")),
    }


def _float_delta(rhs: Optional[float], lhs: Optional[float]) -> Optional[float]:
    if rhs is None or lhs is None:
        return None
    return float(rhs - lhs)


def _has_diff(lhs_row: Dict[str, Any], rhs_row: Dict[str, Any], keys: List[str]) -> bool:
    for key in keys:
        if lhs_row.get(key) != rhs_row.get(key):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two live profile snapshot artifacts.")
    parser.add_argument("--lhs-snapshot", type=Path, required=True)
    parser.add_argument("--rhs-snapshot", type=Path, required=True)
    parser.add_argument("--lhs-label", type=str, required=True)
    parser.add_argument("--rhs-label", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lhs_snapshot.exists():
        raise FileNotFoundError(args.lhs_snapshot)
    if not args.rhs_snapshot.exists():
        raise FileNotFoundError(args.rhs_snapshot)

    lhs_payload = _load(args.lhs_snapshot)
    rhs_payload = _load(args.rhs_snapshot)

    lhs_horizons = lhs_payload.get("predictions") if isinstance(lhs_payload.get("predictions"), dict) else {}
    rhs_horizons = rhs_payload.get("predictions") if isinstance(rhs_payload.get("predictions"), dict) else {}
    labels = sorted(set(lhs_horizons.keys()) | set(rhs_horizons.keys()), key=_horizon_sort_key)

    per_horizon: Dict[str, Any] = {}
    operational_diff_horizons: List[str] = []
    score_only_diff_horizons: List[str] = []
    decision_state_only_diff_horizons: List[str] = []

    operational_keys = [
        "signal_ensemble",
        "trade_action",
        "execution_status",
        "execution_reason",
    ]
    decision_state_keys = [
        "trade_decision_triggered",
        "abstention_triggered",
        "midband_veto_triggered",
        "forecast_triggered",
        "forecast_reasons",
    ]
    score_keys = [
        "p_up",
        "ret_pred",
        "expected_value",
        "expected_net",
        "trade_probability",
        "regime_state",
        "volatility_realized_24h",
        "timestamp",
    ]

    lhs_all_hold = True
    rhs_all_hold = True
    either_profile_actionable = False

    for label in labels:
        lhs_row = _extract_horizon(lhs_payload, label)
        rhs_row = _extract_horizon(rhs_payload, label)

        lhs_actionable = str(lhs_row.get("trade_action") or "").lower() not in {"", "hold"}
        rhs_actionable = str(rhs_row.get("trade_action") or "").lower() not in {"", "hold"}
        lhs_all_hold = lhs_all_hold and (not lhs_actionable)
        rhs_all_hold = rhs_all_hold and (not rhs_actionable)
        either_profile_actionable = either_profile_actionable or lhs_actionable or rhs_actionable

        op_diff = _has_diff(lhs_row, rhs_row, operational_keys)
        decision_diff = _has_diff(lhs_row, rhs_row, decision_state_keys)
        score_diff = _has_diff(lhs_row, rhs_row, score_keys)

        if op_diff:
            operational_diff_horizons.append(label)
        elif decision_diff:
            decision_state_only_diff_horizons.append(label)
        elif score_diff:
            score_only_diff_horizons.append(label)

        per_horizon[label] = {
            str(args.lhs_label): lhs_row,
            str(args.rhs_label): rhs_row,
            f"deltas_{args.rhs_label}_minus_{args.lhs_label}": {
                "p_up": _float_delta(rhs_row.get("p_up"), lhs_row.get("p_up")),
                "ret_pred": _float_delta(rhs_row.get("ret_pred"), lhs_row.get("ret_pred")),
                "expected_value": _float_delta(rhs_row.get("expected_value"), lhs_row.get("expected_value")),
                "expected_net": _float_delta(rhs_row.get("expected_net"), lhs_row.get("expected_net")),
                "trade_probability": _float_delta(rhs_row.get("trade_probability"), lhs_row.get("trade_probability")),
            },
            "flags": {
                "differs_operationally": op_diff,
                "differs_decision_state": decision_diff,
                "differs_score_level": score_diff,
                f"{args.lhs_label}_actionable": lhs_actionable,
                f"{args.rhs_label}_actionable": rhs_actionable,
            },
        }

    comparison = {
        str(args.lhs_label): {
            "snapshot": str(args.lhs_snapshot),
            "generated_at": lhs_payload.get("generated_at"),
        },
        str(args.rhs_label): {
            "snapshot": str(args.rhs_snapshot),
            "generated_at": rhs_payload.get("generated_at"),
        },
        "per_horizon": per_horizon,
        "overall_summary": {
            "profiles_differ": bool(
                operational_diff_horizons or decision_state_only_diff_horizons or score_only_diff_horizons
            ),
            "difference_only_probabilities_or_scores": bool(
                not operational_diff_horizons and not decision_state_only_diff_horizons and score_only_diff_horizons
            ),
            "either_profile_actionable": either_profile_actionable,
            "both_resolve_to_hold": lhs_all_hold and rhs_all_hold,
            "operationally_meaningful_difference": bool(operational_diff_horizons),
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