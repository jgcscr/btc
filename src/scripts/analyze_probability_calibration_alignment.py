from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_history(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Prediction history JSON must be a dict or list of dicts.")


def _direction_from_probability(value: Any, *, neutral_band: float = 0.02) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _direction_from_ret_pred(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close: Any, projected_price: Any) -> str:
    try:
        close_value = float(close)
        projected_value = float(projected_price)
    except (TypeError, ValueError):
        return "neutral"
    if close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"
    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def _iter_rows(history: Iterable[Dict[str, Any]], *, neutral_band: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in history:
        generated_at = snapshot.get("generated_at")
        predictions = snapshot.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for label, entry in predictions.items():
            if not isinstance(entry, dict):
                continue
            probability_calibration = entry.get("probability_calibration") if isinstance(entry.get("probability_calibration"), dict) else {}
            raw_probability = probability_calibration.get("raw_probability", entry.get("raw_p_up", entry.get("p_up")))
            resolved_probability = probability_calibration.get("resolved_probability", entry.get("p_up"))
            raw_side = _direction_from_probability(raw_probability, neutral_band=neutral_band)
            resolved_side = _direction_from_probability(resolved_probability, neutral_band=neutral_band)
            ret_side = _direction_from_ret_pred(entry.get("ret_pred"))
            projected_side = _direction_from_projected_price(entry.get("close"), entry.get("projected_price"))
            guard = probability_calibration.get("forecast_alignment_guard") if isinstance(probability_calibration.get("forecast_alignment_guard"), dict) else {}
            rows.append(
                {
                    "generated_at": generated_at,
                    "horizon": str(label),
                    "raw_probability": None if raw_probability is None else float(raw_probability),
                    "resolved_probability": None if resolved_probability is None else float(resolved_probability),
                    "absolute_gap": abs(float(resolved_probability) - float(raw_probability)) if raw_probability is not None and resolved_probability is not None else None,
                    "raw_side": raw_side,
                    "resolved_side": resolved_side,
                    "ret_side": ret_side,
                    "projected_side": projected_side,
                    "raw_ret_mismatch": bool(raw_side in {"up", "down"} and ret_side in {"up", "down"} and raw_side != ret_side),
                    "resolved_ret_mismatch": bool(resolved_side in {"up", "down"} and ret_side in {"up", "down"} and resolved_side != ret_side),
                    "raw_projected_mismatch": bool(raw_side in {"up", "down"} and projected_side in {"up", "down"} and raw_side != projected_side),
                    "resolved_projected_mismatch": bool(resolved_side in {"up", "down"} and projected_side in {"up", "down"} and resolved_side != projected_side),
                    "guard_applied": bool(guard.get("applied")),
                    "used_regime_key": bool(probability_calibration.get("used_regime_key", False)),
                    "applied_key": probability_calibration.get("applied_key"),
                    "requested_key": probability_calibration.get("requested_key"),
                }
            )
    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_horizon: Dict[str, Dict[str, Any]] = {}
    horizons = sorted({str(row["horizon"]) for row in rows})
    for horizon in horizons:
        subset = [row for row in rows if str(row["horizon"]) == horizon]
        if not subset:
            continue
        absolute_gaps = [float(row["absolute_gap"]) for row in subset if row.get("absolute_gap") is not None]
        raw_ret_mismatch_rate = sum(bool(row["raw_ret_mismatch"]) for row in subset) / len(subset)
        resolved_ret_mismatch_rate = sum(bool(row["resolved_ret_mismatch"]) for row in subset) / len(subset)
        by_horizon[horizon] = {
            "rows": len(subset),
            "mean_absolute_gap": (sum(absolute_gaps) / len(absolute_gaps)) if absolute_gaps else None,
            "raw_ret_mismatch_rate": raw_ret_mismatch_rate,
            "resolved_ret_mismatch_rate": resolved_ret_mismatch_rate,
            "raw_projected_mismatch_rate": sum(bool(row["raw_projected_mismatch"]) for row in subset) / len(subset),
            "resolved_projected_mismatch_rate": sum(bool(row["resolved_projected_mismatch"]) for row in subset) / len(subset),
            "ret_mismatch_improvement": raw_ret_mismatch_rate - resolved_ret_mismatch_rate,
            "guard_applied_rate": sum(bool(row["guard_applied"]) for row in subset) / len(subset),
            "used_regime_key_rate": sum(bool(row["used_regime_key"]) for row in subset) / len(subset),
        }

    latest_generated_at = max((row.get("generated_at") for row in rows), default=None)
    latest_rows = [row for row in rows if row.get("generated_at") == latest_generated_at]
    return {
        "snapshots_with_rows": len({row.get("generated_at") for row in rows if row.get("generated_at")}),
        "latest_generated_at": latest_generated_at,
        "latest_rows": latest_rows,
        "historical_summary": by_horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw-vs-resolved probability alignment across live prediction history."
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("artifacts/predictions/history.json"),
        help="Prediction history JSON produced by run_refresh_and_predict.",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.02,
        help="Neutral band around 0.5 used when converting probabilities into direction labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/probability_calibration_alignment_latest.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    history = _load_history(args.history_path)
    rows = _iter_rows(history, neutral_band=max(float(args.neutral_band), 0.0))
    payload = _summarize_rows(rows)
    payload["history_path"] = str(args.history_path)
    payload["neutral_band"] = max(float(args.neutral_band), 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()