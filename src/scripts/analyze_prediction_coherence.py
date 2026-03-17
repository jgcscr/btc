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


def _direction_from_probability(value: Any, *, neutral_band: float = 0.0) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _iter_live_rows(history: Iterable[Dict[str, Any]], *, neutral_band: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in history:
        generated_at = snapshot.get("generated_at")
        predictions = snapshot.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        for label, entry in predictions.items():
            if not isinstance(entry, dict):
                continue
            try:
                close = float(entry.get("close"))
                projected_price = float(entry.get("projected_price"))
                ret_pred = float(entry.get("ret_pred"))
                p_up = float(entry.get("p_up"))
            except (TypeError, ValueError):
                continue
            if close <= 0.0:
                continue
            direction = str(entry.get("direction_next", "neutral")).lower()
            ret_side = _direction_from_ret_pred(ret_pred)
            projected_side = _direction_from_projected_price(close, projected_price)
            p_up_side = _direction_from_probability(p_up, neutral_band=neutral_band)
            rows.append(
                {
                    "generated_at": generated_at,
                    "horizon": label,
                    "direction_next": direction,
                    "ret_pred_side": ret_side,
                    "projected_price_side": projected_side,
                    "p_up_side": p_up_side,
                    "ret_pred": ret_pred,
                    "p_up": p_up,
                    "projected_delta_pct": (projected_price / close - 1.0) * 100.0,
                    "direction_ret_mismatch": bool(direction in {"up", "down"} and ret_side != "neutral" and direction != ret_side),
                    "direction_projected_price_mismatch": bool(
                        direction in {"up", "down"} and projected_side != "neutral" and direction != projected_side
                    ),
                    "p_up_ret_mismatch": bool(p_up_side != "neutral" and ret_side != "neutral" and p_up_side != ret_side),
                    "p_up_direction_mismatch": bool(direction in {"up", "down"} and p_up_side != "neutral" and direction != p_up_side),
                }
            )
    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_horizon: Dict[str, Dict[str, Any]] = {}
    for horizon in sorted({str(row["horizon"]) for row in rows}):
        subset = [row for row in rows if str(row["horizon"]) == horizon]
        count = len(subset)
        if count == 0:
            continue
        by_horizon[horizon] = {
            "rows": count,
            "direction_ret_mismatch_rate": sum(row["direction_ret_mismatch"] for row in subset) / count,
            "direction_projected_price_mismatch_rate": sum(row["direction_projected_price_mismatch"] for row in subset) / count,
            "p_up_ret_mismatch_rate": sum(row["p_up_ret_mismatch"] for row in subset) / count,
            "p_up_direction_mismatch_rate": sum(row["p_up_direction_mismatch"] for row in subset) / count,
        }

    latest_generated_at = max((row.get("generated_at") for row in rows), default=None)
    latest_rows = [row for row in rows if row.get("generated_at") == latest_generated_at]
    return {
        "snapshots_with_live_rows": len({row.get("generated_at") for row in rows if row.get("generated_at")}),
        "latest_generated_at": latest_generated_at,
        "latest_rows": latest_rows,
        "historical_summary": by_horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit agreement between direction, return, projected-price, and p_up outputs in prediction history.")
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
        help="Neutral band around 0.5 used when turning p_up into a direction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/prediction_coherence_latest.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    history = _load_history(args.history_path)
    rows = _iter_live_rows(history, neutral_band=max(float(args.neutral_band), 0.0))
    payload = _summarize_rows(rows)
    payload["history_path"] = str(args.history_path)
    payload["neutral_band"] = max(float(args.neutral_band), 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()