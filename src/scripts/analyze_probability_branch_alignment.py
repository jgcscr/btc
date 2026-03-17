from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _load_history(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Prediction history JSON must be a dict or list of dicts.")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _direction_from_probability(value: Any, *, neutral_band: float) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _direction_from_return(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close: Any, projected_price: Any) -> str:
    close_value = _safe_float(close)
    projected_value = _safe_float(projected_price)
    if close_value is None or projected_value is None:
        return "neutral"
    if close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"
    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def _alignment_bucket(*, raw_side: str, calibrated_side: str, target_side: str) -> str:
    raw_match = raw_side == target_side and target_side != "neutral"
    calibrated_match = calibrated_side == target_side and target_side != "neutral"
    if raw_match and calibrated_match:
        return "both_match"
    if not raw_match and not calibrated_match:
        return "both_mismatch"
    if raw_match and not calibrated_match:
        return "calibration_introduced_mismatch"
    return "calibration_fixed_mismatch"


def _extract_rows(
    history: Iterable[Mapping[str, Any]],
    *,
    horizons: set[str],
    neutral_band: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snapshot in history:
        generated_at = snapshot.get("generated_at")
        predictions = snapshot.get("predictions")
        if not isinstance(predictions, Mapping):
            continue
        for label, entry in predictions.items():
            if label not in horizons or not isinstance(entry, Mapping):
                continue
            close = _safe_float(entry.get("close"))
            if close is None or close <= 0.0:
                continue
            direction_output = entry.get("direction_output") if isinstance(entry.get("direction_output"), Mapping) else {}
            probability_calibration = (
                entry.get("probability_calibration") if isinstance(entry.get("probability_calibration"), Mapping) else {}
            )
            raw_probability = _safe_float(direction_output.get("raw_probability"))
            calibrated_probability = _safe_float(entry.get("p_up"))
            ret_pred = _safe_float(entry.get("ret_pred"))
            projected_price = _safe_float(entry.get("projected_price"))
            if raw_probability is None or calibrated_probability is None or ret_pred is None:
                continue

            ret_side = _direction_from_return(ret_pred)
            projected_side = _direction_from_projected_price(close, projected_price)
            raw_side = _direction_from_probability(raw_probability, neutral_band=neutral_band)
            calibrated_side = _direction_from_probability(calibrated_probability, neutral_band=neutral_band)

            rows.append(
                {
                    "generated_at": generated_at,
                    "horizon": label,
                    "regime_state": entry.get("regime_state"),
                    "close": close,
                    "ret_pred": ret_pred,
                    "projected_price": projected_price,
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "calibration_delta": calibrated_probability - raw_probability,
                    "direction_next": str(entry.get("direction_next") or "neutral").lower(),
                    "raw_side": raw_side,
                    "calibrated_side": calibrated_side,
                    "ret_side": ret_side,
                    "projected_side": projected_side,
                    "ret_alignment_bucket": _alignment_bucket(
                        raw_side=raw_side,
                        calibrated_side=calibrated_side,
                        target_side=ret_side,
                    ),
                    "projected_alignment_bucket": _alignment_bucket(
                        raw_side=raw_side,
                        calibrated_side=calibrated_side,
                        target_side=projected_side,
                    ),
                    "calibration_key": probability_calibration.get("applied_key"),
                    "used_regime_key": bool(probability_calibration.get("used_regime_key", False)),
                }
            )
    rows.sort(key=lambda row: (str(row.get("generated_at") or ""), str(row.get("horizon") or "")))
    return rows


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _average(values: List[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: List[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _summarize_horizon(rows: List[Dict[str, Any]], *, recent_window: int) -> Dict[str, Any]:
    calibration_deltas = [float(row["calibration_delta"]) for row in rows]
    recent_rows = rows[-recent_window:] if recent_window > 0 else rows
    ret_bucket_counts = Counter(str(row["ret_alignment_bucket"]) for row in rows)
    projected_bucket_counts = Counter(str(row["projected_alignment_bucket"]) for row in rows)
    calibration_keys = Counter(str(row.get("calibration_key") or "none") for row in rows)

    def _branch_match_rate(target_field: str, side_field: str, subset: List[Dict[str, Any]]) -> float:
        comparable = [
            row for row in subset if str(row.get(target_field) or "neutral") != "neutral"
        ]
        if not comparable:
            return 0.0
        matched = sum(1 for row in comparable if row.get(side_field) == row.get(target_field))
        return _rate(matched, len(comparable))

    return {
        "rows": int(len(rows)),
        "avg_calibration_delta": _average(calibration_deltas),
        "median_calibration_delta": _median(calibration_deltas),
        "raw_vs_ret_match_rate": _branch_match_rate("ret_side", "raw_side", rows),
        "calibrated_vs_ret_match_rate": _branch_match_rate("ret_side", "calibrated_side", rows),
        "raw_vs_projected_match_rate": _branch_match_rate("projected_side", "raw_side", rows),
        "calibrated_vs_projected_match_rate": _branch_match_rate("projected_side", "calibrated_side", rows),
        "ret_alignment_buckets": {
            name: {
                "count": int(ret_bucket_counts.get(name, 0)),
                "rate": _rate(int(ret_bucket_counts.get(name, 0)), len(rows)),
            }
            for name in (
                "both_match",
                "both_mismatch",
                "calibration_introduced_mismatch",
                "calibration_fixed_mismatch",
            )
        },
        "projected_alignment_buckets": {
            name: {
                "count": int(projected_bucket_counts.get(name, 0)),
                "rate": _rate(int(projected_bucket_counts.get(name, 0)), len(rows)),
            }
            for name in (
                "both_match",
                "both_mismatch",
                "calibration_introduced_mismatch",
                "calibration_fixed_mismatch",
            )
        },
        "calibration_keys": dict(calibration_keys),
        "used_regime_key_rate": _rate(sum(1 for row in rows if bool(row.get("used_regime_key"))), len(rows)),
        "recent_window": {
            "rows": int(len(recent_rows)),
            "avg_calibration_delta": _average([float(row["calibration_delta"]) for row in recent_rows]),
            "raw_vs_ret_match_rate": _branch_match_rate("ret_side", "raw_side", recent_rows),
            "calibrated_vs_ret_match_rate": _branch_match_rate("ret_side", "calibrated_side", recent_rows),
            "recent_examples": recent_rows[-5:],
        },
        "latest_row": rows[-1] if rows else None,
    }


def build_probability_branch_alignment_report(
    history_path: Path,
    *,
    horizons: List[str],
    neutral_band: float,
    recent_window: int,
) -> Dict[str, Any]:
    history = _load_history(history_path)
    rows = _extract_rows(history, horizons=set(horizons), neutral_band=neutral_band)
    rows_by_horizon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_horizon[str(row["horizon"])].append(row)

    latest_generated_at = max((str(row.get("generated_at") or "") for row in rows), default=None)
    latest_rows = [row for row in rows if row.get("generated_at") == latest_generated_at]
    return {
        "history_path": str(history_path),
        "neutral_band": float(neutral_band),
        "horizons": list(horizons),
        "rows": int(len(rows)),
        "extracted_rows": rows,
        "latest_generated_at": latest_generated_at,
        "latest_rows": latest_rows,
        "by_horizon": {
            horizon: _summarize_horizon(horizon_rows, recent_window=recent_window)
            for horizon, horizon_rows in sorted(rows_by_horizon.items())
        },
    }


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw classifier probability, calibrated trade probability, and the regression branch over prediction history."
        )
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("artifacts/predictions/history.json"),
        help="Prediction history JSON produced by run_refresh_and_predict.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["1h", "4h", "8h", "12h"],
        help="Horizons to audit.",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.02,
        help="Neutral band around 0.5 used when turning probabilities into a direction side.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=25,
        help="Number of most recent rows per horizon to summarize separately.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/probability_branch_alignment_latest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("artifacts/analysis/probability_branch_alignment_rows.csv"),
        help="Detailed extracted rows CSV path.",
    )
    args = parser.parse_args()

    payload = build_probability_branch_alignment_report(
        args.history_path,
        horizons=[str(value) for value in args.horizons],
        neutral_band=max(float(args.neutral_band), 0.0),
        recent_window=max(int(args.recent_window), 1),
    )
    extracted_rows = payload.pop("extracted_rows", [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_rows_csv(args.rows_output, extracted_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()