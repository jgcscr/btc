from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class TriggerConfig:
    baseline_window: int
    recent_window: int
    max_abs_prob_drift: float
    max_signal_rate_drift: float
    min_samples: int


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction history not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prediction history JSON must be a list.")
    return payload


def _parse_horizons(values: str | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for raw in values.split(","):
        key = raw.strip()
        if key:
            out.append(key)
    return out


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _extract_series(history: List[Dict[str, Any]], horizon: str) -> Tuple[np.ndarray, np.ndarray]:
    p_up: List[float] = []
    signal_rate: List[float] = []

    for row in history:
        predictions = row.get("predictions", {})
        if not isinstance(predictions, dict):
            continue
        horizon_payload = predictions.get(horizon)
        if not isinstance(horizon_payload, dict):
            continue

        p = horizon_payload.get("p_up")
        s = horizon_payload.get("signal_ensemble")
        try:
            p_up.append(float(p))
            signal_rate.append(float(s))
        except (TypeError, ValueError):
            continue

    return np.asarray(p_up, dtype=float), np.asarray(signal_rate, dtype=float)


def _split_baseline_recent(series: np.ndarray, baseline_window: int, recent_window: int) -> Tuple[np.ndarray, np.ndarray]:
    if series.size < baseline_window + recent_window:
        return np.array([], dtype=float), np.array([], dtype=float)
    baseline = series[-(baseline_window + recent_window) : -recent_window]
    recent = series[-recent_window:]
    return baseline, recent


def _load_trigger_config(path: Path) -> TriggerConfig:
    if not path.exists():
        raise FileNotFoundError(f"Trigger config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    monitoring = payload.get("monitoring", {}) if isinstance(payload, dict) else {}
    triggers = monitoring.get("triggers", {}) if isinstance(monitoring, dict) else {}

    return TriggerConfig(
        baseline_window=int(triggers.get("baseline_window", 120)),
        recent_window=int(triggers.get("recent_window", 24)),
        max_abs_prob_drift=float(triggers.get("max_abs_prob_drift", 0.08)),
        max_signal_rate_drift=float(triggers.get("max_signal_rate_drift", 0.25)),
        min_samples=int(triggers.get("min_samples", 150)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prediction drift and emit retrain trigger recommendations.")
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("artifacts/predictions/history.json"),
        help="Path to prediction history JSON.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/reliability_workflow.default.yaml"),
        help="Path to reliability workflow config YAML.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1h,4h,8h,12h",
        help="Comma-separated horizon labels to evaluate (matching history keys).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/monitoring/reliability_triggers.json"),
        help="Where to write trigger assessment output.",
    )
    args = parser.parse_args()

    config = _load_trigger_config(args.config_path)
    history = _load_history(args.history_path)
    horizons = _parse_horizons(args.horizons)

    if not horizons:
        raise ValueError("At least one horizon must be provided.")

    horizon_reports: Dict[str, Any] = {}
    global_trigger = False

    for horizon in horizons:
        p_up, signal = _extract_series(history, horizon)
        baseline_p, recent_p = _split_baseline_recent(p_up, config.baseline_window, config.recent_window)
        baseline_s, recent_s = _split_baseline_recent(signal, config.baseline_window, config.recent_window)

        enough_samples = p_up.size >= config.min_samples
        prob_drift = abs(_mean_or_nan(recent_p) - _mean_or_nan(baseline_p)) if baseline_p.size else float("nan")
        signal_drift = abs(_mean_or_nan(recent_s) - _mean_or_nan(baseline_s)) if baseline_s.size else float("nan")

        trigger_reasons: List[str] = []
        if enough_samples and np.isfinite(prob_drift) and prob_drift > config.max_abs_prob_drift:
            trigger_reasons.append("probability_drift")
        if enough_samples and np.isfinite(signal_drift) and signal_drift > config.max_signal_rate_drift:
            trigger_reasons.append("signal_rate_drift")

        triggered = len(trigger_reasons) > 0
        global_trigger = global_trigger or triggered

        horizon_reports[horizon] = {
            "samples": int(p_up.size),
            "enough_samples": bool(enough_samples),
            "probability": {
                "baseline_mean": _mean_or_nan(baseline_p),
                "recent_mean": _mean_or_nan(recent_p),
                "abs_drift": prob_drift,
                "limit": config.max_abs_prob_drift,
            },
            "signal_rate": {
                "baseline_mean": _mean_or_nan(baseline_s),
                "recent_mean": _mean_or_nan(recent_s),
                "abs_drift": signal_drift,
                "limit": config.max_signal_rate_drift,
            },
            "triggered": triggered,
            "reasons": trigger_reasons,
        }

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "history_path": str(args.history_path),
        "config_path": str(args.config_path),
        "global_trigger": global_trigger,
        "horizons": horizon_reports,
        "recommendation": (
            "retrain_and_recalibrate" if global_trigger else "keep_current_models"
        ),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote reliability trigger report to {args.output_path}")


if __name__ == "__main__":
    main()
