from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess recent regime weakness from calibration and walk-forward diagnostics.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--walkforward", type=Path, required=True)
    parser.add_argument("--horizon", type=str, default="1h")
    parser.add_argument("--max-ece-drift", type=float, default=0.02)
    parser.add_argument("--min-recent-auc", type=float, default=0.0)
    parser.add_argument("--min-net-return", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calib = _load_json(args.calibration)
    wf = _load_json(args.walkforward)

    horizons = calib.get("horizons", {}) if isinstance(calib, dict) else {}
    calib_horizon = horizons.get(args.horizon, {}) if isinstance(horizons, dict) else {}
    if not isinstance(calib_horizon, dict):
        calib_horizon = {}
    promotion_hardening = calib_horizon.get("promotion_hardening") if isinstance(calib_horizon, dict) else {}
    if not isinstance(promotion_hardening, dict):
        promotion_hardening = calib.get("promotion_hardening", {}) if isinstance(calib, dict) else {}
    if not isinstance(promotion_hardening, dict):
        promotion_hardening = {}

    recent = calib_horizon.get("recent", {}) if isinstance(calib_horizon, dict) else {}
    baseline = calib_horizon.get("baseline", {}) if isinstance(calib_horizon, dict) else {}
    ece_drift = _to_float(calib_horizon.get("ece_drift"), 0.0)
    auc_recent = _to_float(recent.get("auc"), 0.0)
    auc_baseline = _to_float(baseline.get("auc"), 0.0)
    auc_delta = auc_recent - auc_baseline
    net_total = _to_float(wf.get("cum_ret_net_total"), 0.0)
    failed_checks = promotion_hardening.get("failed_checks", []) if isinstance(promotion_hardening.get("failed_checks", []), list) else []

    weakness = bool(
        ece_drift > float(args.max_ece_drift)
        or net_total < float(args.min_net_return)
        or auc_recent < float(args.min_recent_auc)
        or auc_delta < 0.0
        or bool(failed_checks)
    )
    cadence = {
        "recalibration": "weekly" if weakness else "biweekly",
        "retrain": "weekly_recent_window" if weakness else "monthly",
        "recent_regime_model": bool(weakness),
    }

    payload = {
        "regime_weakness_detected": weakness,
        "horizon": str(args.horizon),
        "metrics": {
            "ece_drift": ece_drift,
            "auc_baseline": auc_baseline,
            "auc_recent": auc_recent,
            "auc_delta": auc_delta,
            "cum_ret_net_total": net_total,
            "promotion_failed_checks": failed_checks,
        },
        "thresholds": {
            "max_ece_drift": float(args.max_ece_drift),
            "min_recent_auc": float(args.min_recent_auc),
            "min_net_return": float(args.min_net_return),
        },
        "recommendations": cadence,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
