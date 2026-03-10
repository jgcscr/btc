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
    parser.add_argument("--max-ece-drift", type=float, default=0.02)
    parser.add_argument("--min-net-return", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calib = _load_json(args.calibration)
    wf = _load_json(args.walkforward)

    ece_drift = _to_float(calib.get("ece_drift"), 0.0)
    auc_recent = _to_float(calib.get("auc_recent"), 0.0)
    auc_baseline = _to_float(calib.get("auc_baseline"), 0.0)
    auc_delta = auc_recent - auc_baseline
    net_total = _to_float(wf.get("cum_ret_net_total"), 0.0)

    weakness = bool(ece_drift > float(args.max_ece_drift) or net_total < float(args.min_net_return) or auc_delta < 0.0)
    cadence = {
        "recalibration": "weekly" if weakness else "biweekly",
        "retrain": "weekly_recent_window" if weakness else "monthly",
        "recent_regime_model": bool(weakness),
    }

    payload = {
        "regime_weakness_detected": weakness,
        "metrics": {
            "ece_drift": ece_drift,
            "auc_baseline": auc_baseline,
            "auc_recent": auc_recent,
            "auc_delta": auc_delta,
            "cum_ret_net_total": net_total,
        },
        "thresholds": {
            "max_ece_drift": float(args.max_ece_drift),
            "min_net_return": float(args.min_net_return),
        },
        "recommendations": cadence,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
