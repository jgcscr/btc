from __future__ import annotations

import json
import sys
import math
from pathlib import Path
from typing import Any, Dict


def _coerce_threshold_entry(entry: Any) -> Dict[str, float] | None:
    """Attempt to coerce a threshold entry into floats."""
    if not isinstance(entry, dict):
        return None

    p_up_min = entry.get("p_up_min")
    ret_min = entry.get("ret_min")
    if p_up_min is None or ret_min is None:
        return None

    try:
        resolved: Dict[str, float] = {
            "p_up_min": float(p_up_min),
            "ret_min": float(ret_min),
        }
    except (TypeError, ValueError):
        return None

    max_drawdown = entry.get("max_drawdown")
    if max_drawdown is not None:
        try:
            resolved["max_drawdown"] = float(max_drawdown)
        except (TypeError, ValueError):
            pass

    volatility_ceiling = entry.get("volatility_ceiling")
    if volatility_ceiling is not None:
        try:
            resolved["volatility_ceiling"] = float(volatility_ceiling)
        except (TypeError, ValueError):
            pass

    volatility_mult = entry.get("volatility_mult")
    if volatility_mult is not None:
        try:
            resolved["volatility_mult"] = float(volatility_mult)
        except (TypeError, ValueError):
            pass

    volatility_metric = entry.get("volatility_metric")
    if isinstance(volatility_metric, str) and volatility_metric.strip():
        resolved["volatility_metric"] = volatility_metric.strip()

    return resolved


def _normalize_horizon_key(value: Any) -> int | float | None:
    """Convert arbitrary horizon keys (str/int/float) into numeric identifiers."""
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None

    if math.isnan(numeric) or numeric <= 0:
        return None

    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 6)


def load_calibrated_thresholds(path: Path | str | None) -> Dict[int | float, Dict[str, float]]:
    """Load per-horizon thresholds from a JSON file with optional max drawdown."""
    if path is None:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        return {}

    try:
        data = json.loads(path_obj.read_text())
    except json.JSONDecodeError as exc:
        print(f"Warning: failed to parse thresholds JSON at {path_obj} ({exc}).", file=sys.stderr)
        return {}

    horizons = data.get("horizons", {})
    loaded: Dict[int | float, Dict[str, float]] = {}
    for key, entry in horizons.items():
        horizon = _normalize_horizon_key(key)
        if horizon is None:
            continue
        threshold_entry = _coerce_threshold_entry(entry)
        if threshold_entry is None:
            continue
        loaded[horizon] = threshold_entry
    return loaded
