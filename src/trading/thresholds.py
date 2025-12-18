from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict


def load_calibrated_thresholds(path: Path | str | None) -> Dict[int, Dict[str, float]]:
    """Load per-horizon thresholds from a JSON file.

    Returns an empty dict when the path is missing or cannot be parsed.
    """
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
    loaded: Dict[int, Dict[str, float]] = {}
    for key, entry in horizons.items():
        try:
            horizon = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue

        p_up_min = entry.get("p_up_min")
        ret_min = entry.get("ret_min")
        if p_up_min is None or ret_min is None:
            continue
        try:
            loaded[horizon] = {
                "p_up_min": float(p_up_min),
                "ret_min": float(ret_min),
            }
        except (TypeError, ValueError):
            continue
    return loaded
