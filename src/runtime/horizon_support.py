from __future__ import annotations

import math


HORIZON_PRECISION = 6


def normalize_horizon_value(value: float | int | str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid horizon value: {value}") from exc
    if math.isnan(numeric) or numeric <= 0.0:
        raise ValueError(f"Horizons must be positive numbers (got {value}).")
    return round(numeric, HORIZON_PRECISION)


def coerce_numeric_horizon(value: int | float | str) -> float | None:
    try:
        if isinstance(value, str) and value.endswith("m"):
            minutes = float(value[:-1])
            return round(minutes / 60.0, HORIZON_PRECISION)
        if isinstance(value, str) and value.endswith("h"):
            return normalize_horizon_value(value[:-1])
        return normalize_horizon_value(value)
    except (TypeError, ValueError):
        return None


def format_horizon_label(value: float) -> str:
    numeric = normalize_horizon_value(value)
    if numeric >= 1.0:
        if numeric.is_integer():
            return f"{int(numeric)}h"
        return f"{numeric:g}h"
    minutes = round(numeric * 60)
    if minutes % 1 == 0:
        return f"{int(minutes)}m"
    return f"{minutes:g}m"


def horizon_sort_key(label: str) -> float | str:
    value = str(label).strip()
    if value.endswith("h"):
        body = value[:-1]
        if body.replace(".", "", 1).isdigit():
            return normalize_horizon_value(body)
    if value.endswith("m"):
        body = value[:-1]
        if body.replace(".", "", 1).isdigit():
            return normalize_horizon_value(float(body) / 60.0)
    return value