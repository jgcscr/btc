from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


def simple_average(values: Iterable[float]) -> float:
    values_list = [float(v) for v in values]
    if not values_list:
        raise ValueError("simple_average requires at least one value")
    return sum(values_list) / len(values_list)


def weighted_average(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0

    for key, value in values.items():
        weight = float(weights.get(key, 0.0))
        if weight == 0.0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight

    if total_weight == 0.0:
        raise ValueError("weighted_average requires at least one positive weight")

    return weighted_sum / total_weight


def parse_weight_spec(spec: Optional[str]) -> Dict[str, float]:
    if spec is None or spec.strip() == "":
        return {}

    weights: Dict[str, float] = {}
    parts = spec.split(",")
    for part in parts:
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Invalid weight spec chunk '{part}'. Expected format name:weight")
        name, value = part.split(":", 1)
        name = name.strip().lower()
        if not name:
            raise ValueError(f"Weight spec chunk '{part}' missing model name")
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight value '{value}' in chunk '{part}'") from exc
        weights[name] = weight

    return weights


__all__ = ["simple_average", "weighted_average", "parse_weight_spec"]
