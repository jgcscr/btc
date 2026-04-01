from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


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


def _safe_history(history: Mapping[str, Sequence[float]] | None, name: str) -> np.ndarray:
    if not history:
        return np.asarray([], dtype=float)
    values = history.get(name)
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr[np.isfinite(arr)]


def _pairwise_history_stats(left: np.ndarray, right: np.ndarray) -> tuple[float, float, int]:
    if left.size == 0 or right.size == 0:
        return float("nan"), float("nan"), 0
    common = min(int(left.size), int(right.size))
    if common <= 1:
        return float("nan"), float("nan"), common
    left_tail = left[-common:]
    right_tail = right[-common:]
    gap = float(np.mean(np.abs(left_tail - right_tail)))
    if np.allclose(left_tail, left_tail[0]) or np.allclose(right_tail, right_tail[0]):
        return float("nan"), gap, common
    corr = float(np.corrcoef(left_tail, right_tail)[0, 1])
    return corr, gap, common


def _group_name(model_groups: Mapping[str, str] | None, name: str) -> str:
    return str((model_groups or {}).get(name, "default"))


def select_diverse_models(
    probabilities: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    *,
    history: Mapping[str, Sequence[float]] | None = None,
    priority_order: Sequence[str] | None = None,
    preferred_groups: Sequence[str] | None = None,
    max_active_models: int | None = None,
    model_groups: Mapping[str, str] | None = None,
    max_models_per_group: Mapping[str, int] | None = None,
    max_correlation: float | None = None,
    min_mean_abs_probability_gap: float | None = None,
    min_history_points: int = 0,
) -> Dict[str, Any]:
    available = {
        str(name): float(value)
        for name, value in probabilities.items()
        if math.isfinite(float(value))
    }
    base_weights = {
        name: max(float((weights or {}).get(name, 1.0)), 0.0)
        for name in available.keys()
    }
    ranked: list[tuple[int, float, str]] = []
    priority_lookup = {str(name): idx for idx, name in enumerate(priority_order or [])}
    for name in available.keys():
        ranked.append((priority_lookup.get(name, len(priority_lookup)), -base_weights.get(name, 0.0), name))
    ranked.sort()

    selected: list[str] = []
    rejected: list[Dict[str, Any]] = []
    group_counts: Dict[str, int] = {}
    pairwise: list[Dict[str, Any]] = []

    def _record_rejection(payload: Dict[str, Any]) -> None:
        if any(
            existing.get("name") == payload.get("name")
            and existing.get("reason") == payload.get("reason")
            and existing.get("against") == payload.get("against")
            for existing in rejected
        ):
            return
        rejected.append(payload)

    def _attempt_select(name: str) -> bool:
        if name in selected:
            return True

        group = _group_name(model_groups, name)
        group_limit = None if not max_models_per_group else max_models_per_group.get(group)
        if group_limit is not None and group_counts.get(group, 0) >= int(group_limit):
            _record_rejection({"name": name, "reason": f"group_cap:{group}"})
            return False
        if max_active_models is not None and len(selected) >= int(max_active_models):
            _record_rejection({"name": name, "reason": "active_cap"})
            return False

        candidate_history = _safe_history(history, name)
        for selected_name in selected:
            peer_history = _safe_history(history, selected_name)
            corr, gap, common = _pairwise_history_stats(candidate_history, peer_history)
            pairwise.append(
                {
                    "left": selected_name,
                    "right": name,
                    "correlation": corr,
                    "mean_abs_probability_gap": gap,
                    "history_points": common,
                }
            )
            enough_history = common >= int(max(min_history_points, 0)) if min_history_points else common > 1
            if not enough_history:
                continue
            correlation_block = (
                max_correlation is not None
                and math.isfinite(corr)
                and corr >= float(max_correlation)
            )
            gap_block = (
                min_mean_abs_probability_gap is not None
                and math.isfinite(gap)
                and gap <= float(min_mean_abs_probability_gap)
            )
            if correlation_block and gap_block:
                _record_rejection(
                    {
                        "name": name,
                        "reason": "orthogonality",
                        "against": selected_name,
                        "correlation": corr,
                        "mean_abs_probability_gap": gap,
                        "history_points": common,
                    }
                )
                return False

        selected.append(name)
        group_counts[group] = group_counts.get(group, 0) + 1
        return True

    preferred_group_names = [str(group).strip().lower() for group in (preferred_groups or []) if str(group).strip()]
    missing_preferred_groups: list[str] = []
    for group in preferred_group_names:
        if max_active_models is not None and len(selected) >= int(max_active_models):
            missing_preferred_groups.append(group)
            continue

        matched = False
        for _, _, name in ranked:
            if _group_name(model_groups, name) != group:
                continue
            if _attempt_select(name):
                matched = True
                break
        if not matched:
            missing_preferred_groups.append(group)

    for _, _, name in ranked:
        _attempt_select(name)

    if not selected and ranked:
        fallback = ranked[0][2]
        selected = [fallback]

    effective_weights = {
        name: float(base_weights.get(name, 1.0))
        for name in selected
    }
    return {
        "selected_models": selected,
        "selected_groups": [_group_name(model_groups, name) for name in selected],
        "missing_preferred_groups": missing_preferred_groups,
        "effective_weights": effective_weights,
        "rejected_models": rejected,
        "pairwise": pairwise,
        "base_weights": base_weights,
    }


__all__ = [
    "parse_weight_spec",
    "select_diverse_models",
    "simple_average",
    "weighted_average",
]
