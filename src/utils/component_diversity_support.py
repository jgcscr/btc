from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_COMPONENT_GROUPS: Dict[str, str] = {
    "xgb": "tree",
    "lgbm": "tree",
    "regime_logit": "regime",
    "lstm": "recurrent",
    "bilstm": "recurrent",
    "gru": "recurrent",
    "cnn_lstm": "recurrent",
    "cnn_bilstm": "recurrent",
    "garch_lstm": "volatility",
    "transformer": "attention",
    "transformer_large": "attention",
}


def resolve_component_group_map(model_groups: Mapping[str, str] | None = None) -> Dict[str, str]:
    resolved = dict(DEFAULT_COMPONENT_GROUPS)
    if isinstance(model_groups, Mapping):
        for raw_name, raw_group in model_groups.items():
            name = str(raw_name).strip().lower()
            group = str(raw_group).strip().lower()
            if name and group:
                resolved[name] = group
    return resolved


def component_feature_column_names(model_groups: Mapping[str, str] | None = None) -> list[str]:
    groups = sorted(set(resolve_component_group_map(model_groups).values()))
    base = [
        "component_count",
        "component_group_count",
        "component_probability_std",
        "component_probability_range",
        "component_mean_abs_gap",
        "component_max_abs_gap",
        "component_entropy",
        "component_agreement_ratio",
        "component_disagreement_ratio",
    ]
    for group in groups:
        base.append(f"component_group_{group}_p_up")
        base.append(f"component_group_{group}_count")
    return base


def pairwise_feature_column_names() -> list[str]:
    return [
        "component_pairwise_count",
        "component_pairwise_correlation_mean",
        "component_pairwise_correlation_max",
        "component_pairwise_correlation_min",
        "component_pairwise_probability_gap_mean",
        "component_pairwise_probability_gap_max",
        "component_pairwise_probability_gap_min",
    ]


def _direction_side(probability: float, *, neutral_band: float) -> str:
    if probability >= 0.5 + neutral_band:
        return "up"
    if probability <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def summarize_component_probabilities(
    probabilities: Mapping[str, float],
    *,
    model_groups: Mapping[str, str] | None = None,
    neutral_band: float = 0.02,
) -> Dict[str, float]:
    group_map = resolve_component_group_map(model_groups)
    feature_names = component_feature_column_names(group_map)
    empty = {name: 0.0 for name in feature_names}

    cleaned: Dict[str, float] = {}
    for raw_name, raw_value in probabilities.items():
        name = str(raw_name).strip().lower()
        if not name:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        cleaned[name] = value

    if not cleaned:
        return empty

    names = sorted(cleaned)
    values = np.asarray([cleaned[name] for name in names], dtype=float)
    mean_probability = float(np.mean(values))
    sides = [_direction_side(float(value), neutral_band=neutral_band) for value in values]
    ensemble_side = _direction_side(mean_probability, neutral_band=neutral_band)
    directional_count = sum(1 for side in sides if side != "neutral")
    agreement_count = sum(1 for side in sides if ensemble_side != "neutral" and side == ensemble_side)
    disagreement_count = sum(1 for side in sides if side not in {"neutral", ensemble_side})

    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    entropy = float(np.mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)) / math.log(2.0)))
    summary = {
        "component_count": float(len(values)),
        "component_group_count": float(len({group_map.get(name, "default") for name in names})),
        "component_probability_std": float(np.std(values)),
        "component_probability_range": float(np.max(values) - np.min(values)),
        "component_mean_abs_gap": float(np.mean(np.abs(values - mean_probability))),
        "component_max_abs_gap": float(np.max(np.abs(values - mean_probability))),
        "component_entropy": entropy,
        "component_agreement_ratio": float(agreement_count / directional_count) if directional_count else 0.0,
        "component_disagreement_ratio": float(disagreement_count / directional_count) if directional_count else 0.0,
    }

    groups = sorted(set(group_map.values()))
    for group in groups:
        group_values = [cleaned[name] for name in names if group_map.get(name, "default") == group]
        summary[f"component_group_{group}_p_up"] = float(np.mean(group_values)) if group_values else 0.0
        summary[f"component_group_{group}_count"] = float(len(group_values))

    return summary


def build_component_feature_frame(
    frame: pd.DataFrame,
    component_columns: Sequence[str],
    *,
    model_groups: Mapping[str, str] | None = None,
    neutral_band: float = 0.02,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(0.0, index=frame.index, columns=component_feature_column_names(model_groups), dtype=float)

    valid_columns = [str(column) for column in component_columns if str(column) in frame.columns]
    if not valid_columns:
        return pd.DataFrame(0.0, index=frame.index, columns=component_feature_column_names(model_groups), dtype=float)

    records = []
    for _, row in frame[valid_columns].iterrows():
        probabilities = {
            str(column).removeprefix("p_up_"): row[column]
            for column in valid_columns
        }
        records.append(
            summarize_component_probabilities(
                probabilities,
                model_groups=model_groups,
                neutral_band=neutral_band,
            )
        )
    return pd.DataFrame(records, index=frame.index).fillna(0.0)


def summarize_component_history(
    frame: pd.DataFrame,
    component_columns: Sequence[str],
    *,
    model_groups: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    valid_columns = [str(column) for column in component_columns if str(column) in frame.columns]
    if len(valid_columns) < 2:
        return {
            "available": False,
            "component_columns": valid_columns,
            "group_map": resolve_component_group_map(model_groups),
            "reason": "insufficient_component_columns",
        }

    numeric = frame[valid_columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(min_periods=2)
    pairs = []
    for idx, left in enumerate(valid_columns):
        for right in valid_columns[idx + 1 :]:
            corr_value = corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan
            overlap = int(numeric[[left, right]].dropna().shape[0])
            mean_gap = float((numeric[left] - numeric[right]).abs().mean(skipna=True))
            if math.isfinite(float(corr_value)):
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": float(corr_value),
                        "mean_abs_probability_gap": mean_gap,
                        "overlap_rows": overlap,
                    }
                )
    if not pairs:
        return {
            "available": False,
            "component_columns": valid_columns,
            "group_map": resolve_component_group_map(model_groups),
            "reason": "insufficient_pairwise_history",
        }

    correlations = np.asarray([pair["correlation"] for pair in pairs], dtype=float)
    gaps = np.asarray([pair["mean_abs_probability_gap"] for pair in pairs], dtype=float)
    ranked = sorted(pairs, key=lambda item: (item["correlation"], -item["mean_abs_probability_gap"]), reverse=True)
    return {
        "available": True,
        "component_columns": valid_columns,
        "group_map": resolve_component_group_map(model_groups),
        "pair_count": len(pairs),
        "pairwise_correlation_mean": float(np.mean(correlations)),
        "pairwise_correlation_max": float(np.max(correlations)),
        "pairwise_correlation_min": float(np.min(correlations)),
        "pairwise_probability_gap_mean": float(np.mean(gaps)),
        "pairwise_probability_gap_max": float(np.max(gaps)),
        "pairwise_probability_gap_min": float(np.min(gaps)),
        "top_correlation_pairs": ranked[:5],
    }


def summarize_pairwise_history(pairwise: Sequence[Mapping[str, Any]] | None) -> Dict[str, float]:
    summary = {name: 0.0 for name in pairwise_feature_column_names()}
    if not pairwise:
        return summary

    correlations = []
    gaps = []
    for entry in pairwise:
        try:
            corr = float(entry.get("correlation"))
        except (TypeError, ValueError):
            corr = float("nan")
        try:
            gap = float(entry.get("mean_abs_probability_gap"))
        except (TypeError, ValueError):
            gap = float("nan")
        if math.isfinite(corr):
            correlations.append(corr)
        if math.isfinite(gap):
            gaps.append(gap)

    if correlations:
        corr_arr = np.asarray(correlations, dtype=float)
        summary["component_pairwise_count"] = float(len(corr_arr))
        summary["component_pairwise_correlation_mean"] = float(np.mean(corr_arr))
        summary["component_pairwise_correlation_max"] = float(np.max(corr_arr))
        summary["component_pairwise_correlation_min"] = float(np.min(corr_arr))
    if gaps:
        gap_arr = np.asarray(gaps, dtype=float)
        summary["component_pairwise_probability_gap_mean"] = float(np.mean(gap_arr))
        summary["component_pairwise_probability_gap_max"] = float(np.max(gap_arr))
        summary["component_pairwise_probability_gap_min"] = float(np.min(gap_arr))
    return summary
