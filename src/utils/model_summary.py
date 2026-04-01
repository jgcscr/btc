from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def metrics_by_split(metrics: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    if isinstance(metrics, Mapping):
        normalized: dict[str, dict[str, float]] = {}
        for split, values in metrics.items():
            if not isinstance(values, Mapping):
                continue
            normalized[str(split)] = {
                str(key): float(value)
                for key, value in values.items()
                if value is not None
            }
        return normalized

    normalized = {}
    for entry in metrics:
        if not isinstance(entry, Mapping) or "split" not in entry:
            continue
        split = str(entry["split"])
        normalized[split] = {
            str(key): float(value)
            for key, value in entry.items()
            if key != "split" and value is not None
        }
    return normalized


def build_model_summary(
    *,
    model_type: str,
    target: str,
    dataset_path: str,
    model_path: str,
    metrics: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    hyperparams: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    horizon_hours: int | float | None = None,
    seq_len: int | None = None,
    scaler_path: str | None = None,
    trained_at: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_metrics = metrics_by_split(metrics)
    payload: dict[str, Any] = {
        "summary_schema_version": 2,
        "model_type": str(model_type),
        "target": str(target),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "metrics": normalized_metrics,
    }
    for split_name in ("train", "val", "test", "trainval", "oof"):
        split_metrics = normalized_metrics.get(split_name)
        if split_metrics:
            payload[f"{split_name}_metrics"] = dict(split_metrics)
    if feature_names is not None:
        payload["feature_names"] = [str(name) for name in feature_names]
    if params is not None:
        payload["params"] = dict(params)
    if hyperparams is not None:
        payload["hyperparams"] = dict(hyperparams)
    if threshold is not None:
        payload["threshold"] = float(threshold)
    if horizon_hours is not None:
        payload["horizon_hours"] = float(horizon_hours)
    if seq_len is not None:
        payload["seq_len"] = int(seq_len)
    if scaler_path is not None:
        payload["scaler_path"] = str(scaler_path)
    if trained_at is not None:
        payload["trained_at"] = str(trained_at)
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def write_model_summary(path: str | Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


__all__ = ["build_model_summary", "metrics_by_split", "write_model_summary"]