from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping


def resolve_feature_coverage_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    max_imputed_zero_columns = cfg.get("max_imputed_zero_columns")
    max_imputed_zero_ratio = cfg.get("max_imputed_zero_ratio")
    max_source_lag_hours = cfg.get("max_source_lag_hours")
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "max_imputed_zero_columns": max(float(1e9 if max_imputed_zero_columns is None else max_imputed_zero_columns), 0.0),
        "max_imputed_zero_ratio": max(float(1.0 if max_imputed_zero_ratio is None else max_imputed_zero_ratio), 0.0),
        "max_source_lag_hours": max(float(1e9 if max_source_lag_hours is None else max_source_lag_hours), 0.0),
        "block_on_violation": bool(cfg.get("block_on_violation", True)),
        "ignored_columns": sorted({str(column).strip() for column in (cfg.get("ignored_columns") or []) if str(column).strip()}),
    }


def evaluate_feature_coverage(metadata: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    feature_alignment = metadata.get("feature_alignment", {}) if isinstance(metadata, Mapping) else {}
    source_freshness = metadata.get("source_freshness", {}) if isinstance(metadata, Mapping) else {}
    imputed_zero_columns = feature_alignment.get("imputed_zero_columns", []) if isinstance(feature_alignment, Mapping) else []
    required_columns = int(feature_alignment.get("required_columns", 0) or 0) if isinstance(feature_alignment, Mapping) else 0
    ignored_columns = set(policy.get("ignored_columns", [])) if isinstance(policy, Mapping) else set()
    ignored_imputed_zero_columns = []
    effective_imputed_zero_columns = []
    if isinstance(imputed_zero_columns, list):
        ignored_imputed_zero_columns = [column for column in imputed_zero_columns if column in ignored_columns]
        effective_imputed_zero_columns = [column for column in imputed_zero_columns if column not in ignored_columns]
    effective_required_columns = max(required_columns - len(ignored_imputed_zero_columns), 0)
    imputed_zero_count = len(effective_imputed_zero_columns)
    imputed_zero_ratio = (imputed_zero_count / effective_required_columns) if effective_required_columns > 0 else 0.0
    max_lag_hours = 0.0
    stale_sources: list[str] = []
    if isinstance(source_freshness, Mapping):
        for source_name, payload in source_freshness.items():
            if not isinstance(payload, Mapping):
                continue
            lag_hours = float(payload.get("lag_hours") or 0.0)
            max_lag_hours = max(max_lag_hours, lag_hours)
            if lag_hours > float(policy.get("max_source_lag_hours", 1e9)):
                stale_sources.append(str(source_name))

    failed_checks: list[str] = []
    if imputed_zero_count > float(policy.get("max_imputed_zero_columns", 1e9)):
        failed_checks.append("imputed_zero_columns")
    if imputed_zero_ratio > float(policy.get("max_imputed_zero_ratio", 1.0)):
        failed_checks.append("imputed_zero_ratio")
    if stale_sources:
        failed_checks.append("stale_sources")

    return {
        "enabled": bool(policy.get("enabled", False)),
        "ok": not failed_checks,
        "imputed_zero_count": int(imputed_zero_count),
        "imputed_zero_ratio": float(imputed_zero_ratio),
        "effective_required_columns": int(effective_required_columns),
        "ignored_columns": sorted(ignored_columns),
        "ignored_imputed_zero_columns": ignored_imputed_zero_columns,
        "effective_imputed_zero_columns": effective_imputed_zero_columns,
        "max_source_lag_hours_observed": float(max_lag_hours),
        "stale_sources": stale_sources,
        "failed_checks": failed_checks,
        "block_on_violation": bool(policy.get("block_on_violation", True)),
    }


def resolve_data_quality_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = config or {}
    max_staleness_hours = cfg.get("max_staleness_hours")
    max_missing_ratio = cfg.get("max_missing_ratio")
    max_zero_volume_ratio = cfg.get("max_zero_volume_ratio")
    min_rows = cfg.get("min_rows")
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "max_staleness_hours": float(2.0 if max_staleness_hours is None else max_staleness_hours),
        "max_missing_ratio": float(0.01 if max_missing_ratio is None else max_missing_ratio),
        "max_zero_volume_ratio": float(0.2 if max_zero_volume_ratio is None else max_zero_volume_ratio),
        "min_rows": int(120 if min_rows is None else min_rows),
    }


def evaluate_data_quality(
    frame: Any,
    policy_config: Mapping[str, Any] | None,
    *,
    data_quality_policy_type: Callable[..., Any],
    evaluate_ohlcv_quality: Callable[[Any, Any], Mapping[str, Any]],
    data_quality_error_type: type[Exception],
    write_data_quality_payload: Callable[[Mapping[str, Any]], None],
) -> Dict[str, Any]:
    policy_values = resolve_data_quality_policy(policy_config)
    policy = data_quality_policy_type(
        max_staleness_hours=float(policy_values["max_staleness_hours"]),
        max_missing_ratio=float(policy_values["max_missing_ratio"]),
        max_zero_volume_ratio=float(policy_values["max_zero_volume_ratio"]),
        min_rows=int(policy_values["min_rows"]),
    )
    payload: Dict[str, Any] = {
        "ok": True,
        "policy": {
            "enabled": bool(policy_values["enabled"]),
            "max_staleness_hours": policy.max_staleness_hours,
            "max_missing_ratio": policy.max_missing_ratio,
            "max_zero_volume_ratio": policy.max_zero_volume_ratio,
            "min_rows": policy.min_rows,
        },
    }
    try:
        payload.update(evaluate_ohlcv_quality(frame, policy))
    except data_quality_error_type as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["row_count"] = int(len(frame))
    write_data_quality_payload(payload)
    return payload


def write_data_quality_payload(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")