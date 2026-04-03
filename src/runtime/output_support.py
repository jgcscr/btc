from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


def build_trade_ready_monitoring_payload(
    predictions_payload: dict[str, Any],
    args: Any,
    *,
    horizon_sort_key: Callable[[Any], Any],
    format_horizon_label: Callable[[float], str],
    confidence_min_default: float,
    position_size_floor_default: float,
    position_size_cap_default: float,
) -> dict[str, Any]:
    predictions = predictions_payload.get("predictions", {})
    horizons: list[dict[str, Any]] = []
    for horizon_key in sorted(predictions.keys(), key=horizon_sort_key):
        entry = predictions[horizon_key]
        if isinstance(entry, dict):
            horizons.append(entry)

    request = {
        "targets": args.targets,
        "spot_provider": args.spot_provider,
        "hours": args.hours,
        "dry_run": bool(args.dry_run),
        "confidence_min": float(getattr(args, "confidence_min", confidence_min_default)),
        "position_size_floor": float(getattr(args, "position_size_floor", position_size_floor_default)),
        "position_size_cap": float(getattr(args, "position_size_cap", position_size_cap_default)),
    }
    position_size_cap_by_horizon = getattr(args, "position_size_cap_by_horizon", None)
    if isinstance(position_size_cap_by_horizon, Mapping) and position_size_cap_by_horizon:
        request["position_size_cap_by_horizon"] = {
            format_horizon_label(float(key)): float(value)
            for key, value in sorted(position_size_cap_by_horizon.items(), key=lambda item: float(item[0]))
        }
    confidence_min_by_horizon_regime = getattr(args, "confidence_min_by_horizon_regime", None)
    if isinstance(confidence_min_by_horizon_regime, Mapping) and confidence_min_by_horizon_regime:
        request["confidence_min_by_horizon_regime"] = {
            format_horizon_label(float(horizon)): {str(regime): float(value) for regime, value in regimes.items()}
            for horizon, regimes in sorted(confidence_min_by_horizon_regime.items(), key=lambda item: float(item[0]))
            if isinstance(regimes, Mapping)
        }
    data_quality_cfg = getattr(args, "data_quality", None)
    if isinstance(data_quality_cfg, Mapping):
        request["data_quality"] = dict(data_quality_cfg)
    metadata = getattr(args, "local_feature_metadata", None)
    if metadata:
        request["local_feature_overrides"] = metadata

    payload = {
        "generated_at": predictions_payload.get("generated_at"),
        "source": "run_refresh_and_predict",
        "request": request,
        "horizons": horizons,
    }
    for key in ("blocked_trade_analytics", "degradation_monitoring", "prompt_ready_summary"):
        if isinstance(predictions_payload.get(key), Mapping):
            payload[key] = predictions_payload.get(key)
    return payload


def write_monitoring_payload_file(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_monitoring_artifact(
    predictions_payload: dict[str, Any],
    args: Any,
    *,
    output_path: Path,
    horizon_sort_key: Callable[[Any], Any],
    format_horizon_label: Callable[[float], str],
    confidence_min_default: float,
    position_size_floor_default: float,
    position_size_cap_default: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monitoring_payload = payload or build_trade_ready_monitoring_payload(
        predictions_payload,
        args,
        horizon_sort_key=horizon_sort_key,
        format_horizon_label=format_horizon_label,
        confidence_min_default=confidence_min_default,
        position_size_floor_default=position_size_floor_default,
        position_size_cap_default=position_size_cap_default,
    )
    write_monitoring_payload_file(monitoring_payload, output_path)
    return monitoring_payload


def refresh_meta_baseline(
    *,
    source_csv: Path,
    json_path: Path,
    parquet_path: Path,
    load_dataframe: Callable[..., Any],
    compute_baseline: Callable[..., dict[str, Any]],
    baseline_to_dataframe: Callable[[dict[str, Any]], Any],
    append_detected_meta_columns: Callable[[Any, list[str]], list[str]],
    default_columns: list[str],
    stderr_write: Callable[[str], None],
) -> None:
    if not source_csv.exists():
        stderr_write(f"Meta baseline CSV not found at {source_csv.as_posix()}; skipping baseline refresh.\n")
        return
    df = load_dataframe(source_csv, limit=0)
    if df.empty:
        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": 0,
            "columns": {},
            "column_order": list(default_columns),
        }
    else:
        columns = append_detected_meta_columns(df, default_columns)
        baseline = compute_baseline(df, columns)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df = baseline_to_dataframe(baseline)
    baseline_df.to_parquet(parquet_path, index=False)
