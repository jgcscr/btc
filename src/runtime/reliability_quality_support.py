from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import pandas as pd

from src.runtime.reliability_workflow_common import load_json


def write_calibrated_quality_input(
    *,
    source_path: Path,
    calibration_path: Path,
    output_path: Path,
    regime_col: str = "regime_state",
    safe_float: Callable[[Any, float], float],
    calibration_label_from_value: Callable[[Any], str],
    resolve_trade_probability_for_horizon: Callable[..., tuple[float, str | None, bool, Mapping[str, Any] | None]],
) -> Dict[str, Any]:
    if not source_path.exists():
        return {
            "written": False,
            "reason": "source_missing",
            "source": str(source_path),
            "output": str(output_path),
        }
    if not calibration_path.exists():
        return {
            "written": False,
            "reason": "calibration_missing",
            "source": str(source_path),
            "output": str(output_path),
            "calibration": str(calibration_path),
        }

    frame = pd.read_csv(source_path)
    if "p_up" not in frame.columns:
        return {
            "written": False,
            "reason": "missing_p_up",
            "source": str(source_path),
            "output": str(output_path),
            "calibration": str(calibration_path),
        }

    calibration_payload = load_json(calibration_path)
    working = frame.copy()
    raw_p_up = pd.to_numeric(working.get("raw_p_up", working["p_up"]), errors="coerce")
    raw_p_up = raw_p_up.fillna(pd.to_numeric(working["p_up"], errors="coerce")).fillna(0.5)

    calibrated_values: List[float] = []
    applied_keys: List[str] = []
    guard_applied: List[float] = []
    used_regime_key: List[float] = []

    for row in working.to_dict(orient="records"):
        raw_probability = safe_float(row.get("raw_p_up", row.get("p_up", 0.5)), 0.5)
        horizon_label = calibration_label_from_value(row.get("horizon", "1h"))
        regime_state = str(row.get(regime_col, "unknown") or "unknown").strip().lower() or "unknown"
        close = safe_float(row.get("close", 0.0), 0.0)
        projected_price = safe_float(row.get("projected_price", close), close)
        ret_pred = safe_float(row.get("ret_pred", 0.0), 0.0)
        calibrated_probability, applied_key, used_regime, guard_payload = resolve_trade_probability_for_horizon(
            platt_calibration=calibration_payload,
            label=horizon_label,
            regime_state=regime_state,
            raw_probability=raw_probability,
            close=close,
            projected_price=projected_price,
            ret_pred=ret_pred,
        )
        calibrated_values.append(float(calibrated_probability))
        applied_keys.append(str(applied_key or ""))
        used_regime_key.append(float(bool(used_regime)))
        guard_applied.append(float(bool(isinstance(guard_payload, Mapping) and guard_payload.get("applied"))))

    calibrated_series = pd.Series(calibrated_values, index=working.index, dtype=float)
    working["raw_p_up"] = raw_p_up.astype(float)
    working["p_up"] = calibrated_series
    working["raw_calibrated_probability_gap"] = calibrated_series - working["raw_p_up"]
    working["probability_calibration_used_regime_key"] = pd.Series(used_regime_key, index=working.index, dtype=float)
    working["probability_calibration_guard_applied"] = pd.Series(guard_applied, index=working.index, dtype=float)
    working["probability_calibration_applied_key"] = pd.Series(applied_keys, index=working.index, dtype=object)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    working.to_csv(output_path, index=False)
    return {
        "written": True,
        "source": str(source_path),
        "output": str(output_path),
        "calibration": str(calibration_path),
        "rows": int(len(working)),
        "mean_abs_delta": float((working["raw_calibrated_probability_gap"].abs().mean())),
        "regime_col": regime_col,
    }


def write_meta_component_frame(
    *,
    source_path: Path,
    output_path: Path,
    requested_columns: Sequence[str] | None = None,
) -> Dict[str, Any]:
    from src.utils.component_diversity_support import (
        build_component_feature_frame,
        summarize_component_history,
    )

    if not source_path.exists():
        return {
            "written": False,
            "reason": "source_missing",
            "source": str(source_path),
            "output": str(output_path),
        }

    frame = pd.read_csv(source_path)
    ts_col = "ts" if "ts" in frame.columns else None
    if ts_col is None:
        return {
            "written": False,
            "reason": "missing_ts",
            "source": str(source_path),
            "output": str(output_path),
        }

    ret_col = None
    for candidate in ("ret_1h", "ret_realized"):
        if candidate in frame.columns:
            ret_col = candidate
            break
    if ret_col is None:
        return {
            "written": False,
            "reason": "missing_return_column",
            "source": str(source_path),
            "output": str(output_path),
        }

    candidate_columns = [
        str(column)
        for column in frame.columns
        if str(column).startswith("p_up_") and str(column) not in {"p_up_meta", "p_up_gate"}
    ]
    if requested_columns:
        aliases = {column: column for column in candidate_columns}
        aliases.update(
            {
                column.removeprefix("p_up_"): column
                for column in candidate_columns
                if column.startswith("p_up_")
            }
        )
        component_columns: List[str] = []
        for raw_column in requested_columns:
            key = str(raw_column).strip().lower()
            if not key:
                continue
            resolved = aliases.get(key)
            if resolved and resolved not in component_columns:
                component_columns.append(resolved)
    else:
        component_columns = sorted(candidate_columns)

    if not component_columns:
        return {
            "written": False,
            "reason": "missing_component_columns",
            "source": str(source_path),
            "output": str(output_path),
        }

    derived = frame[[ts_col, ret_col, *component_columns]].copy()
    if ret_col != "ret_1h":
        derived = derived.rename(columns={ret_col: "ret_1h"})
    component_features = build_component_feature_frame(derived, component_columns)
    for column in component_features.columns:
        derived[column] = pd.to_numeric(component_features[column], errors="coerce").fillna(0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(output_path, index=False)
    return {
        "written": True,
        "source": str(source_path),
        "output": str(output_path),
        "rows": int(len(derived)),
        "component_columns": component_columns,
        "component_diversity": summarize_component_history(derived, component_columns),
    }