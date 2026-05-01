from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.runtime.reliability_config_support import build_audit_weighted_runtime_config
from src.runtime.reliability_workflow_common import load_json, load_yaml
from src.scripts.build_direction_output_shadow_config import build_shadow_config


def resolve_direction_output_shadow_horizons(
    shadow_cfg: Mapping[str, Any],
    *,
    default_horizons: Sequence[float | int] = (1.0,),
) -> List[float]:
    raw_values = shadow_cfg.get("horizons") if isinstance(shadow_cfg, Mapping) else None
    values = raw_values if isinstance(raw_values, list) and raw_values else list(default_horizons)
    resolved: List[float] = []
    for value in values:
        try:
            horizon = float(value)
        except (TypeError, ValueError):
            continue
        if horizon <= 0:
            continue
        if horizon not in resolved:
            resolved.append(horizon)
    return resolved or [1.0]


def write_direction_output_shadow_config(
    *,
    base_config_path: Path,
    direction_output_calibration_path: Path,
    output_path: Path,
    meta_output_path: Path,
    marginal_audit_path: Path | None = None,
    neutral_band: float = 0.02,
    horizons: Sequence[float] = (1.0,),
) -> Dict[str, Any]:
    payload = build_shadow_config(
        base_config_path=base_config_path,
        direction_output_calibration_path=direction_output_calibration_path,
        output_path=output_path,
        marginal_audit_path=marginal_audit_path,
        neutral_band=neutral_band,
        horizons=horizons,
    )
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_upstream_direction_candidate_config(
    *,
    base_config_path: Path,
    marginal_audit_path: Path,
    output_path: Path,
    meta_output_path: Path,
    apply_to_paper_live: bool = False,
) -> Dict[str, Any]:
    audit_payload = load_json(marginal_audit_path)
    applied = build_audit_weighted_runtime_config(
        base_config_path=base_config_path,
        audit_payload=audit_payload,
        output_path=output_path,
    )
    weight_recommendations = audit_payload.get("weight_recommendations") if isinstance(audit_payload, dict) else None
    payload = {
        "base_config": str(base_config_path),
        "marginal_audit_path": str(marginal_audit_path),
        "output_path": str(output_path),
        "apply_to_paper_live": bool(apply_to_paper_live),
        "internal_direction_weight_update_applied": bool(applied),
        "horizon_scope": [1.0],
        "recommended_weight_spec_1h": (
            weight_recommendations.get("recommended_weight_spec_1h") if isinstance(weight_recommendations, Mapping) else None
        ),
        "recommended_regime_weights_1h": (
            weight_recommendations.get("recommended_regime_weights_1h")
            if isinstance(weight_recommendations, Mapping)
            else None
        ),
        "apply_fallback_for_missing_regimes": (
            weight_recommendations.get("apply_fallback_for_missing_regimes")
            if isinstance(weight_recommendations, Mapping)
            else None
        ),
    }
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _round_down_to_step(value: float, *, step: float) -> float:
    if step <= 0.0:
        return float(value)
    return float(np.floor(float(value) / float(step)) * float(step))


def _round_up_to_step(value: float, *, step: float) -> float:
    if step <= 0.0:
        return float(value)
    return float(np.ceil(float(value) / float(step)) * float(step))


def derive_trade_decision_regime_midband_candidate(
    *,
    candidate_path: Path,
    recent_window_rows: int,
    signal_col: str,
    p_col: str,
    ret_pred_col: str,
    return_col: str,
    regime_col: str,
    volatility_col: str = "volatility_realized_24h",
    min_regime_rows: int,
    require_overall_regime_negative: bool,
    band_step: float = 0.01,
    derive_recent_triggered_regime_volatility_rule: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    recent_rule = derive_recent_triggered_regime_volatility_rule(
        candidate_path=candidate_path,
        recent_window_rows=recent_window_rows,
        signal_col=signal_col,
        return_col=return_col,
        regime_col=regime_col,
        volatility_col=volatility_col,
        min_regime_rows=min_regime_rows,
        require_overall_regime_negative=require_overall_regime_negative,
    )
    if not bool(recent_rule.get("enabled", False)):
        return {
            "enabled": False,
            "reason": recent_rule.get("reason", "recent_rule_disabled"),
            "candidate_path": str(candidate_path),
            "recent_rule": recent_rule,
        }

    df = pd.read_csv(candidate_path)
    required = {signal_col, p_col, ret_pred_col, regime_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "enabled": False,
            "reason": "missing_required_columns",
            "candidate_path": str(candidate_path),
            "missing_columns": missing,
            "recent_rule": recent_rule,
        }

    working = df.copy()
    if "ts" in working.columns:
        working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
        working = working.sort_values("ts")
    recent = working.tail(max(int(recent_window_rows), 1)).copy()
    signal = pd.to_numeric(recent[signal_col], errors="coerce").fillna(0.0)
    recent = recent.loc[signal > 0.0].copy()
    if recent.empty:
        return {
            "enabled": False,
            "reason": "no_recent_active_trades",
            "candidate_path": str(candidate_path),
            "recent_rule": recent_rule,
        }

    recent["_regime"] = recent[regime_col].map(
        lambda value: str(value).strip().lower() if pd.notna(value) else "missing"
    )
    selected_regimes = [
        str(value).strip().lower()
        for value in (
            recent_rule.get("selected_regimes", []) if isinstance(recent_rule.get("selected_regimes"), list) else []
        )
        if str(value).strip()
    ]
    scoped = recent.loc[recent["_regime"].isin(selected_regimes)].copy()
    if scoped.empty:
        return {
            "enabled": False,
            "reason": "no_rows_for_selected_regimes",
            "candidate_path": str(candidate_path),
            "selected_regimes": selected_regimes,
            "recent_rule": recent_rule,
        }

    p_values = pd.to_numeric(scoped[p_col], errors="coerce")
    abs_ret_pred = pd.to_numeric(scoped[ret_pred_col], errors="coerce").abs()
    valid = p_values.notna() & abs_ret_pred.notna()
    scoped = scoped.loc[valid].copy()
    p_values = p_values.loc[valid]
    abs_ret_pred = abs_ret_pred.loc[valid]
    if scoped.empty:
        return {
            "enabled": False,
            "reason": "no_valid_probability_rows",
            "candidate_path": str(candidate_path),
            "selected_regimes": selected_regimes,
            "recent_rule": recent_rule,
        }

    p_up_low = round(max(0.0, min(1.0, _round_down_to_step(float(p_values.min()), step=band_step))), 6)
    p_up_high = round(max(0.0, min(1.0, _round_up_to_step(float(p_values.max()), step=band_step))), 6)
    min_abs_ret_pred = round(max(0.0, _round_down_to_step(float(abs_ret_pred.min()), step=0.0001)), 6)
    if p_up_high < p_up_low:
        p_up_high = p_up_low

    return {
        "enabled": True,
        "reason": "ready",
        "candidate_path": str(candidate_path),
        "selected_regimes": selected_regimes,
        "recent_window_rows": int(recent_window_rows),
        "row_count": int(scoped.shape[0]),
        "p_up_low": float(p_up_low),
        "p_up_high": float(p_up_high),
        "high_inclusive": True,
        "min_abs_ret_pred": float(min_abs_ret_pred),
        "max_abs_ret_pred": None,
        "p_up_min_observed": float(p_values.min()),
        "p_up_max_observed": float(p_values.max()),
        "abs_ret_pred_min_observed": float(abs_ret_pred.min()),
        "abs_ret_pred_max_observed": float(abs_ret_pred.max()),
        "recent_rule": recent_rule,
    }


def write_trade_decision_midband_candidate_config(
    *,
    base_config_path: Path,
    candidate_path: Path,
    output_path: Path,
    meta_output_path: Path,
    recent_window_rows: int = 288,
    signal_col: str = "signal_ensemble",
    p_col: str = "p_up",
    ret_pred_col: str = "ret_pred",
    return_col: str = "ret_ensemble_net",
    regime_col: str = "regime_state",
    volatility_col: str = "volatility_realized_24h",
    min_regime_rows: int = 2,
    require_overall_regime_negative: bool = True,
    apply_to_paper_live: bool = False,
    derive_recent_triggered_regime_volatility_rule: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_rule = derive_trade_decision_regime_midband_candidate(
        candidate_path=candidate_path,
        recent_window_rows=recent_window_rows,
        signal_col=signal_col,
        p_col=p_col,
        ret_pred_col=ret_pred_col,
        return_col=return_col,
        regime_col=regime_col,
        volatility_col=volatility_col,
        min_regime_rows=min_regime_rows,
        require_overall_regime_negative=require_overall_regime_negative,
        derive_recent_triggered_regime_volatility_rule=derive_recent_triggered_regime_volatility_rule,
    )

    payload = load_yaml(base_config_path)
    trade_decision_policy = payload.get("trade_decision_policy")
    if not isinstance(trade_decision_policy, dict):
        trade_decision_policy = {}
    current_midband = trade_decision_policy.get("midband_veto")
    if not isinstance(current_midband, dict):
        current_midband = {}

    applied = False
    if bool(candidate_rule.get("enabled", False)):
        updated_midband = dict(current_midband)
        updated_midband.update(
            {
                "enabled": True,
                "p_up_low": float(candidate_rule["p_up_low"]),
                "p_up_high": float(candidate_rule["p_up_high"]),
                "high_inclusive": bool(candidate_rule.get("high_inclusive", True)),
                "min_abs_ret_pred": float(candidate_rule.get("min_abs_ret_pred", 0.0)),
                "max_abs_ret_pred": candidate_rule.get("max_abs_ret_pred"),
                "regime_states": list(candidate_rule.get("selected_regimes", [])),
            }
        )
        trade_decision_policy["midband_veto"] = updated_midband
        payload["trade_decision_policy"] = trade_decision_policy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        applied = True

    meta_payload = {
        "base_config": str(base_config_path),
        "candidate_path": str(candidate_path),
        "output_path": str(output_path),
        "apply_to_paper_live": bool(apply_to_paper_live),
        "trade_decision_midband_update_applied": bool(applied),
        "candidate_rule": candidate_rule,
        "previous_midband_veto": current_midband,
    }
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_output_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return meta_payload