from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd


def load_last_trigger_ts(state_path: Path) -> Optional[str]:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    ts_value = payload.get("last_trigger_ts")
    if isinstance(ts_value, str) and ts_value.strip():
        return ts_value
    return None


def write_last_trigger_ts(state_path: Path, ts_value: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_trigger_ts": ts_value}, indent=2))


def resolve_trend_ignition_payload(
    config: Mapping[str, Any] | None,
    *,
    load_trend_ignition_classifier: Callable[[str], Dict[str, Any]],
    load_state: Callable[[], Optional[str]],
    stderr_write: Callable[[str], None],
) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    enabled = config.get("enabled")
    if enabled is False:
        return None
    model_path = config.get("model_path")
    if not model_path:
        return None

    try:
        payload = load_trend_ignition_classifier(str(model_path))
    except FileNotFoundError as exc:
        stderr_write(f"Warning: {exc}; trend ignition support disabled.\n")
        return None

    threshold = config.get("probability_threshold")
    cooldown = config.get("cooldown_hours")
    payload["threshold"] = float(threshold) if threshold is not None else 0.6
    payload["cooldown_hours"] = max(float(cooldown) if cooldown is not None else 0.0, 0.0)
    payload["last_trigger_ts"] = load_state()
    return payload


def inactive_direction_fallback(
    reason: str,
    *,
    side: Optional[str] = None,
    cooldown_active: bool = False,
    size_factor: float = 0.0,
) -> Dict[str, Any]:
    return {
        "active": False,
        "side": side,
        "size_factor": size_factor,
        "stop_loss_fallback": None,
        "take_profit_fallback": None,
        "reason": reason,
        "cooldown_active": cooldown_active,
    }


def resolve_direction_fallback_policy(
    config: Mapping[str, Any] | None,
    *,
    load_state: Callable[[], Optional[str]],
) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    enabled = config.get("enabled")
    if enabled is False:
        return {
            "enabled": False,
            "prob_threshold": float(config.get("prob_threshold") or 0.5),
            "max_negative_ev": float(config.get("max_negative_ev") or 0.0),
            "size_factor": float(config.get("size_factor") or 0.0),
            "stop_take_ratio": float(config.get("stop_take_ratio") or 0.0),
            "cooldown_hours": float(config.get("cooldown_hours") or 0.0),
            "ignition_ev_extension": float(config.get("ignition_ev_extension") or 0.0),
            "last_trigger_ts": load_state(),
        }

    return {
        "enabled": True,
        "prob_threshold": float(config.get("prob_threshold") or 0.6),
        "max_negative_ev": max(float(config.get("max_negative_ev") or 0.0), 0.0),
        "size_factor": max(float(config.get("size_factor") or 1.0), 0.0),
        "stop_take_ratio": max(float(config.get("stop_take_ratio") or 0.0), 0.0),
        "cooldown_hours": max(float(config.get("cooldown_hours") or 0.0), 0.0),
        "ignition_ev_extension": max(float(config.get("ignition_ev_extension") or 0.0), 0.0),
        "last_trigger_ts": load_state(),
    }


def resolve_adaptive_thresholds_policy(config: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not config:
        return None

    policy: Dict[str, Any] = {
        "enabled": bool(config.get("enabled", False)),
        "breakout_score_threshold": float(config.get("breakout_score_threshold") or 0.8),
        "chop_score_threshold": float(config.get("chop_score_threshold") or 0.3),
        "breakout_scale": max(float(config.get("breakout_scale") or 0.9), 0.0),
        "chop_scale": max(float(config.get("chop_scale") or 1.1), 0.0),
    }
    for clamp_key in ("p_up_min_floor", "p_up_min_ceiling", "ret_min_floor", "ret_min_ceiling"):
        clamp_value = config.get(clamp_key)
        policy[clamp_key] = float(clamp_value) if clamp_value is not None else None
    return policy


def compute_profile_breakout_score(
    prepared: Any,
    index: int,
    volatility_snapshot: Mapping[str, Any] | None,
    *,
    breakout_vol_normalizer: float,
    breakout_ret_normalizer: float,
) -> float:
    snapshot = volatility_snapshot or {}
    vol_component = 0.0
    for value in snapshot.values():
        try:
            vol_component = max(vol_component, float(value))
        except (TypeError, ValueError):
            continue

    ret_component = 0.0
    if index > 0 and "close" in prepared.df_all.columns:
        try:
            current_close = float(prepared.df_all["close"].iloc[index])
            prev_close = float(prepared.df_all["close"].iloc[index - 1])
            if current_close > 0 and prev_close > 0:
                ret_component = abs(math.log(current_close / prev_close))
        except (ValueError, ZeroDivisionError, IndexError):
            ret_component = 0.0

    norm_vol = min(vol_component / breakout_vol_normalizer, 2.0) if breakout_vol_normalizer else 0.0
    norm_ret = min(ret_component / breakout_ret_normalizer, 2.0) if breakout_ret_normalizer else 0.0
    score = (norm_vol + norm_ret) / 2.0
    return round(score, 6)


def derive_regime_labels_from_frame(
    frame: pd.DataFrame,
    *,
    volatility_col: str,
    breakout_score_threshold: float,
    chop_score_threshold: float,
    breakout_vol_normalizer: float,
    breakout_ret_normalizer: float,
    regime_trend: str,
    regime_neutral: str,
    regime_chop: str,
) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce") if "close" in frame.columns else pd.Series(np.nan, index=frame.index)
    volatility = pd.to_numeric(frame.get(volatility_col), errors="coerce") if volatility_col in frame.columns else pd.Series(0.0, index=frame.index)
    ret_component = pd.Series(0.0, index=frame.index, dtype=float)
    valid_close = close > 0.0
    if valid_close.any():
        ret_component = np.log(close.where(valid_close)).diff().abs().fillna(0.0)
    vol_component = volatility.fillna(0.0).abs()
    norm_vol = (vol_component / breakout_vol_normalizer).clip(lower=0.0, upper=2.0) if breakout_vol_normalizer else vol_component * 0.0
    norm_ret = (ret_component / breakout_ret_normalizer).clip(lower=0.0, upper=2.0) if breakout_ret_normalizer else ret_component * 0.0
    score = ((norm_vol + norm_ret) / 2.0).fillna(0.0)

    labels = pd.Series(regime_neutral, index=frame.index, dtype=object)
    labels.loc[score >= breakout_score_threshold] = regime_trend
    labels.loc[score <= chop_score_threshold] = regime_chop
    return labels


def compute_breakout_scores(
    prepared_bundles: Mapping[str, tuple[Any, int, float, str]],
    volatility_snapshots: Mapping[str, Mapping[str, float]],
    *,
    compute_profile_breakout_score: Callable[[Any, int, Mapping[str, Any] | None], float],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for key, bundle in prepared_bundles.items():
        prepared, index, _close, _ts = bundle
        snapshot = volatility_snapshots.get(key, {})
        scores[key] = compute_profile_breakout_score(prepared, index, snapshot)
    return scores


def classify_regime_from_score(
    score: float,
    policy: Mapping[str, Any],
    *,
    regime_trend: str,
    regime_neutral: str,
    regime_chop: str,
) -> str:
    breakout_threshold = float(policy.get("breakout_score_threshold", 1.0))
    chop_threshold = float(policy.get("chop_score_threshold", 0.0))
    if score >= breakout_threshold:
        return regime_trend
    if score <= chop_threshold:
        return regime_chop
    return regime_neutral


def apply_adaptive_thresholds(
    policy: Mapping[str, Any],
    base_p_up: float,
    base_ret: float,
    regime_state: str,
    *,
    regime_trend: str,
    regime_chop: str,
) -> tuple[float, float, float]:
    if not policy.get("enabled"):
        return base_p_up, base_ret, 1.0

    if regime_state == regime_trend:
        scale = float(policy.get("breakout_scale", 1.0))
    elif regime_state == regime_chop:
        scale = float(policy.get("chop_scale", 1.0))
    else:
        scale = 1.0

    scaled_p = base_p_up * scale
    scaled_ret = base_ret * scale

    floor = policy.get("p_up_min_floor")
    if floor is not None:
        scaled_p = max(scaled_p, float(floor))
    ceiling = policy.get("p_up_min_ceiling")
    if ceiling is not None:
        scaled_p = min(scaled_p, float(ceiling))

    ret_floor = policy.get("ret_min_floor")
    if ret_floor is not None:
        scaled_ret = max(scaled_ret, float(ret_floor))
    ret_ceiling = policy.get("ret_min_ceiling")
    if ret_ceiling is not None:
        scaled_ret = min(scaled_ret, float(ret_ceiling))

    return scaled_p, scaled_ret, scale