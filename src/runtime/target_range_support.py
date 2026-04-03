from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd


def resolve_target_range_policy(
    config: Mapping[str, Any] | None,
    *,
    target_range_model_dir: Path,
    default_override_ratio: float,
    default_confidence_scale: float,
    default_horizons: Sequence[float],
) -> Optional[Dict[str, Any]]:
    if not config:
        return None

    policy = {
        "enabled": bool(config.get("enabled", False)),
        "model_dir": Path(config.get("model_dir") or target_range_model_dir).expanduser(),
        "override_ratio": max(float(config.get("override_ratio") or default_override_ratio), 0.0),
        "confidence_rmse_scale": max(
            float(config.get("confidence_rmse_scale") or default_confidence_scale),
            1e-6,
        ),
    }

    horizons = config.get("horizons")
    if horizons is None:
        policy["horizons"] = list(default_horizons)
    else:
        policy["horizons"] = sorted({float(horizon) for horizon in horizons if float(horizon) > 0})
    return policy


def target_range_label(horizon: float) -> str:
    if float(horizon).is_integer():
        return f"{int(round(horizon))}h"
    return f"{horizon:g}h"


def load_target_range_model(
    path: Path,
    *,
    stderr_write: Callable[[str], None],
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:  # pragma: no cover - corrupted artifact guard
        stderr_write(f"Warning: failed to load target-range model at {path}: {exc}\n")
        return None
    if not isinstance(payload, Mapping) or "model" not in payload:
        stderr_write(f"Warning: malformed target-range payload at {path}; skipping.\n")
        return None
    feature_names = payload.get("feature_names") or []
    normalized = dict(payload)
    normalized["feature_names"] = [str(name) for name in feature_names]
    metrics = payload.get("metrics") or {}
    normalized["metrics"] = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    return normalized


def load_target_range_models(
    policy: Mapping[str, Any] | None,
    horizons: Sequence[float],
    *,
    target_range_model_dir: Path,
    load_target_range_model_fn: Callable[[Path], Optional[Dict[str, Any]]],
    stderr_write: Callable[[str], None],
) -> Dict[float, Dict[str, Any]]:
    if not policy or not policy.get("enabled"):
        return {}

    model_dir = Path(policy.get("model_dir") or target_range_model_dir)
    target_horizons = {float(value) for value in (policy.get("horizons") or horizons)}

    bundles: Dict[float, Dict[str, Any]] = {}
    for horizon in horizons:
        if float(horizon) not in target_horizons:
            continue
        label = target_range_label(horizon)
        high_path = model_dir / f"{label}_high.joblib"
        low_path = model_dir / f"{label}_low.joblib"
        high_payload = load_target_range_model_fn(high_path)
        low_payload = load_target_range_model_fn(low_path)
        if not high_payload or not low_payload:
            missing = []
            if not high_payload:
                missing.append(high_path.name)
            if not low_payload:
                missing.append(low_path.name)
            stderr_write(
                f"Warning: skipping target-range models for {label} horizon (missing {', '.join(missing)}).\n"
            )
            continue
        bundles[float(horizon)] = {
            "high": high_payload,
            "low": low_payload,
        }
    return bundles


def predict_single_target_model(payload: Mapping[str, Any], row: pd.Series) -> float:
    model = payload.get("model")
    feature_names: Sequence[str] = payload.get("feature_names") or []
    if not feature_names:
        raise RuntimeError("Target-range model payload missing feature_names for inference")
    values = [float(row.get(name, 0.0)) for name in feature_names]
    vector = np.asarray(values, dtype=float).reshape(1, -1)
    prediction = model.predict(vector)
    return float(prediction[0])


def confidence_from_rmse(rmse: float | None, scale: float) -> float:
    if rmse is None:
        return 0.0
    return max(0.0, min(1.0, math.exp(-rmse / max(scale, 1e-6))))


def predict_target_range_prices(
    bundle: Mapping[str, Any],
    row: pd.Series,
    *,
    close: float,
    confidence_scale: float,
    finite_float_or_none: Callable[[Any], float | None],
) -> Dict[str, float]:
    high_payload = bundle.get("high")
    low_payload = bundle.get("low")
    if not high_payload or not low_payload:
        raise RuntimeError("Incomplete target-range bundle supplied for inference")

    high_ret = predict_single_target_model(high_payload, row)
    low_ret = predict_single_target_model(low_payload, row)
    projected_high = close * math.exp(high_ret)
    projected_low = close * math.exp(low_ret)

    rmse_high = high_payload.get("metrics", {}).get("val_rmse")
    rmse_low = low_payload.get("metrics", {}).get("val_rmse")
    return {
        "projected_high": projected_high,
        "projected_low": projected_low,
        "projected_high_confidence": confidence_from_rmse(rmse_high, confidence_scale),
        "projected_low_confidence": confidence_from_rmse(rmse_low, confidence_scale),
        "projected_high_rmse": finite_float_or_none(rmse_high),
        "projected_low_rmse": finite_float_or_none(rmse_low),
        "projected_high_residual_std": finite_float_or_none(high_payload.get("metrics", {}).get("val_residual_std")),
        "projected_low_residual_std": finite_float_or_none(low_payload.get("metrics", {}).get("val_residual_std")),
    }


def apply_target_range_overrides(
    stop_loss: float,
    take_profit: float,
    projection: Mapping[str, float],
    override_ratio: float,
    direction: int,
) -> tuple[Dict[str, Dict[str, float] | None], float, float]:
    overrides = {
        "stop_loss": None,
        "take_profit": None,
    }
    updated_stop = stop_loss
    updated_take = take_profit

    projected_high = projection.get("projected_high")
    projected_low = projection.get("projected_low")

    if projected_high is not None and direction >= 1:
        if projected_high >= take_profit * (1.0 + override_ratio):
            overrides["take_profit"] = {
                "previous": take_profit,
                "updated": projected_high,
                "reason": "target_range_high",
            }
            updated_take = projected_high
    elif projected_low is not None and direction <= 0:
        if projected_low <= take_profit * (1.0 - override_ratio):
            overrides["take_profit"] = {
                "previous": take_profit,
                "updated": projected_low,
                "reason": "target_range_low",
            }
            updated_take = projected_low

    if projected_low is not None and direction >= 1:
        if projected_low <= stop_loss * (1.0 - override_ratio):
            overrides["stop_loss"] = {
                "previous": stop_loss,
                "updated": projected_low,
                "reason": "target_range_low",
            }
            updated_stop = projected_low
    elif projected_high is not None and direction <= 0:
        if projected_high >= stop_loss * (1.0 + override_ratio):
            overrides["stop_loss"] = {
                "previous": stop_loss,
                "updated": projected_high,
                "reason": "target_range_high",
            }
            updated_stop = projected_high

    return overrides, updated_stop, updated_take


def evaluate_direction_only_fallback(
    policy: Optional[Dict[str, Any]],
    *,
    p_up: float,
    signal_dir_only: int,
    expected_value: float,
    projected_price: float,
    signal_ts: str,
    trend_prob: float,
    trend_threshold: Optional[float],
    inactive_direction_fallback: Callable[..., Dict[str, Any]],
    parse_iso_timestamp: Callable[[str], Any],
) -> tuple[Dict[str, Any], bool]:
    if policy is None:
        return inactive_direction_fallback("not_configured"), False
    size_factor = float(policy.get("size_factor", 0.0))
    if not policy.get("enabled", True):
        return inactive_direction_fallback("disabled", size_factor=size_factor), False

    side = "long" if int(signal_dir_only or 0) == 1 else "short"
    side_prob = p_up if side == "long" else 1.0 - p_up
    threshold = float(policy.get("prob_threshold", 0.5))
    if side_prob < threshold:
        return inactive_direction_fallback("insufficient_probability", side=side, size_factor=size_factor), False

    if expected_value >= 0.0:
        return inactive_direction_fallback("non_negative_ev", side=side, size_factor=size_factor), False

    allowed_negative = float(policy.get("max_negative_ev", 0.0))
    ignition_extension_reason = False
    ignition_extension = float(policy.get("ignition_ev_extension", 0.0))
    if ignition_extension and trend_threshold is not None and trend_prob >= trend_threshold:
        allowed_negative += ignition_extension
        ignition_extension_reason = True

    if expected_value < -allowed_negative:
        reason = "ev_below_band_ignition_extension" if ignition_extension_reason else "ev_below_band"
        return inactive_direction_fallback(reason, side=side, size_factor=size_factor), False

    cooldown_hours = float(policy.get("cooldown_hours", 0.0))
    last_ts = policy.get("last_trigger_ts")
    cooldown_active = False
    if cooldown_hours > 0 and isinstance(last_ts, str) and last_ts.strip():
        try:
            elapsed = (parse_iso_timestamp(signal_ts) - parse_iso_timestamp(last_ts)).total_seconds() / 3600.0
            if elapsed < cooldown_hours:
                cooldown_active = True
        except ValueError:
            cooldown_active = False
    if cooldown_active:
        return inactive_direction_fallback(
            "cooldown_active",
            side=side,
            cooldown_active=True,
            size_factor=size_factor,
        ), False

    ratio = max(float(policy.get("stop_take_ratio", 0.0)), 0.0)
    projected = float(projected_price)
    if ratio == 0.0:
        stop_loss = projected
        take_profit = projected
    elif side == "long":
        stop_loss = projected * (1.0 - ratio)
        take_profit = projected * (1.0 + ratio)
    else:
        stop_loss = projected * (1.0 + ratio)
        take_profit = projected * (1.0 - ratio)

    reason = "ev_within_band_ignition_extension" if ignition_extension_reason else "ev_within_band"
    payload = {
        "active": True,
        "side": side,
        "size_factor": size_factor,
        "stop_loss_fallback": stop_loss,
        "take_profit_fallback": take_profit,
        "reason": reason,
        "cooldown_active": False,
    }
    policy["last_trigger_ts"] = signal_ts
    return payload, True