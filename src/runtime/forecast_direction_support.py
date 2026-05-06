from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def coerce_result_horizon(
    value: Any,
    *,
    normalize_horizon_value: Callable[[Any], float] | None = None,
) -> float | None:
    try:
        if normalize_horizon_value is not None:
            return normalize_horizon_value(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def direction_vote(entry: Mapping[str, Any]) -> str:
    return "up" if str(entry.get("direction_next", "down")).lower() == "up" else "down"


def direction_from_ret_pred(value: Any) -> str:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def direction_from_projected_price(close: Any, projected_price: Any) -> str:
    close_value = _finite_float_or_none(close)
    projected_value = _finite_float_or_none(projected_price)
    if close_value is None or projected_value is None:
        return "neutral"
    if close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"
    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def direction_from_probability(value: Any, *, neutral_band: float = 0.0) -> str:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def resolve_direction_signal_for_horizon(
    *,
    raw_probability: float,
    calibrated_probability: float,
    threshold: float,
    close: float,
    projected_price: float,
    ret_pred: float,
    calibration_key: str | None,
    calibration_used_regime_key: bool,
) -> int:
    directional_threshold = max(float(threshold), 0.5)
    calibrated_signal = int(float(calibrated_probability) >= directional_threshold)
    raw_signal = int(float(raw_probability) >= directional_threshold)
    raw_side = "up" if raw_signal == 1 else "down"
    calibrated_side = "up" if calibrated_signal == 1 else "down"
    ret_side = direction_from_ret_pred(ret_pred)
    projected_side = direction_from_projected_price(close, projected_price)

    if ret_side == projected_side and ret_side in {"up", "down"}:
        forecast_consensus_signal = 1 if ret_side == "up" else 0
        if calibrated_side != ret_side:
            return forecast_consensus_signal

    if raw_signal == calibrated_signal:
        return calibrated_signal
    if ret_side == raw_side and projected_side == raw_side:
        return raw_signal
    if ret_side == calibrated_side and projected_side == calibrated_side:
        return calibrated_signal
    if calibration_key is None or calibration_used_regime_key:
        return calibrated_signal
    return calibrated_signal


def derive_probability_alignment_features(
    *,
    close: float,
    projected_price: float,
    ret_pred: float,
    raw_probability: float,
    resolved_probability: float,
    direction: str,
    neutral_band: float,
    probability_guard: Mapping[str, Any] | None,
    calibration_used_regime_key: bool,
) -> Dict[str, float | str]:
    direction_side = str(direction).strip().lower()
    ret_side = direction_from_ret_pred(ret_pred)
    projected_side = direction_from_projected_price(close, projected_price)
    raw_side = direction_from_probability(raw_probability, neutral_band=neutral_band)
    resolved_side = direction_from_probability(resolved_probability, neutral_band=neutral_band)
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else "neutral"
    raw_gap = float(resolved_probability) - float(raw_probability)
    return {
        "raw_p_up": float(raw_probability),
        "raw_calibrated_probability_gap": float(raw_gap),
        "probability_alignment_gap": float(abs(raw_gap)),
        "raw_p_up_side": raw_side,
        "resolved_p_up_side": resolved_side,
        "ret_pred_side": ret_side,
        "projected_price_side": projected_side,
        "forecast_consensus_side": consensus_side,
        "raw_p_up_ret_mismatch": float(raw_side in {"up", "down"} and ret_side in {"up", "down"} and raw_side != ret_side),
        "p_up_ret_mismatch": float(resolved_side in {"up", "down"} and ret_side in {"up", "down"} and resolved_side != ret_side),
        "raw_p_up_direction_mismatch": float(raw_side in {"up", "down"} and direction_side in {"up", "down"} and raw_side != direction_side),
        "p_up_direction_mismatch": float(resolved_side in {"up", "down"} and direction_side in {"up", "down"} and resolved_side != direction_side),
        "ret_projected_price_consensus": float(consensus_side in {"up", "down"}),
        "probability_calibration_guard_applied": float(bool(isinstance(probability_guard, Mapping) and probability_guard.get("applied"))),
        "probability_calibration_used_regime_key": float(bool(calibration_used_regime_key)),
    }