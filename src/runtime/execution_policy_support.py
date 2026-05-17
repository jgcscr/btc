from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.runtime.forecast_coherence_support import forecast_coherence_excluded
from src.runtime.regime_policy_support import derive_regime_labels_from_frame
from src.runtime.summary_support import coerce_result_horizon, finite_float_or_none


SummaryPayload = Dict[str, Dict[str, Any]]
ExecutionContexts = Mapping[str, Dict[str, Any]]


def finite_float(value: Any, default: float = 0.0) -> float:
    numeric = finite_float_or_none(value)
    return float(default) if numeric is None else float(numeric)


def direction_vote(entry: Mapping[str, Any]) -> str:
    return "up" if str(entry.get("direction_next", "down")).lower() == "up" else "down"


def execution_side(entry: Mapping[str, Any]) -> str:
    return "long" if direction_vote(entry) == "up" else "short"


def lookup_horizon_value(mapping: Mapping[float, float] | None, horizon: float, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        numeric_horizon = float(horizon)
    except (TypeError, ValueError):
        return float(default)
    if numeric_horizon in mapping:
        return float(mapping[numeric_horizon])
    for key, value in mapping.items():
        try:
            if abs(float(key) - numeric_horizon) <= 1e-6:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def dominant_direction_from_scores(up_score: float, down_score: float) -> tuple[str, float]:
    total = max(float(up_score) + float(down_score), 0.0)
    if total <= 0.0:
        return "neutral", 0.0
    if up_score > down_score:
        return "up", float(up_score / total)
    if down_score > up_score:
        return "down", float(down_score / total)
    return "neutral", 0.5


def _direction_from_ret_pred(value: Any) -> str:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric > 0.0:
        return "up"
    if numeric < 0.0:
        return "down"
    return "neutral"


def _direction_from_projected_price(close: Any, projected_price: Any) -> str:
    close_value = finite_float_or_none(close)
    projected_value = finite_float_or_none(projected_price)
    if close_value is None or projected_value is None or close_value <= 0.0 or projected_value <= 0.0:
        return "neutral"
    if projected_value > close_value:
        return "up"
    if projected_value < close_value:
        return "down"
    return "neutral"


def _direction_from_probability(value: Any, *, neutral_band: float = 0.0) -> str:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return "neutral"
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def coherence_weight_multiplier(
    entry: Mapping[str, Any],
    *,
    horizon: float,
    policy: Mapping[str, Any],
) -> float:
    weighting_cfg = policy.get("coherence_weighting") if isinstance(policy.get("coherence_weighting"), Mapping) else {}
    base_multiplier = lookup_horizon_value(
        weighting_cfg.get("by_horizon", {}) if isinstance(weighting_cfg.get("by_horizon"), Mapping) else {},
        horizon,
        1.0,
    )
    base_multiplier = max(float(base_multiplier), 0.0)
    if not bool(weighting_cfg.get("enabled", False)):
        return base_multiplier

    multiplier = base_multiplier
    min_multiplier = max(min(float(weighting_cfg.get("min_multiplier", 0.1) or 0.1), 1.5), 0.0)
    coherence = entry.get("forecast_coherence") if isinstance(entry.get("forecast_coherence"), Mapping) else {}
    low_trust_penalty = max(min(float(weighting_cfg.get("low_trust_penalty", 0.35) or 0.35), 1.0), 0.0)
    blocked_penalty = max(min(float(weighting_cfg.get("blocked_penalty", 1.0) or 1.0), 1.0), 0.0)
    p_up_conflict_penalty = max(min(float(weighting_cfg.get("p_up_conflict_penalty", 0.2) or 0.2), 1.0), 0.0)
    consensus_bonus = max(float(weighting_cfg.get("consensus_bonus", 0.1) or 0.1), 0.0)

    if bool(coherence.get("triggered")):
        multiplier *= max(0.0, 1.0 - blocked_penalty)
    elif bool(coherence.get("low_trust")):
        multiplier *= max(0.0, 1.0 - low_trust_penalty)

    ret_side = str(coherence.get("ret_pred_side") or _direction_from_ret_pred(entry.get("ret_pred")))
    projected_side = str(coherence.get("projected_price_side") or _direction_from_projected_price(entry.get("close"), entry.get("projected_price")))
    p_up_side = str(
        coherence.get("p_up_side")
        or _direction_from_probability(entry.get("p_up"), neutral_band=float(weighting_cfg.get("neutral_band", 0.02) or 0.02))
    )
    consensus_side = ret_side if ret_side == projected_side and ret_side in {"up", "down"} else None
    if consensus_side is not None and p_up_side in {"up", "down"}:
        if p_up_side != consensus_side:
            multiplier *= max(0.0, 1.0 - p_up_conflict_penalty)
        else:
            multiplier *= 1.0 + consensus_bonus

    return max(float(multiplier), min_multiplier)


def compute_weighted_direction_scores(
    labeled_entries: Sequence[tuple[str, Mapping[str, Any], float]],
    *,
    weights: Mapping[float, float] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_weights = weights or {}
    up_score = 0.0
    down_score = 0.0
    details: list[Dict[str, Any]] = []
    for label, entry, horizon in labeled_entries:
        direction = direction_vote(entry)
        if direction not in {"up", "down"}:
            continue
        base_weight = max(lookup_horizon_value(resolved_weights, horizon, 1.0), 0.0)
        confidence = max(float(entry.get("confidence_score") or 0.0), 0.0)
        coherence_multiplier_value = coherence_weight_multiplier(entry, horizon=horizon, policy=policy or {})
        trust_weight = max(float(entry.get("voting_weight_after_trust") or 1.0), 0.0)
        weighted_vote = base_weight * coherence_multiplier_value * trust_weight * (0.5 + 0.5 * min(confidence, 1.0))
        if direction == "up":
            up_score += weighted_vote
        else:
            down_score += weighted_vote
        details.append(
            {
                "label": label,
                "horizon_hours": float(horizon),
                "direction": direction,
                "base_weight": float(base_weight),
                "confidence_score": float(confidence),
                "coherence_multiplier": float(coherence_multiplier_value),
                "trust_weight": float(trust_weight),
                "weighted_vote": float(weighted_vote),
            }
        )
    dominant_direction, dominant_ratio = dominant_direction_from_scores(up_score, down_score)
    return {
        "dominant_direction": dominant_direction,
        "dominant_ratio": float(dominant_ratio),
        "up_score": float(up_score),
        "down_score": float(down_score),
        "total_score": float(up_score + down_score),
        "details": details,
    }


def resolve_execution_upstream_hold_reason(entry: Mapping[str, Any]) -> str:
    trade_decision = entry.get("trade_decision") if isinstance(entry.get("trade_decision"), Mapping) else {}
    if trade_decision.get("confluence_gate_triggered"):
        return "confluence_gate"

    blocking_reason = str(trade_decision.get("blocking_reason") or "").strip()
    if blocking_reason:
        return blocking_reason

    weak_band_veto = trade_decision.get("weak_band_veto") if isinstance(trade_decision.get("weak_band_veto"), Mapping) else {}
    if weak_band_veto.get("triggered"):
        return str(weak_band_veto.get("reason") or "weak_band_veto")

    midband_veto = trade_decision.get("midband_veto") if isinstance(trade_decision.get("midband_veto"), Mapping) else {}
    if midband_veto.get("triggered"):
        return str(midband_veto.get("reason") or "midband_veto")

    abstention = entry.get("abstention") if isinstance(entry.get("abstention"), Mapping) else {}
    if abstention.get("triggered"):
        return str(abstention.get("reason") or "abstention_gate")

    return "upstream_model_hold"


def compute_atr_like_price_distance(
    frame: pd.DataFrame,
    *,
    index: int,
    fallback_close: float,
    fallback_return_std: float,
    min_residual_std: float,
    window: int = 14,
) -> float:
    start = max(0, index - max(window, 2) + 1)
    history = frame.iloc[start : index + 1].copy()
    if {"high", "low", "close"}.issubset(history.columns):
        high = pd.to_numeric(history["high"], errors="coerce")
        low = pd.to_numeric(history["low"], errors="coerce")
        close = pd.to_numeric(history["close"], errors="coerce")
        valid_close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_close.empty:
            anchor = float(valid_close.tail(window).median())
            if anchor > 0.0 and fallback_close > 0.0:
                deviation = abs(anchor / fallback_close - 1.0)
                if deviation > 0.5:
                    return max(float(fallback_close) * max(abs(float(fallback_return_std)), min_residual_std), 1e-8)
            elif anchor <= 0.0 and fallback_close > 0.0:
                return max(float(fallback_close) * max(abs(float(fallback_return_std)), min_residual_std), 1e-8)
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1, skipna=True)
        atr = pd.to_numeric(true_range, errors="coerce").tail(window).mean()
        if pd.notna(atr) and float(atr) > 0.0:
            return float(atr)
    return max(float(fallback_close) * max(abs(float(fallback_return_std)), min_residual_std), 1e-8)


def compute_recent_structure(
    frame: pd.DataFrame,
    *,
    index: int,
    session_lookback_bars: int,
    swing_lookback_bars: int,
    atr_distance: float,
    fallback_price: float,
) -> Dict[str, float]:
    start_session = max(0, index - max(session_lookback_bars, 2) + 1)
    start_swing = max(0, index - max(swing_lookback_bars, 2) + 1)
    session_frame = frame.iloc[start_session : index + 1].copy()
    swing_frame = frame.iloc[start_swing : index + 1].copy()

    def _safe_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column not in df.columns:
            return pd.Series([default], dtype=float)
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return pd.Series([default], dtype=float)
        return series.astype(float)

    high_session = float(_safe_series(session_frame, "high", fallback_price).max())
    low_session = float(_safe_series(session_frame, "low", fallback_price).min())
    swing_high = float(_safe_series(swing_frame, "high", fallback_price).max())
    swing_low = float(_safe_series(swing_frame, "low", fallback_price).min())
    close_series = _safe_series(session_frame, "close", fallback_price)
    volume_series = _safe_series(session_frame, "volume", 0.0)
    if float(volume_series.sum()) > 0.0 and len(close_series) == len(volume_series):
        vwap = float((close_series * volume_series).sum() / volume_series.sum())
    else:
        vwap = float(close_series.iloc[-1]) if not close_series.empty else float(fallback_price)

    if fallback_price > 0.0:
        structure_values = (high_session, low_session, swing_high, swing_low, vwap)
        invalid_structure = any(value <= 0.0 for value in structure_values)
        if not invalid_structure:
            invalid_structure = any(abs(value / fallback_price - 1.0) > 0.5 for value in structure_values)
        if invalid_structure:
            high_session = float(fallback_price + atr_distance)
            low_session = float(max(fallback_price - atr_distance, 1e-8))
            swing_high = float(fallback_price + atr_distance * 1.5)
            swing_low = float(max(fallback_price - atr_distance * 1.5, 1e-8))
            vwap = float(fallback_price)

    return {
        "session_high": high_session,
        "session_low": low_session,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "vwap": vwap,
        "atr_distance": float(max(atr_distance, 1e-8)),
    }


def compute_excursion_priors(
    frame: pd.DataFrame,
    *,
    index: int,
    horizon_steps: int,
    side: str,
    lookback_bars: int,
    min_samples: int,
    mae_quantile: float,
    mfe_quantile: float,
    current_regime: str | None = None,
    current_volatility: float | None = None,
    bucket_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "sample_count": 0,
        "mae_distance": None,
        "mfe_distance": None,
        "peak_step_p50": None,
        "adverse_step_p50": None,
        "source": "global",
        "matched_regime": None,
        "volatility_bucket": None,
        "bucket_threshold": None,
    }
    if horizon_steps <= 0 or index <= horizon_steps or not {"high", "low", "close"}.issubset(frame.columns):
        return result

    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    end = max(index - horizon_steps, 0)
    start = max(0, end - max(lookback_bars, min_samples))
    selected_indices = list(range(start, end))

    normalized_regime = str(current_regime or "").strip().lower() or None
    if bucket_policy and bool(bucket_policy.get("enabled", False)) and start < end:
        regime_col = str(bucket_policy.get("regime_col") or "regime_state")
        volatility_col = str(bucket_policy.get("volatility_col") or "volatility_realized_24h")
        min_bucket_samples = max(int(bucket_policy.get("min_bucket_samples") or min_samples), 1)
        low_vol_quantile = max(min(float(bucket_policy.get("low_vol_quantile") or 0.5), 0.95), 0.05)
        breakout_score_threshold = float(bucket_policy.get("breakout_score_threshold") or 0.8)
        chop_score_threshold = float(bucket_policy.get("chop_score_threshold") or 0.3)

        regime_matches: list[int] = []
        bucket_matches: list[int] = []
        regime_bucket_matches: list[int] = []
        regime_match_used = False

        if regime_col in frame.columns:
            regime_series = frame[regime_col].iloc[start:end].fillna("").astype(str).str.strip().str.lower()
            if normalized_regime is not None:
                regime_matches = [start + offset for offset, value in enumerate(regime_series) if value == normalized_regime]
                regime_match_used = bool(regime_matches)
        elif normalized_regime is not None:
            derived_regimes = derive_regime_labels_from_frame(
                frame.iloc[start:end].copy(),
                volatility_col=volatility_col,
                breakout_score_threshold=breakout_score_threshold,
                chop_score_threshold=chop_score_threshold,
                breakout_vol_normalizer=0.01,
                breakout_ret_normalizer=0.01,
                regime_trend="trend_ignition",
                regime_neutral="neutral",
                regime_chop="chop",
            )
            regime_matches = [start + offset for offset, value in enumerate(derived_regimes.astype(str).str.lower()) if value == normalized_regime]
            regime_match_used = bool(regime_matches)

        if volatility_col in frame.columns and current_volatility is not None and math.isfinite(float(current_volatility)):
            volatility_history = pd.to_numeric(frame[volatility_col].iloc[start:end], errors="coerce")
            valid_history = volatility_history.dropna()
            if not valid_history.empty:
                bucket_threshold = float(valid_history.quantile(low_vol_quantile))
                current_bucket = "low_vol" if float(current_volatility) <= bucket_threshold else "high_vol"
                bucket_matches = [
                    start + offset
                    for offset, value in enumerate(volatility_history)
                    if pd.notna(value)
                    and ((current_bucket == "low_vol" and float(value) <= bucket_threshold) or (current_bucket == "high_vol" and float(value) > bucket_threshold))
                ]
                result["volatility_bucket"] = current_bucket
                result["bucket_threshold"] = bucket_threshold

        if regime_matches and bucket_matches:
            regime_bucket_matches = sorted(set(regime_matches).intersection(bucket_matches))

        if len(regime_bucket_matches) >= min_bucket_samples:
            selected_indices = regime_bucket_matches
            result["source"] = "regime_volatility_bucket"
        elif len(regime_matches) >= min_bucket_samples:
            selected_indices = regime_matches
            result["source"] = "regime_bucket"
        elif len(bucket_matches) >= min_bucket_samples:
            selected_indices = bucket_matches
            result["source"] = "volatility_bucket"

        if result["source"] in {"regime_bucket", "regime_volatility_bucket"} and regime_match_used:
            result["matched_regime"] = normalized_regime

    maes: list[float] = []
    mfes: list[float] = []
    peak_steps: list[int] = []
    adverse_steps: list[int] = []
    for cursor in selected_indices:
        entry = close[cursor]
        if not math.isfinite(entry) or entry <= 0.0:
            continue
        future_high = high[cursor + 1 : cursor + 1 + horizon_steps]
        future_low = low[cursor + 1 : cursor + 1 + horizon_steps]
        if future_high.size == 0 or future_low.size == 0:
            continue
        if side == "long":
            favorable_idx = int(np.nanargmax(future_high))
            adverse_idx = int(np.nanargmin(future_low))
            favorable = max(float(future_high[favorable_idx]) / entry - 1.0, 0.0)
            adverse = max(1.0 - float(future_low[adverse_idx]) / entry, 0.0)
        else:
            favorable_idx = int(np.nanargmin(future_low))
            adverse_idx = int(np.nanargmax(future_high))
            favorable = max(entry / float(future_low[favorable_idx]) - 1.0, 0.0)
            adverse = max(float(future_high[adverse_idx]) / entry - 1.0, 0.0)
        if not math.isfinite(favorable) or not math.isfinite(adverse):
            continue
        mfes.append(favorable)
        maes.append(adverse)
        peak_steps.append(favorable_idx + 1)
        adverse_steps.append(adverse_idx + 1)

    if len(maes) < min_samples or len(mfes) < min_samples:
        result["sample_count"] = len(maes)
        return result

    result.update(
        {
            "available": True,
            "sample_count": len(maes),
            "mae_distance": float(np.quantile(np.asarray(maes, dtype=float), mae_quantile)),
            "mfe_distance": float(np.quantile(np.asarray(mfes, dtype=float), mfe_quantile)),
            "peak_step_p50": int(round(float(np.quantile(np.asarray(peak_steps, dtype=float), 0.5)))),
            "adverse_step_p50": int(round(float(np.quantile(np.asarray(adverse_steps, dtype=float), 0.5)))),
        }
    )
    return result


def summarize_bias_context(summary: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Any]:
    bias_horizons = set(policy.get("bias_horizons", []))
    execution_horizons = set(policy.get("execution_horizons", []))
    short_term_horizons = set(policy.get("short_term_strict_horizons", []))
    weights = policy.get("horizon_bias_weights") if isinstance(policy.get("horizon_bias_weights"), Mapping) else {}
    bias_entries: list[tuple[str, Mapping[str, Any], float]] = []
    execution_entries: list[tuple[str, Mapping[str, Any], float]] = []
    short_entries: list[tuple[str, Mapping[str, Any], float]] = []
    mid_entries: list[tuple[str, Mapping[str, Any], float]] = []
    for label, entry in summary.items():
        if forecast_coherence_excluded(entry) or bool(entry.get("excluded_from_voting", False)):
            continue
        horizon = coerce_result_horizon(entry.get("horizon_hours"))
        if horizon is None:
            continue
        if horizon in bias_horizons:
            bias_entries.append((label, entry, horizon))
            mid_entries.append((label, entry, horizon))
        if horizon in execution_horizons:
            execution_entries.append((label, entry, horizon))
        if horizon in short_term_horizons:
            short_entries.append((label, entry, horizon))

    bias_scores = compute_weighted_direction_scores(bias_entries, weights=weights, policy=policy)
    execution_scores = compute_weighted_direction_scores(execution_entries, weights=weights, policy=policy)
    short_term_scores = compute_weighted_direction_scores(short_entries, weights=weights, policy=policy)
    mid_term_scores = compute_weighted_direction_scores(mid_entries, weights=weights, policy=policy)

    bias_direction = str(bias_scores.get("dominant_direction", "neutral"))
    bias_alignment_ratio = float(bias_scores.get("dominant_ratio", 0.0))
    min_bias_alignment_ratio = max(min(float(policy.get("min_bias_alignment_ratio", 0.0) or 0.0), 1.0), 0.0)
    if bias_direction != "neutral" and bias_alignment_ratio < min_bias_alignment_ratio:
        bias_direction = "neutral"

    direction_support_horizons: Dict[str, list[str]] = {"up": [], "down": []}
    for label, entry, _horizon in bias_entries:
        direction = direction_vote(entry)
        if direction in direction_support_horizons:
            direction_support_horizons[direction].append(label)

    return {
        "bias_direction": bias_direction,
        "bias_direction_pre_threshold": str(bias_scores.get("dominant_direction", "neutral")),
        "bias_alignment_ratio": bias_alignment_ratio,
        "bias_scores": bias_scores,
        "execution_scores": execution_scores,
        "short_term_scores": short_term_scores,
        "mid_term_scores": mid_term_scores,
        "short_term_direction": str(short_term_scores.get("dominant_direction", "neutral")),
        "short_term_alignment_ratio": float(short_term_scores.get("dominant_ratio", 0.0)),
        "mid_term_direction": str(mid_term_scores.get("dominant_direction", "neutral")),
        "mid_term_alignment_ratio": float(mid_term_scores.get("dominant_ratio", 0.0)),
        "min_bias_alignment_ratio": float(min_bias_alignment_ratio),
        "direction_support_horizons": direction_support_horizons,
        "execution_entries": execution_entries,
    }


def execution_alignment_ratio(
    execution_entries: Sequence[tuple[str, Mapping[str, Any], float]],
    *,
    direction: str,
    weights: Mapping[float, float] | None = None,
) -> float:
    if not execution_entries:
        return 0.0
    score_payload = compute_weighted_direction_scores(execution_entries, weights=weights)
    total = float(score_payload.get("total_score", 0.0) or 0.0)
    if total <= 0.0:
        return 0.0
    if direction == "up":
        return float(score_payload.get("up_score", 0.0) or 0.0) / total
    if direction == "down":
        return float(score_payload.get("down_score", 0.0) or 0.0) / total
    return 0.0


def classify_execution_tier(
    entry: Mapping[str, Any],
    *,
    bias_direction: str,
    execution_alignment_ratio: float,
    policy: Mapping[str, Any],
) -> str:
    direction = direction_vote(entry)
    horizon = coerce_result_horizon(entry.get("horizon_hours")) or 0.0
    support_ratio = float(entry.get("confluence_support_ratio") or 0.0)
    mid_ratio = float(entry.get("confluence_mid_term_ratio") or 0.0)
    if bias_direction != "neutral" and direction != bias_direction:
        return "low"
    if horizon in set(policy.get("short_term_strict_horizons", [])):
        strict_mid_ratio = lookup_horizon_value(
            policy.get("short_term_min_mid_ratio_by_horizon", {}),
            horizon,
            float(policy.get("short_term_min_mid_ratio", 0.67)),
        )
        strict_support_ratio = lookup_horizon_value(
            policy.get("short_term_min_support_ratio_by_horizon", {}),
            horizon,
            float(policy.get("short_term_min_support_ratio", 0.75)),
        )
        if support_ratio < strict_support_ratio or mid_ratio < strict_mid_ratio:
            return "low"
    if support_ratio >= float(policy.get("immediate_entry_min_support_ratio", 0.8)) and mid_ratio >= float(policy.get("immediate_entry_min_mid_ratio", 0.67)) and execution_alignment_ratio >= float(policy.get("high_execution_alignment_ratio", 1.0)):
        return "high"
    if support_ratio >= float(policy.get("pullback_entry_min_support_ratio", 0.6)) and mid_ratio >= float(policy.get("pullback_entry_min_mid_ratio", 0.5)) and execution_alignment_ratio >= float(policy.get("medium_execution_alignment_ratio", 0.5)):
        return "medium"
    return "low"


def build_entry_zone(
    *,
    market_price: float,
    side: str,
    structure: Mapping[str, float],
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any] | None = None,
) -> Dict[str, float | bool | str]:
    atr_distance = float(structure.get("atr_distance", 0.0))
    template_zone_mult = float((regime_template or {}).get("entry_zone_atr_mult") or 0.0)
    entry_zone_width = atr_distance * (template_zone_mult if template_zone_mult > 0.0 else float(policy.get("entry_zone_atr_mult", 0.25)))
    session_high = float(structure.get("session_high", market_price))
    session_low = float(structure.get("session_low", market_price))
    range_size = max(session_high - session_low, atr_distance)
    vwap = float(structure.get("vwap", market_price))
    if side == "long":
        preferred = min(market_price, max(vwap, session_low + range_size * 0.382))
    else:
        preferred = max(market_price, min(vwap, session_high - range_size * 0.382))
    zone_low = preferred - entry_zone_width
    zone_high = preferred + entry_zone_width
    return {
        "preferred_entry_price": float(preferred),
        "entry_zone_low": float(zone_low),
        "entry_zone_high": float(zone_high),
        "entry_ready": bool(zone_low <= market_price <= zone_high),
        "vwap_reference": vwap,
    }


def _compute_recent_candle_expansion(frame: pd.DataFrame, *, index: int, window: int) -> float:
    if frame.empty:
        return 1.0
    start = max(0, index - max(window, 2) + 1)
    history = frame.iloc[start : index + 1].copy()
    if history.empty:
        return 1.0
    if {"high", "low"}.issubset(history.columns):
        ranges = (pd.to_numeric(history["high"], errors="coerce") - pd.to_numeric(history["low"], errors="coerce")).abs()
    else:
        closes = pd.to_numeric(history.get("close"), errors="coerce")
        ranges = closes.diff().abs()
    clean = ranges.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 1.0
    latest = float(clean.iloc[-1])
    baseline = float(clean.iloc[:-1].median()) if clean.size > 1 else float(clean.median())
    if baseline <= 0.0:
        return 1.0
    return float(latest / baseline)


def compute_pullback_quality_score(
    *,
    entry: Mapping[str, Any],
    frame: pd.DataFrame,
    index: int,
    market_price: float,
    side: str,
    structure: Mapping[str, float],
    atr_distance: float,
    horizon: float,
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any],
) -> Dict[str, Any]:
    pullback_cfg = policy.get("pullback_quality") if isinstance(policy.get("pullback_quality"), Mapping) else {}
    if not pullback_cfg.get("enabled"):
        return {"enabled": False, "score": 1.0, "min_score": 0.0, "triggered": False, "vwap_deviation_atr": 0.0, "range_expansion_1h": finite_float(entry.get("range_expansion_1h"), 0.0), "candle_expansion_ratio": 1.0}

    vwap = float(structure.get("vwap", market_price))
    safe_atr = max(float(atr_distance), 1e-8)
    vwap_deviation_atr = abs(market_price - vwap) / safe_atr
    max_vwap_deviation = float(pullback_cfg.get("max_vwap_deviation_atr", 1.5))
    vwap_score = max(0.0, 1.0 - (vwap_deviation_atr / max(max_vwap_deviation, 1e-8)))
    range_expansion = abs(finite_float(entry.get("range_expansion_1h"), 0.0))
    range_threshold = float(pullback_cfg.get("range_expansion_penalty_threshold", 1.25))
    range_score = 1.0 if range_expansion <= range_threshold else max(0.0, 1.0 - min(range_expansion - range_threshold, 1.0))
    candle_expansion_ratio = _compute_recent_candle_expansion(frame, index=index, window=int(pullback_cfg.get("candle_expansion_window", 8)))
    max_candle_expansion = float(pullback_cfg.get("max_candle_expansion_ratio", 2.0))
    candle_score = 1.0 if candle_expansion_ratio <= 1.0 else max(0.0, 1.0 - ((candle_expansion_ratio - 1.0) / max(max_candle_expansion - 1.0, 1e-8)))
    momentum_penalty = 0.0
    if side == "long" and finite_float(entry.get("momentum_slope_2h"), 0.0) < 0.0:
        momentum_penalty = min(abs(finite_float(entry.get("momentum_slope_2h"), 0.0)) * 10.0, 0.15)
    if side == "short" and finite_float(entry.get("momentum_slope_2h"), 0.0) > 0.0:
        momentum_penalty = min(abs(finite_float(entry.get("momentum_slope_2h"), 0.0)) * 10.0, 0.15)
    score = max(0.0, min(1.0, 0.45 * vwap_score + 0.30 * range_score + 0.25 * candle_score - momentum_penalty))
    min_score = lookup_horizon_value(pullback_cfg.get("min_score_by_horizon", {}), horizon, max(float(regime_template.get("pullback_quality_floor", 0.0) or 0.0), 0.0))
    min_score = max(min_score, float(regime_template.get("pullback_quality_floor", 0.0) or 0.0))
    return {
        "enabled": True,
        "score": float(score),
        "min_score": float(min_score),
        "triggered": bool(score < min_score),
        "vwap_deviation_atr": float(vwap_deviation_atr),
        "range_expansion_1h": float(range_expansion),
        "candle_expansion_ratio": float(candle_expansion_ratio),
    }


def compute_disagreement_severity(
    entry: Mapping[str, Any],
    *,
    bias_context: Mapping[str, Any],
    policy: Mapping[str, Any],
    atr_distance: float,
    structure: Mapping[str, float],
) -> Dict[str, Any]:
    disagreement_cfg = policy.get("disagreement_severity") if isinstance(policy.get("disagreement_severity"), Mapping) else {}
    if not disagreement_cfg.get("enabled", True):
        return {"enabled": False, "score": 0.0, "triggered": False, "pullback_only": False, "reasons": []}
    direction = direction_vote(entry)
    short_direction = str(bias_context.get("short_term_direction", "neutral"))
    mid_direction = str(bias_context.get("mid_term_direction", "neutral"))
    short_ratio = float(bias_context.get("short_term_alignment_ratio", 0.0) or 0.0)
    mid_ratio = float(bias_context.get("mid_term_alignment_ratio", 0.0) or 0.0)
    score = 0.0
    reasons: list[str] = []
    if short_direction in {"up", "down"} and mid_direction in {"up", "down"} and short_direction != mid_direction:
        score += 0.5
        reasons.append("short_mid_direction_conflict")
    if mid_direction in {"up", "down"} and direction == mid_direction and short_direction not in {"neutral", mid_direction}:
        score += 0.15
        reasons.append("short_term_countertrend")
    alignment_gap = abs(mid_ratio - short_ratio)
    if alignment_gap > 0.1:
        score += min(alignment_gap, 0.2)
        reasons.append("alignment_gap")
    vwap = float(structure.get("vwap", finite_float(entry.get("close"), 0.0)))
    if atr_distance > 0.0:
        vwap_deviation_atr = abs(finite_float(entry.get("close"), 0.0) - vwap) / max(atr_distance, 1e-8)
        if vwap_deviation_atr >= float(disagreement_cfg.get("vwap_extension_penalty_atr", 0.75)):
            score += 0.1
            reasons.append("vwap_extension")
    range_expansion = abs(finite_float(entry.get("range_expansion_1h"), 0.0))
    if range_expansion >= float(disagreement_cfg.get("range_expansion_penalty_threshold", 1.0)):
        score += 0.1
        reasons.append("range_expansion")
    score = max(0.0, min(1.0, score))
    block_threshold = float(disagreement_cfg.get("block_threshold", 0.7))
    pullback_threshold = float(disagreement_cfg.get("pullback_threshold", 0.45))
    return {
        "enabled": True,
        "score": float(score),
        "triggered": bool(score >= block_threshold),
        "pullback_only": bool(score >= pullback_threshold and score < block_threshold),
        "reasons": reasons,
        "short_term_direction": short_direction,
        "mid_term_direction": mid_direction,
        "short_term_alignment_ratio": float(short_ratio),
        "mid_term_alignment_ratio": float(mid_ratio),
    }


def resolve_stop_with_guardrails(
    *,
    side: str,
    planned_entry: float,
    existing_stop: float,
    structure_stop: float,
    analytic_stop: float | None,
    atr_distance: float,
    guards_cfg: Mapping[str, Any],
    analytic_stop_preferred: bool = False,
) -> Dict[str, Any]:
    def _valid_stop(stop_value: float | None) -> bool:
        if stop_value is None or not math.isfinite(float(stop_value)):
            return False
        numeric_stop = float(stop_value)
        return numeric_stop < planned_entry if side == "long" else numeric_stop > planned_entry

    def _distance(stop_value: float) -> float:
        return planned_entry - stop_value if side == "long" else stop_value - planned_entry

    candidates: list[Dict[str, Any]] = []
    for source_name, stop_value in (("existing", existing_stop), ("structure", structure_stop), ("analytics", analytic_stop)):
        if _valid_stop(stop_value):
            numeric_stop = float(stop_value)
            candidates.append({"source": source_name, "stop_loss": numeric_stop, "risk_unit": _distance(numeric_stop)})
    if not candidates:
        fallback_risk = max(atr_distance * 0.5, 1e-8)
        fallback_stop = planned_entry - fallback_risk if side == "long" else planned_entry + fallback_risk
        return {"stop_loss": fallback_stop, "risk_unit": fallback_risk, "source": "atr_fallback", "adjustment": {"applied": True, "type": "atr_fallback", "reason": "no_valid_stop_candidates", "risk_unit_before": None, "risk_unit_after": fallback_risk}}
    if analytic_stop_preferred:
        priority = {"analytics": 0, "structure": 1, "existing": 2}
        selected = min(candidates, key=lambda item: (priority.get(str(item.get("source")), 99), abs(float(item["risk_unit"]) - atr_distance)))
    else:
        selected = max(candidates, key=lambda item: float(item["risk_unit"]))
    selected_stop = float(selected["stop_loss"])
    risk_unit = float(selected["risk_unit"])
    adjustment: Dict[str, Any] | None = None
    if guards_cfg.get("enabled"):
        min_stop = float(guards_cfg.get("min_stop_distance_atr_mult", 0.35)) * atr_distance
        max_stop = float(guards_cfg.get("max_stop_distance_atr_mult", 3.0)) * atr_distance
        if min_stop > 0.0 and risk_unit < min_stop:
            adjusted_stop = planned_entry - min_stop if side == "long" else planned_entry + min_stop
            adjustment = {"applied": True, "type": "expanded_to_min_stop_distance", "reason": "stop_too_tight_near_invalidation", "from_source": str(selected["source"]), "risk_unit_before": risk_unit, "risk_unit_after": min_stop}
            selected_stop = float(adjusted_stop)
            risk_unit = float(min_stop)
        elif max_stop > 0.0 and risk_unit > max_stop:
            within_band = [item for item in candidates if float(item["risk_unit"]) <= max_stop]
            if within_band:
                replacement = max(within_band, key=lambda item: float(item["risk_unit"]))
                adjustment = {"applied": True, "type": "replaced_with_guardrail_candidate", "reason": "stop_too_wide", "from_source": str(selected["source"]), "to_source": str(replacement["source"]), "risk_unit_before": risk_unit, "risk_unit_after": float(replacement["risk_unit"])}
                selected = replacement
                selected_stop = float(replacement["stop_loss"])
                risk_unit = float(replacement["risk_unit"])
            else:
                adjusted_stop = planned_entry - max_stop if side == "long" else planned_entry + max_stop
                adjustment = {"applied": True, "type": "capped_to_max_stop_distance", "reason": "stop_too_wide", "from_source": str(selected["source"]), "risk_unit_before": risk_unit, "risk_unit_after": max_stop}
                selected_stop = float(adjusted_stop)
                risk_unit = float(max_stop)
    return {"stop_loss": float(selected_stop), "risk_unit": float(max(risk_unit, 1e-8)), "source": str(selected.get("source", "unknown")), "adjustment": adjustment}


def refine_stop_with_target_range(
    *,
    side: str,
    planned_entry: float,
    selected_stop: float,
    risk_unit: float,
    atr_distance: float,
    horizon: float,
    projected_high: float | None,
    projected_low: float | None,
    projected_high_confidence: float | None,
    projected_low_confidence: float | None,
    projected_high_residual_std: float | None,
    projected_low_residual_std: float | None,
    policy: Mapping[str, Any],
    guards_cfg: Mapping[str, Any],
    normalize_horizon_value: Callable[[Any], float],
    default_confidence_min: float,
    default_buffer_std_mult: float,
    default_min_tighten_fraction: float,
) -> Dict[str, Any]:
    refinement_cfg = policy.get("target_range_stop_refinement") if isinstance(policy.get("target_range_stop_refinement"), Mapping) else {}
    if not refinement_cfg.get("enabled"):
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    scoped_horizons = set(refinement_cfg.get("horizons", []))
    if scoped_horizons and normalize_horizon_value(horizon) not in scoped_horizons:
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    if side == "long":
        projected_adverse = finite_float_or_none(projected_low)
        confidence = finite_float_or_none(projected_low_confidence)
        residual_std = finite_float_or_none(projected_low_residual_std)
        projection_field = "projected_low"
        tighten_only = projected_adverse is not None and projected_adverse > selected_stop and projected_adverse < planned_entry
    else:
        projected_adverse = finite_float_or_none(projected_high)
        confidence = finite_float_or_none(projected_high_confidence)
        residual_std = finite_float_or_none(projected_high_residual_std)
        projection_field = "projected_high"
        tighten_only = projected_adverse is not None and projected_adverse < selected_stop and projected_adverse > planned_entry
    if not tighten_only or confidence is None:
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    confidence_min = float(refinement_cfg.get("confidence_min", default_confidence_min))
    if confidence < confidence_min:
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    residual_std_value = max(float(residual_std or 0.0), 0.0)
    uncertainty_buffer = max(planned_entry * residual_std_value * float(refinement_cfg.get("buffer_std_mult", default_buffer_std_mult)), atr_distance * 0.1, 1e-8)
    if side == "long":
        candidate_stop = min(float(projected_adverse) - uncertainty_buffer, planned_entry - 1e-8)
        candidate_risk = planned_entry - candidate_stop
    else:
        candidate_stop = max(float(projected_adverse) + uncertainty_buffer, planned_entry + 1e-8)
        candidate_risk = candidate_stop - planned_entry
    min_stop = float(guards_cfg.get("min_stop_distance_atr_mult", 0.35)) * atr_distance if guards_cfg.get("enabled") else 0.0
    candidate_risk = max(candidate_risk, min_stop, 1e-8)
    candidate_stop = planned_entry - candidate_risk if side == "long" else planned_entry + candidate_risk
    if candidate_risk >= risk_unit:
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    tighten_fraction = (float(risk_unit) - float(candidate_risk)) / max(float(risk_unit), 1e-8)
    min_tighten_fraction = float(refinement_cfg.get("min_tighten_fraction", default_min_tighten_fraction))
    if tighten_fraction < min_tighten_fraction:
        return {"applied": False, "stop_loss": float(selected_stop), "risk_unit": float(risk_unit), "details": None}
    return {"applied": True, "stop_loss": float(candidate_stop), "risk_unit": float(candidate_risk), "details": {"applied": True, "type": "target_range_stop_tightened", "projection_field": projection_field, "projected_level": float(projected_adverse), "confidence": float(confidence), "confidence_min": float(confidence_min), "uncertainty_buffer": float(uncertainty_buffer), "risk_unit_before": float(risk_unit), "risk_unit_after": float(candidate_risk), "tighten_fraction": float(tighten_fraction)}}


def resolve_execution_target_reward(
    *,
    side: str,
    planned_entry: float,
    existing_take: float,
    projected_high: float | None,
    projected_low: float | None,
    analytics_payload: Mapping[str, Any],
    risk_unit: float,
    horizon: float,
    policy: Mapping[str, Any],
    regime_template: Mapping[str, Any],
    regime_state: str,
) -> Dict[str, Any]:
    rr_floor = lookup_horizon_value(policy.get("minimum_rr_by_horizon", {}), horizon, 1.0)
    rr_floor *= float(regime_template.get("tp_multiplier", 1.0) or 1.0)
    dynamic_rr_cfg = policy.get("dynamic_rr_floor") if isinstance(policy.get("dynamic_rr_floor"), Mapping) else {}
    dynamic_rr_applied = False
    dynamic_rr_ratio = None
    if bool(dynamic_rr_cfg.get("enabled", False)) and bool(analytics_payload.get("available")):
        sample_count = int(analytics_payload.get("sample_count") or 0)
        min_samples = max(int(dynamic_rr_cfg.get("min_samples", 40) or 40), 1)
        mae_distance = finite_float_or_none(analytics_payload.get("mae_distance"))
        mfe_distance = finite_float_or_none(analytics_payload.get("mfe_distance"))
        if sample_count >= min_samples and mae_distance is not None and mfe_distance is not None and mae_distance > 0.0:
            realized_ratio = max(mfe_distance / max(mae_distance, 1e-8), 0.0)
            regime_multiplier = 1.0
            regime_map = dynamic_rr_cfg.get("regime_multiplier") if isinstance(dynamic_rr_cfg.get("regime_multiplier"), Mapping) else {}
            if regime_map:
                regime_multiplier = max(float(regime_map.get(str(regime_state).strip().lower(), 1.0) or 1.0), 0.0)
            scaled_ratio = realized_ratio * max(float(dynamic_rr_cfg.get("mfe_mae_scale", 0.9) or 0.9), 0.0) * regime_multiplier
            max_adjustment = max(min(float(dynamic_rr_cfg.get("max_adjustment", 0.35) or 0.35), 1.0), 0.0)
            floor_reduction = rr_floor * max_adjustment
            bounded_floor = max(rr_floor - floor_reduction, scaled_ratio)
            min_floor = lookup_horizon_value(dynamic_rr_cfg.get("min_floor_by_horizon", {}), horizon, float(dynamic_rr_cfg.get("default_floor", 0.0) or 0.0))
            max_floor = lookup_horizon_value(dynamic_rr_cfg.get("max_floor_by_horizon", {}), horizon, rr_floor)
            rr_floor = max(min(bounded_floor, max_floor), min_floor)
            dynamic_rr_applied = True
            dynamic_rr_ratio = realized_ratio
    effective_rr_floor = rr_floor
    existing_reward = abs(existing_take - planned_entry)
    projection_reward = 0.0
    if side == "long" and projected_high is not None:
        projection_reward = max(projected_high - planned_entry, 0.0)
    elif side == "short" and projected_low is not None:
        projection_reward = max(planned_entry - projected_low, 0.0)
    analytics_available = bool(analytics_payload.get("available"))
    analytic_mfe_reward = planned_entry * float(analytics_payload.get("mfe_distance") or 0.0)
    if analytics_available and analytic_mfe_reward > 0.0:
        projection_cap_ratio = float(((policy.get("analytics", {}).get("regime_volatility_buckets") or {}).get("max_projection_mfe_ratio") or 1.25))
        projection_reward = min(projection_reward, analytic_mfe_reward * projection_cap_ratio) if projection_reward > 0.0 else 0.0
    min_reward = rr_floor * risk_unit
    adapted = False
    status = "pass"
    reason = "pass"
    if analytics_available and analytic_mfe_reward > 0.0:
        feasible_reward = max(analytic_mfe_reward, projection_reward)
        if feasible_reward < min_reward:
            adaptive_cfg = policy.get("adaptive_take_profit", {}) if isinstance(policy.get("adaptive_take_profit"), Mapping) else {}
            adaptive_rr_floor = rr_floor * float(adaptive_cfg.get("min_rr_fraction_of_floor", 1.0) or 1.0)
            feasible_rr = feasible_reward / max(risk_unit, 1e-8)
            if bool(adaptive_cfg.get("enabled", False)) and feasible_rr >= adaptive_rr_floor:
                min_reward = feasible_reward
                adapted = True
                status = "pass"
            else:
                status = "blocked"
                reason = "insufficient_mfe_headroom"
    selected_reward = max(existing_reward, projection_reward, min_reward)
    if side == "long":
        selected_take = planned_entry + selected_reward
    else:
        selected_take = planned_entry - selected_reward
    risk_reward_ratio = selected_reward / max(risk_unit, 1e-8)
    return {
        "status": status,
        "reason": reason,
        "selected_take": float(selected_take),
        "risk_reward_ratio": float(risk_reward_ratio),
        "target_management": {
            "source": "analytics_mfe" if analytics_available and analytic_mfe_reward > 0.0 else ("projection" if projection_reward > 0.0 else "existing_take"),
            "adapted_to_mfe_headroom": bool(adapted),
            "analytics_available": analytics_available,
            "original_rr_floor": float(lookup_horizon_value(policy.get("minimum_rr_by_horizon", {}), horizon, 1.0) * float(regime_template.get("tp_multiplier", 1.0) or 1.0)),
            "effective_rr_floor": float(effective_rr_floor),
            "analytic_mfe_reward": float(analytic_mfe_reward),
            "projection_reward": float(projection_reward),
            "selected_reward": float(selected_reward),
            "dynamic_rr_floor_applied": bool(dynamic_rr_applied),
            "dynamic_realized_rr_ratio": None if dynamic_rr_ratio is None else float(dynamic_rr_ratio),
        },
    }


def resolve_execution_policy(
    config: Mapping[str, Any] | None,
    *,
    normalize_horizon_value: Callable[[Any], float],
    coerce_numeric_horizon: Callable[[Any], float | None],
    default_lookback_bars: int,
    default_min_samples: int,
    default_target_range_stop_horizons: Sequence[float],
    default_target_range_stop_confidence_min: float,
    default_target_range_stop_buffer_std_mult: float,
    default_target_range_stop_min_tighten_fraction: float,
) -> Dict[str, Any]:
    cfg = config or {}

    def normalize_float_map(raw: Any, *, minimum: float = 0.0) -> Dict[float, float]:
        if not isinstance(raw, Mapping):
            return {}
        resolved: Dict[float, float] = {}
        for key, value in raw.items():
            horizon = coerce_numeric_horizon(key)
            if horizon is None:
                continue
            try:
                resolved[horizon] = max(float(value), minimum)
            except (TypeError, ValueError):
                continue
        return resolved

    partial_cfg = cfg.get("partial_take_profit") if isinstance(cfg.get("partial_take_profit"), Mapping) else {}
    trailing_cfg = cfg.get("trailing_stop") if isinstance(cfg.get("trailing_stop"), Mapping) else {}
    analytics_cfg = cfg.get("analytics") if isinstance(cfg.get("analytics"), Mapping) else {}
    analytics_bucket_cfg = (
        analytics_cfg.get("regime_volatility_buckets")
        if isinstance(analytics_cfg.get("regime_volatility_buckets"), Mapping)
        else {}
    )
    guards_cfg = cfg.get("no_trade_guards") if isinstance(cfg.get("no_trade_guards"), Mapping) else {}
    adaptive_tp_cfg = cfg.get("adaptive_take_profit") if isinstance(cfg.get("adaptive_take_profit"), Mapping) else {}
    target_range_stop_cfg = (
        cfg.get("target_range_stop_refinement")
        if isinstance(cfg.get("target_range_stop_refinement"), Mapping)
        else {}
    )
    raw_regime_templates = cfg.get("regime_templates") if isinstance(cfg.get("regime_templates"), Mapping) else {}
    regime_templates: Dict[str, Dict[str, Any]] = {}
    for regime_name, raw_template in raw_regime_templates.items():
        if not isinstance(raw_template, Mapping):
            continue
        entry_mode_by_tier = raw_template.get("entry_mode_by_tier") if isinstance(raw_template.get("entry_mode_by_tier"), Mapping) else {}
        regime_templates[str(regime_name)] = {
            "tp_multiplier": max(float(raw_template.get("tp_multiplier", 1.0) or 1.0), 0.1),
            "time_stop_multiplier": max(float(raw_template.get("time_stop_multiplier", 1.0) or 1.0), 0.1),
            "size_multiplier": max(float(raw_template.get("size_multiplier", 1.0) or 1.0), 0.0),
            "entry_zone_atr_mult": max(float(raw_template.get("entry_zone_atr_mult", 0.0) or 0.0), 0.0),
            "max_chase_atr_mult": max(float(raw_template.get("max_chase_atr_mult", 0.0) or 0.0), 0.0),
            "pullback_quality_floor": max(float(raw_template.get("pullback_quality_floor", 0.0) or 0.0), 0.0),
            "entry_mode_by_tier": {
                str(tier).strip().lower(): str(mode).strip().lower()
                for tier, mode in entry_mode_by_tier.items()
                if str(tier).strip() and str(mode).strip()
            },
        }

    pullback_quality_cfg = cfg.get("pullback_quality") if isinstance(cfg.get("pullback_quality"), Mapping) else {}
    disagreement_cfg = cfg.get("disagreement_severity") if isinstance(cfg.get("disagreement_severity"), Mapping) else {}
    coherence_weighting_cfg = cfg.get("coherence_weighting") if isinstance(cfg.get("coherence_weighting"), Mapping) else {}
    dynamic_rr_floor_cfg = cfg.get("dynamic_rr_floor") if isinstance(cfg.get("dynamic_rr_floor"), Mapping) else {}
    volatility_expansion_stop_cfg = (
        cfg.get("volatility_expansion_stop") if isinstance(cfg.get("volatility_expansion_stop"), Mapping) else {}
    )

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "bias_horizons": sorted({normalize_horizon_value(value) for value in (cfg.get("bias_horizons") or [4.0, 8.0, 12.0])}),
        "execution_horizons": sorted({normalize_horizon_value(value) for value in (cfg.get("execution_horizons") or [0.25, 1.0])}),
        "horizon_bias_weights": normalize_float_map(cfg.get("horizon_bias_weights"), minimum=0.0),
        "short_term_strict_horizons": sorted(
            {
                normalize_horizon_value(value)
                for value in (
                    [1.0]
                    if cfg.get("short_term_strict_horizons") is None
                    else cfg.get("short_term_strict_horizons")
                )
            }
        ),
        "short_term_min_mid_ratio": max(min(float(cfg.get("short_term_min_mid_ratio") or 0.67), 1.0), 0.0),
        "short_term_min_support_ratio": max(min(float(cfg.get("short_term_min_support_ratio") or 0.75), 1.0), 0.0),
        "short_term_min_mid_ratio_by_horizon": normalize_float_map(
            cfg.get("short_term_min_mid_ratio_by_horizon"),
            minimum=0.0,
        ),
        "min_bias_alignment_ratio": max(min(float(cfg.get("min_bias_alignment_ratio") or 0.0), 1.0), 0.0),
        "short_term_min_support_ratio_by_horizon": normalize_float_map(
            cfg.get("short_term_min_support_ratio_by_horizon"),
            minimum=0.0,
        ),
        "require_bias_alignment": bool(cfg.get("require_bias_alignment", True)),
        "immediate_entry_min_support_ratio": max(min(float(cfg.get("immediate_entry_min_support_ratio") or 0.8), 1.0), 0.0),
        "pullback_entry_min_support_ratio": max(min(float(cfg.get("pullback_entry_min_support_ratio") or 0.6), 1.0), 0.0),
        "immediate_entry_min_mid_ratio": max(min(float(cfg.get("immediate_entry_min_mid_ratio") or 0.67), 1.0), 0.0),
        "pullback_entry_min_mid_ratio": max(min(float(cfg.get("pullback_entry_min_mid_ratio") or 0.5), 1.0), 0.0),
        "high_execution_alignment_ratio": max(min(float(cfg.get("high_execution_alignment_ratio") or 1.0), 1.0), 0.0),
        "medium_execution_alignment_ratio": max(min(float(cfg.get("medium_execution_alignment_ratio") or 0.5), 1.0), 0.0),
        "entry_zone_atr_mult": max(float(cfg.get("entry_zone_atr_mult") or 0.25), 0.01),
        "max_chase_atr_mult": max(float(cfg.get("max_chase_atr_mult") or 0.35), 0.0),
        "session_lookback_bars": max(int(cfg.get("session_lookback_bars") or 8), 2),
        "swing_lookback_bars": max(int(cfg.get("swing_lookback_bars") or 6), 2),
        "structure_buffer_atr_mult": max(float(cfg.get("structure_buffer_atr_mult") or 0.2), 0.0),
        "minimum_rr_by_horizon": normalize_float_map(cfg.get("minimum_rr_by_horizon"), minimum=0.0),
        "time_stop_bars_by_horizon": {
            horizon: max(int(round(value)), 1)
            for horizon, value in normalize_float_map(cfg.get("time_stop_bars_by_horizon"), minimum=1.0).items()
        },
        "partial_take_profit": {
            "enabled": bool(partial_cfg.get("enabled", False)),
            "tp1_r_multiple": max(float(partial_cfg.get("tp1_r_multiple") or 1.0), 0.1),
            "tp1_size_fraction": max(min(float(partial_cfg.get("tp1_size_fraction") or 0.5), 1.0), 0.0),
            "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
        },
        "trailing_stop": {
            "enabled": bool(trailing_cfg.get("enabled", False)),
            "activation_r_multiple": max(float(trailing_cfg.get("activation_r_multiple") or 1.0), 0.1),
            "trail_buffer_atr_mult": max(float(trailing_cfg.get("trail_buffer_atr_mult") or 0.75), 0.0),
        },
        "analytics": {
            "enabled": bool(analytics_cfg.get("enabled", False)),
            "lookback_bars": max(int(analytics_cfg.get("lookback_bars") or default_lookback_bars), 10),
            "mae_quantile": max(min(float(analytics_cfg.get("mae_quantile") or 0.75), 0.99), 0.5),
            "mfe_quantile": max(min(float(analytics_cfg.get("mfe_quantile") or 0.6), 0.99), 0.5),
            "min_samples": max(int(analytics_cfg.get("min_samples") or default_min_samples), 10),
            "regime_volatility_buckets": {
                "enabled": bool(analytics_bucket_cfg.get("enabled", False)),
                "regime_col": str(analytics_bucket_cfg.get("regime_col") or "regime_state"),
                "volatility_col": str(analytics_bucket_cfg.get("volatility_col") or "volatility_realized_24h"),
                "min_bucket_samples": max(int(analytics_bucket_cfg.get("min_bucket_samples") or 12), 1),
                "low_vol_quantile": max(min(float(analytics_bucket_cfg.get("low_vol_quantile") or 0.5), 0.95), 0.05),
                "max_projection_mfe_ratio": max(float(analytics_bucket_cfg.get("max_projection_mfe_ratio") or 1.25), 0.5),
                "breakout_score_threshold": float(analytics_bucket_cfg.get("breakout_score_threshold") or 0.8),
                "chop_score_threshold": float(analytics_bucket_cfg.get("chop_score_threshold") or 0.3),
            },
        },
        "no_trade_guards": {
            "enabled": bool(guards_cfg.get("enabled", False)),
            "min_stop_distance_atr_mult": max(float(guards_cfg.get("min_stop_distance_atr_mult") or 0.35), 0.0),
            "max_stop_distance_atr_mult": max(float(guards_cfg.get("max_stop_distance_atr_mult") or 3.0), 0.0),
            "max_entry_deviation_atr_mult": max(float(guards_cfg.get("max_entry_deviation_atr_mult") or 1.25), 0.0),
            "require_favorable_entry_zone": bool(guards_cfg.get("require_favorable_entry_zone", True)),
        },
        "adaptive_take_profit": {
            "enabled": bool(adaptive_tp_cfg.get("enabled", True)),
            "min_rr_fraction_of_floor": max(min(float(adaptive_tp_cfg.get("min_rr_fraction_of_floor") or 0.85), 1.0), 0.0),
        },
        "target_range_stop_refinement": {
            "enabled": bool(target_range_stop_cfg.get("enabled", False)),
            "horizons": sorted(
                {
                    normalize_horizon_value(value)
                    for value in (target_range_stop_cfg.get("horizons") or default_target_range_stop_horizons)
                }
            ),
            "confidence_min": max(
                min(float(target_range_stop_cfg.get("confidence_min") or default_target_range_stop_confidence_min), 1.0),
                0.0,
            ),
            "buffer_std_mult": max(float(target_range_stop_cfg.get("buffer_std_mult") or default_target_range_stop_buffer_std_mult), 0.0),
            "min_tighten_fraction": max(
                min(float(target_range_stop_cfg.get("min_tighten_fraction") or default_target_range_stop_min_tighten_fraction), 1.0),
                0.0,
            ),
        },
        "pullback_quality": {
            "enabled": bool(pullback_quality_cfg.get("enabled", False)),
            "min_score_by_horizon": normalize_float_map(pullback_quality_cfg.get("min_score_by_horizon"), minimum=0.0),
            "max_vwap_deviation_atr": max(float(pullback_quality_cfg.get("max_vwap_deviation_atr") or 1.5), 0.1),
            "max_candle_expansion_ratio": max(float(pullback_quality_cfg.get("max_candle_expansion_ratio") or 2.0), 0.1),
            "candle_expansion_window": max(int(pullback_quality_cfg.get("candle_expansion_window") or 8), 2),
            "range_expansion_penalty_threshold": max(float(pullback_quality_cfg.get("range_expansion_penalty_threshold") or 1.25), 0.0),
        },
        "disagreement_severity": {
            "enabled": bool(disagreement_cfg.get("enabled", True)),
            "block_threshold": max(min(float(disagreement_cfg.get("block_threshold") or 0.7), 1.0), 0.0),
            "pullback_threshold": max(min(float(disagreement_cfg.get("pullback_threshold") or 0.45), 1.0), 0.0),
            "vwap_extension_penalty_atr": max(float(disagreement_cfg.get("vwap_extension_penalty_atr") or 0.75), 0.0),
            "range_expansion_penalty_threshold": max(float(disagreement_cfg.get("range_expansion_penalty_threshold") or 1.0), 0.0),
        },
        "coherence_weighting": {
            "enabled": bool(coherence_weighting_cfg.get("enabled", False)),
            "low_trust_penalty": max(min(float(coherence_weighting_cfg.get("low_trust_penalty") or 0.35), 1.0), 0.0),
            "blocked_penalty": max(min(float(coherence_weighting_cfg.get("blocked_penalty") or 1.0), 1.0), 0.0),
            "p_up_conflict_penalty": max(min(float(coherence_weighting_cfg.get("p_up_conflict_penalty") or 0.2), 1.0), 0.0),
            "consensus_bonus": max(float(coherence_weighting_cfg.get("consensus_bonus") or 0.1), 0.0),
            "neutral_band": max(float(coherence_weighting_cfg.get("neutral_band") or 0.02), 0.0),
            "min_multiplier": max(min(float(coherence_weighting_cfg.get("min_multiplier") or 0.1), 1.0), 0.0),
            "by_horizon": normalize_float_map(coherence_weighting_cfg.get("by_horizon"), minimum=0.0),
        },
        "dynamic_rr_floor": {
            "enabled": bool(dynamic_rr_floor_cfg.get("enabled", False)),
            "mfe_mae_scale": max(float(dynamic_rr_floor_cfg.get("mfe_mae_scale") or 0.9), 0.0),
            "max_adjustment": max(min(float(dynamic_rr_floor_cfg.get("max_adjustment") or 0.35), 1.0), 0.0),
            "min_samples": max(int(dynamic_rr_floor_cfg.get("min_samples") or 40), 1),
            "default_floor": max(float(dynamic_rr_floor_cfg.get("default_floor") or 0.0), 0.0),
            "min_floor_by_horizon": normalize_float_map(dynamic_rr_floor_cfg.get("min_floor_by_horizon"), minimum=0.0),
            "max_floor_by_horizon": normalize_float_map(dynamic_rr_floor_cfg.get("max_floor_by_horizon"), minimum=0.0),
            "regime_multiplier": {
                str(key).strip().lower(): max(float(value), 0.0)
                for key, value in (dynamic_rr_floor_cfg.get("regime_multiplier") or {}).items()
                if str(key).strip()
            }
            if isinstance(dynamic_rr_floor_cfg.get("regime_multiplier"), Mapping)
            else {},
        },
        "volatility_expansion_stop": {
            "enabled": bool(volatility_expansion_stop_cfg.get("enabled", False)),
            "expansion_threshold": max(float(volatility_expansion_stop_cfg.get("expansion_threshold") or 1.15), 0.0),
            "stop_multiplier": max(float(volatility_expansion_stop_cfg.get("stop_multiplier") or 1.1), 0.1),
            "max_multiplier": max(float(volatility_expansion_stop_cfg.get("max_multiplier") or 1.5), 0.1),
            "regimes": [
                str(value).strip().lower()
                for value in (volatility_expansion_stop_cfg.get("regimes") or [])
                if str(value).strip()
            ],
        },
        "regime_templates": regime_templates,
    }


def apply_execution_policy(
    summary: SummaryPayload,
    contexts: ExecutionContexts,
    policy: Mapping[str, Any],
    *,
    regime_neutral: str,
    execution_policy_default_lookback_bars: int,
    execution_policy_default_min_samples: int,
    summarize_bias_context: Callable[[SummaryPayload, Mapping[str, Any]], Mapping[str, Any]],
    execution_side: Callable[[Mapping[str, Any]], str],
    direction_vote: Callable[[Mapping[str, Any]], str],
    execution_alignment_ratio: Callable[[Any, str, Mapping[str, Any]], float],
    classify_execution_tier: Callable[[Mapping[str, Any], str, float, Mapping[str, Any]], str],
    compute_atr_like_price_distance: Callable[..., float],
    compute_recent_structure: Callable[..., Mapping[str, Any]],
    build_entry_zone: Callable[..., Mapping[str, Any]],
    compute_pullback_quality_score: Callable[..., Mapping[str, Any]],
    compute_disagreement_severity: Callable[..., Mapping[str, Any]],
    compute_excursion_priors: Callable[..., Mapping[str, Any]],
    finite_float_or_none: Callable[[Any], float | None],
    finite_float: Callable[[Any, float], float],
    resolve_stop_with_guardrails: Callable[..., Mapping[str, Any]],
    refine_stop_with_target_range: Callable[..., Mapping[str, Any]],
    resolve_execution_target_reward: Callable[..., Mapping[str, Any]],
    lookup_horizon_value: Callable[[Mapping[str, Any], float, Any], Any],
    resolve_execution_upstream_hold_reason: Callable[[Mapping[str, Any]], str],
) -> SummaryPayload:
    if not summary:
        return summary

    bias_context = summarize_bias_context(summary, policy)
    bias_direction = str(bias_context.get("bias_direction", "neutral"))
    bias_alignment_ratio = float(bias_context.get("bias_alignment_ratio", 0.0))
    execution_entries = bias_context.get("execution_entries", [])
    weights = policy.get("horizon_bias_weights") if isinstance(policy.get("horizon_bias_weights"), Mapping) else {}

    for label, entry in summary.items():
        market_price = float(entry.get("close", entry.get("entry_price", 0.0)) or 0.0)
        entry["market_price"] = market_price
        entry["execution_prior_provenance"] = {
            "analytics_source": "unavailable",
            "matched_regime": None,
            "volatility_bucket": None,
            "bucket_threshold": None,
            "sample_count": 0,
            "stop_source": None,
            "stop_adjustment_type": None,
            "target_source": "existing_or_projection",
        }
        side = execution_side(entry)
        direction = direction_vote(entry)
        upstream_hold = str(entry.get("trade_action", "hold")) == "hold"
        alignment_ratio = execution_alignment_ratio(execution_entries, direction=direction, weights=weights)
        tier = classify_execution_tier(
            entry,
            bias_direction=bias_direction,
            execution_alignment_ratio=alignment_ratio,
            policy=policy,
        )
        bias_scores = bias_context.get("bias_scores") if isinstance(bias_context.get("bias_scores"), Mapping) else {}
        execution_scores = bias_context.get("execution_scores") if isinstance(bias_context.get("execution_scores"), Mapping) else {}
        bias_score_value = float((bias_scores.get("up_score") if direction == "up" else bias_scores.get("down_score")) or 0.0)
        execution_score_value = float(
            (execution_scores.get("up_score") if direction == "up" else execution_scores.get("down_score")) or 0.0
        )
        support_horizons = list((bias_context.get("direction_support_horizons") or {}).get(direction, []))
        entry["bias_score"] = bias_score_value
        entry["execution_score"] = execution_score_value
        entry["bias_support_horizons"] = support_horizons
        entry["bias_support_is_8h_standalone"] = support_horizons == ["8h"]
        plan: Dict[str, Any] = {
            "enabled": bool(policy.get("enabled", False)),
            "bias_direction": bias_direction,
            "bias_alignment_ratio": bias_alignment_ratio,
            "execution_alignment_ratio": float(alignment_ratio),
            "bias_score": float(bias_score_value),
            "execution_score": float(execution_score_value),
            "confluence_tier": tier,
            "status": "ready",
            "reason": "pass",
            "side": side,
            "entry_mode": "disabled",
            "pending_trade_action": side,
            "partial_take_profit": None,
            "time_stop": None,
            "trailing_stop": None,
            "analytics": {"available": False},
            "structure": None,
            "stop_management": None,
        }
        if not bool(policy.get("enabled", False)):
            entry["execution_plan"] = plan
            continue

        forecast_coherence = entry.get("forecast_coherence")
        if isinstance(forecast_coherence, Mapping) and forecast_coherence.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "forecast_coherence_gate"
            entry["execution_plan"] = plan
            continue

        if bool(policy.get("require_bias_alignment", True)) and bias_direction != "neutral" and direction != bias_direction:
            plan["status"] = "rejected"
            plan["reason"] = "bias_direction_conflict"
            entry["execution_plan"] = plan
            continue

        context = contexts.get(label)
        if not context:
            plan["status"] = "rejected"
            plan["reason"] = "missing_execution_context"
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
            entry["execution_plan"] = plan
            continue

        prepared = context["prepared"]
        index = int(context["index"])
        horizon = float(context["horizon"])
        residual_std = float(context["residual_std"])
        regime_state = str(entry.get("regime_state", regime_neutral))
        regime_template = (policy.get("regime_templates") or {}).get(regime_state, {})
        horizon_steps = max(int(round(horizon)), 1)
        atr_distance = compute_atr_like_price_distance(
            prepared.df_all,
            index=index,
            fallback_close=market_price,
            fallback_return_std=residual_std,
        )
        structure = compute_recent_structure(
            prepared.df_all,
            index=index,
            session_lookback_bars=int(policy.get("session_lookback_bars", 8)),
            swing_lookback_bars=int(policy.get("swing_lookback_bars", 6)),
            atr_distance=atr_distance,
            fallback_price=market_price,
        )
        plan["structure"] = structure
        entry_zone = build_entry_zone(
            market_price=market_price,
            side=side,
            structure=structure,
            policy=policy,
            regime_template=regime_template,
        )
        preferred_entry = float(entry_zone["preferred_entry_price"])
        plan.update(entry_zone)

        pullback_quality = compute_pullback_quality_score(
            entry=entry,
            frame=prepared.df_all,
            index=index,
            market_price=market_price,
            side=side,
            structure=structure,
            atr_distance=atr_distance,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
        )
        disagreement_severity = compute_disagreement_severity(
            entry,
            bias_context=bias_context,
            policy=policy,
            atr_distance=atr_distance,
            structure=structure,
        )
        plan["pullback_quality"] = pullback_quality
        plan["disagreement_severity"] = disagreement_severity
        entry["disagreement_severity"] = disagreement_severity

        template_max_chase = float(regime_template.get("max_chase_atr_mult", 0.0) or 0.0)
        max_chase = (template_max_chase if template_max_chase > 0.0 else float(policy.get("max_chase_atr_mult", 0.35))) * atr_distance
        market_deviation = abs(market_price - preferred_entry)
        if tier == "high" and (bool(entry_zone["entry_ready"]) or market_deviation <= max_chase):
            entry_mode = "immediate"
            planned_entry = market_price
        elif tier in {"high", "medium"}:
            entry_mode = "pullback"
            planned_entry = preferred_entry
        else:
            entry_mode = "blocked"
            planned_entry = preferred_entry

        template_entry_modes = regime_template.get("entry_mode_by_tier") if isinstance(regime_template.get("entry_mode_by_tier"), Mapping) else {}
        template_entry_mode = str(template_entry_modes.get(tier) or "").strip().lower()
        if template_entry_mode in {"immediate", "pullback", "blocked"}:
            if template_entry_mode == "blocked":
                entry_mode = "blocked"
            elif template_entry_mode == "pullback" and entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif template_entry_mode == "immediate" and entry_mode == "pullback" and bool(entry_zone["entry_ready"]):
                entry_mode = "immediate"
                planned_entry = market_price

        if disagreement_severity.get("triggered"):
            plan["status"] = "rejected"
            plan["reason"] = "short_term_disagreement"
        elif disagreement_severity.get("pullback_only") and entry_mode == "immediate":
            entry_mode = "pullback"
            planned_entry = preferred_entry

        if pullback_quality.get("triggered"):
            if entry_mode == "immediate":
                entry_mode = "pullback"
                planned_entry = preferred_entry
            elif entry_mode == "pullback":
                plan["status"] = "rejected"
                plan["reason"] = "pullback_quality_insufficient"
        plan["entry_mode"] = entry_mode

        analytics_cfg = policy.get("analytics", {}) if isinstance(policy.get("analytics"), Mapping) else {}
        analytics_payload: Mapping[str, Any] = {"available": False}
        if analytics_cfg.get("enabled"):
            analytics_payload = compute_excursion_priors(
                prepared.df_all,
                index=index,
                horizon_steps=horizon_steps,
                side=side,
                lookback_bars=int(analytics_cfg.get("lookback_bars", execution_policy_default_lookback_bars)),
                min_samples=int(analytics_cfg.get("min_samples", execution_policy_default_min_samples)),
                mae_quantile=float(analytics_cfg.get("mae_quantile", 0.75)),
                mfe_quantile=float(analytics_cfg.get("mfe_quantile", 0.6)),
                current_regime=regime_state,
                current_volatility=finite_float_or_none((entry.get("volatility") or {}).get("current")),
                bucket_policy=analytics_cfg.get("regime_volatility_buckets"),
            )
        plan["analytics"] = analytics_payload

        existing_stop = float(entry.get("stop_loss", planned_entry))
        existing_take = float(entry.get("take_profit", planned_entry))
        structure_buffer = atr_distance * float(policy.get("structure_buffer_atr_mult", 0.2))
        if side == "long":
            structure_stop = min(float(structure["session_low"]), float(structure["swing_low"])) - structure_buffer
            analytic_stop = planned_entry * (1.0 - float(analytics_payload.get("mae_distance") or 0.0))
        else:
            structure_stop = max(float(structure["session_high"]), float(structure["swing_high"])) + structure_buffer
            analytic_stop = planned_entry * (1.0 + float(analytics_payload.get("mae_distance") or 0.0))
        analytic_stop_value = analytic_stop if analytics_payload.get("available") else None

        guards_cfg = policy.get("no_trade_guards", {}) if isinstance(policy.get("no_trade_guards"), Mapping) else {}
        stop_resolution = resolve_stop_with_guardrails(
            side=side,
            planned_entry=planned_entry,
            existing_stop=existing_stop,
            structure_stop=structure_stop,
            analytic_stop=analytic_stop_value,
            atr_distance=atr_distance,
            guards_cfg=guards_cfg,
            analytic_stop_preferred=bool(analytics_payload.get("available")) and str(analytics_payload.get("source")) != "global",
        )
        selected_stop = float(stop_resolution["stop_loss"])
        risk_unit = float(stop_resolution["risk_unit"])
        stop_refinement = refine_stop_with_target_range(
            side=side,
            planned_entry=planned_entry,
            selected_stop=selected_stop,
            risk_unit=risk_unit,
            atr_distance=atr_distance,
            horizon=horizon,
            projected_high=finite_float_or_none(entry.get("projected_high")),
            projected_low=finite_float_or_none(entry.get("projected_low")),
            projected_high_confidence=finite_float_or_none(entry.get("projected_high_confidence")),
            projected_low_confidence=finite_float_or_none(entry.get("projected_low_confidence")),
            projected_high_residual_std=finite_float_or_none(entry.get("projected_high_residual_std")),
            projected_low_residual_std=finite_float_or_none(entry.get("projected_low_residual_std")),
            policy=policy,
            guards_cfg=guards_cfg,
        )
        if stop_refinement.get("applied"):
            selected_stop = float(stop_refinement["stop_loss"])
            risk_unit = float(stop_refinement["risk_unit"])
        stop_scaling_payload = {
            "applied": False,
            "reason": "not_triggered",
            "multiplier": 1.0,
            "risk_unit_before": float(risk_unit),
            "risk_unit_after": float(risk_unit),
        }
        regime_stop_multiplier = max(float(regime_template.get("stop_multiplier", 1.0) or 1.0), 0.1)
        if regime_stop_multiplier > 1.0:
            scaled_risk = risk_unit * regime_stop_multiplier
            selected_stop = planned_entry - scaled_risk if side == "long" else planned_entry + scaled_risk
            risk_unit = float(max(scaled_risk, 1e-8))
            stop_scaling_payload = {
                "applied": True,
                "reason": "regime_stop_multiplier",
                "multiplier": float(regime_stop_multiplier),
                "risk_unit_before": float(stop_scaling_payload["risk_unit_before"]),
                "risk_unit_after": float(risk_unit),
            }

        vol_stop_cfg = policy.get("volatility_expansion_stop") if isinstance(policy.get("volatility_expansion_stop"), Mapping) else {}
        if bool(vol_stop_cfg.get("enabled", False)):
            expansion_value = abs(finite_float(entry.get("range_expansion_1h"), 0.0))
            expansion_threshold = float(vol_stop_cfg.get("expansion_threshold", 1.15) or 1.15)
            scoped_regimes = {str(v).strip().lower() for v in (vol_stop_cfg.get("regimes") or []) if str(v).strip()}
            regime_allowed = (not scoped_regimes) or (regime_state in scoped_regimes)
            if regime_allowed and expansion_value >= expansion_threshold:
                stop_multiplier = max(float(vol_stop_cfg.get("stop_multiplier", 1.1) or 1.1), 0.1)
                max_multiplier = max(float(vol_stop_cfg.get("max_multiplier", 1.5) or 1.5), 0.1)
                stop_multiplier = min(stop_multiplier, max_multiplier)
                scaled_risk = risk_unit * stop_multiplier
                selected_stop = planned_entry - scaled_risk if side == "long" else planned_entry + scaled_risk
                risk_unit = float(max(scaled_risk, 1e-8))
                stop_scaling_payload = {
                    "applied": True,
                    "reason": "volatility_expansion_stop",
                    "multiplier": float(stop_multiplier),
                    "expansion_value": float(expansion_value),
                    "expansion_threshold": float(expansion_threshold),
                    "risk_unit_before": float(stop_scaling_payload.get("risk_unit_after", stop_scaling_payload["risk_unit_before"])),
                    "risk_unit_after": float(risk_unit),
                }
        plan["stop_management"] = {
            "source": stop_resolution.get("source"),
            "adjustment": stop_resolution.get("adjustment"),
            "target_range_refinement": stop_refinement.get("details"),
            "stop_scaling": stop_scaling_payload,
        }

        if guards_cfg.get("enabled"):
            max_entry_dev = float(guards_cfg.get("max_entry_deviation_atr_mult", 1.25)) * atr_distance
            if bool(guards_cfg.get("require_favorable_entry_zone", True)) and market_deviation > max_entry_dev and entry_mode == "immediate":
                plan["status"] = "rejected"
                plan["reason"] = "entry_too_extended"

        target_resolution = resolve_execution_target_reward(
            side=side,
            planned_entry=planned_entry,
            existing_take=existing_take,
            projected_high=finite_float_or_none(entry.get("projected_high")),
            projected_low=finite_float_or_none(entry.get("projected_low")),
            analytics_payload=analytics_payload,
            risk_unit=risk_unit,
            horizon=horizon,
            policy=policy,
            regime_template=regime_template,
            regime_state=regime_state,
        )
        selected_take = float(target_resolution["selected_take"])
        risk_reward_ratio = float(target_resolution["risk_reward_ratio"])
        plan["target_management"] = dict(target_resolution["target_management"])
        if target_resolution["status"] != "pass":
            plan["status"] = "rejected"
            plan["reason"] = str(target_resolution["reason"])

        partial_cfg = policy.get("partial_take_profit", {}) if isinstance(policy.get("partial_take_profit"), Mapping) else {}
        partial_take_profit = None
        if partial_cfg.get("enabled"):
            tp1_distance = risk_unit * float(partial_cfg.get("tp1_r_multiple", 1.0))
            tp1_price = planned_entry + tp1_distance if side == "long" else planned_entry - tp1_distance
            partial_take_profit = {
                "enabled": True,
                "tp1_price": tp1_price,
                "tp1_size_fraction": float(partial_cfg.get("tp1_size_fraction", 0.5)),
                "tp2_price": selected_take,
                "move_stop_to_break_even": bool(partial_cfg.get("move_stop_to_break_even", True)),
            }

        trailing_cfg = policy.get("trailing_stop", {}) if isinstance(policy.get("trailing_stop"), Mapping) else {}
        trailing_stop = None
        if trailing_cfg.get("enabled"):
            activation_distance = risk_unit * float(trailing_cfg.get("activation_r_multiple", 1.0))
            trailing_stop = {
                "enabled": True,
                "activation_price": planned_entry + activation_distance if side == "long" else planned_entry - activation_distance,
                "trail_buffer": atr_distance * float(trailing_cfg.get("trail_buffer_atr_mult", 0.75)),
            }

        time_stop_map = policy.get("time_stop_bars_by_horizon", {}) if isinstance(policy.get("time_stop_bars_by_horizon"), Mapping) else {}
        base_time_stop = max(int(round(lookup_horizon_value(time_stop_map, horizon, max(horizon_steps, 1)))), 1)
        time_stop_mult = float(regime_template.get("time_stop_multiplier", 1.0) or 1.0)
        recommended_time_stop = max(int(round(base_time_stop * time_stop_mult)), 1)
        if analytics_payload.get("available") and analytics_payload.get("peak_step_p50"):
            recommended_time_stop = min(recommended_time_stop, max(int(analytics_payload["peak_step_p50"] * 1.25), 1))
        time_stop_payload = {
            "enabled": True,
            "bars": recommended_time_stop,
            "reason": "stagnation_exit",
        }

        if plan["status"] == "ready" and entry_mode == "pullback":
            if bool(entry_zone["entry_ready"]):
                plan["status"] = "ready"
                plan["reason"] = "pass"
            else:
                plan["status"] = "waiting_pullback"
                plan["reason"] = "await_pullback_entry_zone"
        elif plan["status"] == "ready" and entry_mode == "blocked":
            plan["status"] = "rejected"
            plan["reason"] = "low_execution_confluence"

        position_size = float(entry.get("position_size", 0.0))
        position_size *= float(regime_template.get("size_multiplier", 1.0) or 1.0)
        if tier == "medium":
            position_size *= 0.85
        elif tier == "low":
            position_size = 0.0

        plan["partial_take_profit"] = partial_take_profit
        plan["time_stop"] = time_stop_payload
        plan["trailing_stop"] = trailing_stop

        entry["entry_price"] = float(planned_entry)
        entry["stop_loss"] = float(selected_stop)
        entry["take_profit"] = float(selected_take)
        entry["risk_reward_ratio"] = float(risk_reward_ratio)
        entry["position_size"] = float(max(position_size, 0.0))
        analytics_payload_final = plan.get("analytics") if isinstance(plan.get("analytics"), Mapping) else {}
        stop_management = plan.get("stop_management") if isinstance(plan.get("stop_management"), Mapping) else {}
        entry["execution_prior_provenance"] = {
            "analytics_source": analytics_payload_final.get("source", "unavailable") if analytics_payload_final else "unavailable",
            "matched_regime": analytics_payload_final.get("matched_regime"),
            "volatility_bucket": analytics_payload_final.get("volatility_bucket"),
            "bucket_threshold": analytics_payload_final.get("bucket_threshold"),
            "sample_count": analytics_payload_final.get("sample_count"),
            "stop_source": stop_management.get("source"),
            "stop_adjustment_type": (stop_management.get("adjustment") or {}).get("type") if stop_management else None,
            "target_source": str((plan.get("target_management") or {}).get("source") or "existing_or_projection"),
        }
        entry["execution_plan"] = plan

        if upstream_hold and plan["status"] == "ready":
            plan["status"] = "bias_only_ready"
            plan["reason"] = resolve_execution_upstream_hold_reason(entry)
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        elif plan["status"] != "ready":
            entry["trade_action"] = "hold"
            entry["signal_ensemble"] = 0
        else:
            entry["trade_action"] = side
    return summary
