from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


WarningHandler = Optional[Callable[[str, str], None]]


def _emit_warning(handler: WarningHandler, key: str, message: str) -> None:
    if handler is not None:
        handler(key, message)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def apply_funding_rate_features(
    df: pd.DataFrame,
    *,
    strict_missing: bool,
    warn: WarningHandler = None,
) -> pd.DataFrame:
    result = df.copy()
    candidates = (
        "funding_rate",
        "funding_BTCUSDT_funding_rate",
        "fut_funding_rate",
    )
    source = next((col for col in candidates if col in result.columns), None)
    if source is None:
        if strict_missing:
            raise ValueError(
                "Cannot compute funding_rate_zscore_24h; missing funding-rate source columns.",
            )
        if "funding_rate_zscore_24h" not in result.columns:
            result["funding_rate_zscore_24h"] = 0.0
            _emit_warning(
                warn,
                "funding_rate",
                "Funding rate columns missing; refresh data/processed/funding/hourly_features.parquet or BigQuery curated feeds to unlock funding_rate_zscore_24h.",
            )
        return result

    funding = result[source].astype(float)
    rolling_mean = funding.rolling(window=24, min_periods=6).mean()
    rolling_std = funding.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
    oscillator = ((funding - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["funding_rate_zscore_24h"] = oscillator.clip(-10.0, 10.0)
    return result


def augment_hourly_price_features(
    frame: pd.DataFrame,
    *,
    strict_missing: bool,
    warn: WarningHandler = None,
) -> pd.DataFrame:
    result = frame.copy()

    def _safe_diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _safe_pct(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for base in ("close", "volume", "fut_close", "fut_volume"):
        if base not in result.columns:
            continue
        result[f"{base}_delta_1h"] = _safe_diff(result[base])
        result[f"{base}_pct_change_1h"] = _safe_pct(result[base])

    if "close" in result.columns:
        std_7 = result["close"].rolling(window=7, min_periods=3).std(ddof=0)
        std_24 = result["close"].rolling(window=24, min_periods=6).std(ddof=0)
        if "ma_close_7h" in result.columns:
            denom = std_7.replace(0.0, np.nan)
            result["close_zscore_7h"] = ((result["close"] - result["ma_close_7h"]) / denom).fillna(0.0)
        if "ma_close_24h" in result.columns:
            denom = std_24.replace(0.0, np.nan)
            result["close_zscore_24h"] = ((result["close"] - result["ma_close_24h"]) / denom).fillna(0.0)

    if "fut_close" in result.columns:
        rolling_mean = result["fut_close"].rolling(window=7, min_periods=3).mean()
        rolling_std = result["fut_close"].rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
        result["fut_close_zscore_7h"] = ((result["fut_close"] - rolling_mean) / rolling_std).fillna(0.0)

    required_cvd = {"volume", "taker_buy_base_volume"}
    if required_cvd.issubset(result.columns):
        total_volume = result["volume"].astype(float)
        taker_buy = result["taker_buy_base_volume"].astype(float)
        taker_sell = (total_volume - taker_buy).clip(lower=0.0)
        cvd_raw = taker_buy - taker_sell
        cvd_window = cvd_raw.rolling(window=6, min_periods=2).sum()
        vol_window = total_volume.rolling(window=6, min_periods=2).sum().replace(0.0, np.nan)
        ratio = (cvd_window / vol_window).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        result["cvd_ratio_6h"] = ratio
        cvd_mean = cvd_window.rolling(window=24, min_periods=6).mean()
        cvd_std = cvd_window.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
        zscore = ((cvd_window - cvd_mean) / cvd_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["cvd_zscore_6h"] = zscore.clip(-10.0, 10.0)
    elif strict_missing:
        missing = ", ".join(sorted(required_cvd - set(result.columns)))
        raise ValueError(f"Cannot compute CVD features; missing columns: {missing}.")
    else:
        if "cvd_ratio_6h" not in result.columns:
            result["cvd_ratio_6h"] = 0.0
        if "cvd_zscore_6h" not in result.columns:
            result["cvd_zscore_6h"] = 0.0
        _emit_warning(
            warn,
            "cvd_ratio_6h",
            "Missing taker volume columns (volume + taker_buy_base_volume); hydrate Binance spot klines before relying on CVD breakout signals.",
        )

    required_liquidity = {"high", "low", "close"}
    if required_liquidity.issubset(result.columns):
        high = result["high"].astype(float)
        low = result["low"].astype(float)
        close = result["close"].astype(float)
        true_range = _true_range(high, low, close)
        atr_6h = true_range.rolling(window=6, min_periods=2).mean().replace(0.0, np.nan)
        range_span = (high - low).abs()
        liquidity_ratio = (range_span / atr_6h).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["liquidity_range_ratio_6h"] = liquidity_ratio.clip(0.0, 10.0)

        mid_price = (high + low) / 2.0
        half_range = (high - low).replace(0.0, np.nan) / 2.0
        close_position = ((close - mid_price) / half_range)
        close_position = close_position.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        result["liquidity_close_position_ratio"] = close_position

        atr_24h = true_range.rolling(window=24, min_periods=6).mean().replace(0.0, np.nan)
        result["range_expansion_1h"] = (true_range / atr_24h).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 10.0)
        session_high = high.rolling(window=8, min_periods=2).max().replace(0.0, np.nan)
        session_low = low.rolling(window=8, min_periods=2).min().replace(0.0, np.nan)
        result["distance_from_session_high_8h"] = ((close / session_high) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        result["distance_from_session_low_8h"] = ((close / session_low) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

        if "volume" in result.columns:
            volume = result["volume"].astype(float)
            typical_price = (high + low + close) / 3.0
            rolling_notional = (typical_price * volume).rolling(window=8, min_periods=2).sum()
            rolling_volume = volume.rolling(window=8, min_periods=2).sum().replace(0.0, np.nan)
            rolling_vwap = (rolling_notional / rolling_volume).replace([np.inf, -np.inf], np.nan)
            result["vwap_deviation_8h"] = ((close / rolling_vwap) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        else:
            result["vwap_deviation_8h"] = 0.0

        result["momentum_slope_2h"] = _safe_pct(close).rolling(window=2, min_periods=1).mean().fillna(0.0)
        result["momentum_slope_4h"] = _safe_pct(close).rolling(window=4, min_periods=1).mean().fillna(0.0)
    elif strict_missing:
        missing = ", ".join(sorted(required_liquidity - set(result.columns)))
        raise ValueError(f"Cannot compute liquidity features; missing OHLC columns: {missing}.")
    else:
        if "liquidity_range_ratio_6h" not in result.columns:
            result["liquidity_range_ratio_6h"] = 0.0
        if "liquidity_close_position_ratio" not in result.columns:
            result["liquidity_close_position_ratio"] = 0.0
        if "range_expansion_1h" not in result.columns:
            result["range_expansion_1h"] = 0.0
        if "distance_from_session_high_8h" not in result.columns:
            result["distance_from_session_high_8h"] = 0.0
        if "distance_from_session_low_8h" not in result.columns:
            result["distance_from_session_low_8h"] = 0.0
        if "vwap_deviation_8h" not in result.columns:
            result["vwap_deviation_8h"] = 0.0
        if "momentum_slope_2h" not in result.columns:
            result["momentum_slope_2h"] = 0.0
        if "momentum_slope_4h" not in result.columns:
            result["momentum_slope_4h"] = 0.0
        _emit_warning(
            warn,
            "liquidity_features",
            "Missing OHLC columns; Binance-native liquidity stress metrics default to zeros.",
        )

    return result


__all__ = [
    "apply_funding_rate_features",
    "augment_hourly_price_features",
]