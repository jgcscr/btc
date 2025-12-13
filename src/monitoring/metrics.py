"""Monitoring metric helper utilities."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
EPSILON = 1e-9


def _to_series(values: Iterable[float] | Sequence[float] | pd.Series | np.ndarray) -> pd.Series:
    """Coerce various iterables into a float pandas Series."""
    if isinstance(values, pd.Series):
        return values.astype(float, copy=False)
    if isinstance(values, np.ndarray):
        return pd.Series(values.astype(float, copy=False))
    if isinstance(values, (list, tuple)):
        return pd.Series(values, dtype=float)
    return pd.Series(list(values), dtype=float)


def rolling_mean(series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling mean with a minimum of one observation per window."""
    if window <= 0:
        raise ValueError("window must be positive")

    s = _to_series(series)
    if s.empty:
        return np.array([], dtype=float)

    return s.rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def rolling_std(series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling population standard deviation."""
    if window <= 0:
        raise ValueError("window must be positive")

    s = _to_series(series)
    if s.empty:
        return np.array([], dtype=float)

    return s.rolling(window=window, min_periods=1).std(ddof=0).to_numpy(dtype=float)


def z_scores(
    series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    *,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Compute z-scores relative to a baseline distribution."""
    arr = _to_series(series).to_numpy(dtype=float, copy=False)
    if arr.size == 0:
        return np.array([], dtype=float)

    if (not math.isfinite(baseline_mean)) or (not math.isfinite(baseline_std)) or abs(baseline_std) <= epsilon:
        return np.full(arr.shape, np.nan, dtype=float)

    return (arr - baseline_mean) / baseline_std


def ks_statistic(
    sample_series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray,
    baseline_series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic between two samples."""
    sample = _to_series(sample_series).dropna().to_numpy(dtype=float, copy=False)
    baseline = _to_series(baseline_series).dropna().to_numpy(dtype=float, copy=False)

    if sample.size == 0 or baseline.size == 0:
        return math.nan

    sample.sort()
    baseline.sort()

    combined = np.concatenate([sample, baseline])
    sample_cdf = np.searchsorted(sample, combined, side="right") / sample.size
    baseline_cdf = np.searchsorted(baseline, combined, side="right") / baseline.size

    return float(np.max(np.abs(sample_cdf - baseline_cdf)))


def summary_stats(
    series: Iterable[float] | Sequence[float] | pd.Series | np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float | dict[str, float] | int]:
    """Return summary statistics for the provided series."""
    s = _to_series(series).dropna()
    if s.empty:
        return {}

    stats: dict[str, float | dict[str, float] | int] = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
    }

    if quantiles:
        quantile_values = s.quantile(list(quantiles))
        stats["quantiles"] = {f"{q:.2f}": float(quantile_values.loc[q]) for q in quantiles}

    return stats


def ensure_meta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach meta-ensemble monitoring columns when source data permits."""

    result = df.copy()

    meta_net_candidates: list[str] = []

    for column in list(result.columns):
        if column.startswith("ret_net_fee_"):
            alias = "meta_" + column[len("ret_") :]
            if alias not in result.columns:
                result[alias] = pd.to_numeric(result[column], errors="coerce")
            meta_net_candidates.append(alias)
        elif column.startswith("meta_net_fee_"):
            meta_net_candidates.append(column)

    if "signal_meta" in result.columns:
        result["signal_meta"] = pd.to_numeric(result["signal_meta"], errors="coerce")

    net_source = next((col for col in meta_net_candidates if col in result.columns), None)
    if net_source and "signal_meta" in result.columns:
        signal_series = pd.to_numeric(result["signal_meta"], errors="coerce")
        net_series = pd.to_numeric(result[net_source], errors="coerce")
        meta_hits = np.where(signal_series > 0, (net_series > 0).astype(float), np.nan)
        result["meta_hit_rate"] = meta_hits

    return result


__all__ = [
    "rolling_mean",
    "rolling_std",
    "z_scores",
    "ks_statistic",
    "summary_stats",
    "ensure_meta_metrics",
]
