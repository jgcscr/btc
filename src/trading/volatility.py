from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_REALIZED_WINDOWS: tuple[int, ...] = (24, 72)
DEFAULT_GARCH_ALPHA = 0.05
DEFAULT_GARCH_BETA = 0.9
DEFAULT_GARCH_OMEGA = 1e-6


def _safe_log_returns(close: pd.Series) -> pd.Series:
    series = pd.Series(close, dtype=float).replace(0.0, np.nan)
    log_prices = np.log(series)
    returns = log_prices.diff().fillna(0.0)
    return returns.replace([np.inf, -np.inf], 0.0)


def compute_realized_volatility(
    close: pd.Series,
    window: int,
    *,
    annualize_to_daily: bool = True,
) -> pd.Series:
    if window <= 1:
        raise ValueError("Realized volatility window must be > 1 hour.")
    returns = _safe_log_returns(close)
    min_periods = max(2, window // 2)
    squared_sum = returns.pow(2).rolling(window=window, min_periods=min_periods).sum()
    realized = np.sqrt(squared_sum)
    if annualize_to_daily:
        realized *= math.sqrt(24.0 / float(window))
    return realized.fillna(0.0)


def compute_ewm_volatility(close: pd.Series, window: int) -> pd.Series:
    returns = _safe_log_returns(close)
    span = max(2, window)
    variance = returns.pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(variance).fillna(0.0)


def estimate_garch_like_volatility(
    close: pd.Series,
    *,
    alpha: float = DEFAULT_GARCH_ALPHA,
    beta: float = DEFAULT_GARCH_BETA,
    omega: float = DEFAULT_GARCH_OMEGA,
) -> pd.Series:
    if beta + alpha >= 1.0:
        raise ValueError("GARCH-like parameters must satisfy alpha + beta < 1.")
    returns = _safe_log_returns(close)
    variances = np.zeros(len(returns), dtype=float)
    prev_var = float(returns.iloc[0] ** 2) if not returns.empty else 0.0
    for idx, ret in enumerate(returns):
        prev_var = omega + alpha * (ret ** 2) + beta * prev_var
        variances[idx] = prev_var
    return pd.Series(np.sqrt(np.maximum(variances, 0.0)), index=returns.index).fillna(0.0)


def add_volatility_columns(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    realized_windows: Sequence[int] = DEFAULT_REALIZED_WINDOWS,
    include_garch: bool = True,
    periods_per_hour: int = 1,
) -> tuple[pd.DataFrame, List[str]]:
    if close_col not in df.columns:
        return df, []

    augmented = df.copy()
    close = pd.Series(augmented[close_col], dtype=float)
    added_columns: List[str] = []

    for window in realized_windows:
        effective_window = max(2, window * periods_per_hour)
        column_realized = f"volatility_realized_{window}h"
        augmented[column_realized] = compute_realized_volatility(close, effective_window)
        added_columns.append(column_realized)

        column_ewm = f"volatility_ewm_{window}h"
        augmented[column_ewm] = compute_ewm_volatility(close, effective_window)
        added_columns.append(column_ewm)

    if include_garch:
        column_garch = "volatility_garch_like"
        augmented[column_garch] = estimate_garch_like_volatility(close)
        added_columns.append(column_garch)

    return augmented, added_columns


def split_volatility_arrays(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    n_train: int,
    n_val: int,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    n_total = len(df)
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].to_numpy(dtype=np.float32, copy=False)
        arrays[f"{column}_train"] = values[:n_train]
        arrays[f"{column}_val"] = values[n_train:n_train + n_val]
        arrays[f"{column}_test"] = values[n_train + n_val : n_total]
    return arrays


def latest_volatility_snapshot(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    index: Optional[int] = None,
) -> Dict[str, float]:
    if index is None:
        index = len(df) - 1
    if index < 0:
        return {}

    result: Dict[str, float] = {}
    for column in columns:
        if column not in df.columns:
            continue
        try:
            value = df[column].iloc[index]
        except IndexError:
            continue
        if pd.notna(value):
            result[column] = float(value)
    return result