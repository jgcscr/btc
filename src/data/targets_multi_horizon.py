"""Utilities for computing multi-horizon BTC return and direction targets."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

RANGE_TARGET_HORIZONS: tuple[int, ...] = (4, 8, 12)


def add_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Return a copy of *df* with forward log-return and direction targets added.

    Parameters
    ----------
    df: pd.DataFrame
        Time-ordered 1h OHLCV dataframe that must contain the *price_col* column and
        optionally a ``ts`` column used purely for sorting.
    horizons: Iterable[int], optional
        Collection of horizon lengths (in hours) to compute. Defaults to ``[1, 4]``.
    price_col: str
        Column name containing the closing price used for return computation.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with new columns ``ret_{H}h`` and ``dir_{H}h``
        appended for each requested horizon ``H``. Trailing rows that do not have
        sufficient future data receive ``NaN`` targets so that callers can drop them
        prior to training or evaluation.
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataframe.")

    if horizons is None:
        horizons = [1, 4]

    horizons_list: List[int] = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons_list:
        raise ValueError("At least one positive horizon must be provided.")

    df_sorted = df.sort_values("ts" if "ts" in df.columns else df.index).reset_index(drop=True)
    close = df_sorted[price_col].astype(float)
    log_close = np.log(close)

    result = df_sorted.copy()

    for horizon in horizons_list:
        future_log = log_close.shift(-horizon)
        ret_h = future_log - log_close
        dir_h = np.where(ret_h.notna(), (ret_h > 0).astype(int), np.nan)

        result[f"ret_{horizon}h"] = ret_h
        result[f"dir_{horizon}h"] = dir_h

    def _extrema_targets(window: int) -> tuple[pd.Series, pd.Series]:
        if window <= 0:
            raise ValueError("range target horizon must be positive")
        deltas = [
            log_close.shift(-step) - log_close
            for step in range(1, window + 1)
        ]
        if not deltas:
            return pd.Series(np.nan, index=log_close.index), pd.Series(np.nan, index=log_close.index)
        matrix = pd.concat(deltas, axis=1)
        valid = matrix.notna().any(axis=1)
        max_forward = pd.Series(np.nan, index=log_close.index, dtype=float)
        min_forward = pd.Series(np.nan, index=log_close.index, dtype=float)
        if valid.any():
            max_forward.loc[valid] = matrix.max(axis=1, skipna=True).loc[valid]
            min_forward.loc[valid] = matrix.min(axis=1, skipna=True).loc[valid]
        return max_forward, min_forward

    for range_horizon in RANGE_TARGET_HORIZONS:
        if range_horizon not in horizons_list:
            # Only compute range targets when corresponding return horizon is available.
            continue
        max_forward, min_forward = _extrema_targets(range_horizon)
        result[f"ret_max_{range_horizon}h"] = max_forward
        result[f"ret_min_{range_horizon}h"] = min_forward

    return result


def add_trend_ignition_label(
    df: pd.DataFrame,
    *,
    horizon_hours: int = 6,
    threshold: float = 0.01,
    price_col: str = "close",
    label_col: str = "trend_ignition_6h",
) -> pd.DataFrame:
    """Append a binary "trend ignition" label based on forward log-return extremes.

    The label is ``1`` when the maximum *or* minimum forward log return within the
    ``horizon_hours`` window crosses ``±threshold``. Rows without sufficient future
    data retain ``NaN`` so callers can drop them prior to training.
    """

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataframe.")

    df_sorted = df.sort_values("ts" if "ts" in df.columns else df.index).reset_index(drop=True)
    close = df_sorted[price_col].astype(float)
    log_close = np.log(close)

    forward_deltas = []
    for step in range(1, int(horizon_hours) + 1):
        delta = log_close.shift(-step) - log_close
        forward_deltas.append(delta.rename(step))

    if not forward_deltas:
        df_sorted[label_col] = np.nan
        return df_sorted

    matrix = pd.concat(forward_deltas, axis=1)
    max_forward = matrix.max(axis=1, skipna=True)
    min_forward = matrix.min(axis=1, skipna=True)

    trigger_mask = (matrix.notna().any(axis=1))
    label = pd.Series(np.nan, index=df_sorted.index, dtype=float)
    label.loc[trigger_mask] = (
        (max_forward >= threshold) | (min_forward <= -threshold)
    ).loc[trigger_mask].astype(int)

    df_sorted[label_col] = label
    return df_sorted
