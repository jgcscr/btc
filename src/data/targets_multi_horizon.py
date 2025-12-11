"""Utilities for computing multi-horizon BTC return and direction targets."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


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

    return result
