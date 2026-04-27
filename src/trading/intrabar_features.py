from __future__ import annotations

import numpy as np
import pandas as pd


def _sign_persistence(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    signs = np.sign(values.to_numpy(dtype=float))
    non_zero = signs[signs != 0.0]
    if non_zero.size == 0:
        return 0.0
    return float(np.abs(non_zero.mean()))


def _mean_or_zero(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def _segment_return(bucket: pd.DataFrame) -> float:
    if bucket.empty:
        return 0.0
    open_ = pd.to_numeric(bucket["open"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    close = pd.to_numeric(bucket["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if open_.empty or close.empty:
        return 0.0
    first_open = open_.iloc[0]
    last_close = close.iloc[-1]
    if pd.isna(first_open) or pd.isna(last_close) or np.isclose(float(first_open), 0.0):
        return 0.0
    return float((last_close / first_open) - 1.0)


def compute_hourly_intrabar_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Intrabar frame missing required columns: {missing}")

    working = frame.copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
    working = working.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if working.empty:
        return pd.DataFrame(columns=["ts"])

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce")

    deltas = working["ts"].diff().dropna()
    if deltas.empty:
        return pd.DataFrame(columns=["ts"])
    median_minutes = float(deltas.dt.total_seconds().median() / 60.0)
    if median_minutes > 20.0:
        return pd.DataFrame(columns=["ts"])

    close = working["close"].replace(0.0, np.nan)
    open_ = working["open"]
    high = working["high"]
    low = working["low"]
    volume = working["volume"].replace(0.0, np.nan)
    taker_buy = working["taker_buy_base_volume"].fillna(0.0)

    working["hour_ts"] = working["ts"].dt.ceil("h")
    working["_log_ret_15m"] = np.log(close).diff()
    working["_bar_ret"] = working["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    working["_bar_direction"] = np.sign(working["_bar_ret"]).replace([np.inf, -np.inf], np.nan)
    spread = (high - low).replace(0.0, np.nan)
    upper_wick = high - np.maximum(open_, working["close"])
    lower_wick = np.minimum(open_, working["close"]) - low
    working["_wick_asym"] = ((upper_wick - lower_wick) / spread).replace([np.inf, -np.inf], np.nan)
    working["_imbalance"] = ((2.0 * taker_buy - volume) / volume).replace([np.inf, -np.inf], np.nan)
    working["_abs_path_step"] = working["close"].diff().abs()

    grouped = working.groupby("hour_ts", as_index=False)
    hourly = grouped.agg(
        intrabar_path_high=("high", "max"),
        intrabar_path_low=("low", "min"),
        intrabar_close_last=("close", "last"),
        intrabar_open_first=("open", "first"),
        intrabar_realized_vol_15m=("_log_ret_15m", "std"),
        intrabar_return_dispersion_15m=("_bar_ret", "std"),
        intrabar_taker_imbalance_mean=("_imbalance", "mean"),
        intrabar_wick_asymmetry_mean=("_wick_asym", "mean"),
        intrabar_volume_sum=("volume", "sum"),
        intrabar_quote_volume_sum=("quote_volume", "sum"),
        intrabar_num_trades_mean=("num_trades", "mean"),
        intrabar_taker_buy_base_sum=("taker_buy_base_volume", "sum"),
        intrabar_abs_path_sum=("_abs_path_step", "sum"),
    )
    hourly["intrabar_path_range"] = (
        (hourly["intrabar_path_high"] - hourly["intrabar_path_low"])
        / hourly["intrabar_close_last"].replace(0.0, np.nan)
    )
    net_move = (hourly["intrabar_close_last"] - hourly["intrabar_open_first"]).abs()
    hourly["intrabar_path_efficiency_1h"] = (
        net_move / hourly["intrabar_abs_path_sum"].replace(0.0, np.nan)
    )
    hourly["intrabar_taker_buy_ratio"] = (
        hourly["intrabar_taker_buy_base_sum"] / hourly["intrabar_volume_sum"].replace(0.0, np.nan)
    )

    persist_rows = []
    for hour_ts, bucket in grouped:
        ordered_bucket = bucket.sort_values("ts").reset_index(drop=True)
        split_idx = max(int(len(ordered_bucket) / 2), 1)
        first_half = ordered_bucket.iloc[:split_idx]
        second_half = ordered_bucket.iloc[split_idx:]
        if second_half.empty:
            second_half = first_half

        early_imbalance = _mean_or_zero(first_half["_imbalance"])
        late_imbalance = _mean_or_zero(second_half["_imbalance"])
        first_half_ret = _segment_return(first_half)
        second_half_ret = _segment_return(second_half)
        reversal_score = 0.0
        if first_half_ret * second_half_ret < 0.0:
            reversal_score = float(np.sign(second_half_ret) * (abs(first_half_ret) + abs(second_half_ret)))

        persist_rows.append(
            {
                "hour_ts": hour_ts,
                "intrabar_taker_imbalance_persistence": _sign_persistence(bucket["_imbalance"]),
                "intrabar_wick_asymmetry_persistence": _sign_persistence(bucket["_wick_asym"]),
                "intrabar_directional_persistence_1h": _sign_persistence(bucket["_bar_direction"]),
                "intrabar_taker_imbalance_early_late_delta": late_imbalance - early_imbalance,
                "intrabar_reversal_score_1h": reversal_score,
                "intrabar_wick_asymmetry_shift": _mean_or_zero(second_half["_wick_asym"]) - _mean_or_zero(first_half["_wick_asym"]),
            }
        )
    hourly = hourly.merge(pd.DataFrame(persist_rows), on="hour_ts", how="left")
    hourly = hourly.sort_values("hour_ts").reset_index(drop=True)

    rv_short = hourly["intrabar_realized_vol_15m"].rolling(window=6, min_periods=3).mean()
    rv_long = hourly["intrabar_realized_vol_15m"].rolling(window=24, min_periods=8).mean().replace(0.0, np.nan)
    hourly["intrabar_vol_term_structure_6h_24h"] = (rv_short / rv_long).replace([np.inf, -np.inf], np.nan)

    vol_mean = hourly["intrabar_volume_sum"].rolling(window=24, min_periods=8).mean()
    vol_std = hourly["intrabar_volume_sum"].rolling(window=24, min_periods=8).std(ddof=0).replace(0.0, np.nan)
    volume_z = ((hourly["intrabar_volume_sum"] - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)
    hourly["intrabar_volume_regime_zscore_24h"] = volume_z
    hourly["intrabar_volume_regime_transition"] = volume_z.diff().abs()
    hourly["intrabar_flow_acceleration_3h"] = hourly["intrabar_taker_imbalance_mean"].diff().rolling(window=3, min_periods=1).mean()
    hourly["intrabar_breakout_failure_1h"] = (
        hourly["intrabar_path_range"].clip(lower=0.0)
        * (1.0 - hourly["intrabar_path_efficiency_1h"].clip(lower=0.0, upper=1.0))
    )
    dispersion_3h = hourly["intrabar_return_dispersion_15m"].rolling(window=3, min_periods=2).mean().replace(0.0, np.nan)
    dispersion_6h = hourly["intrabar_return_dispersion_15m"].rolling(window=6, min_periods=3).mean().replace(0.0, np.nan)
    hourly["intrabar_return_dispersion_regime_3h"] = (
        hourly["intrabar_return_dispersion_15m"] / dispersion_3h
    ).replace([np.inf, -np.inf], np.nan)
    hourly["intrabar_return_dispersion_regime_6h"] = (
        hourly["intrabar_return_dispersion_15m"] / dispersion_6h
    ).replace([np.inf, -np.inf], np.nan)

    intrabar_cols = [column for column in hourly.columns if column.startswith("intrabar_")]
    for column in intrabar_cols:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    output = hourly.rename(columns={"hour_ts": "ts"})
    return output.loc[:, ["ts", *intrabar_cols]].copy()


__all__ = ["compute_hourly_intrabar_features"]