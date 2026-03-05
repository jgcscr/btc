"""Multi-horizon paper trading loop.

This script supersedes the 4h-only implementation and can simulate
paper trading for any supported prediction horizon by loading the
appropriate dataset/model pair and logging per-bar results with
volatility-gating metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config_trading import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    format_ts_iso,
    load_models,
    populate_sequence_cache_from_prepared,
)
from src.trading.volatility import DEFAULT_REALIZED_WINDOWS, add_volatility_columns, latest_volatility_snapshot


SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass
class HorizonDefaults:
    dataset_path: str
    return_key: Optional[str]
    reg_model_dir: str
    dir_model_dir: str
    output_dir: str
    p_up_min: float
    ret_min: float
    volatility_metric: str
    volatility_ceiling: float
    volatility_mult: float
    expected_value_multiplier: float = 1.0


HORIZON_DEFAULTS: Dict[float, HorizonDefaults] = {
    0.25: HorizonDefaults(
        dataset_path="artifacts/datasets/btc_features_15m_splits.npz",
        return_key=None,
        reg_model_dir="artifacts/models/xgb_ret15m_v1",
        dir_model_dir="artifacts/models/xgb_dir15m_v1",
        output_dir="artifacts/analysis/paper_trade_15m_v1",
        p_up_min=0.55,
        ret_min=0.0002,
        volatility_metric="volatility_realized_24h",
        volatility_ceiling=0.04,
        volatility_mult=1.25,
    ),
    1.0: HorizonDefaults(
        dataset_path="artifacts/datasets/btc_features_1h_splits.npz",
        return_key=None,
        reg_model_dir="artifacts/models/xgb_ret1h_v2",
        dir_model_dir="artifacts/models/xgb_dir1h_v2",
        output_dir="artifacts/analysis/paper_trade_1h_v2",
        p_up_min=0.62,
        ret_min=0.00045,
        volatility_metric="volatility_realized_24h",
        volatility_ceiling=0.032,
        volatility_mult=1.25,
    ),
    4.0: HorizonDefaults(
        dataset_path="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        return_key="ret4h",
        reg_model_dir="artifacts/models/xgb_ret4h_v2",
        dir_model_dir="artifacts/models/xgb_dir4h_v2",
        output_dir="artifacts/analysis/paper_trade_4h_v2",
        p_up_min=0.58,
        ret_min=0.00065,
        volatility_metric="volatility_realized_24h",
        volatility_ceiling=0.03,
        volatility_mult=1.25,
    ),
    8.0: HorizonDefaults(
        dataset_path="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        return_key="ret8h",
        reg_model_dir="artifacts/models/xgb_ret8h_v2",
        dir_model_dir="artifacts/models/xgb_dir8h_v2",
        output_dir="artifacts/analysis/paper_trade_8h_v2",
        p_up_min=0.605,
        ret_min=0.0009,
        volatility_metric="volatility_realized_72h",
        volatility_ceiling=0.18,
        volatility_mult=1.25,
    ),
    12.0: HorizonDefaults(
        dataset_path="artifacts/datasets/btc_features_multi_horizon_splits.npz",
        return_key="ret12h",
        reg_model_dir="artifacts/models/xgb_ret12h_v2",
        dir_model_dir="artifacts/models/xgb_dir12h_v2",
        output_dir="artifacts/analysis/paper_trade_12h_v2",
        p_up_min=0.57,
        ret_min=0.0013,
        volatility_metric="volatility_ewm_72h",
        volatility_ceiling=0.045,
        volatility_mult=1.1,
    ),
}


def _format_horizon_label(hours: float) -> str:
    if hours >= 1.0:
        if float(hours).is_integer():
            return f"{int(hours)}h"
        return f"{hours:g}h"
    minutes = int(round(hours * 60))
    return f"{minutes}m"


def _reg_model_filename(label: str) -> str:
    return f"xgb_ret{label}_model.json"


def _dir_model_filename(label: str) -> str:
    return f"xgb_dir{label}_model.json"


def _infer_frequency(ts_values: pd.Series) -> pd.Timedelta:
    diffs = ts_values.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)
    median_ns = int(np.median(diffs.values.astype("timedelta64[ns]")))
    if median_ns <= 0:
        return pd.Timedelta(hours=1)
    return pd.to_timedelta(median_ns, unit="ns")


def _scale_split_lengths(total_len: int, split_lengths: Mapping[str, int]) -> Dict[str, int]:
    expected = sum(split_lengths.get(split, 0) for split in SPLITS)
    if expected <= 0 or expected == total_len:
        return {split: split_lengths.get(split, 0) for split in SPLITS}
    scale = total_len / expected
    adjusted: Dict[str, int] = {}
    cumulative = 0
    for split in SPLITS:
        raw = split_lengths.get(split, 0)
        length = int(round(raw * scale))
        adjusted[split] = max(length, 0)
        cumulative += adjusted[split]
    adjusted["test"] = adjusted.get("test", 0) + (total_len - cumulative)
    adjusted["test"] = max(adjusted["test"], 0)
    return adjusted


def _inject_volatility_columns(df: pd.DataFrame, npz: Mapping[str, np.ndarray]) -> List[str]:
    vol_columns: List[str] = []
    for base in (
        "volatility_realized_24h",
        "volatility_realized_72h",
        "volatility_ewm_24h",
        "volatility_ewm_72h",
        "volatility_garch_like",
    ):
        arrays: List[np.ndarray] = []
        for split in SPLITS:
            key = f"{base}_{split}"
            if key not in npz:
                arrays = []
                break
            arrays.append(np.asarray(npz[key], dtype=float))
        if not arrays:
            continue
        merged = np.concatenate(arrays, axis=0)
        if len(merged) != len(df):
            continue
        df[base] = merged
        vol_columns.append(base)
    return vol_columns


def _load_return_series(npz: Mapping[str, np.ndarray], return_key: Optional[str]) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for split in SPLITS:
        if return_key:
            key = f"y_{return_key}_{split}"
        else:
            key = f"y_{split}"
        if key not in npz:
            raise KeyError(f"Dataset is missing return array: {key}")
        arrays.append(np.asarray(npz[key], dtype=float))
    return np.concatenate(arrays, axis=0)


def _load_prepared_from_npz(dataset_path: str, return_key: Optional[str]) -> tuple[PreparedData, np.ndarray, Dict[str, int]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset npz not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        if "feature_names" not in data.files:
            raise KeyError("Dataset NPZ missing feature_names")
        feature_names = data["feature_names"].tolist()

        split_lengths: Dict[str, int] = {split: 0 for split in SPLITS}
        feature_blocks: List[np.ndarray] = []
        for split in SPLITS:
            key = f"X_{split}"
            if key not in data.files:
                continue
            block = np.array(data[key], dtype=float, copy=False)
            split_lengths[split] = len(block)
            if len(block):
                feature_blocks.append(block)

        if not feature_blocks:
            raise RuntimeError(f"Dataset {dataset_path} does not contain any feature splits.")

        X_all = np.concatenate(feature_blocks, axis=0)
        df = pd.DataFrame(X_all, columns=feature_names)

        ts_values: Optional[pd.Series]
        if "ts_all" in data.files:
            ts_values = pd.to_datetime(data["ts_all"], utc=True)
        else:
            ts_segments: List[np.ndarray] = []
            for split in SPLITS:
                key = f"ts_{split}"
                if key in data.files:
                    ts_segments.append(data[key])
            if ts_segments:
                ts_values = pd.to_datetime(np.concatenate(ts_segments, axis=0), utc=True)
            else:
                ts_values = None

        if ts_values is None or len(ts_values) != len(df):
            ts_values = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="H", tz="UTC")

        df.insert(0, "ts", ts_values)

        if "close_all" in data.files and len(data["close_all"]) == len(df):
            df["close"] = data["close_all"]

        vol_columns = _inject_volatility_columns(df, data)

        freq = _infer_frequency(df["ts"])
        periods_per_hour = max(int(round(pd.Timedelta(hours=1) / freq)), 1)
        df, derived_vols = add_volatility_columns(
            df,
            realized_windows=DEFAULT_REALIZED_WINDOWS,
            periods_per_hour=periods_per_hour,
        )
        combined_vol_columns = sorted(set(vol_columns + derived_vols + [col for col in df.columns if col.startswith("volatility_")]))

        df = df.sort_values("ts").reset_index(drop=True)

        ret_series = _load_return_series(data, return_key)

    total_len = min(len(ret_series), len(df))
    if total_len == 0:
        raise RuntimeError("Aligned dataset has zero rows; cannot run paper trading.")
    if total_len != len(df) or total_len != len(ret_series):
        print(
            f"Info: truncating to {total_len} rows to align features ({len(df)}) and returns ({len(ret_series)})."
        )

    df = df.iloc[:total_len].reset_index(drop=True)
    ret_series = ret_series[:total_len]
    X_all_ordered = df[feature_names].copy()

    adjusted_lengths = _scale_split_lengths(total_len, split_lengths)
    train_len = max(adjusted_lengths.get("train", 0), 1)

    scaler = StandardScaler()
    scaler.fit(X_all_ordered.iloc[:train_len])

    prepared = PreparedData(
        df_all=df,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names,
        volatility_columns=combined_vol_columns,
    )

    return prepared, ret_series, adjusted_lengths


def _rolling_percentile(values: pd.Series, window: int, min_periods: Optional[int] = None) -> np.ndarray:
    if window <= 1:
        return np.ones(len(values), dtype=float)

    if min_periods is None:
        min_periods = max(10, window // 10)

    def percentile_rank(window_values: np.ndarray) -> float:
        current = window_values[-1]
        if np.isnan(current):
            return math.nan
        valid = window_values[~np.isnan(window_values)]
        if valid.size == 0:
            return math.nan
        return float((valid <= current).sum()) / float(valid.size)

    rolled = values.rolling(window=window, min_periods=min_periods).apply(percentile_rank, raw=True)
    return rolled.to_numpy()


def _compute_volatility_percentiles(
    df: pd.DataFrame,
    metric: str,
    freq: pd.Timedelta,
    window_days: float,
) -> np.ndarray:
    if metric not in df.columns:
        raise KeyError(f"Volatility metric '{metric}' not found in dataset; cannot compute percentiles.")
    if window_days <= 0:
        raise ValueError("volatility_percentile_days must be positive when using percentile gating.")
    window_td = pd.Timedelta(days=window_days)
    if freq <= pd.Timedelta(0):
        raise ValueError("Inferred frequency must be positive to compute volatility percentiles.")
    window = max(int(round(window_td / freq)), 1)
    series = df[metric].astype(float)
    return _rolling_percentile(series, window=window)


def _compute_index_range(total_len: int, use_test_split: bool, split_lengths: Mapping[str, int]) -> range:
    if not use_test_split:
        return range(total_len)
    start = split_lengths.get("train", 0) + split_lengths.get("val", 0)
    start = min(start, total_len)
    return range(start, total_len)


def _compute_trade_metrics(trade_pnls: Iterable[float], equity_log_series: np.ndarray) -> Dict[str, float]:
    trade_array = np.array(list(trade_pnls), dtype=float)
    n_trades = int(len(trade_array))
    if n_trades:
        hit_rate = float((trade_array > 0.0).mean())
        avg_pnl = float(trade_array.mean())
    else:
        hit_rate = math.nan
        avg_pnl = math.nan

    cum_ret = float(equity_log_series[-1] if equity_log_series.size else 0.0)
    if equity_log_series.size:
        peak = np.maximum.accumulate(equity_log_series)
        drawdowns = equity_log_series - peak
        max_drawdown = float(drawdowns.min())
    else:
        max_drawdown = 0.0

    if equity_log_series.size > 1:
        net_returns = np.diff(equity_log_series)
        active = net_returns[net_returns != 0.0]
        if active.size > 1:
            mu = float(active.mean())
            std = float(active.std(ddof=1))
            sharpe_like = float(mu / std) if std > 0 else math.nan
        else:
            sharpe_like = math.nan
    else:
        sharpe_like = math.nan

    return {
        "n_trades": float(n_trades),
        "hit_rate": hit_rate,
        "avg_pnl_trade": avg_pnl,
        "cum_ret": cum_ret,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a position-aware paper-trading loop for flexible horizons "
            "(supports 15m / 4h / 8h / 12h by default)."
        ),
    )
    parser.add_argument("--horizon", type=float, default=4.0, help="Prediction horizon in hours (e.g. 0.25, 4, 8, 12).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional override for the dataset NPZ path.")
    parser.add_argument("--return-key", type=str, default=None, help="Prefix for return arrays (e.g. ret4h => y_ret4h_*).")
    parser.add_argument("--reg-model-dir", type=str, default=None, help="Directory containing the regression model JSON.")
    parser.add_argument("--dir-model-dir", type=str, default=None, help="Directory containing the direction model JSON.")
    parser.add_argument("--p-up-min", type=float, default=None, help="Ensemble probability threshold override.")
    parser.add_argument("--ret-min", type=float, default=None, help="Return threshold override.")
    parser.add_argument(
        "--expected-value-multiplier",
        type=float,
        default=None,
        help="Optional scaling factor applied to expected-value outputs before logging.",
    )
    parser.add_argument("--volatility-metric", type=str, default=None, help="Volatility metric key (default depends on horizon).")
    parser.add_argument("--volatility-ceiling", type=float, default=None, help="Volatility ceiling for gating.")
    parser.add_argument("--volatility-mult", type=float, default=None, help="Multiplier applied when volatility exceeds the ceiling.")
    parser.add_argument(
        "--volatility-gating-mode",
        choices=("ceiling", "percentile"),
        default="ceiling",
        help="Choose between hard-ceiling gating or percentile-based scaling.",
    )
    parser.add_argument(
        "--volatility-percentile-days",
        type=float,
        default=90.0,
        help="Trailing window (in days) for percentile-based volatility gating.",
    )
    parser.add_argument(
        "--volatility-calm-percentile",
        type=float,
        default=0.7,
        help="Percentile threshold below which no volatility scaling is applied.",
    )
    parser.add_argument(
        "--volatility-extreme-percentile",
        type=float,
        default=0.9,
        help="Percentile threshold above which extreme volatility logic triggers.",
    )
    parser.add_argument(
        "--volatility-elevated-scale",
        type=float,
        default=0.5,
        help="Scaling factor applied to p_up_min inside the elevated percentile band.",
    )
    parser.add_argument(
        "--volatility-extreme-scale",
        type=float,
        default=1.0,
        help="Additional scaling applied to p_up_min once the extreme percentile is breached.",
    )
    parser.add_argument(
        "--volatility-ret-scale",
        type=float,
        default=0.0,
        help="Optional multiplier applied to ret_min in elevated/extreme regimes.",
    )
    parser.add_argument(
        "--volatility-block-extreme",
        dest="volatility_block_extreme",
        action="store_true",
        help="Force trades to flat when volatility percentile exceeds the extreme threshold.",
    )
    parser.add_argument(
        "--no-volatility-block-extreme",
        dest="volatility_block_extreme",
        action="store_false",
        help="Allow trades even when volatility percentile exceeds the extreme threshold.",
    )
    parser.set_defaults(volatility_block_extreme=True)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS, help="Per-trade fee in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS, help="Per-trade slippage in basis points.")
    parser.add_argument("--use-test-split", action="store_true", help="Restrict the simulation to the test split only.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the per-bar CSV log.")
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=None,
        help="Optional calibrated thresholds JSON to auto-apply per-horizon settings.",
    )
    return parser


def _normalize_horizon_key(hours: float) -> str:
    if float(hours).is_integer():
        return str(int(hours))
    return format(hours, "g")


def _load_threshold_profile(path: Optional[str], horizon: float) -> Dict[str, Any]:
    if not path:
        return {}
    resolved = os.path.expanduser(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Thresholds JSON not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    profile: Dict[str, Any] = {}
    base = payload.get("thresholds")
    if isinstance(base, dict):
        profile.update(base)
    horizons = payload.get("horizons") or {}
    horizon_key = _normalize_horizon_key(horizon)
    horizon_overrides = horizons.get(horizon_key)
    if isinstance(horizon_overrides, dict):
        profile.update(horizon_overrides)
    return profile


def _maybe_set_from_sources(
    args: argparse.Namespace,
    attr: str,
    profile: Mapping[str, Any],
    profile_key: Optional[str],
    default_value: Any,
) -> None:
    if getattr(args, attr) is not None:
        return
    if profile_key and profile_key in profile:
        setattr(args, attr, profile[profile_key])
        return
    if default_value is not None:
        setattr(args, attr, default_value)


def _apply_horizon_defaults(args: argparse.Namespace) -> tuple[float, str]:
    horizon = round(float(args.horizon), 6)
    label = _format_horizon_label(horizon)
    defaults = HORIZON_DEFAULTS.get(horizon)
    profile = _load_threshold_profile(args.thresholds_json, horizon)

    if defaults:
        if args.return_key is None and defaults.return_key is not None:
            args.return_key = defaults.return_key
    _maybe_set_from_sources(args, "dataset_path", profile, "dataset_path", defaults.dataset_path if defaults else None)
    _maybe_set_from_sources(args, "reg_model_dir", profile, "reg_model_dir", defaults.reg_model_dir if defaults else None)
    _maybe_set_from_sources(args, "dir_model_dir", profile, "dir_model_dir", defaults.dir_model_dir if defaults else None)
    _maybe_set_from_sources(args, "p_up_min", profile, "p_up_min", defaults.p_up_min if defaults else None)
    _maybe_set_from_sources(args, "ret_min", profile, "ret_min", defaults.ret_min if defaults else None)
    _maybe_set_from_sources(args, "volatility_metric", profile, "volatility_metric", defaults.volatility_metric if defaults else None)
    _maybe_set_from_sources(
        args,
        "volatility_ceiling",
        profile,
        "volatility_ceiling",
        defaults.volatility_ceiling if defaults else None,
    )
    _maybe_set_from_sources(args, "volatility_mult", profile, "volatility_mult", defaults.volatility_mult if defaults else None)
    _maybe_set_from_sources(
        args,
        "expected_value_multiplier",
        profile,
        "expected_value_multiplier",
        defaults.expected_value_multiplier if defaults else 1.0,
    )

    if defaults and not args.output_dir:
        args.output_dir = defaults.output_dir

    if not args.dataset_path or not args.reg_model_dir or not args.dir_model_dir:
        raise ValueError(
            "Horizon configuration requires dataset/model directories via defaults, thresholds JSON, or explicit overrides."
        )
    if args.p_up_min is None or args.ret_min is None:
        raise ValueError("p_up_min and ret_min must be provided via CLI, thresholds JSON, or defaults.")
    if args.expected_value_multiplier is None:
        args.expected_value_multiplier = 1.0
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts/analysis", f"paper_trade_{label}")
    return horizon, label


def run(args: argparse.Namespace) -> None:
    horizon, label = _apply_horizon_defaults(args)

    prepared, ret_series, split_lengths = _load_prepared_from_npz(args.dataset_path, args.return_key)

    reg_model_path = os.path.join(args.reg_model_dir, _reg_model_filename(label))
    dir_model_path = os.path.join(args.dir_model_dir, _dir_model_filename(label))
    models = load_models(
        reg_model_path=reg_model_path,
        dir_model_path=dir_model_path,
    )
    populate_sequence_cache_from_prepared(prepared, models)

    freq = _infer_frequency(prepared.df_all["ts"])

    volatility_policy: Optional[Dict[str, Any]] = None
    if args.volatility_gating_mode == "percentile":
        if not args.volatility_metric:
            raise ValueError("Percentile gating requires --volatility-metric to be set.")
        percentiles = _compute_volatility_percentiles(
            prepared.df_all,
            args.volatility_metric,
            freq=freq,
            window_days=args.volatility_percentile_days,
        )
        volatility_policy = {
            "mode": "percentile",
            "volatility_metric": args.volatility_metric,
            "percentiles": percentiles,
            "calm_pct": args.volatility_calm_percentile,
            "extreme_pct": args.volatility_extreme_percentile,
            "elevated_scale": args.volatility_elevated_scale,
            "extreme_scale": args.volatility_extreme_scale,
            "ret_scale": args.volatility_ret_scale,
            "block_extreme": bool(args.volatility_block_extreme),
        }
    elif args.volatility_metric and args.volatility_ceiling is not None:
        volatility_policy = {
            "mode": "ceiling",
            "volatility_metric": args.volatility_metric,
            "volatility_ceiling": args.volatility_ceiling,
            "volatility_mult": args.volatility_mult or 1.25,
        }

    cost_per_trade = (args.fee_bps + args.slippage_bps) / 10_000.0
    ev_multiplier = float(args.expected_value_multiplier or 1.0)
    total_len = len(ret_series)
    idx_range = _compute_index_range(total_len, args.use_test_split, split_lengths)

    position = 0
    equity_log = 0.0
    equity_log_series: List[float] = []
    trade_pnls: List[float] = []
    entry_equity_log: Optional[float] = None

    ts_list: List[str] = []
    ret_list: List[float] = []
    p_up_list: List[float] = []
    ret_pred_list: List[float] = []
    signal_list: List[int] = []
    position_list: List[int] = []
    ret_net_list: List[float] = []
    expected_value_list: List[float] = []
    vol_flag_list: List[bool] = []
    vol_metric_list: List[Optional[str]] = []
    vol_current_list: List[Optional[float]] = []
    vol_ceiling_list: List[Optional[float]] = []
    vol_multiplier_list: List[Optional[float]] = []
    vol_percentile_list: List[Optional[float]] = []
    p_up_min_effective_list: List[float] = []

    for i in idx_range:
        snapshot = latest_volatility_snapshot(
            prepared.df_all,
            prepared.volatility_columns or [],
            index=i,
        )
        signal = compute_signal_for_index(
            prepared=prepared,
            index=i,
            models=models,
            p_up_min=args.p_up_min,
            ret_min=args.ret_min,
            volatility_snapshot=snapshot,
            volatility_policy=volatility_policy,
        )

        ts_value = format_ts_iso(prepared.df_all["ts"].iloc[i])
        p_up = float(signal.get("p_up", 0.0))
        ret_pred = float(signal.get("ret_pred", 0.0))
        expected_value = float(signal.get("expected_value", ret_pred)) * ev_multiplier
        signal_ens = int(signal.get("signal_ensemble", 0))
        ret_bar = float(ret_series[i])

        entry = position == 0 and signal_ens == 1
        exit_ = position == 1 and signal_ens == 0
        if entry:
            entry_equity_log = equity_log
        position = signal_ens

        gross_ret = ret_bar if position == 1 else 0.0
        cost = 0.0
        if entry:
            cost -= cost_per_trade
        if exit_:
            cost -= cost_per_trade
        net_ret = gross_ret + cost
        equity_log += net_ret
        equity_log_series.append(equity_log)
        if exit_ and entry_equity_log is not None:
            trade_pnls.append(equity_log - entry_equity_log)
            entry_equity_log = None

        vol_block = signal.get("volatility") or {}
        vol_flag = bool(signal.get("volatility_flag", False))

        ts_list.append(ts_value)
        ret_list.append(ret_bar)
        p_up_list.append(p_up)
        ret_pred_list.append(ret_pred)
        signal_list.append(signal_ens)
        position_list.append(position)
        ret_net_list.append(net_ret)
        expected_value_list.append(expected_value)
        vol_flag_list.append(vol_flag)
        vol_metric_list.append(vol_block.get("metric"))
        vol_current_list.append(vol_block.get("current"))
        vol_ceiling_list.append(vol_block.get("ceiling"))
        vol_multiplier_list.append(vol_block.get("multiplier"))
        vol_percentile_list.append(vol_block.get("percentile"))
        p_up_min_effective_list.append(float(vol_block.get("p_up_min_effective", args.p_up_min)))

    metrics = _compute_trade_metrics(trade_pnls, np.array(equity_log_series, dtype=float))
    print(f"=== {label} ensemble paper-trading ===")
    print(f"n_trades: {metrics['n_trades']:.0f}")
    print(f"hit_rate: {metrics['hit_rate']:.3f}")
    print(f"avg_pnl_trade: {metrics['avg_pnl_trade']:.6f}")
    print(f"cum_ret (log): {metrics['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics['max_drawdown']:.4f}")
    sharpe_like = metrics['sharpe_like']
    if math.isnan(sharpe_like):
        print("sharpe_like: nan")
    else:
        print(f"sharpe_like: {sharpe_like:.3f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        equity_curve = np.exp(np.array(equity_log_series, dtype=float))
        prefix = label
        df_out = pd.DataFrame(
            {
                "ts": ts_list,
                f"ret_{prefix}": ret_list,
                f"p_up_{prefix}": p_up_list,
                f"ret_pred_{prefix}": ret_pred_list,
                f"expected_value_{prefix}": expected_value_list,
                f"signal_ensemble_{prefix}": signal_list,
                f"position_{prefix}": position_list,
                f"ret_net_{prefix}": ret_net_list,
                f"equity_{prefix}": equity_curve,
                f"volatility_flag_{prefix}": vol_flag_list,
                f"volatility_metric_{prefix}": vol_metric_list,
                f"volatility_current_{prefix}": vol_current_list,
                f"volatility_ceiling_{prefix}": vol_ceiling_list,
                f"volatility_multiplier_{prefix}": vol_multiplier_list,
                f"volatility_percentile_{prefix}": vol_percentile_list,
                f"p_up_min_effective_{prefix}": p_up_min_effective_list,
            }
        )
        out_path = os.path.join(args.output_dir, f"paper_trade_{label}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"Saved paper-trade log to {out_path}")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()