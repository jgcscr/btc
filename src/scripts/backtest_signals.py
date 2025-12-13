import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config_trading import (
    DEFAULT_DIR_MODEL_DIR_1H,
    DEFAULT_FEE_BPS,
    DEFAULT_P_UP_MIN,
    DEFAULT_REG_MODEL_DIR_1H,
    DEFAULT_RET_MIN,
    DEFAULT_SLIPPAGE_BPS,
    OPTUNA_DIR_MODEL_DIR_1H,
    OPTUNA_P_UP_MIN_1H,
    OPTUNA_REG_MODEL_DIR_1H,
    OPTUNA_RET_MIN_1H,
)
from src.trading.ensembles import parse_weight_spec
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    load_models,
    populate_sequence_cache_from_prepared,
    prepare_data_for_signals,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest ensemble and direction-only signals using the same logic "
            "as run_signal_once over a historical period."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression NPZ file (used for feature names and split ratios).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default=DEFAULT_REG_MODEL_DIR_1H,
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default=DEFAULT_DIR_MODEL_DIR_1H,
        help="Directory containing direction model JSON (xgb_dir1h_model.json).",
    )
    parser.add_argument(
        "--lstm-dir-model",
        type=str,
        default=None,
        help="Optional directory containing an LSTM direction model (model.pt, summary.json).",
    )
    parser.add_argument(
        "--transformer-dir-model",
        type=str,
        default=None,
        help="Optional directory containing a transformer direction model (model.pt, summary.json).",
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated weights for direction models, e.g. transformer:1,lstm:1,xgb:1. "
            "Weights are normalized automatically; unspecified models default to equal weighting."
        ),
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Ensemble threshold for P(up).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Ensemble threshold for predicted ret_1h.",
    )
    parser.add_argument(
        "--use-optuna-profile",
        action="store_true",
        help="Override default 1h model dirs and thresholds with the Optuna-tuned profile.",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help="Per-trade fee in basis points.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help="Per-trade slippage in basis points.",
    )
    parser.add_argument(
        "--use-test-split",
        action="store_true",
        help=(
            "If set, restrict the backtest to the test split defined by "
            "the original 70/15/15 train/val/test fractions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write a per-bar CSV log of the backtest.",
    )
    return parser.parse_args()


def _apply_optuna_profile(args: argparse.Namespace) -> None:
    if not getattr(args, "use_optuna_profile", False):
        return

    if args.reg_model_dir == DEFAULT_REG_MODEL_DIR_1H:
        args.reg_model_dir = OPTUNA_REG_MODEL_DIR_1H

    if args.dir_model_dir == DEFAULT_DIR_MODEL_DIR_1H:
        args.dir_model_dir = OPTUNA_DIR_MODEL_DIR_1H

    if args.p_up_min == DEFAULT_P_UP_MIN:
        args.p_up_min = OPTUNA_P_UP_MIN_1H

    if args.ret_min == DEFAULT_RET_MIN:
        args.ret_min = OPTUNA_RET_MIN_1H

    print(
        (
            "Optuna profile active (reg_model_dir="
            f"{args.reg_model_dir}, dir_model_dir={args.dir_model_dir}, "
            f"p_up_min={args.p_up_min}, ret_min={args.ret_min})"
        ),
    )


def _trading_metrics(ret: np.ndarray, signal: np.ndarray) -> Dict[str, Any]:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)

    ret_active = ret[signal]
    n_trades = int(signal.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "hit_rate": np.nan,
            "avg_ret_trade": np.nan,
            "cum_ret": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": np.nan,
        }

    hit_rate = float((ret_active > 0.0).mean())
    avg_ret = float(ret_active.mean())
    cum_ret = float(ret_active.sum())

    equity_log = np.cumsum(ret * signal)
    peak = np.maximum.accumulate(equity_log)
    drawdowns = equity_log - peak
    max_drawdown = float(drawdowns.min())

    if n_trades > 1:
        std_ret = float(ret_active.std(ddof=1))
        sharpe_like = float(avg_ret / std_ret) if std_ret > 0 else np.nan
    else:
        sharpe_like = np.nan

    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "avg_ret_trade": avg_ret,
        "cum_ret": cum_ret,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
    }


def _apply_costs(ret: np.ndarray, signal: np.ndarray, fee_bps: float, slippage_bps: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)

    cost_per_trade = (fee_bps + slippage_bps) / 10_000.0
    ret_net = ret.copy()
    ret_net[signal] = ret_net[signal] - cost_per_trade
    return ret_net


def _build_equity_curve(ret: np.ndarray, signal: np.ndarray) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)
    equity_log = np.cumsum(ret * signal)
    return np.exp(equity_log)


def _load_ret_series_from_npz(dataset_path: str) -> np.ndarray:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset npz not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    # y_train, y_val, y_test are 1D arrays of ret_1h; we stitch them back
    parts = [data["y_train"], data["y_val"], data["y_test"]]
    return np.concatenate(parts, axis=0)


def _compute_index_range(n: int, use_test_split: bool) -> range:
    if not use_test_split:
        return range(0, n)

    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    start = n_train + n_val
    return range(start, n)


def backtest_signals(args: argparse.Namespace) -> None:
    _apply_optuna_profile(args)

    # Prepare data and models shared with run_signal_once
    prepared: PreparedData = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")
    models = load_models(
        reg_model_path=os.path.join(args.reg_model_dir, "xgb_ret1h_model.json"),
        dir_model_path=os.path.join(args.dir_model_dir, "xgb_dir1h_model.json") if args.dir_model_dir else None,
        lstm_model_dir=args.lstm_dir_model,
        transformer_model_dir=args.transformer_dir_model,
    )

    dir_model_weights = parse_weight_spec(args.dir_model_weights)

    populate_sequence_cache_from_prepared(prepared, models)

    # Realized returns series (ret_1h) reconstructed from the NPZ splits
    ret_series = _load_ret_series_from_npz(args.dataset_path)
    if len(ret_series) != len(prepared.df_all):
        raise RuntimeError(
            f"Length mismatch between NPZ returns ({len(ret_series)}) and curated df ({len(prepared.df_all)}).",
        )

    idx_range = _compute_index_range(len(prepared.df_all), args.use_test_split)

    ts_list = []
    ret_list = []
    p_up_list = []
    ret_pred_list = []
    sig_ens_list = []
    sig_dir_list = []

    for i in idx_range:
        sig = compute_signal_for_index(
            prepared=prepared,
            index=i,
            models=models,
            p_up_min=args.p_up_min,
            ret_min=args.ret_min,
            dir_model_weights=dir_model_weights if dir_model_weights else None,
        )

        ts_list.append(sig["ts"])
        ret_value = float(ret_series[i])
        ret_list.append(ret_value)
        p_up_list.append(sig["p_up"])
        ret_pred_list.append(sig["ret_pred"])
        sig_ens_list.append(int(sig["signal_ensemble"]))
        sig_dir_list.append(int(sig["signal_dir_only"]))

    ret_arr = np.array(ret_list, dtype=float)
    sig_ens = np.array(sig_ens_list, dtype=bool)
    sig_dir = np.array(sig_dir_list, dtype=bool)

    # Gross metrics
    metrics_ens = _trading_metrics(ret_arr, sig_ens)
    metrics_dir = _trading_metrics(ret_arr, sig_dir)

    # Net metrics (after simple fees/slippage)
    ret_ens_net = _apply_costs(ret_arr, sig_ens, args.fee_bps, args.slippage_bps)
    ret_dir_net = _apply_costs(ret_arr, sig_dir, args.fee_bps, args.slippage_bps)

    metrics_ens_net = _trading_metrics(ret_ens_net, sig_ens)
    metrics_dir_net = _trading_metrics(ret_dir_net, sig_dir)

    print("=== Ensemble (thresholded) ===")
    print(f"n_trades: {metrics_ens['n_trades']}")
    print(f"hit_rate: {metrics_ens['hit_rate']:.3f}")
    print(f"avg_ret_trade: {metrics_ens['avg_ret_trade']:.6f}")
    print(f"cum_ret (log-sum): {metrics_ens['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics_ens['max_drawdown']:.4f}")
    print(f"sharpe_like: {metrics_ens['sharpe_like']:.3f}")

    print("\n--- Net of fees/slippage ---")
    print(f"n_trades: {metrics_ens_net['n_trades']}")
    print(f"hit_rate: {metrics_ens_net['hit_rate']:.3f}")
    print(f"avg_ret_trade: {metrics_ens_net['avg_ret_trade']:.6f}")
    print(f"cum_ret (log-sum): {metrics_ens_net['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics_ens_net['max_drawdown']:.4f}")
    print(f"sharpe_like: {metrics_ens_net['sharpe_like']:.3f}")

    print("\n=== Direction-only baseline (p_up >= 0.5) ===")
    print(f"n_trades: {metrics_dir['n_trades']}")
    print(f"hit_rate: {metrics_dir['hit_rate']:.3f}")
    print(f"avg_ret_trade: {metrics_dir['avg_ret_trade']:.6f}")
    print(f"cum_ret (log-sum): {metrics_dir['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics_dir['max_drawdown']:.4f}")
    print(f"sharpe_like: {metrics_dir['sharpe_like']:.3f}")

    print("\n--- Net of fees/slippage ---")
    print(f"n_trades: {metrics_dir_net['n_trades']}")
    print(f"hit_rate: {metrics_dir_net['hit_rate']:.3f}")
    print(f"avg_ret_trade: {metrics_dir_net['avg_ret_trade']:.6f}")
    print(f"cum_ret (log-sum): {metrics_dir_net['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics_dir_net['max_drawdown']:.4f}")
    print(f"sharpe_like: {metrics_dir_net['sharpe_like']:.3f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        sig_ens_int = np.array(sig_ens_list, dtype=int)
        sig_dir_int = np.array(sig_dir_list, dtype=int)

        ret_ens_gross = ret_arr * sig_ens_int
        ret_dir_gross = ret_arr * sig_dir_int
        ret_ens_net_bar = ret_ens_net * sig_ens_int
        ret_dir_net_bar = ret_dir_net * sig_dir_int

        equity_ens_net = _build_equity_curve(ret_ens_net, sig_ens)
        equity_dir_net = _build_equity_curve(ret_dir_net, sig_dir)

        df = pd.DataFrame(
            {
                "ts": ts_list,
                "ret_1h": ret_arr,
                "p_up": p_up_list,
                "ret_pred": ret_pred_list,
                "signal_ensemble": sig_ens_int,
                "signal_dir_only": sig_dir_int,
                "ret_ensemble_gross": ret_ens_gross,
                "ret_dir_only_gross": ret_dir_gross,
                "ret_ensemble_net": ret_ens_net_bar,
                "ret_dir_only_net": ret_dir_net_bar,
                "equity_ensemble_net": equity_ens_net,
                "equity_dir_only_net": equity_dir_net,
            },
        )

        out_path = os.path.join(args.output_dir, "backtest_signals.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved backtest log to {out_path}")


def main() -> None:
    args = _parse_args()
    backtest_signals(args)


if __name__ == "__main__":
    main()
