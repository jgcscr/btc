import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config_trading import (
    DEFAULT_FEE_BPS,
    DEFAULT_P_UP_MIN,
    DEFAULT_RET_MIN,
    DEFAULT_SLIPPAGE_BPS,
)
from src.trading.signals import PreparedData, compute_signal_for_index, load_models, prepare_data_for_signals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a simple position-aware paper-trading loop using the same "
            "signal logic as run_signal_once over a historical period."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression NPZ file (used for returns and feature names).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default="artifacts/models/xgb_ret1h_v1",
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default="artifacts/models/xgb_dir1h_v1",
        help="Directory containing direction model JSON (xgb_dir1h_model.json).",
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
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help="Per-trade fee in basis points (charged on entries/exits).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help="Per-trade slippage in basis points (charged on entries/exits).",
    )
    parser.add_argument(
        "--use-test-split",
        action="store_true",
        help=(
            "If set, simulate only over the test split defined by the "
            "original 70/15/15 train/val/test fractions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write a per-bar CSV log of the paper trading run.",
    )
    return parser.parse_args()


def _load_ret_series_from_npz(dataset_path: str) -> np.ndarray:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset npz not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    parts = [data["y_train"], data["y_val"], data["y_test"]]
    return np.concatenate(parts, axis=0)


def _compute_index_range(n: int, use_test_split: bool) -> range:
    if not use_test_split:
        return range(0, n)

    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    start = n_train + n_val
    return range(start, n)


def _compute_trade_metrics(trade_pnls: List[float], equity_log_series: np.ndarray) -> Dict[str, Any]:
    trade_pnls_arr = np.array(trade_pnls, dtype=float)
    n_trades = int(len(trade_pnls_arr))

    if n_trades == 0:
        hit_rate = np.nan
        avg_pnl = np.nan
    else:
        hit_rate = float((trade_pnls_arr > 0.0).mean())
        avg_pnl = float(trade_pnls_arr.mean())

    cum_ret = float(equity_log_series[-1] if equity_log_series.size > 0 else 0.0)

    if equity_log_series.size == 0:
        max_drawdown = 0.0
    else:
        peak = np.maximum.accumulate(equity_log_series)
        drawdowns = equity_log_series - peak
        max_drawdown = float(drawdowns.min())

    # For a Sharpe-like metric, use bar-level net returns where there was some activity
    if equity_log_series.size <= 1:
        sharpe_like = np.nan
    else:
        net_returns = np.diff(equity_log_series)
        active = net_returns[net_returns != 0.0]
        if active.size > 1:
            mu = float(active.mean())
            std = float(active.std(ddof=1))
            sharpe_like = float(mu / std) if std > 0 else np.nan
        else:
            sharpe_like = np.nan

    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "avg_pnl_trade": avg_pnl,
        "cum_ret": cum_ret,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
    }


def paper_trade_loop(args: argparse.Namespace) -> None:
    # Prepare data and models shared with run_signal_once/backtest_signals
    prepared: PreparedData = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")
    models = load_models(
        reg_model_path=os.path.join(args.reg_model_dir, "xgb_ret1h_model.json"),
        dir_model_path=os.path.join(args.dir_model_dir, "xgb_dir1h_model.json"),
    )

    ret_series = _load_ret_series_from_npz(args.dataset_path)
    if len(ret_series) != len(prepared.df_all):
        raise RuntimeError(
            f"Length mismatch between NPZ returns ({len(ret_series)}) and curated df ({len(prepared.df_all)}).",
        )

    idx_range = _compute_index_range(len(prepared.df_all), args.use_test_split)

    cost_per_trade = (args.fee_bps + args.slippage_bps) / 10_000.0

    position = 0  # 0 = flat, 1 = long
    equity_log = 0.0
    equity_log_series: List[float] = []
    trade_pnls: List[float] = []
    entry_equity_log: float | None = None

    ts_list: List[str] = []
    ret_list: List[float] = []
    p_up_list: List[float] = []
    ret_pred_list: List[float] = []
    signal_list: List[int] = []
    position_list: List[int] = []
    ret_net_list: List[float] = []

    for i in idx_range:
        sig = compute_signal_for_index(
            prepared=prepared,
            index=i,
            models=models,
            p_up_min=args.p_up_min,
            ret_min=args.ret_min,
        )

        ts = sig["ts"]
        p_up = float(sig["p_up"])
        ret_pred = float(sig["ret_pred"])
        signal_ens = int(sig["signal_ensemble"])

        ret_bar = float(ret_series[i])

        entry = False
        exit_ = False

        if position == 0 and signal_ens == 1:
            entry = True
            entry_equity_log = equity_log
        elif position == 1 and signal_ens == 0:
            exit_ = True

        # Position for this bar equals the current ensemble signal
        position = signal_ens

        gross_ret_bar = ret_bar if position == 1 else 0.0

        cost = 0.0
        if entry:
            cost -= cost_per_trade
        if exit_:
            cost -= cost_per_trade

        net_ret_bar = gross_ret_bar + cost
        equity_log += net_ret_bar
        equity_log_series.append(equity_log)

        if exit_ and entry_equity_log is not None:
            trade_pnls.append(equity_log - entry_equity_log)
            entry_equity_log = None

        ts_list.append(ts)
        ret_list.append(ret_bar)
        p_up_list.append(p_up)
        ret_pred_list.append(ret_pred)
        signal_list.append(signal_ens)
        position_list.append(position)
        ret_net_list.append(net_ret_bar)

    equity_log_arr = np.array(equity_log_series, dtype=float)
    metrics = _compute_trade_metrics(trade_pnls, equity_log_arr)

    print("=== Ensemble paper-trading (position-aware) ===")
    print(f"n_trades (entries): {metrics['n_trades']}")
    print(f"hit_rate: {metrics['hit_rate']:.3f}")
    print(f"avg_pnl_trade: {metrics['avg_pnl_trade']:.6f}")
    print(f"cum_ret (log-sum): {metrics['cum_ret']:.4f}")
    print(f"max_drawdown (log): {metrics['max_drawdown']:.4f}")
    print(f"sharpe_like: {metrics['sharpe_like']:.3f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        equity_curve = np.exp(equity_log_arr)

        df = pd.DataFrame(
            {
                "ts": ts_list,
                "ret_1h": ret_list,
                "p_up": p_up_list,
                "ret_pred": ret_pred_list,
                "signal_ensemble": signal_list,
                "position": position_list,
                "ret_net": ret_net_list,
                "equity": equity_curve,
            },
        )

        out_path = os.path.join(args.output_dir, "paper_trade.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved paper-trade log to {out_path}")


def main() -> None:
    args = _parse_args()
    paper_trade_loop(args)


if __name__ == "__main__":
    main()
