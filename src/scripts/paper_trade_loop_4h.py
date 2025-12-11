import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config_trading import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
from src.trading.signals import PreparedData, compute_signal_for_index, load_models, prepare_data_for_signals


DEFAULT_P_UP_MIN_4H = 0.55
DEFAULT_RET_MIN_4H = 0.0
DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_REG_MODEL_DIR = "artifacts/models/xgb_ret4h_v1"
DEFAULT_DIR_MODEL_DIR = "artifacts/models/xgb_dir4h_v1"
DEFAULT_OUTPUT_DIR = "artifacts/analysis/paper_trade_4h_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a simple 4h position-aware paper-trading loop using 4h models.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the multi-horizon NPZ file (needs 4h labels).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default=DEFAULT_REG_MODEL_DIR,
        help="Directory containing the trained 4h regression model (xgb_ret4h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default=DEFAULT_DIR_MODEL_DIR,
        help="Directory containing the trained 4h direction model (xgb_dir4h_model.json).",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN_4H,
        help="4h ensemble threshold for P(up).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN_4H,
        help="4h ensemble threshold for predicted ret_4h (log return).",
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
        help="If set, simulate only over the test split (last 15 percent).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the per-bar 4h paper-trade CSV (created if missing).",
    )
    return parser.parse_args()


def _load_ret_series(dataset_path: str, horizon: int) -> np.ndarray:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset npz not found: {dataset_path}")

    key_prefix = f"y_ret{horizon}h"
    data = np.load(dataset_path, allow_pickle=True)
    required = [f"{key_prefix}_train", f"{key_prefix}_val", f"{key_prefix}_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset missing required 4h return arrays: {missing}")

    parts = [data[f"{key_prefix}_train"], data[f"{key_prefix}_val"], data[f"{key_prefix}_test"]]
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


def paper_trade_loop_4h(args: argparse.Namespace) -> None:
    prepared: PreparedData = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")

    models = load_models(
        reg_model_path=os.path.join(args.reg_model_dir, "xgb_ret4h_model.json"),
        dir_model_path=os.path.join(args.dir_model_dir, "xgb_dir4h_model.json"),
    )

    ret_series = _load_ret_series(args.dataset_path, horizon=4)
    if len(ret_series) == 0:
        raise RuntimeError("Loaded empty 4h return series from dataset; ensure multi-horizon NPZ is valid.")

    if len(ret_series) > len(prepared.df_all):
        raise RuntimeError(
            f"4h returns ({len(ret_series)}) exceed curated frame ({len(prepared.df_all)}); this should not happen.",
        )

    if len(ret_series) != len(prepared.df_all):
        diff = len(prepared.df_all) - len(ret_series)
        print(
            f"Truncating the last {diff} rows from curated features to align with available 4h returns.",
        )

    idx_range = _compute_index_range(len(ret_series), args.use_test_split)

    cost_per_trade = (args.fee_bps + args.slippage_bps) / 10_000.0

    position = 0
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

        # We reuse the 1h loop mechanics: each hour is a decision point, but the
        # models are trained on 4h targets so entries remain active for multiple
        # bars until the ensemble drops below thresholds.
        if position == 0 and signal_ens == 1:
            entry = True
            entry_equity_log = equity_log
        elif position == 1 and signal_ens == 0:
            exit_ = True

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

    print("=== 4h Ensemble paper-trading (net) ===")
    print(f"n_trades_4h: {metrics['n_trades']}")
    print(f"hit_rate_4h: {metrics['hit_rate']:.3f}" if metrics["n_trades"] > 0 else "hit_rate_4h: nan")
    print(f"cum_ret_4h: {metrics['cum_ret']:.4f}")
    print(f"max_drawdown_4h: {metrics['max_drawdown']:.4f}")
    print(f"sharpe_like_4h: {metrics['sharpe_like']:.3f}" if not np.isnan(metrics["sharpe_like"]) else "sharpe_like_4h: nan")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        equity_curve = np.exp(equity_log_arr)

        df = pd.DataFrame(
            {
                "ts": ts_list,
                "position_4h": position_list,
                "signal_ensemble_4h": signal_list,
                "ret_4h": ret_list,
                "ret_net_4h": ret_net_list,
                "p_up_4h": p_up_list,
                "ret_pred_4h": ret_pred_list,
                "equity_4h": equity_curve,
            },
        )

        out_path = os.path.join(args.output_dir, "paper_trade_4h.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved 4h paper-trade log to {out_path}")


if __name__ == "__main__":
    args = _parse_args()
    paper_trade_loop_4h(args)
