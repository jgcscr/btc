import argparse
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config_trading import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
from src.trading.signals import PreparedData, compute_signal_for_index, load_models, prepare_data_for_signals


DEFAULT_P_UP_MIN_4H = 0.55
DEFAULT_RET_MIN_4H = 0.0
DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_REG_MODEL_DIR = "artifacts/models/xgb_ret4h_v1"
DEFAULT_DIR_MODEL_DIR = "artifacts/models/xgb_dir4h_v1"
DEFAULT_OUTPUT_DIR = "artifacts/analysis/backtest_signals_4h_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest 4h ensemble and direction-only signals using the multi-horizon dataset.",
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
        help="If set, restrict evaluation to the test split (last 15 percent).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the per-bar 4h backtest CSV (created if missing).",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Optional CSV with 1h features for offline backtests.",
    )
    parser.add_argument(
        "--onchain-path",
        type=str,
        default=None,
        help="Optional cached on-chain metrics to merge with --features-path.",
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


def _apply_costs(ret: np.ndarray, signal: np.ndarray, fee_bps: float, slippage_bps: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)

    cost_per_trade = (fee_bps + slippage_bps) / 10_000.0
    ret_net = ret.copy()
    ret_net[signal] = ret_net[signal] - cost_per_trade
    return ret_net


def _trading_metrics(ret_net: np.ndarray, signal: np.ndarray) -> Dict[str, Any]:
    ret_net = np.asarray(ret_net, dtype=float)
    signal = np.asarray(signal).astype(bool)

    ret_active = ret_net[signal]
    n_trades = int(signal.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "hit_rate": np.nan,
            "cum_ret": 0.0,
        }

    hit_rate = float((ret_active > 0.0).mean())
    cum_ret = float(ret_active.sum())

    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "cum_ret": cum_ret,
    }


def _build_equity_curve(ret_net: np.ndarray, signal: np.ndarray) -> np.ndarray:
    ret_net = np.asarray(ret_net, dtype=float)
    signal = np.asarray(signal).astype(bool)
    equity_log = np.cumsum(ret_net * signal)
    return np.exp(equity_log)


def backtest_signals_4h(args: argparse.Namespace) -> None:
    prepared: PreparedData = prepare_data_for_signals(
        args.dataset_path,
        target_column="ret_1h",
        features_path=args.features_path,
        onchain_path=args.onchain_path,
    )

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
        )

        ts_list.append(sig["ts"])
        ret_value = float(ret_series[i])
        ret_list.append(ret_value)
        p_up_list.append(float(sig["p_up"]))
        ret_pred_list.append(float(sig["ret_pred"]))
        sig_ens_list.append(int(sig["signal_ensemble"]))
        sig_dir_list.append(int(sig["signal_dir_only"]))

    ret_arr = np.array(ret_list, dtype=float)
    sig_ens = np.array(sig_ens_list, dtype=bool)
    sig_dir = np.array(sig_dir_list, dtype=bool)

    ret_net_ens = _apply_costs(ret_arr, sig_ens, args.fee_bps, args.slippage_bps)
    ret_net_dir = _apply_costs(ret_arr, sig_dir, args.fee_bps, args.slippage_bps)

    metrics_ens = _trading_metrics(ret_net_ens, sig_ens)
    metrics_dir = _trading_metrics(ret_net_dir, sig_dir)

    print("=== 4h Ensemble (net of costs) ===")
    print(f"n_trades_ens_4h: {metrics_ens['n_trades']}")
    print(f"hit_rate_ens_4h: {metrics_ens['hit_rate']:.3f}" if metrics_ens["n_trades"] > 0 else "hit_rate_ens_4h: nan")
    print(f"cum_ret_ens_4h: {metrics_ens['cum_ret']:.4f}")

    print("\n=== 4h Direction-only baseline (net of costs) ===")
    print(f"n_trades_dir_4h: {metrics_dir['n_trades']}")
    print(f"hit_rate_dir_4h: {metrics_dir['hit_rate']:.3f}" if metrics_dir["n_trades"] > 0 else "hit_rate_dir_4h: nan")
    print(f"cum_ret_dir_4h: {metrics_dir['cum_ret']:.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        sig_ens_int = np.array(sig_ens_list, dtype=int)
        sig_dir_int = np.array(sig_dir_list, dtype=int)

        ret_net_ens_bar = ret_net_ens * sig_ens_int
        ret_net_dir_bar = ret_net_dir * sig_dir_int

        equity_ens = _build_equity_curve(ret_net_ens, sig_ens)
        equity_dir = _build_equity_curve(ret_net_dir, sig_dir)

        df = pd.DataFrame(
            {
                "ts": ts_list,
                "ret_4h": ret_arr,
                "p_up_4h": p_up_list,
                "ret_pred_4h": ret_pred_list,
                "signal_ensemble_4h": sig_ens_int,
                "signal_dir_only_4h": sig_dir_int,
                "ret_net_ens_4h": ret_net_ens_bar,
                "ret_net_dir_4h": ret_net_dir_bar,
                "equity_ens_4h": equity_ens,
                "equity_dir_4h": equity_dir,
            },
        )

        out_path = os.path.join(args.output_dir, "backtest_signals_4h.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved 4h backtest log to {out_path}")


def main() -> None:
    args = _parse_args()
    backtest_signals_4h(args)


if __name__ == "__main__":
    main()
