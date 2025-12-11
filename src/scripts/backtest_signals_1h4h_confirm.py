import argparse
import math
import os
from typing import Dict

import numpy as np
import pandas as pd

DEFAULT_BT1H_PATH = "artifacts/analysis/backtest_signals_v1/backtest_signals.csv"
DEFAULT_BT4H_PATH = "artifacts/analysis/backtest_signals_4h_v1/backtest_signals_4h.csv"
DEFAULT_OUTPUT_DIR = "artifacts/analysis/backtest_signals_1h4h_confirm_v1"
DEFAULT_P_UP_MIN_4H = 0.55
DEFAULT_FEE_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0

REQUIRED_4H_COLUMNS = ["p_up_4h", "ret_pred_4h", "signal_ensemble_4h"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a 1h ensemble strategy filtered by 4h confirmation signals.",
    )
    parser.add_argument(
        "--bt1h-path",
        type=str,
        default=DEFAULT_BT1H_PATH,
        help="Path to the 1h backtest CSV (ensemble baseline).",
    )
    parser.add_argument(
        "--bt4h-path",
        type=str,
        default=DEFAULT_BT4H_PATH,
        help="Path to the 4h backtest CSV with 4h predictions.",
    )
    parser.add_argument(
        "--p-up-min-4h",
        type=float,
        default=DEFAULT_P_UP_MIN_4H,
        help="Threshold applied to 4h p_up for confirmation.",
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
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the combined backtest CSV (optional).",
    )
    return parser.parse_args()


def _load_backtest(path: str, parse_dates: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Backtest CSV not found: {path}")

    kwargs: Dict[str, object] = {}
    if parse_dates:
        kwargs["parse_dates"] = ["ts"]

    df = pd.read_csv(path, **kwargs)
    return df


def _resolve_ret_column(df: pd.DataFrame) -> str:
    candidates = ["ret_1h", "ret", "ret_true"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not locate the 1h return column; expected one of ret_1h, ret, ret_true.",
    )


def _compute_metrics(ret_true: np.ndarray, signal: np.ndarray, cost_per_trade: float) -> Dict[str, float]:
    signal_bool = signal.astype(bool)
    ret_net = (ret_true - cost_per_trade) * signal_bool

    n_trades = int(signal_bool.sum())
    if n_trades == 0:
        hit_rate = math.nan
    else:
        hit_rate = float((ret_true[signal_bool] > 0.0).mean())

    cum_ret = float(ret_net.sum())

    equity_log = np.cumsum(ret_net)
    if equity_log.size == 0:
        max_drawdown = 0.0
    else:
        peak = np.maximum.accumulate(equity_log)
        drawdowns = equity_log - peak
        max_drawdown = float(drawdowns.min())

    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "cum_ret": cum_ret,
        "max_drawdown": max_drawdown,
        "ret_net_series": ret_net,
        "equity_log": equity_log,
    }


def backtest_with_confirmation(args: argparse.Namespace) -> None:
    df_1h = _load_backtest(args.bt1h_path)
    df_4h = _load_backtest(args.bt4h_path)

    ret_col = _resolve_ret_column(df_1h)

    if "signal_ensemble" not in df_1h.columns:
        raise ValueError("1h backtest CSV missing signal_ensemble column.")

    missing_4h = [col for col in REQUIRED_4H_COLUMNS if col not in df_4h.columns]
    if missing_4h:
        raise ValueError(f"4h backtest CSV missing required columns: {missing_4h}")

    df_1h_renamed = df_1h.rename(columns={"signal_ensemble": "signal_ensemble_1h"})

    merged = pd.merge(df_1h_renamed, df_4h, on="ts", how="inner", suffixes=("_1h", "_4h"))
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between 1h and 4h backtests.")

    merged = merged.sort_values("ts").reset_index(drop=True)

    ret_1h_true = merged[ret_col].astype(float).to_numpy()
    signal_1h = merged["signal_ensemble_1h"].astype(int).to_numpy()
    p_up_4h = merged["p_up_4h"].astype(float).to_numpy()

    filter_4h = (p_up_4h >= args.p_up_min_4h).astype(int)
    signal_combined = (signal_1h == 1) & (filter_4h == 1)

    cost_per_trade = (args.fee_bps + args.slippage_bps) / 10_000.0

    metrics_1h = _compute_metrics(ret_1h_true, signal_1h, cost_per_trade)
    metrics_combined = _compute_metrics(ret_1h_true, signal_combined.astype(int), cost_per_trade)

    print("=== 1h ensemble only (net) ===")
    print(f"n_trades_1h: {metrics_1h['n_trades']}")
    print(
        f"hit_rate_1h: {metrics_1h['hit_rate']:.3f}" if not math.isnan(metrics_1h["hit_rate"]) else "hit_rate_1h: nan",
    )
    print(f"cum_ret_1h: {metrics_1h['cum_ret']:.4f}")

    print("\n=== 1h + 4h confirmation (net) ===")
    print(f"n_trades_1h4h: {metrics_combined['n_trades']}")
    print(
        "hit_rate_1h4h: nan"
        if math.isnan(metrics_combined["hit_rate"])
        else f"hit_rate_1h4h: {metrics_combined['hit_rate']:.3f}",
    )
    print(f"cum_ret_1h4h: {metrics_combined['cum_ret']:.4f}")
    print(f"max_drawdown_1h4h: {metrics_combined['max_drawdown']:.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ret_net_combined = metrics_combined["ret_net_series"]
        equity_combined = np.exp(np.cumsum(ret_net_combined))

        output_df = pd.DataFrame(
            {
                "ts": merged["ts"],
                "ret_1h": ret_1h_true,
                "signal_ensemble_1h": signal_1h,
                "p_up_4h": p_up_4h,
                "ret_pred_4h": merged["ret_pred_4h"].astype(float).to_numpy(),
                "filter_4h": filter_4h,
                "signal_combined": signal_combined.astype(int),
                "ret_net_combined": ret_net_combined,
                "equity_combined": equity_combined,
            },
        )

        out_path = os.path.join(args.output_dir, "backtest_signals_1h4h.csv")
        output_df.to_csv(out_path, index=False)
        print(f"\nSaved combined backtest log to {out_path}")


def main() -> None:
    args = _parse_args()
    backtest_with_confirmation(args)


if __name__ == "__main__":
    main()
