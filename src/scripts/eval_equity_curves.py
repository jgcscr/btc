import argparse
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

try:  # Allow importing this module without xgboost (for tests)
    from xgboost import XGBClassifier, XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without xgboost
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment]
    _XGBOOST_AVAILABLE = False

from src.config_trading import (
    DEFAULT_FEE_BPS,
    DEFAULT_P_UP_MIN,
    DEFAULT_RET_MIN,
    DEFAULT_SLIPPAGE_BPS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble vs direction-only equity curves on test set.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to NPZ with train/val/test splits (expects X_test, y_test).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        required=True,
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        required=True,
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
        help=(
            "Per-trade fee in basis points (applied once per executed trade) "
            "expressed in log-return approximation."
        ),
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=(
            "Per-trade slippage in basis points (applied once per executed trade) "
            "expressed in log-return approximation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write CSV with per-bar equity curves.",
    )
    return parser.parse_args()


def _load_dataset(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset npz not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = {"X_test", "y_test"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing keys in dataset npz {path}: {sorted(missing)}")

    return {
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "ts_test": data["ts_test"] if "ts_test" in data.files else None,
    }


def _load_models(reg_dir: str, dir_dir: str) -> Dict[str, Any]:
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is required to load models; install it via 'pip install -r requirements.txt' or 'pip install xgboost'."
        )

    reg_path = os.path.join(reg_dir, "xgb_ret1h_model.json")
    dir_path = os.path.join(dir_dir, "xgb_dir1h_model.json")

    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regression model not found at {reg_path}")
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Direction model not found at {dir_path}")

    reg = XGBRegressor()
    reg.load_model(reg_path)

    dir_model = XGBClassifier()
    dir_model.load_model(dir_path)

    return {"reg": reg, "dir": dir_model}


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


def _build_equity_curve(ret: np.ndarray, signal: np.ndarray) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)
    equity_log = np.cumsum(ret * signal)
    return np.exp(equity_log)


def _apply_costs(ret: np.ndarray, signal: np.ndarray, fee_bps: float, slippage_bps: float) -> np.ndarray:
    """Apply simple transaction costs to returns.

    fee_bps and slippage_bps are in *basis points* per trade. We approximate
    their impact in log-return space as:

        cost_log ≈ (fee_bps + slippage_bps) / 10_000

    and subtract this cost once for each active bar (signal == 1).
    """
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)

    cost_per_trade = (fee_bps + slippage_bps) / 10_000.0
    ret_net = ret.copy()
    ret_net[signal] = ret_net[signal] - cost_per_trade
    return ret_net


def main() -> None:
    args = _parse_args()

    dataset = _load_dataset(args.dataset_path)
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    ts_test = dataset["ts_test"]

    models = _load_models(args.reg_model_dir, args.dir_model_dir)
    reg = models["reg"]
    dir_model = models["dir"]

    # Predictions
    y_pred = reg.predict(X_test)
    p_up = dir_model.predict_proba(X_test)[:, 1]

    # Signals
    signal_ens = (p_up >= args.p_up_min) & (y_pred >= args.ret_min)
    signal_dir = p_up >= 0.5

    # Gross metrics (no cost)
    metrics_ens = _trading_metrics(y_test, signal_ens)
    metrics_dir = _trading_metrics(y_test, signal_dir)

    # Net metrics (after costs)
    y_test_ens_net = _apply_costs(y_test, signal_ens, args.fee_bps, args.slippage_bps)
    y_test_dir_net = _apply_costs(y_test, signal_dir, args.fee_bps, args.slippage_bps)

    metrics_ens_net = _trading_metrics(y_test_ens_net, signal_ens)
    metrics_dir_net = _trading_metrics(y_test_dir_net, signal_dir)

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

        equity_ens_gross = _build_equity_curve(y_test, signal_ens)
        equity_dir_gross = _build_equity_curve(y_test, signal_dir)

        equity_ens_net = _build_equity_curve(y_test_ens_net, signal_ens)
        equity_dir_net = _build_equity_curve(y_test_dir_net, signal_dir)

        if ts_test is None:
            ts = np.arange(len(y_test))
        else:
            ts = ts_test

        df = pd.DataFrame(
            {
                "ts": ts,
                "ret_1h": y_test,
                "signal_ensemble": signal_ens.astype(int),
                "signal_dir_only": signal_dir.astype(int),
                "ret_ensemble_gross": y_test * signal_ens,
                "ret_dir_only_gross": y_test * signal_dir,
                "ret_ensemble_net": y_test_ens_net * signal_ens,
                "ret_dir_only_net": y_test_dir_net * signal_dir,
                "equity_ensemble_gross": equity_ens_gross,
                "equity_dir_only_gross": equity_dir_gross,
                "equity_ensemble_net": equity_ens_net,
                "equity_dir_only_net": equity_dir_net,
            },
        )

        out_path = os.path.join(args.output_dir, "equity_curves_test.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved equity curves to {out_path}")


if __name__ == "__main__":
    main()
