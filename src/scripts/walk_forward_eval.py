import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.config_trading import DEFAULT_P_UP_MIN, DEFAULT_RET_MIN


def _load_npz_dataset(path: str) -> Dict[str, Any]:
    """Load train/val/test splits and feature names from an npz file.

    We reconstruct the full chronological series by concatenating train/val/test
    in that order.
    """
    data = np.load(path, allow_pickle=True)

    required_keys = [
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
        "feature_names",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in dataset npz: {path}")

    X_full = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)

    return {
        "X_full": X_full,
        "y_full": y_full,
        "feature_names": data["feature_names"].tolist(),
    }


def _make_folds(n_samples: int, n_folds: int) -> List[Tuple[int, int]]:
    """Create non-overlapping test windows at the end of the series.

    We divide the tail of the series into `n_folds` equal-sized windows.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    window_size = n_samples // (n_folds + 1)
    if window_size <= 0:
        raise ValueError("Not enough samples to create folds")

    folds: List[Tuple[int, int]] = []
    for i in range(n_folds):
        start = n_samples - (n_folds - i) * window_size
        end = start + window_size
        folds.append((start, end))
    return folds


def _create_reg_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def _create_dir_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
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


def _evaluate_window(
    X_full: np.ndarray,
    y_full: np.ndarray,
    start: int,
    end: int,
    inner_train_frac: float,
    p_up_min: float,
    ret_min: float,
) -> Dict[str, Any]:
    """Train models on data before [start, end) and evaluate on that window."""

    X_train_full = X_full[:start]
    y_train_full = y_full[:start]
    X_test_win = X_full[start:end]
    y_test_win = y_full[start:end]

    n_pre = X_train_full.shape[0]
    if n_pre < 10:
        raise ValueError("Not enough pre-window samples to train models")

    # Inner split of pre-window data
    n_inner_train = max(1, int(n_pre * inner_train_frac))
    if n_inner_train >= n_pre:
        n_inner_train = n_pre - 1

    X_train_inner = X_train_full[:n_inner_train]
    y_train_inner = y_train_full[:n_inner_train]
    X_val_inner = X_train_full[n_inner_train:]
    y_val_inner = y_train_full[n_inner_train:]

    # Regression model
    reg = _create_reg_model()
    reg.fit(
        X_train_inner,
        y_train_inner,
        eval_set=[(X_train_inner, y_train_inner), (X_val_inner, y_val_inner)],
        verbose=False,
    )

    # Direction model (labels from sign of ret_1h)
    y_train_dir = (y_train_inner > 0.0).astype(int)
    y_val_dir = (y_val_inner > 0.0).astype(int)

    dir_model = _create_dir_model()
    dir_model.fit(
        X_train_inner,
        y_train_dir,
        eval_set=[(X_train_inner, y_train_dir), (X_val_inner, y_val_dir)],
        verbose=False,
    )

    # Predictions on window
    y_pred = reg.predict(X_test_win)
    p_up = dir_model.predict_proba(X_test_win)[:, 1]

    # Regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_test_win, y_pred)))
    mae = float(mean_absolute_error(y_test_win, y_pred))

    # Direction metrics
    y_true_dir = (y_test_win > 0.0).astype(int)
    y_hat_dir = (p_up >= 0.5).astype(int)
    acc = float(accuracy_score(y_true_dir, y_hat_dir))
    prec = float(precision_score(y_true_dir, y_hat_dir, zero_division=0))
    rec = float(recall_score(y_true_dir, y_hat_dir, zero_division=0))
    f1 = float(f1_score(y_true_dir, y_hat_dir, zero_division=0))

    # Trading metrics
    signal_ens = (p_up >= p_up_min) & (y_pred >= ret_min)
    signal_dir_only = p_up >= 0.5

    metrics_ens = _trading_metrics(y_test_win, signal_ens)
    metrics_dir = _trading_metrics(y_test_win, signal_dir_only)

    return {
        "start": start,
        "end": end,
        "n_samples": int(end - start),
        "regression": {"rmse": rmse, "mae": mae},
        "direction": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "ensemble": metrics_ens,
        "direction_only": metrics_dir,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation for BTC 1h models using NPZ splits.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to NPZ with train/val/test splits.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of sequential test windows near the end of the series.",
    )
    parser.add_argument(
        "--inner-train-frac",
        type=float,
        default=0.8,
        help="Fraction of pre-window data used as inner train (rest is inner val).",
    )
    parser.add_argument(
        "--min-pre-window-samples",
        type=int,
        default=500,
        help="Minimum number of samples required before a window to train models.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset = _load_npz_dataset(args.dataset_path)
    X_full = dataset["X_full"]
    y_full = dataset["y_full"]
    n = X_full.shape[0]

    folds = _make_folds(n, args.n_folds)

    print(f"Total samples: {n}")
    print("Walk-forward folds (test windows):")
    for i, (s, e) in enumerate(folds):
        print(f"  Fold {i}: [{s}, {e}) size={e - s}")

    print("\n=== Walk-forward results ===")
    header = (
        "fold\tstart\tend\tn\tRMSE\tMAE\tAcc\tPrec\tRec\tF1\t"
        "Ens_n\tEns_hit\tEns_avg\tEns_cum\tEns_dd\tEns_sharpe\t"
        "Dir_n\tDir_hit\tDir_avg\tDir_cum\tDir_dd\tDir_sharpe"
    )
    print(header)

    for i, (start, end) in enumerate(folds):
        if start < args.min_pre_window_samples:
            print(
                f"{i}\t{start}\t{end}\t{end - start}\t" "SKIP (insufficient pre-window samples)",
            )
            continue

        res = _evaluate_window(
            X_full=X_full,
            y_full=y_full,
            start=start,
            end=end,
            inner_train_frac=args.inner_train_frac,
            p_up_min=args.p_up_min,
            ret_min=args.ret_min,
        )

        reg = res["regression"]
        direc = res["direction"]
        ens = res["ensemble"]
        base = res["direction_only"]

        line = (
            f"{i}\t{res['start']}\t{res['end']}\t{res['n_samples']}\t"
            f"{reg['rmse']:.6f}\t{reg['mae']:.6f}\t"
            f"{direc['accuracy']:.3f}\t{direc['precision']:.3f}\t{direc['recall']:.3f}\t{direc['f1']:.3f}\t"
            f"{ens['n_trades']}\t{ens['hit_rate']:.3f}\t{ens['avg_ret_trade']:.6f}\t{ens['cum_ret']:.4f}\t{ens['max_drawdown']:.4f}\t{ens['sharpe_like']:.3f}\t"
            f"{base['n_trades']}\t{base['hit_rate']:.3f}\t{base['avg_ret_trade']:.6f}\t{base['cum_ret']:.4f}\t{base['max_drawdown']:.4f}\t{base['sharpe_like']:.3f}"
        )
        print(line)


if __name__ == "__main__":
    main()
