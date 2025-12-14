import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from xgboost import Booster, DMatrix


def _load_npz_dataset(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset npz not found: {path}")

    data = np.load(path, allow_pickle=True)

    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    return {
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


def _load_xgb_booster(model_dir: str, model_filename: str, meta_filename: str) -> Tuple[Booster, List[str]]:
    model_path = os.path.join(model_dir, model_filename)
    meta_path = os.path.join(model_dir, meta_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    booster = Booster()
    booster.load_model(model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError(f"Invalid or missing 'feature_names' in {meta_path}")

    return booster, feature_names


def _align_features(X: np.ndarray, dataset_feature_names: List[str], model_feature_names: List[str]) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape [N, F], got {X.shape}")

    name_to_idx = {name: i for i, name in enumerate(dataset_feature_names)}

    indices = []
    for name in model_feature_names:
        if name not in name_to_idx:
            raise KeyError(f"Feature '{name}' required by model is missing from dataset")
        indices.append(name_to_idx[name])

    X_aligned = X[:, indices]
    return X_aligned


def _compute_trade_stats(ret: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    mask = signal.astype(bool)
    n_trades = int(mask.sum())

    if n_trades == 0:
        return {
            "n_trades": 0.0,
            "hit_rate": 0.0,
            "avg_ret_per_trade": 0.0,
            "cum_ret": 0.0,
            "max_drawdown": 0.0,
        }

    ret_trades = ret[mask]
    hit_rate = float(np.mean(ret_trades > 0))
    avg_ret = float(np.mean(ret_trades))
    cum_ret = float(np.sum(ret_trades))

    strategy_ret = ret * signal.astype(float)
    equity_curve = np.cumsum(strategy_ret)
    if equity_curve.size == 0:
        max_drawdown = 0.0
    else:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = running_max - equity_curve
        max_drawdown = float(np.max(drawdowns))

    return {
        "n_trades": float(n_trades),
        "hit_rate": hit_rate,
        "avg_ret_per_trade": avg_ret,
        "cum_ret": cum_ret,
        "max_drawdown": max_drawdown,
    }


def evaluate_ensemble_signals(
    dataset_path_reg: str,
    reg_model_dir: str,
    dir_model_dir: str,
    p_up_min: float,
    ret_min: float,
) -> None:
    data = _load_npz_dataset(dataset_path_reg)
    X_test = data["X_test"]
    y_test = data["y_test"]  # realized ret_1h
    dataset_feature_names = data["feature_names"]

    # Load regression model
    reg_booster, reg_feature_names = _load_xgb_booster(
        model_dir=reg_model_dir,
        model_filename="xgb_ret1h_model.json",
        meta_filename="model_metadata.json",
    )

    # Load direction model
    dir_booster, dir_feature_names = _load_xgb_booster(
        model_dir=dir_model_dir,
        model_filename="xgb_dir1h_model.json",
        meta_filename="model_metadata_direction.json",
    )

    # Align features for each model separately
    X_test_reg = _align_features(X_test, dataset_feature_names, reg_feature_names)
    X_test_dir = _align_features(X_test, dataset_feature_names, dir_feature_names)

    # Predictions
    dmat_reg = DMatrix(X_test_reg, feature_names=reg_feature_names)
    dmat_dir = DMatrix(X_test_dir, feature_names=dir_feature_names)

    ret_pred = reg_booster.predict(dmat_reg)  # predicted ret_1h
    p_up = dir_booster.predict(dmat_dir)      # P(up), since objective=binary:logistic

    # Ensemble signal: require both high p_up and sufficiently positive ret_pred
    signal_ensemble = ((p_up >= p_up_min) & (ret_pred >= ret_min)).astype(int)

    # Direction-only baseline: only p_up threshold
    signal_dir_only = (p_up >= 0.5).astype(int)

    ret = y_test.astype(float)

    ens_stats = _compute_trade_stats(ret, signal_ensemble)
    dir_stats = _compute_trade_stats(ret, signal_dir_only)

    print(
        f"Ensemble strategy (p_up >= {p_up_min}, ret_pred >= {ret_min}):\n"
        f"  n_trades: {int(ens_stats['n_trades'])}\n"
        f"  hit_rate: {ens_stats['hit_rate']:.4f}\n"
        f"  avg_ret_per_trade: {ens_stats['avg_ret_per_trade']:.6f}\n"
        f"  cum_ret: {ens_stats['cum_ret']:.6f}\n"
        f"  max_drawdown: {ens_stats['max_drawdown']:.6f}\n"
    )

    print(
        f"Direction-only baseline (p_up >= 0.5):\n"
        f"  n_trades: {int(dir_stats['n_trades'])}\n"
        f"  hit_rate: {dir_stats['hit_rate']:.4f}\n"
        f"  avg_ret_per_trade: {dir_stats['avg_ret_per_trade']:.6f}\n"
        f"  cum_ret: {dir_stats['cum_ret']:.6f}\n"
        f"  max_drawdown: {dir_stats['max_drawdown']:.6f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble trading signals from regression and direction models.",
    )
    parser.add_argument(
        "--dataset-path-reg",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression npz file with test split.",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default="artifacts/models/xgb_ret1h_v1",
        help="Directory containing regression model and metadata.",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default="artifacts/models/xgb_dir1h_v1",
        help="Directory containing direction model and metadata.",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=0.6,
        help="Minimum P(up) for ensemble signal.",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=0.0005,
        help="Minimum predicted ret_1h for ensemble signal.",
    )

    args = parser.parse_args()

    evaluate_ensemble_signals(
        dataset_path_reg=args.dataset_path_reg,
        reg_model_dir=args.reg_model_dir,
        dir_model_dir=args.dir_model_dir,
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
    )


if __name__ == "__main__":
    main()
