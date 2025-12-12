import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from xgboost import Booster, DMatrix

from src.config_trading import (
    DEFAULT_FEE_BPS,
    DEFAULT_P_UP_MIN,
    DEFAULT_RET_MIN,
    DEFAULT_SLIPPAGE_BPS,
)


def _load_npz_dataset(path: str) -> Dict[str, Any]:
    """Load train/val/test splits and feature names from an npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset npz not found: {path}")

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

    feature_names = data["feature_names"].tolist()

    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "feature_names": feature_names,
    }


def _load_xgb_booster(model_dir: str, model_filename: str, meta_filename: str) -> Tuple[Booster, List[str]]:
    """Load an XGBoost Booster and its feature names from metadata."""
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
    """Reorder columns of X to match the model's expected feature order."""
    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape [N, F], got {X.shape}")

    name_to_idx = {name: i for i, name in enumerate(dataset_feature_names)}

    indices: List[int] = []
    for name in model_feature_names:
        if name not in name_to_idx:
            raise KeyError(f"Feature '{name}' required by model is missing from dataset")
        indices.append(name_to_idx[name])

    return X[:, indices]


def _apply_costs(ret: np.ndarray, signal: np.ndarray, fee_bps: float, slippage_bps: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    signal = np.asarray(signal).astype(bool)
    if ret.shape != signal.shape:
        raise ValueError("Return and signal arrays must share the same shape for cost adjustments.")

    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative.")

    cost_per_trade = (fee_bps + slippage_bps) / 10_000.0
    if cost_per_trade == 0:
        return ret

    ret_adj = ret.copy()
    ret_adj[signal] = ret_adj[signal] - cost_per_trade
    return ret_adj


def _compute_trade_stats(ret: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    """Compute trade-level statistics for a given return series and signal."""
    mask = signal.astype(bool)
    n_trades = int(mask.sum())

    if n_trades == 0:
        return {
            "n_trades": 0.0,
            "hit_rate": 0.0,
            "avg_ret_per_trade": 0.0,
            "cum_ret": 0.0,
            "ret_std": 0.0,
            "max_drawdown": 0.0,
        }

    ret_trades = ret[mask]
    hit_rate = float(np.mean(ret_trades > 0))
    avg_ret = float(np.mean(ret_trades))
    cum_ret = float(np.sum(ret_trades))
    ret_std = float(np.std(ret_trades, ddof=0))

    equity_curve = np.cumsum(ret_trades)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0

    return {
        "n_trades": float(n_trades),
        "hit_rate": hit_rate,
        "avg_ret_per_trade": avg_ret,
        "cum_ret": cum_ret,
        "ret_std": ret_std,
        "max_drawdown": max_drawdown,
    }


@dataclass
class ThresholdCandidate:
    p_up_min: float
    ret_min: float
    n_trades_val: float
    hit_rate_val: float
    avg_ret_per_trade_val: float
    cum_ret_val: float
    ret_std_val: float
    max_drawdown_val: float

    @property
    def sharpe_like_val(self) -> float:
        if self.ret_std_val <= 0.0:
            return float("-inf")
        return self.cum_ret_val / self.ret_std_val


def _normalize_grid_arg(value: Optional[Any]) -> Optional[str]:
    """Normalize argparse inputs that may be strings or lists into a comma-separated string."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return ""
        return ",".join(str(v) for v in value)
    return str(value)


def _parse_float_list(values: Optional[str], default: List[float]) -> List[float]:
    """Parse a comma-separated list of floats from CLI, with a default."""
    if values is None:
        return list(default)
    values_str = values.strip()
    if values_str == "":
        return list(default)
    parts = [v.strip() for v in values_str.split(",") if v.strip() != ""]
    if not parts:
        return list(default)
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError as exc:
            raise ValueError(f"Could not parse float value '{p}' in list '{values}'") from exc
    return out


def search_ensemble_thresholds(
    dataset_path: str,
    reg_model_dir: str,
    dir_model_dir: str,
    p_up_min_list: List[float],
    ret_min_list: List[float],
    fee_bps: float,
    slippage_bps: float,
    output_dir: Optional[str],
    min_trades_preferred: int = 10,
    min_trades_fallback: int = 5,
    top_k: int = 5,
    objective: str = "cumret",
    max_dd: Optional[float] = None,
    min_trades_constraint: Optional[int] = None,
) -> None:
    """Grid-search ensemble thresholds on validation and report test metrics."""
    data = _load_npz_dataset(dataset_path)
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
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

    # Align features for each model separately (validation and test)
    X_val_reg = _align_features(X_val, dataset_feature_names, reg_feature_names)
    X_val_dir = _align_features(X_val, dataset_feature_names, dir_feature_names)
    X_test_reg = _align_features(X_test, dataset_feature_names, reg_feature_names)
    X_test_dir = _align_features(X_test, dataset_feature_names, dir_feature_names)

    # Predictions on validation
    dmat_val_reg = DMatrix(X_val_reg, feature_names=reg_feature_names)
    dmat_val_dir = DMatrix(X_val_dir, feature_names=dir_feature_names)
    ret_pred_val = reg_booster.predict(dmat_val_reg)
    p_up_val = dir_booster.predict(dmat_val_dir)

    # Predictions on test (for the final recommended thresholds)
    dmat_test_reg = DMatrix(X_test_reg, feature_names=reg_feature_names)
    dmat_test_dir = DMatrix(X_test_dir, feature_names=dir_feature_names)
    ret_pred_test = reg_booster.predict(dmat_test_reg)
    p_up_test = dir_booster.predict(dmat_test_dir)

    ret_val = y_val.astype(float)
    ret_test = y_test.astype(float)

    # Evaluate all combinations on validation
    results: List[ThresholdCandidate] = []
    for p_up_min in p_up_min_list:
        for ret_min in ret_min_list:
            signal_val = ((p_up_val >= p_up_min) & (ret_pred_val >= ret_min)).astype(int)
            ret_val_net = _apply_costs(ret_val, signal_val, fee_bps, slippage_bps)
            stats_val = _compute_trade_stats(ret_val_net, signal_val)

            results.append(
                ThresholdCandidate(
                    p_up_min=float(p_up_min),
                    ret_min=float(ret_min),
                    n_trades_val=float(stats_val["n_trades"]),
                    hit_rate_val=float(stats_val["hit_rate"]),
                    avg_ret_per_trade_val=float(stats_val["avg_ret_per_trade"]),
                    cum_ret_val=float(stats_val["cum_ret"]),
                    ret_std_val=float(stats_val["ret_std"]),
                    max_drawdown_val=float(stats_val["max_drawdown"]),
                )
            )

    # Sort all results for reporting (descending cum_ret, then hit_rate)
    results_sorted = sorted(
        results,
        key=lambda r: (r.cum_ret_val, r.hit_rate_val),
        reverse=True,
    )

    # Filter candidates for recommended thresholds
    def apply_min_trade_filters(
        entries: Iterable[ThresholdCandidate],
        constraint: Optional[int],
    ) -> List[ThresholdCandidate]:
        if constraint is not None:
            return [r for r in entries if r.n_trades_val >= float(constraint)]

        filtered = [r for r in entries if r.n_trades_val >= float(min_trades_preferred)]
        if filtered:
            return filtered
        filtered = [r for r in entries if r.n_trades_val >= float(min_trades_fallback)]
        if filtered:
            return filtered
        filtered = [r for r in entries if r.n_trades_val >= 1.0]
        if filtered:
            return filtered
        return list(entries)

    def select_candidate(
        entries: List[ThresholdCandidate],
        objective_name: str,
        max_dd_constraint: Optional[float],
    ) -> ThresholdCandidate:
        # Apply objective-specific constraints
        constrained = list(entries)

        if objective_name == "cumret_with_dd_constraint" and max_dd_constraint is not None:
            constrained = [r for r in constrained if r.max_drawdown_val >= float(max_dd_constraint)]
            if not constrained:
                print(
                    "Warning: No candidates satisfy the max drawdown constraint. "
                    "Falling back to candidates filtered only by trade count.",
                    file=sys.stderr,
                )
                constrained = list(entries)

        if objective_name == "sharpe_like":
            constrained = [r for r in constrained if r.ret_std_val > 0.0]
            if not constrained:
                print(
                    "Warning: No candidates have positive return std for Sharpe-like score. "
                    "Falling back to trade-count filtered candidates.",
                    file=sys.stderr,
                )
                constrained = list(entries)

        if not constrained:
            raise RuntimeError("No candidates available for selection.")

        if objective_name == "sharpe_like":
            return max(constrained, key=lambda r: (r.sharpe_like_val, r.cum_ret_val))

        if objective_name == "cumret_with_dd_constraint":
            return max(constrained, key=lambda r: (r.cum_ret_val, r.hit_rate_val))

        return max(constrained, key=lambda r: (r.cum_ret_val, r.hit_rate_val))

    filtered_candidates = apply_min_trade_filters(results_sorted, min_trades_constraint)

    if not filtered_candidates and min_trades_constraint is not None:
        print(
            "Warning: No candidates satisfy the provided min-trades constraint. "
            "Falling back to automatic trade-count fallback logic.",
            file=sys.stderr,
        )
        filtered_candidates = apply_min_trade_filters(results_sorted, None)

    if not filtered_candidates:
        print("No valid threshold combinations found on validation set.")
        return

    recommended = select_candidate(filtered_candidates, objective, max_dd)

    # Print top-K validation combinations (excluding pure zero-trade ones for readability)
    print("Top validation combos (p_up_min, ret_min):\n")
    shown = 0
    for r in results_sorted:
        if r.n_trades_val <= 0.0:
            continue
        print(
            f"p_up_min={r.p_up_min:.2f}, ret_min={r.ret_min:.5f}:\n"
            f"  n_trades_val: {int(r.n_trades_val)}\n"
            f"  hit_rate_val: {r.hit_rate_val:.4f}\n"
            f"  avg_ret_per_trade_val: {r.avg_ret_per_trade_val:.6f}\n"
            f"  cum_ret_val: {r.cum_ret_val:.6f}\n"
            f"  max_drawdown_val: {r.max_drawdown_val:.6f}\n"
            f"  ret_std_val: {r.ret_std_val:.6f}\n"
        )
        shown += 1
        if shown >= top_k:
            break
    if shown == 0:
        print("(No validation combinations produced any trades.)\n")

    # Recommended thresholds
    p_up_star = recommended.p_up_min
    ret_star = recommended.ret_min

    # Compute ensemble performance on test with recommended thresholds
    signal_test_ensemble = ((p_up_test >= p_up_star) & (ret_pred_test >= ret_star)).astype(int)
    ret_test_net_ensemble = _apply_costs(ret_test, signal_test_ensemble, fee_bps, slippage_bps)
    stats_test_ensemble = _compute_trade_stats(ret_test_net_ensemble, signal_test_ensemble)

    # Direction-only baseline on test (p_up >= 0.5)
    signal_test_dir_only = (p_up_test >= 0.5).astype(int)
    ret_test_net_dir = _apply_costs(ret_test, signal_test_dir_only, fee_bps, slippage_bps)
    stats_test_dir_only = _compute_trade_stats(ret_test_net_dir, signal_test_dir_only)

    print("Recommended thresholds based on validation:")
    print(f"  p_up_min*: {p_up_star:.2f}")
    print(f"  ret_min*: {ret_star:.5f}")
    print(f"  n_trades_val*: {int(recommended.n_trades_val)}")
    print(f"  cum_ret_val*: {recommended.cum_ret_val:.6f}")
    print(f"  max_drawdown_val*: {recommended.max_drawdown_val:.6f}")
    print(f"  ret_std_val*: {recommended.ret_std_val:.6f}")

    if objective == "sharpe_like":
        print(f"  sharpe_like_val*: {recommended.sharpe_like_val:.6f}")
    if objective == "cumret_with_dd_constraint" and max_dd is not None:
        constraint_ok = recommended.max_drawdown_val >= float(max_dd)
        print(f"  satisfied_max_dd_constraint: {constraint_ok}")

    print()

    print("Test performance (ensemble):")
    print(f"  n_trades_test: {int(stats_test_ensemble['n_trades'])}")
    print(f"  hit_rate_test: {stats_test_ensemble['hit_rate']:.4f}")
    print(f"  avg_ret_per_trade_test: {stats_test_ensemble['avg_ret_per_trade']:.6f}")
    print(f"  cum_ret_test: {stats_test_ensemble['cum_ret']:.6f}\n")

    print("Test performance (direction-only baseline, p_up >= 0.5):")
    print(f"  n_trades_test: {int(stats_test_dir_only['n_trades'])}")
    print(f"  hit_rate_test: {stats_test_dir_only['hit_rate']:.4f}")
    print(f"  avg_ret_per_trade_test: {stats_test_dir_only['avg_ret_per_trade']:.6f}")
    print(f"  cum_ret_test: {stats_test_dir_only['cum_ret']:.6f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        best_config = {
            "p_up_min": float(p_up_star),
            "ret_min": float(ret_star),
            "objective": objective,
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "validation": {
                "n_trades": float(recommended.n_trades_val),
                "hit_rate": float(recommended.hit_rate_val),
                "avg_ret_per_trade": float(recommended.avg_ret_per_trade_val),
                "cum_ret": float(recommended.cum_ret_val),
                "ret_std": float(recommended.ret_std_val),
                "max_drawdown": float(recommended.max_drawdown_val),
            },
            "test_ensemble": stats_test_ensemble,
            "test_direction_baseline": stats_test_dir_only,
        }

        best_config_path = os.path.join(output_dir, "best_config.json")
        with open(best_config_path, "w", encoding="utf-8") as handle:
            json.dump(best_config, handle, indent=2)

        summary_lines = [
            "Threshold Search Summary",
            "==========================",
            f"p_up_min*: {p_up_star:.2f}",
            f"ret_min*: {ret_star:.5f}",
            f"Validation trades: {int(recommended.n_trades_val)}",
            f"Validation cum_ret (net): {recommended.cum_ret_val:.6f}",
            f"Validation hit_rate: {recommended.hit_rate_val:.4f}",
            "",
            "Test Ensemble (net):",
            f"  n_trades: {int(stats_test_ensemble['n_trades'])}",
            f"  hit_rate: {stats_test_ensemble['hit_rate']:.4f}",
            f"  avg_ret_per_trade: {stats_test_ensemble['avg_ret_per_trade']:.6f}",
            f"  cum_ret: {stats_test_ensemble['cum_ret']:.6f}",
            "",
            "Test Direction Baseline (net):",
            f"  n_trades: {int(stats_test_dir_only['n_trades'])}",
            f"  hit_rate: {stats_test_dir_only['hit_rate']:.4f}",
            f"  avg_ret_per_trade: {stats_test_dir_only['avg_ret_per_trade']:.6f}",
            f"  cum_ret: {stats_test_dir_only['cum_ret']:.6f}",
        ]

        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(summary_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search ensemble thresholds on validation and report recommended "
            "thresholds plus test performance."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression npz file with train/val/test splits.",
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
        "--p-up-grid",
        "--p-up-min-grid",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of P(up) thresholds to search. "
            "If provided, overrides --p-up-min-list/default grid."
        ),
    )
    parser.add_argument(
        "--ret-min-grid",
        "--ret-min-grid-values",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of ret_1h thresholds to search. "
            "If provided, overrides --ret-min-list/default grid."
        ),
    )
    parser.add_argument(
        "--p-up-min-list",
        type=str,
        default=None,
        help=(
            "Comma-separated list of P(up) thresholds to search. "
            "Defaults to a grid around the v1 default if not provided."
        ),
    )
    parser.add_argument(
        "--ret-min-list",
        type=str,
        default=None,
        help=(
            "Comma-separated list of ret_1h thresholds to search. "
            "Defaults to a grid around the v1 default if not provided."
        ),
    )
    parser.add_argument(
        "--min-trades-preferred",
        type=int,
        default=10,
        help="Preferred minimum validation trades when choosing thresholds.",
    )
    parser.add_argument(
        "--min-trades-fallback",
        type=int,
        default=5,
        help="Fallback minimum validation trades if preferred threshold is unmet.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help=(
            "Optional hard minimum number of validation trades. When provided, only candidates meeting this "
            "threshold are considered across all objectives."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top validation combos to print.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["cumret", "sharpe_like", "cumret_with_dd_constraint"],
        default="cumret",
        help=(
            "Objective used for selecting thresholds. 'cumret' maximizes cumulative return (default). "
            "'sharpe_like' maximizes return/std for active trades. 'cumret_with_dd_constraint' maximizes return "
            "subject to an optional drawdown cap."
        ),
    )
    parser.add_argument(
        "--max-dd",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed validation max drawdown (log space). "
            "Only applied when --objective cumret_with_dd_constraint is selected."
        ),
    )

    parser.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help="Per-trade fee assumption in basis points (applied when a trade is taken).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help="Per-trade slippage assumption in basis points (applied when a trade is taken).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write search artifacts (best_config.json, summary.txt).",
    )

    args = parser.parse_args()

    # Default grids if none provided, centered around v1 defaults
    default_p_up = [DEFAULT_P_UP_MIN - 0.05, DEFAULT_P_UP_MIN, DEFAULT_P_UP_MIN + 0.05]
    default_ret_min = [DEFAULT_RET_MIN, DEFAULT_RET_MIN + 0.00025, DEFAULT_RET_MIN + 0.0005]

    # Determine grids with override precedence: *-grid > *-min-list > default
    p_up_grid_normalized = _normalize_grid_arg(args.p_up_grid)
    ret_min_grid_normalized = _normalize_grid_arg(args.ret_min_grid)

    if p_up_grid_normalized is not None and p_up_grid_normalized.strip() != "":
        p_up_min_list = _parse_float_list(p_up_grid_normalized, default_p_up)
    else:
        p_up_min_list = _parse_float_list(_normalize_grid_arg(args.p_up_min_list), default_p_up)

    if ret_min_grid_normalized is not None and ret_min_grid_normalized.strip() != "":
        ret_min_list = _parse_float_list(ret_min_grid_normalized, default_ret_min)
    else:
        ret_min_list = _parse_float_list(_normalize_grid_arg(args.ret_min_list), default_ret_min)

    search_ensemble_thresholds(
        dataset_path=args.dataset_path,
        reg_model_dir=args.reg_model_dir,
        dir_model_dir=args.dir_model_dir,
        p_up_min_list=p_up_min_list,
        ret_min_list=ret_min_list,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        output_dir=args.output_dir,
        min_trades_preferred=args.min_trades_preferred,
        min_trades_fallback=args.min_trades_fallback,
        top_k=args.top_k,
        objective=args.objective,
        max_dd=args.max_dd,
        min_trades_constraint=args.min_trades,
    )


if __name__ == "__main__":
    main()
