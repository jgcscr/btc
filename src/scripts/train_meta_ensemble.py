import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

CostSchedule = Tuple[float, float, str]


@dataclass
class MetaEnsembleResult:
    fee_bps: float
    slippage_bps: float
    label: str
    n_trades: int
    hit_rate: float
    net_return: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an out-of-fold stacked logistic meta-ensemble over model probabilities.",
    )
    parser.add_argument(
        "--transformer-csv",
        type=Path,
        default=Path("artifacts/backtests/historical_1h_pup060_full_simplified/backtest_signals.csv"),
        help="CSV with transformer per-bar probabilities.",
    )
    parser.add_argument(
        "--lstm-csv",
        type=Path,
        default=Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
        help="CSV with LSTM per-bar probabilities.",
    )
    parser.add_argument(
        "--xgb-csv",
        type=Path,
        default=Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
        help="CSV with XGB per-bar probabilities (baseline).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/backtests/backtest_signals_meta_ensemble.csv"),
        help="Destination for the meta-ensemble backtest log with OOF train/val rows and held-out test rows.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("artifacts/backtests/meta_ensemble_config.json"),
        help="Where to write coefficients/threshold metadata for realtime inference.",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.5,
        help="Probability threshold to activate the meta-ensemble trade signal.",
    )
    parser.add_argument(
        "--oof-splits",
        type=int,
        default=5,
        help="Number of time-series OOF folds for stacked training (default: 5).",
    )
    parser.add_argument(
        "--disable-regime-features",
        action="store_true",
        help="Disable engineered regime/volatility meta features.",
    )
    parser.add_argument(
        "--component-weight-spec",
        type=str,
        default=None,
        help="Optional comma-separated component weights, e.g. transformer:0,lstm:1.5,xgb:1.5.",
    )
    return parser.parse_args()


def parse_component_weight_spec(spec: str | None) -> Dict[str, float]:
    weights = {
        "p_up_transformer": 1.0,
        "p_up_lstm": 1.0,
        "p_up_xgb": 1.0,
    }
    if not spec:
        return weights

    aliases = {
        "transformer": "p_up_transformer",
        "lstm": "p_up_lstm",
        "xgb": "p_up_xgb",
        "p_up_transformer": "p_up_transformer",
        "p_up_lstm": "p_up_lstm",
        "p_up_xgb": "p_up_xgb",
    }
    for raw_chunk in str(spec).split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid component weight chunk '{chunk}'. Expected name:value format.")
        raw_name, raw_value = chunk.split(":", 1)
        key = aliases.get(raw_name.strip())
        if key is None:
            continue
        weights[key] = float(raw_value.strip())

    if sum(max(value, 0.0) for value in weights.values()) <= 0.0:
        raise ValueError("Component weight spec disabled all meta-ensemble inputs.")
    return weights


def weighted_probability_average(df: pd.DataFrame, columns: Sequence[str], component_weights: Dict[str, float]) -> pd.Series:
    numer = pd.Series(0.0, index=df.index, dtype=float)
    denom = 0.0
    for column in columns:
        weight = max(float(component_weights.get(column, 1.0)), 0.0)
        if weight <= 0.0:
            continue
        numer = numer + pd.to_numeric(df[column], errors="coerce").fillna(0.0) * weight
        denom += weight
    if denom <= 0.0:
        raise ValueError("At least one positive meta-ensemble component weight is required.")
    return numer / denom


def load_model_frame(path: Path, prob_column_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ts", "ret_1h", "p_up"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
    selected = ["ts", "ret_1h", "p_up", "signal_ensemble", "ret_ensemble_net"]
    return df[selected].rename(columns={"p_up": prob_column_name})


def validate_alignment(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0].copy()
    for frame in frames[1:]:
        if len(frame) != len(base):
            raise ValueError("Input CSVs have mismatched lengths; ensure they cover identical windows.")
        if not (frame["ts"].values == base["ts"].values).all():
            raise ValueError("Timestamp alignment mismatch between input CSVs.")
        if not np.allclose(frame["ret_1h"].values, base["ret_1h"].values):
            raise ValueError("Mismatch in realized returns across input CSVs.")
    return base


def compute_split_indices(n_rows: int) -> Tuple[int, int]:
    n_train = int(n_rows * 0.70)
    n_val = int(n_rows * 0.15)
    return n_train, n_val


def _expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += (np.sum(mask) / max(n, 1)) * abs(acc - conf)
    return float(ece)


def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def build_meta_features(
    master: pd.DataFrame,
    *,
    add_regime_features: bool,
    component_weights: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str]]:
    master = master.copy()
    base_cols = ["p_up_transformer", "p_up_lstm", "p_up_xgb"]
    feature_cols = list(base_cols)

    if add_regime_features:
        p_stack = master[base_cols].astype(float)
        weighted_mean = weighted_probability_average(master, base_cols, component_weights)
        centered = p_stack.sub(weighted_mean, axis=0)
        weight_vector = np.asarray([max(float(component_weights.get(col, 1.0)), 0.0) for col in base_cols], dtype=float)
        weight_total = float(weight_vector.sum())
        weighted_std = np.sqrt(
            np.maximum(
                0.0,
                (centered.to_numpy(dtype=float) ** 2 @ weight_vector) / max(weight_total, 1e-12),
            )
        )
        master["meta_prob_mean"] = weighted_mean
        master["meta_prob_std"] = p_stack.std(axis=1, ddof=0)
        master["meta_prob_span"] = p_stack.max(axis=1) - p_stack.min(axis=1)
        master["meta_prob_weighted_std"] = weighted_std
        ret = pd.to_numeric(master["ret_1h"], errors="coerce").fillna(0.0)
        master["meta_ret_vol_24"] = ret.rolling(window=24, min_periods=8).std(ddof=0).fillna(0.0)
        master["meta_ret_trend_24"] = ret.rolling(window=24, min_periods=8).mean().fillna(0.0)
        master["meta_ret_abs_6"] = ret.abs().rolling(window=6, min_periods=3).mean().fillna(0.0)
        feature_cols.extend(
            [
                "meta_prob_mean",
                "meta_prob_std",
                "meta_prob_span",
                "meta_prob_weighted_std",
                "meta_ret_vol_24",
                "meta_ret_trend_24",
                "meta_ret_abs_6",
            ]
        )

    return master, feature_cols


def compute_oof_probabilities(X: pd.DataFrame, y: pd.Series, n_splits: int) -> np.ndarray:
    n_rows = len(X)
    if n_rows < 20:
        return np.full(n_rows, np.nan, dtype=float)

    splits = max(2, min(int(n_splits), max(2, n_rows // 30)))
    cv = TimeSeriesSplit(n_splits=splits)
    oof = np.full(n_rows, np.nan, dtype=float)
    for train_idx, val_idx in cv.split(X):
        model = fit_logistic_regression(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def evaluate_probabilities(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(prob)
    if not np.any(mask):
        return {
            "rows": 0.0,
            "accuracy": float("nan"),
            "roc_auc": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "ece_10": float("nan"),
        }

    y = y_true[mask].astype(int)
    p = np.clip(prob[mask], 1e-6, 1.0 - 1e-6)
    pred = (p >= threshold).astype(int)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "rows": float(y.size),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": _expected_calibration_error(y, p, bins=10),
    }


def summarize_meta_backtest(
    df: pd.DataFrame,
    schedule: CostSchedule,
    signal_column: str,
    net_column: str,
) -> MetaEnsembleResult:
    fee_bps, slippage_bps, label = schedule
    trades = int(df[signal_column].sum())
    active = df[df[signal_column] > 0]
    hit_rate = float((active["ret_1h"] > 0).mean()) if trades > 0 else float("nan")
    net_return = float(df[net_column].sum())
    return MetaEnsembleResult(fee_bps, slippage_bps, label, trades, hit_rate, net_return)


def adjust_baseline_net(
    base_net: float,
    base_fee_bps: float,
    base_slip_bps: float,
    target_fee_bps: float,
    target_slip_bps: float,
    trades: int,
) -> float:
    delta_bps = (target_fee_bps + target_slip_bps) - (base_fee_bps + base_slip_bps)
    return base_net - trades * (delta_bps / 10_000.0)


def save_meta_config(
    path: Path,
    feature_columns: Sequence[str],
    intercept: float,
    coefficients: Sequence[float],
    threshold: float,
    schedules: Sequence[Dict[str, float]],
    oof_metrics: Dict[str, float],
    trainval_metrics: Dict[str, float],
    oof_splits: int,
    component_weights: Dict[str, float],
) -> None:
    payload = {
        "feature_columns": list(feature_columns),
        "intercept": float(intercept),
        "coefficients": [float(coef) for coef in coefficients],
        "threshold": float(threshold),
        "oof_splits": int(oof_splits),
        "schedules": [
            {
                "fee_bps": float(schedule["fee_bps"]),
                "slippage_bps": float(schedule["slippage_bps"]),
                "label": str(schedule["label"]),
            }
            for schedule in schedules
        ],
        "oof_metrics": {key: float(value) for key, value in oof_metrics.items()},
        "trainval_metrics": {key: float(value) for key, value in trainval_metrics.items()},
        "component_weights": {key: float(value) for key, value in component_weights.items()},
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved meta-ensemble config to {path}")


def main() -> None:
    args = parse_args()
    component_weights = parse_component_weight_spec(args.component_weight_spec)

    transformer_df = load_model_frame(args.transformer_csv, "p_up_transformer")
    lstm_df = load_model_frame(args.lstm_csv, "p_up_lstm")
    xgb_df = load_model_frame(args.xgb_csv, "p_up_xgb")

    master = validate_alignment([transformer_df, lstm_df, xgb_df])
    master["p_up_lstm"] = lstm_df["p_up_lstm"].values
    master["p_up_xgb"] = xgb_df["p_up_xgb"].values

    master["target"] = (master["ret_1h"] > 0.0).astype(int)
    master, feature_cols = build_meta_features(
        master,
        add_regime_features=not args.disable_regime_features,
        component_weights=component_weights,
    )

    n_rows = len(master)
    n_train, n_val = compute_split_indices(n_rows)
    n_test_start = n_train + n_val

    trainval_df = master.iloc[:n_test_start].copy()
    test_df = master.iloc[n_test_start:].copy()

    X_trainval = trainval_df[feature_cols]
    y_trainval = trainval_df["target"]

    oof_prob = compute_oof_probabilities(X_trainval, y_trainval, n_splits=args.oof_splits)
    oof_metrics = evaluate_probabilities(
        y_trainval.to_numpy(dtype=int),
        oof_prob,
        threshold=args.weight_threshold,
    )

    print("OOF metrics (stacking sanity check):")
    for key, value in oof_metrics.items():
        print(f"  {key}: {value:.6f}")

    final_model = fit_logistic_regression(X_trainval, y_trainval)
    trainval_prob = final_model.predict_proba(X_trainval)[:, 1]
    trainval_metrics = evaluate_probabilities(
        y_trainval.to_numpy(dtype=int),
        trainval_prob,
        threshold=args.weight_threshold,
    )

    print("\nFinal model metrics on train+val:")
    for key, value in trainval_metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\nFinal-fit coefficients:")
    final_coef = pd.Series(final_model.coef_[0], index=feature_cols)
    for name, coef in final_coef.items():
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept: {final_model.intercept_[0]:.6f}")

    full_backtest = master.copy()
    full_backtest["p_up_meta"] = np.nan
    full_backtest.loc[: n_test_start - 1, "p_up_meta"] = oof_prob
    full_backtest.loc[n_test_start:, "p_up_meta"] = final_model.predict_proba(test_df[feature_cols])[:, 1]
    # Gate signal uses a stable average of base probabilities to reduce over-compressed meta scores.
    full_backtest["p_up_gate"] = weighted_probability_average(
        full_backtest,
        ["p_up_transformer", "p_up_lstm", "p_up_xgb"],
        component_weights,
    )
    full_backtest["signal_meta"] = (full_backtest["p_up_gate"] >= args.weight_threshold).astype(int)
    full_backtest["y_true"] = full_backtest["target"].astype(int)
    full_backtest["fold"] = (np.arange(len(full_backtest)) // max(int(n_val), 1)).astype(int)
    full_backtest["backtest_split"] = np.where(
        np.arange(len(full_backtest)) < n_test_start,
        "trainval_oof",
        "test_holdout",
    )

    schedules: List[CostSchedule] = [
        (2.0, 1.0, "fee_20_10"),
        (2.5, 1.2, "fee_25_12"),
        (3.0, 1.5, "fee_30_15"),
    ]
    schedule_dicts = [
        {"fee_bps": fee_bps, "slippage_bps": slippage_bps, "label": label}
        for fee_bps, slippage_bps, label in schedules
    ]

    test_df = full_backtest.iloc[n_test_start:].copy()
    test_df["ret_gross_meta"] = test_df["ret_1h"] * test_df["signal_meta"]
    full_backtest["ret_gross_meta"] = full_backtest["ret_1h"] * full_backtest["signal_meta"]

    meta_results: List[MetaEnsembleResult] = []
    for fee_bps, slippage_bps, label in schedules:
        per_trade_cost = (fee_bps + slippage_bps) / 10_000.0
        net_column = f"ret_net_{label}"
        test_df[net_column] = test_df["ret_gross_meta"] - per_trade_cost * test_df["signal_meta"]
        full_backtest[net_column] = full_backtest["ret_gross_meta"] - per_trade_cost * full_backtest["signal_meta"]
        meta_results.append(
            summarize_meta_backtest(
                test_df,
                (fee_bps, slippage_bps, label),
                "signal_meta",
                net_column,
            ),
        )
        equity_col = f"equity_{label}"
        test_df[equity_col] = np.exp(np.cumsum(test_df[net_column]))
        full_backtest[equity_col] = np.exp(np.cumsum(full_backtest[net_column].fillna(0.0)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "ts",
        "ret_1h",
        *feature_cols,
        "p_up_meta",
        "p_up_gate",
        "signal_meta",
        "y_true",
        "fold",
        "backtest_split",
        "ret_gross_meta",
        "ret_net_fee_20_10",
        "ret_net_fee_25_12",
        "ret_net_fee_30_15",
        "equity_fee_20_10",
        "equity_fee_25_12",
        "equity_fee_30_15",
    ]
    full_backtest["p_up"] = full_backtest["p_up_gate"]
    full_backtest["signal_ensemble"] = full_backtest["signal_meta"]
    full_backtest["ret_ensemble_net"] = full_backtest["ret_net_fee_20_10"]
    full_backtest_output = full_backtest[
        output_columns + ["p_up", "signal_ensemble", "ret_ensemble_net"]
    ]
    full_backtest_output.to_csv(args.output_csv, index=False)
    print(f"\nSaved meta-ensemble backtest to {args.output_csv}")

    print("\nMeta-ensemble net returns (test split):")
    for result in meta_results:
        print(
            f"  {result.label}: net={result.net_return:.6f}, trades={result.n_trades}, hit_rate={result.hit_rate:.3f}",
        )

    save_meta_config(
        args.config_path,
        feature_cols,
        final_model.intercept_[0],
        final_model.coef_[0],
        args.weight_threshold,
        schedule_dicts,
        oof_metrics,
        trainval_metrics,
        args.oof_splits,
        component_weights,
    )

    base_fee = 2.0
    base_slip = 1.0
    xgb_test = xgb_df.iloc[n_test_start:].copy()
    baseline_trades = int(xgb_test["signal_ensemble"].sum())
    baseline_net_base = float(xgb_test["ret_ensemble_net"].sum())

    print("\nPure XGB baseline net returns (adjusted for costs):")
    for fee_bps, slippage_bps, label in schedules:
        adjusted = adjust_baseline_net(
            baseline_net_base,
            base_fee,
            base_slip,
            fee_bps,
            slippage_bps,
            baseline_trades,
        )
        print(f"  {label}: net={adjusted:.6f}, trades={baseline_trades}")


if __name__ == "__main__":
    main()
