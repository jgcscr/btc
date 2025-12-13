import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

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
        description="Train a logistic regression meta-ensemble on per-model probabilities and backtest on the test window.",
    )
    parser.add_argument(
        "--transformer-csv",
        type=Path,
        default=Path("artifacts/analysis/backtest_signals_transformer_dir1h_optuna_v2/backtest_signals.csv"),
        help="CSV with transformer per-bar probabilities.",
    )
    parser.add_argument(
        "--lstm-csv",
        type=Path,
        default=Path("artifacts/analysis/backtest_signals_lstm_dir1h_v2/backtest_signals.csv"),
        help="CSV with LSTM per-bar probabilities.",
    )
    parser.add_argument(
        "--xgb-csv",
        type=Path,
        default=Path("artifacts/analysis/backtest_signals_xgb_dir1h_optuna/backtest_signals.csv"),
        help="CSV with XGB per-bar probabilities (baseline).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/backtests/backtest_signals_meta_ensemble.csv"),
        help="Destination for the meta-ensemble backtest log (test split only).",
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
    return parser.parse_args()


def load_model_frame(path: Path, prob_column_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"ts", "ret_1h", "p_up"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
    return df[["ts", "ret_1h", "p_up", "signal_ensemble", "ret_ensemble_net"]].rename(
        columns={"p_up": prob_column_name},
    )


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


def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def evaluate_validation(model: LogisticRegression, X_val: pd.DataFrame, y_val: pd.Series, threshold: float) -> Dict[str, float]:
    if len(X_val) == 0:
        return {"accuracy": float("nan"), "roc_auc": float("nan"), "log_loss": float("nan")}
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_val, pred),
        "roc_auc": roc_auc_score(y_val, prob),
        "log_loss": log_loss(y_val, prob),
    }
    return metrics


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
    validation_metrics: Dict[str, float],
) -> None:
    payload = {
        "feature_columns": list(feature_columns),
        "intercept": float(intercept),
        "coefficients": [float(coef) for coef in coefficients],
        "threshold": float(threshold),
        "schedules": [
            {
                "fee_bps": float(schedule["fee_bps"]),
                "slippage_bps": float(schedule["slippage_bps"]),
                "label": str(schedule["label"]),
            }
            for schedule in schedules
        ],
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved meta-ensemble config to {path}")


def main() -> None:
    args = parse_args()

    transformer_df = load_model_frame(args.transformer_csv, "p_up_transformer")
    lstm_df = load_model_frame(args.lstm_csv, "p_up_lstm")
    xgb_df = load_model_frame(args.xgb_csv, "p_up_xgb")

    # Validate alignment and build master frame with probabilities
    master = validate_alignment([transformer_df, lstm_df, xgb_df])
    master["p_up_lstm"] = lstm_df["p_up_lstm"].values
    master["p_up_xgb"] = xgb_df["p_up_xgb"].values

    # Prepare features and target
    feature_cols = ["p_up_transformer", "p_up_lstm", "p_up_xgb"]
    master["target"] = (master["ret_1h"] > 0.0).astype(int)

    n_rows = len(master)
    n_train, n_val = compute_split_indices(n_rows)
    n_test_start = n_train + n_val

    train_df = master.iloc[:n_train]
    val_df = master.iloc[n_train:n_test_start]
    test_df = master.iloc[n_test_start:]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    # Train and evaluate on validation split
    val_model = fit_logistic_regression(X_train, y_train)
    val_metrics = evaluate_validation(val_model, X_val, y_val, args.weight_threshold)

    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    coef_series = pd.Series(val_model.coef_[0], index=feature_cols)
    print("\nValidation-fit coefficients:")
    for name, coef in coef_series.items():
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {val_model.intercept_[0]:.4f}")

    # Refit on train+val for final backtest predictions
    combined_df = master.iloc[:n_test_start]
    X_combined = combined_df[feature_cols]
    y_combined = combined_df["target"]
    final_model = fit_logistic_regression(X_combined, y_combined)
    print("\nFinal-fit coefficients (train+val):")
    final_coef = pd.Series(final_model.coef_[0], index=feature_cols)
    for name, coef in final_coef.items():
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {final_model.intercept_[0]:.4f}")

    master["p_up_meta"] = final_model.predict_proba(master[feature_cols])[:, 1]
    master["signal_meta"] = (master["p_up_meta"] >= args.weight_threshold).astype(int)

    # Prepare cost schedules and compute net returns on test split
    schedules: List[CostSchedule] = [
        (2.0, 1.0, "fee_20_10"),
        (2.5, 1.2, "fee_25_12"),
        (3.0, 1.5, "fee_30_15"),
    ]
    schedule_dicts = [
        {"fee_bps": fee_bps, "slippage_bps": slippage_bps, "label": label}
        for fee_bps, slippage_bps, label in schedules
    ]

    test_df = master.iloc[n_test_start:].copy()
    test_df["ret_gross_meta"] = test_df["ret_1h"] * test_df["signal_meta"]

    meta_results: List[MetaEnsembleResult] = []
    for fee_bps, slippage_bps, label in schedules:
        per_trade_cost = (fee_bps + slippage_bps) / 10_000.0
        net_column = f"ret_net_{label}"
        test_df[net_column] = test_df["ret_gross_meta"] - per_trade_cost * test_df["signal_meta"]
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

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_df_output = test_df[
        [
            "ts",
            "ret_1h",
            "p_up_transformer",
            "p_up_lstm",
            "p_up_xgb",
            "p_up_meta",
            "signal_meta",
            "ret_gross_meta",
            "ret_net_fee_20_10",
            "ret_net_fee_25_12",
            "ret_net_fee_30_15",
            "equity_fee_20_10",
            "equity_fee_25_12",
            "equity_fee_30_15",
        ]
    ]
    test_df_output.to_csv(args.output_csv, index=False)
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
        val_metrics,
    )

    # Baseline comparison using XGB backtest (same window)
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
