import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.trading.signals import PreparedData, prepare_data_for_signals


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _build_time_splits(prepared: PreparedData, mode: str) -> DatasetSplits:
    df_all = prepared.df_all.reset_index(drop=True)
    X_all = prepared.X_all_ordered.to_numpy(dtype=np.float32)
    y_ret = df_all["ret_1h"].to_numpy(dtype=np.float32)

    if mode == "reg":
        y_all = y_ret
    else:
        y_all = (y_ret > 0.0).astype(np.int32)

    n_samples = len(df_all)
    if n_samples < 100:
        raise ValueError("Dataset too small for time-aware split (requires >= 100 rows).")

    idx_train_end = int(n_samples * 0.7)
    idx_val_end = int(n_samples * 0.85)

    if idx_val_end <= idx_train_end:
        raise ValueError("Validation split is empty; increase dataset size or adjust split ratios.")

    X_train = X_all[:idx_train_end]
    X_val = X_all[idx_train_end:idx_val_end]
    X_test = X_all[idx_val_end:]

    y_train = y_all[:idx_train_end]
    y_val = y_all[idx_train_end:idx_val_end]
    y_test = y_all[idx_val_end:]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
    )


def _sample_params(trial: optuna.Trial, mode: str) -> Dict[str, float]:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
    }

    if mode == "dir":
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, 5.0)

    return params


def _create_model(mode: str, params: Dict[str, float]):
    common = {
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    }

    if mode == "reg":
        model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **params, **common)
    else:
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params, **common)
    return model


def _evaluate_objective(mode: str, splits: DatasetSplits) -> Tuple[optuna.study.Study, optuna.trial.Trial]:
    X_train, X_val = splits.X_train, splits.X_val
    y_train, y_val = splits.y_train, splits.y_val

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, mode)
        model = _create_model(mode, params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        if mode == "reg":
            val_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            rmse = float(np.sqrt(mse))
            trial.set_user_attr("val_rmse", rmse)
            return rmse

        val_proba = model.predict_proba(X_val)[:, 1]
        val_proba = np.clip(val_proba, 1e-15, 1 - 1e-15)
        logloss = log_loss(y_val, val_proba)
        acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
        auc = roc_auc_score(y_val, val_proba)
        trial.set_user_attr("val_logloss", float(logloss))
        trial.set_user_attr("val_accuracy", float(acc))
        trial.set_user_attr("val_auc", float(auc))
        return float(logloss)

    return objective


def _run_study(args: argparse.Namespace, splits: DatasetSplits) -> optuna.Study:
    direction = "minimize"

    if args.storage:
        study = optuna.create_study(
            direction=direction,
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction=direction, study_name=args.study_name)

    objective = _evaluate_objective(args.mode, splits)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    if len(study.trials) == 0:
        raise RuntimeError("Optuna study completed without trials.")

    return study


def _retrain_best(args: argparse.Namespace, splits: DatasetSplits, best_params: Dict[str, float]):
    X_trainval = np.vstack([splits.X_train, splits.X_val])
    y_trainval = np.concatenate([splits.y_train, splits.y_val])

    model = _create_model(args.mode, best_params)
    model.fit(X_trainval, y_trainval, verbose=False)

    if args.mode == "reg":
        test_pred = model.predict(splits.X_test)
        mse = mean_squared_error(splits.y_test, test_pred)
        rmse = float(np.sqrt(mse))
        return model, {"test_rmse": rmse}

    test_proba = model.predict_proba(splits.X_test)[:, 1]
    test_proba = np.clip(test_proba, 1e-15, 1 - 1e-15)
    logloss = log_loss(splits.y_test, test_proba)
    acc = accuracy_score(splits.y_test, (test_proba >= 0.5).astype(int))
    auc = roc_auc_score(splits.y_test, test_proba)
    return model, {
        "test_logloss": float(logloss),
        "test_accuracy": float(acc),
        "test_auc": float(auc),
    }


def _save_outputs(output_dir: str, model, summary: Dict[str, Dict[str, float]]):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.json")
    model.save_model(model_path)

    summary_path = os.path.join(output_dir, "best_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    legacy_path = os.path.join(output_dir, "summary.json")
    with open(legacy_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for XGBoost 1h models.")
    parser.add_argument("--dataset-path", type=str, default="artifacts/datasets/btc_features_1h_splits.npz")
    parser.add_argument("--mode", choices=["reg", "dir"], required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=None, help="Timeout for the study in seconds.")
    parser.add_argument("--study-name", type=str, default="xgb_optuna_study")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///study.db).")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    prepared = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")
    splits = _build_time_splits(prepared, args.mode)

    print(
        f"Running Optuna search (mode={args.mode}, n_trials={args.n_trials}, timeout={args.timeout}) "
        f"with split sizes train={len(splits.y_train)}, val={len(splits.y_val)}, test={len(splits.y_test)}"
    )

    study = _run_study(args, splits)
    best_trial = study.best_trial
    best_params = best_trial.params

    print(f"Best trial #{best_trial.number} value={best_trial.value:.6f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    model, test_metrics = _retrain_best(args, splits, best_params)

    summary = {
        "mode": args.mode,
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "best_trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": best_params,
        "val_metrics": best_trial.user_attrs,
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": len(splits.y_train),
            "val": len(splits.y_val),
            "test": len(splits.y_test),
        },
    }

    _save_outputs(args.output_dir, model, summary)
    print(f"Saved best model and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
