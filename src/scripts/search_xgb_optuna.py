import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.trading.signals import PreparedData, prepare_data_for_signals
from src.training.time_series_cv import build_time_series_folds


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _build_time_splits(prepared: PreparedData, mode: str, target_column: str, target_scale: float) -> DatasetSplits:
    df_all = prepared.df_all.reset_index(drop=True)
    X_all = prepared.X_all_ordered.to_numpy(dtype=np.float32)

    if target_column not in df_all.columns:
        raise KeyError(f"Target column '{target_column}' missing from prepared dataframe.")

    y_ret = df_all[target_column].to_numpy(dtype=np.float32)

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

    if mode == "reg" and target_scale != 1.0:
        y_train = (y_train * target_scale).astype(np.float32)
        y_val = (y_val * target_scale).astype(np.float32)
        y_test = (y_test * target_scale).astype(np.float32)

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


def _load_npz_splits(path: str, mode: str, target_scale: float, horizon: int) -> Tuple[DatasetSplits, float, List[str]]:
    data = np.load(path, allow_pickle=True)
    required = ["X_train", "X_val", "X_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset npz missing keys {missing}: {path}")

    dataset_scale: Optional[float] = None
    if "target_scale" in data:
        scale_arr = np.asarray(data["target_scale"]).astype(float)
        if scale_arr.size:
            dataset_scale = float(scale_arr.reshape(-1)[0])

    effective_scale = target_scale if target_scale != 1.0 else (dataset_scale or 1.0)

    def _target_key(split: str) -> str:
        if mode == "reg":
            if horizon == 1 and f"y_{split}" in data:
                return f"y_{split}"
            return f"y_ret{horizon}h_{split}"
        if horizon == 1 and f"y_{split}" in data:
            return f"y_{split}"
        return f"y_dir{horizon}h_{split}"

    def _target(split: str) -> np.ndarray:
        scaled_key = f"y_{split}_scaled"
        if mode == "reg" and scaled_key in data:
            return data[scaled_key].astype(np.float32)
        key = _target_key(split)
        if key not in data:
            raise KeyError(f"Dataset npz missing target key '{key}' for horizon {horizon}h")
        arr = data[key].astype(np.float32)
        if mode == "reg" and effective_scale != 1.0:
            arr = arr * effective_scale
        return arr

    y_train = _target("train")
    y_val = _target("val")
    y_test = _target("test")

    splits = DatasetSplits(
        X_train=data["X_train"].astype(np.float32),
        X_val=data["X_val"].astype(np.float32),
        X_test=data["X_test"].astype(np.float32),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=None,
    )

    feature_names = data["feature_names"].tolist() if "feature_names" in data else []

    return splits, (effective_scale if mode == "reg" else 1.0), feature_names


def _load_npz_full(path: str, mode: str, target_scale: float, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    splits, effective_scale, feature_names = _load_npz_splits(path, mode, target_scale, horizon)
    X_all = np.vstack([splits.X_train, splits.X_val, splits.X_test])
    y_all = np.concatenate([splits.y_train, splits.y_val, splits.y_test])
    return X_all, y_all, feature_names, effective_scale


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


def _evaluate_objective_cv(
    mode: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    folds: List[Tuple[slice, slice, slice]],
) -> Tuple[optuna.study.Study, optuna.trial.Trial]:
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, mode)
        fold_metrics: List[float] = []

        for fold_idx, (train_slice, val_slice, test_slice) in enumerate(folds, start=1):
            X_train = X_all[train_slice]
            y_train = y_all[train_slice]
            X_val = X_all[val_slice]
            y_val = y_all[val_slice]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = _create_model(mode, params)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )

            if mode == "reg":
                val_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, val_pred)
                fold_metrics.append(float(np.sqrt(mse)))
            else:
                val_proba = model.predict_proba(X_val_scaled)[:, 1]
                val_proba = np.clip(val_proba, 1e-15, 1 - 1e-15)
                fold_metrics.append(float(log_loss(y_val, val_proba)))

            trial.set_user_attr(f"fold_{fold_idx}_val", fold_metrics[-1])

        mean_metric = float(np.mean(fold_metrics))
        trial.set_user_attr("cv_mean", mean_metric)
        return mean_metric

    return objective


def _run_study(args: argparse.Namespace, objective) -> optuna.Study:
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


def _save_model_safely(model, path: str) -> None:
    """Persist the fitted booster even when sklearn wrappers lack estimator metadata."""

    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        booster = get_booster()
        booster.save_model(path)
        return

    # Fallback to native save_model if booster access is unavailable.
    model.save_model(path)


def _save_outputs(
    output_dir: str,
    mode: str,
    model,
    summary: Dict[str, Dict[str, float]],
    feature_names: List[str],
    horizon_hours: int,
    target_scale: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    legacy_model_path = os.path.join(output_dir, "best_model.json")
    _save_model_safely(model, legacy_model_path)

    summary_path = os.path.join(output_dir, "best_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    legacy_path = os.path.join(output_dir, "summary.json")
    with open(legacy_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    horizon_suffix = f"{horizon_hours}h"

    if mode == "reg":
        model_filename = f"xgb_ret{horizon_suffix}_model.json"
        metadata_filename = "model_metadata.json"
        metadata_payload: Dict[str, Any] = {
            "model_type": "xgboost",
            "target": f"ret_{horizon_suffix}",
            "horizon_hours": horizon_hours,
            "feature_names": feature_names,
            "summary": summary,
            "target_scale": target_scale,
        }
    else:
        model_filename = f"xgb_dir{horizon_suffix}_model.json"
        metadata_filename = "model_metadata_direction.json"
        metadata_payload = {
            "model_type": "xgboost_classifier",
            "target": f"direction_{horizon_suffix}",
            "threshold": 0.0,
            "horizon_hours": horizon_hours,
            "feature_names": feature_names,
            "summary": summary,
        }

    _save_model_safely(model, os.path.join(output_dir, model_filename))
    with open(os.path.join(output_dir, metadata_filename), "w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for XGBoost direction/regression models.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset NPZ path.")
    parser.add_argument("--mode", choices=["reg", "dir"], required=True)
    parser.add_argument("--horizon", type=int, default=1, help="Target horizon in hours (e.g. 1, 4, 8, 12).")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=None, help="Timeout for the study in seconds.")
    parser.add_argument("--study-name", type=str, default="xgb_optuna_study")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///study.db).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-scale", type=float, default=1.0, help="Multiplicative scale applied to regression targets before training.")
    parser.add_argument("--cv-folds", type=int, default=0, help="Enable time-series CV with this many folds.")
    parser.add_argument("--cv-train-size", type=int, default=0, help="Train window size for each fold.")
    parser.add_argument("--cv-val-size", type=int, default=0, help="Validation window size for each fold.")
    parser.add_argument("--cv-test-size", type=int, default=0, help="Test window size for each fold.")
    parser.add_argument("--cv-gap", type=int, default=0, help="Gap size between train/val/test windows.")
    parser.add_argument("--cv-purge-size", type=int, default=0, help="Rows purged between train/val and val/test boundaries.")
    parser.add_argument("--cv-embargo-size", type=int, default=0, help="Rows embargoed between train and validation windows.")
    parser.add_argument("--cv-step-size", type=int, default=None, help="Step size between folds (defaults to test size).")
    parser.add_argument("--cv-mode", type=str, default="expanding", choices=["expanding", "rolling"], help="Time-series CV mode.")
    args = parser.parse_args()

    if args.horizon <= 0:
        parser.error("--horizon must be a positive integer.")

    if args.dataset_path is None:
        if args.horizon == 1:
            args.dataset_path = "artifacts/datasets/btc_features_1h_splits.npz"
        else:
            args.dataset_path = "artifacts/datasets/btc_features_multi_horizon_splits.npz"

    target_column = f"ret_{args.horizon}h"

    npz_mode = args.dataset_path and args.dataset_path.endswith(".npz")
    target_scale_effective = 1.0

    feature_names: List[str]

    folds: Optional[List[Tuple[slice, slice, slice]]] = None
    splits: Optional[DatasetSplits] = None
    X_all: Optional[np.ndarray] = None
    y_all: Optional[np.ndarray] = None

    if npz_mode:
        if args.cv_folds > 0:
            X_all, y_all, feature_names, inferred_scale = _load_npz_full(
                args.dataset_path,
                args.mode,
                args.target_scale,
                args.horizon,
            )
            if args.mode == "reg":
                target_scale_effective = inferred_scale
        else:
            splits, inferred_scale, feature_names = _load_npz_splits(
                args.dataset_path,
                args.mode,
                args.target_scale,
                args.horizon,
            )
            if args.mode == "reg":
                target_scale_effective = inferred_scale
    else:
        prepared = prepare_data_for_signals(args.dataset_path, target_column=target_column)
        if args.cv_folds > 0:
            X_all = prepared.X_all_ordered.to_numpy(dtype=np.float32)
            df_all = prepared.df_all.reset_index(drop=True)
            if target_column not in df_all.columns:
                raise KeyError(f"Target column '{target_column}' missing from prepared dataframe.")
            y_ret = df_all[target_column].to_numpy(dtype=np.float32)
            y_all = y_ret if args.mode == "reg" else (y_ret > 0.0).astype(np.int32)
        else:
            splits = _build_time_splits(prepared, args.mode, target_column, args.target_scale)
        if args.mode == "reg":
            target_scale_effective = args.target_scale
        feature_names = prepared.feature_names

    if args.cv_folds > 0:
        if X_all is None or y_all is None:
            raise RuntimeError("CV requested but full dataset arrays were not loaded.")
        if args.cv_train_size <= 0 or args.cv_val_size <= 0 or args.cv_test_size <= 0:
            raise ValueError("CV sizes must be positive when --cv-folds is set.")
        folds = list(
            build_time_series_folds(
                len(y_all),
                n_splits=args.cv_folds,
                train_size=args.cv_train_size,
                val_size=args.cv_val_size,
                test_size=args.cv_test_size,
                gap=args.cv_gap,
                purge_size=args.cv_purge_size,
                embargo_size=args.cv_embargo_size,
                step_size=args.cv_step_size,
                mode=args.cv_mode,
            )
        )
        fold_slices = [(fold.train_slice, fold.val_slice, fold.test_slice) for fold in folds]
        objective = _evaluate_objective_cv(args.mode, X_all, y_all, fold_slices)
        print(
            f"Running Optuna CV (mode={args.mode}, horizon={args.horizon}h, folds={len(fold_slices)}, n_trials={args.n_trials})"
        )
    else:
        if splits is None:
            raise RuntimeError("Missing dataset splits for non-CV Optuna run.")
        objective = _evaluate_objective(args.mode, splits)
        print(
            f"Running Optuna search (mode={args.mode}, horizon={args.horizon}h, n_trials={args.n_trials}, timeout={args.timeout}) "
            f"with split sizes train={len(splits.y_train)}, val={len(splits.y_val)}, test={len(splits.y_test)}"
        )

    study = _run_study(args, objective)
    best_trial = study.best_trial
    best_params = best_trial.params

    print(f"Best trial #{best_trial.number} value={best_trial.value:.6f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    if folds:
        last_fold = folds[-1]
        X_trainval = np.vstack([X_all[last_fold.train_slice], X_all[last_fold.val_slice]])
        y_trainval = np.concatenate([y_all[last_fold.train_slice], y_all[last_fold.val_slice]])
        X_test = X_all[last_fold.test_slice]
        y_test = y_all[last_fold.test_slice]

        scaler = StandardScaler()
        scaler.fit(X_trainval)
        X_trainval = scaler.transform(X_trainval)
        X_test = scaler.transform(X_test)

        model = _create_model(args.mode, best_params)
        model.fit(X_trainval, y_trainval, verbose=False)

        if args.mode == "reg":
            test_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, test_pred)
            test_metrics = {"test_rmse": float(np.sqrt(mse))}
        else:
            test_proba = model.predict_proba(X_test)[:, 1]
            test_proba = np.clip(test_proba, 1e-15, 1 - 1e-15)
            test_metrics = {
                "test_logloss": float(log_loss(y_test, test_proba)),
                "test_accuracy": float(accuracy_score(y_test, (test_proba >= 0.5).astype(int))),
                "test_auc": float(roc_auc_score(y_test, test_proba)),
            }
    else:
        model, test_metrics = _retrain_best(args, splits, best_params)

    summary = {
        "mode": args.mode,
        "horizon_hours": args.horizon,
        "target_column": target_column,
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "best_trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": best_params,
        "val_metrics": best_trial.user_attrs,
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": len(splits.y_train) if splits else None,
            "val": len(splits.y_val) if splits else None,
            "test": len(splits.y_test) if splits else None,
        },
        "cv": {
            "enabled": bool(folds),
            "folds": len(folds) if folds else 0,
            "train_size": args.cv_train_size if folds else None,
            "val_size": args.cv_val_size if folds else None,
            "test_size": args.cv_test_size if folds else None,
            "gap": args.cv_gap if folds else None,
            "purge_size": args.cv_purge_size if folds else None,
            "embargo_size": args.cv_embargo_size if folds else None,
            "step_size": args.cv_step_size if folds else None,
            "mode": args.cv_mode if folds else None,
        },
    }

    _save_outputs(
        args.output_dir,
        args.mode,
        model,
        summary,
        feature_names,
        args.horizon,
        target_scale_effective,
    )
    print(f"Saved best model and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
