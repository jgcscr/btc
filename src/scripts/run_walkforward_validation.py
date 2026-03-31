from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.training.time_series_cv import build_time_series_folds


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


def _load_npz(path: Path, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    if "X_train" not in data or "X_val" not in data or "X_test" not in data:
        raise KeyError("NPZ missing split arrays")
    if y_key == "y":
        required_target_keys = {"y_train", "y_val", "y_test"}
    else:
        required_target_keys = {f"{y_key}_train", f"{y_key}_val", f"{y_key}_test"}
    missing_targets = [key for key in required_target_keys if key not in data]
    if missing_targets:
        raise KeyError(f"Missing target key {y_key}: required split arrays not found {missing_targets}")

    X = np.vstack([data["X_train"], data["X_val"], data["X_test"]]).astype(np.float32)
    if y_key == "y":
        y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]]).astype(np.float32)
        if {"y_ret_train", "y_ret_val", "y_ret_test"}.issubset(set(data.files)):
            y_ret = np.concatenate([data["y_ret_train"], data["y_ret_val"], data["y_ret_test"]]).astype(np.float32)
        else:
            y_ret = None
    else:
        y = np.concatenate([data[f"{y_key}_train"], data[f"{y_key}_val"], data[f"{y_key}_test"]]).astype(np.float32)
        y_ret = None
    if {"ts_train", "ts_val", "ts_test"}.issubset(set(data.files)):
        ts = np.concatenate([data["ts_train"], data["ts_val"], data["ts_test"]])
    elif "ts_all" in data:
        ts = np.asarray(data["ts_all"])
    else:
        ts = None
    return X, y, y_ret, ts


def _fit_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=4,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _fit_logit(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def _predict_meta_stack(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    oof_splits: int,
) -> np.ndarray:
    n_train = len(X_train)
    oof_1 = np.full(n_train, np.nan, dtype=float)
    oof_2 = np.full(n_train, np.nan, dtype=float)

    splits = max(2, min(int(oof_splits), max(2, n_train // 60)))
    tscv = TimeSeriesSplit(n_splits=splits)

    for tr_idx, val_idx in tscv.split(X_train):
        base_xgb = _fit_xgb(X_train[tr_idx], y_train[tr_idx])
        base_logit = _fit_logit(X_train[tr_idx], y_train[tr_idx])
        oof_1[val_idx] = base_xgb.predict_proba(X_train[val_idx])[:, 1]
        oof_2[val_idx] = base_logit.predict_proba(X_train[val_idx])[:, 1]

    valid = np.isfinite(oof_1) & np.isfinite(oof_2)
    if int(valid.sum()) < 50:
        base_xgb = _fit_xgb(X_train, y_train)
        base_logit = _fit_logit(X_train, y_train)
        p1 = base_xgb.predict_proba(X_test)[:, 1]
        p2 = base_logit.predict_proba(X_test)[:, 1]
        return np.clip(0.5 * p1 + 0.5 * p2, 1e-6, 1.0 - 1e-6)

    meta_X_train = np.column_stack(
        [
            oof_1[valid],
            oof_2[valid],
            0.5 * (oof_1[valid] + oof_2[valid]),
            np.abs(oof_1[valid] - oof_2[valid]),
        ]
    )
    meta_y_train = y_train[valid]
    meta = _fit_logit(meta_X_train, meta_y_train)

    base_xgb_full = _fit_xgb(X_train, y_train)
    base_logit_full = _fit_logit(X_train, y_train)
    p1_test = base_xgb_full.predict_proba(X_test)[:, 1]
    p2_test = base_logit_full.predict_proba(X_test)[:, 1]
    meta_X_test = np.column_stack(
        [
            p1_test,
            p2_test,
            0.5 * (p1_test + p2_test),
            np.abs(p1_test - p2_test),
        ]
    )
    return np.clip(meta.predict_proba(meta_X_test)[:, 1], 1e-6, 1.0 - 1e-6)


def _predict_fold_probabilities(
    model_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    meta_oof_splits: int,
) -> np.ndarray:
    if model_kind == "meta_stack":
        return _predict_meta_stack(X_train, y_train, X_test, oof_splits=meta_oof_splits)
    if model_kind == "selector_simple":
        # Train two simple experts and pick by volatility regime inferred from feature dispersion.
        xgb_model = _fit_xgb(X_train, y_train)
        logit_model = _fit_logit(X_train, y_train)

        train_vol = np.std(X_train, axis=1)
        split_point = float(np.nanmedian(train_vol))
        low_regime = train_vol <= split_point
        high_regime = ~low_regime

        p_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
        p_logit_train = logit_model.predict_proba(X_train)[:, 1]

        def _regime_auc(mask: np.ndarray, probs: np.ndarray) -> float:
            yy = y_train[mask]
            if yy.size < 10 or len(np.unique(yy)) < 2:
                return float("nan")
            return float(roc_auc_score(yy, probs[mask]))

        auc_xgb_low = _regime_auc(low_regime, p_xgb_train)
        auc_logit_low = _regime_auc(low_regime, p_logit_train)
        auc_xgb_high = _regime_auc(high_regime, p_xgb_train)
        auc_logit_high = _regime_auc(high_regime, p_logit_train)

        use_xgb_low = not np.isfinite(auc_logit_low) or (np.isfinite(auc_xgb_low) and auc_xgb_low >= auc_logit_low)
        use_xgb_high = not np.isfinite(auc_logit_high) or (np.isfinite(auc_xgb_high) and auc_xgb_high >= auc_logit_high)

        p_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
        p_logit_test = logit_model.predict_proba(X_test)[:, 1]
        test_vol = np.std(X_test, axis=1)
        test_low = test_vol <= split_point
        test_high = ~test_low
        p = np.empty(len(X_test), dtype=float)
        p[test_low] = p_xgb_test[test_low] if use_xgb_low else p_logit_test[test_low]
        p[test_high] = p_xgb_test[test_high] if use_xgb_high else p_logit_test[test_high]
        return np.clip(p, 1e-6, 1.0 - 1e-6)
    model = _fit_xgb(X_train, y_train)
    return np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1.0 - 1e-6)


def _trading_metrics_from_probs(
    p: np.ndarray,
    y_test: np.ndarray,
    y_ret_test: np.ndarray | None,
    threshold: float,
    fee_bps: float,
    slippage_bps: float,
) -> Tuple[int, float, float]:
    signal = (p >= threshold).astype(int)
    n_trades = int(signal.sum())
    if y_ret_test is None:
        # Direction-only fallback proxy when true returns are unavailable.
        ret_stream = np.where(y_test > 0, 1.0, -1.0)
    else:
        ret_stream = y_ret_test.astype(float)
    per_trade_cost = (float(fee_bps) + float(slippage_bps)) / 10_000.0
    net = ret_stream * signal - per_trade_cost * signal
    active = signal > 0
    hit_rate = float((ret_stream[active] > 0).mean()) if n_trades > 0 else float("nan")
    cum_ret = float(net.sum())
    return n_trades, hit_rate, cum_ret


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for direction model stability.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--y-key", type=str, default="y", help="'y' for flat labels or prefix like y_dir4h")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--purge-size", type=int, default=0)
    parser.add_argument("--embargo-size", type=int, default=0)
    parser.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--model-kind", choices=("xgb", "meta_stack", "selector_simple"), default="meta_stack")
    parser.add_argument("--meta-oof-splits", type=int, default=4)
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/walkforward_validation.json"))
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=None,
        help="Optional CSV path to write per-bar fold predictions and net-return contributions.",
    )
    args = parser.parse_args()

    X, y, y_ret, ts = _load_npz(args.dataset_path, args.y_key)
    folds = build_time_series_folds(
        len(y),
        n_splits=args.folds,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        gap=args.gap,
        purge_size=args.purge_size,
        embargo_size=args.embargo_size,
        mode=args.mode,
    )

    rows: List[dict] = []
    detailed_rows: List[dict] = []
    for i, fold in enumerate(folds, start=1):
        X_train = X[fold.train_slice]
        y_train = y[fold.train_slice].astype(int)
        X_test = X[fold.test_slice]
        y_test = y[fold.test_slice].astype(int)
        y_ret_test = y_ret[fold.test_slice] if y_ret is not None else None
        ts_test = ts[fold.test_slice] if ts is not None else None
        test_indices = np.arange(len(y))[fold.test_slice]

        p = _predict_fold_probabilities(
            args.model_kind,
            X_train,
            y_train,
            X_test,
            meta_oof_splits=args.meta_oof_splits,
        )
        pred = (p >= float(args.signal_threshold)).astype(int)

        auc = float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan")
        acc = float(accuracy_score(y_test, pred))
        brier = float(brier_score_loss(y_test, p))
        nll = float(log_loss(y_test, p, labels=[0, 1]))
        ece = float(_expected_calibration_error(y_test, p, bins=10))
        trades, hit_rate, cum_ret_net = _trading_metrics_from_probs(
            p,
            y_test,
            y_ret_test,
            threshold=float(args.signal_threshold),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        signal = (p >= float(args.signal_threshold)).astype(int)
        if y_ret_test is None:
            ret_stream = np.where(y_test > 0, 1.0, -1.0).astype(float)
        else:
            ret_stream = y_ret_test.astype(float)
        per_trade_cost = (float(args.fee_bps) + float(args.slippage_bps)) / 10_000.0
        ret_net = ret_stream * signal - per_trade_cost * signal
        rows.append(
            {
                "fold": i,
                "auc": auc,
                "acc": acc,
                "brier": brier,
                "log_loss": nll,
                "ece_10": ece,
                "n_test": int(len(y_test)),
                "trade_count": trades,
                "hit_rate": hit_rate,
                "cum_ret_net": cum_ret_net,
            }
        )
        for local_index, global_index in enumerate(test_indices):
            detailed_rows.append(
                {
                    "fold": int(i),
                    "global_index": int(global_index),
                    "ts": (
                        np.datetime_as_string(ts_test[local_index], unit="s")
                        if ts_test is not None
                        else ""
                    ),
                    "y_true": int(y_test[local_index]),
                    "y_ret": float(ret_stream[local_index]),
                    "p_up": float(p[local_index]),
                    "pred_label": int(pred[local_index]),
                    "signal": int(signal[local_index]),
                    "ret_net": float(ret_net[local_index]),
                    "model_kind": str(args.model_kind),
                    "mode": str(args.mode),
                }
            )

    def _finite_mean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    def _finite_std(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.std(arr, ddof=0))

    auc_values = [float(r["auc"]) for r in rows]
    payload = {
        "model_kind": args.model_kind,
        "folds": rows,
        "auc_mean": _finite_mean(auc_values),
        "auc_std": _finite_std(auc_values),
        "auc_cv": (
            _finite_std(auc_values) / abs(_finite_mean(auc_values))
            if np.isfinite(_finite_mean(auc_values)) and abs(_finite_mean(auc_values)) > 1e-12
            else float("nan")
        ),
        "brier_mean": _finite_mean([float(r["brier"]) for r in rows]),
        "ece_10_mean": _finite_mean([float(r["ece_10"]) for r in rows]),
        "cum_ret_net_mean": _finite_mean([float(r["cum_ret_net"]) for r in rows]),
        "cum_ret_net_total": float(np.nansum(np.asarray([r["cum_ret_net"] for r in rows], dtype=float))),
        "trade_count_total": int(np.nansum(np.asarray([r["trade_count"] for r in rows], dtype=float))),
    }
    if args.detailed_output is not None:
        payload["detailed_output"] = str(args.detailed_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.detailed_output is not None:
        args.detailed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.detailed_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "fold",
                    "global_index",
                    "ts",
                    "y_true",
                    "y_ret",
                    "p_up",
                    "pred_label",
                    "signal",
                    "ret_net",
                    "model_kind",
                    "mode",
                ],
            )
            writer.writeheader()
            writer.writerows(detailed_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
