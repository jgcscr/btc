from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "p_up",
    "ret_pred",
    "expected_value_proxy",
    "abs_ret_pred",
    "volatility_realized_24h",
    "volatility_ewm_24h",
    "volatility_garch_like",
    "regime_is_trend",
    "regime_is_neutral",
    "regime_is_chop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trade/no-trade decision model from labeled backtest rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input labeled backtest CSV.")
    parser.add_argument(
        "--target-col",
        type=str,
        default="ret_ensemble_net",
        help="Net return column used to derive profitable-trade target.",
    )
    parser.add_argument(
        "--signal-col",
        type=str,
        default="signal_ensemble",
        help="Signal column to identify candidate trade rows.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Decision probability threshold stored in artifact.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum labeled rows required for reliable deployment of this gate.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Train only on rows where signal-col > 0. Default uses full labeled slice for larger sample size.",
    )
    parser.add_argument("--oof-splits", type=int, default=5, help="Number of time-series OOF splits.")
    parser.add_argument("--ev-bins", type=int, default=6, help="Number of probability bins for OOF expected-net calibration.")
    parser.add_argument(
        "--ev-calibration-source",
        choices=("all", "candidate", "hybrid"),
        default="hybrid",
        help=(
            "Rows used for OOF expected-value bins: all rows, candidate rows only, or hybrid "
            "(candidate when enough samples exist, otherwise all)."
        ),
    )
    parser.add_argument(
        "--ev-min-candidate-rows",
        type=int,
        default=20,
        help="Minimum candidate rows required when ev-calibration-source=hybrid.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=4,
        help="Minimum OOF samples required per expected-net bin.",
    )
    parser.add_argument(
        "--raw-ev-calibration-source",
        choices=("all", "candidate", "weighted_hybrid"),
        default="weighted_hybrid",
        help="Data source used for raw-EV expected-net calibration.",
    )
    parser.add_argument(
        "--raw-ev-candidate-weight",
        type=float,
        default=4.0,
        help="Sample weight for candidate rows when raw-ev-calibration-source=weighted_hybrid.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    return parser.parse_args()


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    def _series(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(float(default))
        return pd.Series(float(default), index=df.index, dtype=float)

    out = pd.DataFrame(index=df.index)
    out["p_up"] = _series("p_up", 0.0)
    out["ret_pred"] = _series("ret_pred", 0.0)
    out["expected_value_proxy"] = out["p_up"] * out["ret_pred"]
    out["abs_ret_pred"] = out["ret_pred"].abs()
    out["volatility_realized_24h"] = _series("volatility_realized_24h", 0.0)
    out["volatility_ewm_24h"] = _series("volatility_ewm_24h", 0.0)
    out["volatility_garch_like"] = _series("volatility_garch_like", 0.0)
    regime = df.get("regime_state")
    if regime is None:
        regime = pd.Series(["neutral"] * len(df), index=df.index)
    regime = regime.astype(str).str.lower().fillna("neutral")
    out["regime_is_trend"] = (regime == "trend_ignition").astype(float)
    out["regime_is_neutral"] = (regime == "neutral").astype(float)
    out["regime_is_chop"] = (regime == "chop").astype(float)
    return out[FEATURE_COLUMNS]


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _build_oof_probabilities(X: pd.DataFrame, y: pd.Series, n_splits: int) -> np.ndarray:
    n_rows = len(X)
    oof = np.full(n_rows, np.nan, dtype=float)
    if n_rows < 24:
        return oof

    resolved_splits = max(2, min(int(n_splits), max(2, n_rows // 30)))
    cv = TimeSeriesSplit(n_splits=resolved_splits)
    for train_idx, val_idx in cv.split(X):
        y_train = y.iloc[train_idx]
        if int(y_train.nunique()) < 2:
            continue
        model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
        model.fit(X.iloc[train_idx], y_train)
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return np.clip(oof, 1e-6, 1.0 - 1e-6)


def _build_expected_net_curve(
    prob: np.ndarray,
    ret_net: np.ndarray,
    *,
    n_bins: int,
    min_bin_samples: int,
) -> Dict[str, object]:
    valid = np.isfinite(prob) & np.isfinite(ret_net)
    if int(valid.sum()) < 20:
        default_mean = float(np.nanmean(ret_net[valid])) if int(valid.sum()) else 0.0
        return {
            "bins": [],
            "default_expected_net": default_mean,
            "valid_samples": int(valid.sum()),
        }

    p = prob[valid]
    r = ret_net[valid]
    # Use fewer bins on small samples to avoid degenerate empty-bin calibration.
    n_bins_eff = min(max(2, int(n_bins)), max(2, int(valid.sum()) // max(3, int(min_bin_samples))))
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins_eff + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return {
            "bins": [],
            "default_expected_net": float(np.mean(r)),
            "valid_samples": int(valid.sum()),
        }

    bins: List[Dict[str, float]] = []
    for idx in range(edges.size - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < edges.size - 2:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count < int(min_bin_samples):
            continue
        bins.append(
            {
                "p_min": lo,
                "p_max": hi,
                "samples": count,
                "mean_ret_net": float(np.mean(r[mask])),
                "hit_rate": float(np.mean((r[mask] > 0.0).astype(float))),
            }
        )

    return {
        "bins": bins,
        "default_expected_net": float(np.mean(r)),
        "valid_samples": int(valid.sum()),
    }


def _build_expected_net_curve_by_raw_ev(
    raw_ev: np.ndarray,
    ret_net: np.ndarray,
    *,
    n_bins: int,
    min_bin_samples: int,
) -> Dict[str, object]:
    valid = np.isfinite(raw_ev) & np.isfinite(ret_net)
    if int(valid.sum()) < 20:
        default_mean = float(np.nanmean(ret_net[valid])) if int(valid.sum()) else 0.0
        return {
            "bins": [],
            "default_expected_net": default_mean,
            "valid_samples": int(valid.sum()),
        }

    x = raw_ev[valid]
    r = ret_net[valid]
    n_bins_eff = min(max(2, int(n_bins)), max(2, int(valid.sum()) // max(3, int(min_bin_samples))))
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins_eff + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return {
            "bins": [],
            "default_expected_net": float(np.mean(r)),
            "valid_samples": int(valid.sum()),
        }

    bins: List[Dict[str, float]] = []
    for idx in range(edges.size - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx < edges.size - 2:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        count = int(mask.sum())
        if count < int(min_bin_samples):
            continue
        bins.append(
            {
                "x_min": lo,
                "x_max": hi,
                "samples": count,
                "mean_ret_net": float(np.mean(r[mask])),
                "hit_rate": float(np.mean((r[mask] > 0.0).astype(float))),
            }
        )

    return {
        "bins": bins,
        "default_expected_net": float(np.mean(r)),
        "valid_samples": int(valid.sum()),
    }


def _fit_raw_ev_isotonic(
    raw_ev: np.ndarray,
    ret_net: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> IsotonicRegression | None:
    valid = np.isfinite(raw_ev) & np.isfinite(ret_net)
    if int(valid.sum()) < 20:
        return None
    x = raw_ev[valid]
    y = ret_net[valid]
    if sample_weight is not None:
        w = sample_weight[valid]
    else:
        w = None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y, sample_weight=w)
    return model


def _oof_raw_ev_isotonic_predictions(
    raw_ev: np.ndarray,
    ret_net: np.ndarray,
    *,
    n_splits: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    n_rows = len(raw_ev)
    oof = np.full(n_rows, np.nan, dtype=float)
    if n_rows < 24:
        return oof

    resolved_splits = max(2, min(int(n_splits), max(2, n_rows // 30)))
    cv = TimeSeriesSplit(n_splits=resolved_splits)
    for train_idx, val_idx in cv.split(np.arange(n_rows)):
        model = _fit_raw_ev_isotonic(
            raw_ev[train_idx],
            ret_net[train_idx],
            sample_weight=(sample_weight[train_idx] if sample_weight is not None else None),
        )
        if model is None:
            continue
        oof[val_idx] = model.predict(raw_ev[val_idx])
    return oof


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")

    signal = pd.to_numeric(df.get(args.signal_col, 1.0), errors="coerce").fillna(0.0)
    trade_mask = signal > 0
    if bool(args.candidate_only):
        train_df = df.loc[trade_mask].copy()
        if int(len(train_df)) < 30:
            raise RuntimeError("candidate-only training requires at least 30 candidate rows.")
    else:
        train_df = df.copy()

    target_net = pd.to_numeric(train_df[args.target_col], errors="coerce").fillna(0.0)
    y = (target_net > 0.0).astype(int)
    X = _extract_features(train_df)

    if int(y.nunique()) < 2:
        raise RuntimeError("Trade decision training requires both positive and negative classes.")

    calibration_source = str(args.ev_calibration_source)
    if calibration_source == "candidate":
        calibration_mask = pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0
    elif calibration_source == "hybrid":
        candidate_mask = pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0
        calibration_mask = candidate_mask if int(candidate_mask.sum()) >= int(args.ev_min_candidate_rows) else pd.Series(True, index=train_df.index)
    else:
        calibration_mask = pd.Series(True, index=train_df.index)

    X_cal = X.loc[calibration_mask].copy()
    y_cal = y.loc[calibration_mask].copy()
    target_net_cal = target_net.loc[calibration_mask].copy()
    if len(X_cal) < 30 or int(y_cal.nunique()) < 2:
        X_cal = X
        y_cal = y
        target_net_cal = target_net
        calibration_source = "fallback_all"

    oof_prob = _build_oof_probabilities(X_cal, y_cal, n_splits=int(args.oof_splits))
    valid_oof = np.isfinite(oof_prob)
    oof_curve = _build_expected_net_curve(
        oof_prob,
        target_net_cal.to_numpy(dtype=float),
        n_bins=int(args.ev_bins),
        min_bin_samples=int(args.min_bin_samples),
    )
    raw_ev_cal = pd.to_numeric(X_cal["expected_value_proxy"], errors="coerce").to_numpy(dtype=float)
    raw_ev_curve = _build_expected_net_curve_by_raw_ev(
        raw_ev_cal,
        target_net_cal.to_numpy(dtype=float),
        n_bins=int(args.ev_bins),
        min_bin_samples=int(args.min_bin_samples),
    )

    raw_ev_source = str(args.raw_ev_calibration_source)
    if raw_ev_source == "candidate":
        raw_mask = pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0
        raw_weights = None
    elif raw_ev_source == "weighted_hybrid":
        raw_mask = pd.Series(True, index=train_df.index)
        is_candidate = pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0
        raw_weights = np.where(is_candidate.to_numpy(dtype=bool), float(args.raw_ev_candidate_weight), 1.0).astype(float)
    else:
        raw_mask = pd.Series(True, index=train_df.index)
        raw_weights = None

    raw_ev_series = pd.to_numeric(X.loc[raw_mask, "expected_value_proxy"], errors="coerce").to_numpy(dtype=float)
    raw_ret_series = target_net.loc[raw_mask].to_numpy(dtype=float)
    raw_weight_series = raw_weights[raw_mask.to_numpy(dtype=bool)] if raw_weights is not None else None

    if len(raw_ev_series) < 20:
        raw_ev_series = pd.to_numeric(X["expected_value_proxy"], errors="coerce").to_numpy(dtype=float)
        raw_ret_series = target_net.to_numpy(dtype=float)
        raw_weight_series = None
        raw_ev_source = "fallback_all"

    raw_iso_model = _fit_raw_ev_isotonic(raw_ev_series, raw_ret_series, sample_weight=raw_weight_series)
    raw_iso_oof = _oof_raw_ev_isotonic_predictions(
        raw_ev_series,
        raw_ret_series,
        n_splits=int(args.oof_splits),
        sample_weight=raw_weight_series,
    )
    raw_iso_valid = np.isfinite(raw_iso_oof)
    raw_iso_mae = float(np.mean(np.abs(raw_iso_oof[raw_iso_valid] - raw_ret_series[raw_iso_valid]))) if int(raw_iso_valid.sum()) else float("nan")
    raw_iso_mse = float(np.mean((raw_iso_oof[raw_iso_valid] - raw_ret_series[raw_iso_valid]) ** 2.0)) if int(raw_iso_valid.sum()) else float("nan")

    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    model.fit(X, y)

    raw_ev_series = target_net.loc[trade_mask.loc[target_net.index]] if int(trade_mask.loc[target_net.index].sum()) > 0 else target_net
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95]
    raw_ev_quantiles = {
        f"q{int(q * 100)}": float(np.quantile(raw_ev_series.to_numpy(dtype=float), q)) if len(raw_ev_series) else 0.0
        for q in quantiles
    }

    prob = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    auc = _safe_auc(y.to_numpy(dtype=int), prob)
    oof_auc = _safe_auc(y_cal.to_numpy(dtype=int)[valid_oof], oof_prob[valid_oof]) if int(valid_oof.sum()) else float("nan")
    payload: Dict[str, object] = {
        "feature_columns": FEATURE_COLUMNS,
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "threshold": float(args.threshold),
        "min_rows_required": int(args.min_rows),
        "deploy_ready": bool(int(len(X)) >= int(args.min_rows)),
        "metrics": {
            "rows": int(len(X)),
            "full_rows": int(len(df)),
            "candidate_rows": int(trade_mask.sum()),
            "auc": auc,
            "oof_auc": oof_auc,
            "oof_rows": int(valid_oof.sum()),
            "brier": float(brier_score_loss(y, prob)),
            "log_loss": float(log_loss(y, prob)),
            "positive_rate": float(y.mean()),
        },
        "oof_expected_value": oof_curve,
        "raw_ev_expected_value": raw_ev_curve,
        "oof_expected_value_calibration": {
            "source": calibration_source,
            "rows": int(len(X_cal)),
            "candidate_rows": int((pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0).sum()),
        },
        "raw_ev_fallback": {
            "source": "candidate_rows" if int(trade_mask.loc[target_net.index].sum()) > 0 else "all_rows",
            "quantiles": raw_ev_quantiles,
        },
        "raw_ev_isotonic": {
            "source": raw_ev_source,
            "rows": int(len(raw_ev_series)),
            "candidate_rows": int((pd.to_numeric(train_df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0).sum()),
            "oof_rows": int(raw_iso_valid.sum()),
            "oof_mae": raw_iso_mae,
            "oof_mse": raw_iso_mse,
            "x_thresholds": [] if raw_iso_model is None else [float(v) for v in raw_iso_model.X_thresholds_],
            "y_thresholds": [] if raw_iso_model is None else [float(v) for v in raw_iso_model.y_thresholds_],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
