from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "p_up",
    "raw_p_up",
    "ret_pred",
    "expected_value_proxy",
    "abs_ret_pred",
    "raw_calibrated_probability_gap",
    "probability_alignment_gap",
    "raw_p_up_ret_mismatch",
    "p_up_ret_mismatch",
    "raw_p_up_direction_mismatch",
    "p_up_direction_mismatch",
    "ret_projected_price_consensus",
    "probability_calibration_guard_applied",
    "probability_calibration_used_regime_key",
    "confidence_score",
    "position_size",
    "volatility_realized_24h",
    "volatility_ewm_24h",
    "volatility_garch_like",
    "range_expansion_1h",
    "distance_from_session_high_8h",
    "distance_from_session_low_8h",
    "vwap_deviation_8h",
    "momentum_slope_2h",
    "momentum_slope_4h",
    "confluence_support_ratio",
    "confluence_short_term_ratio",
    "confluence_mid_term_ratio",
    "confluence_direction_matches_dominant",
    "incumbent_signal_reference",
    "candidate_only_reference",
    "candidate_incumbent_disagreement",
    "regime_is_trend",
    "regime_is_neutral",
    "regime_is_chop",
]

REFERENCE_FEATURE_COLUMNS = [
    "incumbent_signal_reference",
    "candidate_only_reference",
    "candidate_incumbent_disagreement",
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
        "--min-candidate-rows",
        type=int,
        default=60,
        help="Minimum candidate trade rows required before deploying this gate.",
    )
    parser.add_argument(
        "--min-oof-rows",
        type=int,
        default=40,
        help="Minimum OOF probability rows required before deploying this gate.",
    )
    parser.add_argument(
        "--min-positive-oof-bins",
        type=int,
        default=2,
        help="Minimum positive expected-net OOF bins required before deploying this gate.",
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
    parser.add_argument("--midband-focus-enabled", action="store_true")
    parser.add_argument("--midband-focus-pup-low", type=float, default=0.55)
    parser.add_argument("--midband-focus-pup-high", type=float, default=0.60)
    parser.add_argument("--midband-focus-high-inclusive", action="store_true")
    parser.add_argument("--midband-focus-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--midband-focus-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--midband-focus-negative-weight", type=float, default=1.0)
    parser.add_argument("--midband-focus-positive-weight", type=float, default=1.0)
    parser.add_argument("--feature-meta-path", type=Path, default=None)
    parser.add_argument(
        "--reference-feature-mode",
        choices=("allow", "disable", "disable_on_source_mismatch"),
        default="allow",
    )
    parser.add_argument("--reference-feature-expected-source", type=str, default=None)
    parser.add_argument("--reference-feature-max-abs-value", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    return parser.parse_args()


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    def _series(name: str, default: float | pd.Series = 0.0) -> pd.Series:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce")
            if isinstance(default, pd.Series):
                fallback = pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
                return series.fillna(fallback)
            return series.fillna(float(default))
        if isinstance(default, pd.Series):
            return pd.to_numeric(default, errors="coerce").reindex(df.index).fillna(0.0)
        return pd.Series(float(default), index=df.index, dtype=float)

    def _direction_from_ret_pred(series: pd.Series) -> pd.Series:
        out = pd.Series("neutral", index=series.index, dtype=object)
        out.loc[series > 0.0] = "up"
        out.loc[series < 0.0] = "down"
        return out

    def _direction_from_price(close_series: pd.Series, projected_series: pd.Series) -> pd.Series:
        out = pd.Series("neutral", index=close_series.index, dtype=object)
        valid = (close_series > 0.0) & (projected_series > 0.0)
        out.loc[valid & (projected_series > close_series)] = "up"
        out.loc[valid & (projected_series < close_series)] = "down"
        return out

    def _direction_from_probability(series: pd.Series, neutral_band: float = 0.02) -> pd.Series:
        out = pd.Series("neutral", index=series.index, dtype=object)
        out.loc[series >= 0.5 + neutral_band] = "up"
        out.loc[series <= 0.5 - neutral_band] = "down"
        return out

    out = pd.DataFrame(index=df.index)
    out["p_up"] = _series("p_up", 0.0)
    out["raw_p_up"] = _series("raw_p_up", 0.0)
    out["ret_pred"] = _series("ret_pred", 0.0)
    out["expected_value_proxy"] = out["p_up"] * out["ret_pred"]
    out["abs_ret_pred"] = out["ret_pred"].abs()
    out["raw_calibrated_probability_gap"] = _series("raw_calibrated_probability_gap", out["p_up"] - out["raw_p_up"])
    out["probability_alignment_gap"] = _series("probability_alignment_gap", out["raw_calibrated_probability_gap"].abs())

    ret_side = _direction_from_ret_pred(out["ret_pred"])
    close_series = _series("close", 0.0)
    projected_series = _series("projected_price", close_series)
    projected_side = _direction_from_price(close_series, projected_series)
    raw_side = _direction_from_probability(out["raw_p_up"])
    resolved_side = _direction_from_probability(out["p_up"])
    direction_series = df.get("direction_next")
    if direction_series is None:
        direction_series = pd.Series(["up" if value >= 0.5 else "down" for value in out["p_up"]], index=df.index)
    direction_series = direction_series.astype(str).str.lower().fillna("neutral")

    out["raw_p_up_ret_mismatch"] = _series(
        "raw_p_up_ret_mismatch",
        ((raw_side != "neutral") & (ret_side != "neutral") & (raw_side != ret_side)).astype(float),
    )
    out["p_up_ret_mismatch"] = _series(
        "p_up_ret_mismatch",
        ((resolved_side != "neutral") & (ret_side != "neutral") & (resolved_side != ret_side)).astype(float),
    )
    out["raw_p_up_direction_mismatch"] = _series(
        "raw_p_up_direction_mismatch",
        ((raw_side != "neutral") & (direction_series != "neutral") & (raw_side != direction_series)).astype(float),
    )
    out["p_up_direction_mismatch"] = _series(
        "p_up_direction_mismatch",
        ((resolved_side != "neutral") & (direction_series != "neutral") & (resolved_side != direction_series)).astype(float),
    )
    out["ret_projected_price_consensus"] = _series(
        "ret_projected_price_consensus",
        ((ret_side == projected_side) & (ret_side != "neutral")).astype(float),
    )
    out["probability_calibration_guard_applied"] = _series("probability_calibration_guard_applied", 0.0)
    out["probability_calibration_used_regime_key"] = _series("probability_calibration_used_regime_key", 0.0)
    out["confidence_score"] = _series("confidence_score", 0.0)
    out["position_size"] = _series("position_size", 0.0)
    out["volatility_realized_24h"] = _series("volatility_realized_24h", 0.0)
    out["volatility_ewm_24h"] = _series("volatility_ewm_24h", 0.0)
    out["volatility_garch_like"] = _series("volatility_garch_like", 0.0)
    out["range_expansion_1h"] = _series("range_expansion_1h", 0.0)
    out["distance_from_session_high_8h"] = _series("distance_from_session_high_8h", 0.0)
    out["distance_from_session_low_8h"] = _series("distance_from_session_low_8h", 0.0)
    out["vwap_deviation_8h"] = _series("vwap_deviation_8h", 0.0)
    out["momentum_slope_2h"] = _series("momentum_slope_2h", 0.0)
    out["momentum_slope_4h"] = _series("momentum_slope_4h", 0.0)
    out["confluence_support_ratio"] = _series("confluence_support_ratio", 0.0)
    out["confluence_short_term_ratio"] = _series("confluence_short_term_ratio", 0.0)
    out["confluence_mid_term_ratio"] = _series("confluence_mid_term_ratio", 0.0)
    out["confluence_direction_matches_dominant"] = _series("confluence_direction_matches_dominant", 0.0)
    out["incumbent_signal_reference"] = _series("incumbent_signal_reference", 0.0)
    out["candidate_only_reference"] = _series("candidate_only_reference", 0.0)
    out["candidate_incumbent_disagreement"] = _series("candidate_incumbent_disagreement", 0.0)
    regime = df.get("regime_state")
    if regime is None:
        regime = pd.Series(["neutral"] * len(df), index=df.index)
    regime = regime.astype(str).str.lower().fillna("neutral")
    out["regime_is_trend"] = (regime == "trend_ignition").astype(float)
    out["regime_is_neutral"] = (regime == "neutral").astype(float)
    out["regime_is_chop"] = (regime == "chop").astype(float)
    return out[FEATURE_COLUMNS]


def _load_feature_meta(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _apply_reference_feature_controls(
    X: pd.DataFrame,
    *,
    feature_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    adjusted = X.copy()
    incumbent_reference = feature_meta.get("incumbent_reference", {}) if isinstance(feature_meta, dict) else {}
    incumbent_reference = incumbent_reference if isinstance(incumbent_reference, dict) else {}
    source_path = incumbent_reference.get("source")
    source_path = str(source_path) if source_path is not None else None
    expected_source = (
        str(args.reference_feature_expected_source)
        if args.reference_feature_expected_source is not None and str(args.reference_feature_expected_source).strip()
        else None
    )
    source_matches_expected = expected_source is None or source_path == expected_source
    max_abs_value = args.reference_feature_max_abs_value

    before_means = {
        column: float(pd.to_numeric(adjusted.get(column, 0.0), errors="coerce").fillna(0.0).mean())
        for column in REFERENCE_FEATURE_COLUMNS
    }
    clipped_columns: List[str] = []
    if max_abs_value is not None:
        max_abs = abs(float(max_abs_value))
        for column in REFERENCE_FEATURE_COLUMNS:
            if column in adjusted.columns:
                adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce").fillna(0.0).clip(-max_abs, max_abs)
                clipped_columns.append(column)

    disabled = False
    disable_reason = "allowed"
    if args.reference_feature_mode == "disable":
        disabled = True
        disable_reason = "disabled_by_config"
    elif args.reference_feature_mode == "disable_on_source_mismatch" and not source_matches_expected:
        disabled = True
        disable_reason = "source_mismatch"

    if disabled:
        for column in REFERENCE_FEATURE_COLUMNS:
            if column in adjusted.columns:
                adjusted[column] = 0.0

    after_means = {
        column: float(pd.to_numeric(adjusted.get(column, 0.0), errors="coerce").fillna(0.0).mean())
        for column in REFERENCE_FEATURE_COLUMNS
    }
    return adjusted, {
        "mode": str(args.reference_feature_mode),
        "feature_meta_path": (str(args.feature_meta_path) if args.feature_meta_path is not None else None),
        "incumbent_reference_source": source_path,
        "expected_source": expected_source,
        "source_matches_expected": bool(source_matches_expected),
        "max_abs_value": (None if max_abs_value is None else float(max_abs_value)),
        "clipped_columns": clipped_columns,
        "disabled": bool(disabled),
        "disable_reason": disable_reason,
        "feature_columns": list(REFERENCE_FEATURE_COLUMNS),
        "feature_means_before": before_means,
        "feature_means_after": after_means,
    }


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _build_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
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
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]
        model.fit(X.iloc[train_idx], y_train, **fit_kwargs)
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return np.clip(oof, 1e-6, 1.0 - 1e-6)


def _build_midband_focus_weights(df: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> np.ndarray:
    weights = np.ones(len(df), dtype=float)
    if not bool(args.midband_focus_enabled):
        return weights

    signal = pd.to_numeric(df.get(args.signal_col, 0.0), errors="coerce").fillna(0.0) > 0.0
    p_up = pd.to_numeric(df.get("p_up", df.get("p_up_meta", np.nan)), errors="coerce")
    abs_ret_pred = pd.to_numeric(df.get("ret_pred", np.nan), errors="coerce").abs()
    if bool(args.midband_focus_high_inclusive):
        in_band = (p_up >= float(args.midband_focus_pup_low)) & (p_up <= float(args.midband_focus_pup_high))
    else:
        in_band = (p_up >= float(args.midband_focus_pup_low)) & (p_up < float(args.midband_focus_pup_high))
    in_band = in_band & (abs_ret_pred >= float(args.midband_focus_min_abs_ret_pred))
    if args.midband_focus_max_abs_ret_pred is not None:
        in_band = in_band & (abs_ret_pred < float(args.midband_focus_max_abs_ret_pred))
    focus_mask = signal & in_band.fillna(False)

    neg_weight = max(1.0, float(args.midband_focus_negative_weight))
    pos_weight = max(1.0, float(args.midband_focus_positive_weight))
    y_arr = y.to_numpy(dtype=int)
    focus_arr = focus_mask.to_numpy(dtype=bool)
    weights[focus_arr & (y_arr <= 0)] = neg_weight
    weights[focus_arr & (y_arr > 0)] = pos_weight
    return weights


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
    feature_meta = _load_feature_meta(args.feature_meta_path)
    X = _extract_features(train_df)
    X, reference_feature_controls = _apply_reference_feature_controls(
        X,
        feature_meta=feature_meta,
        args=args,
    )
    sample_weight = _build_midband_focus_weights(train_df, y, args)

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

    sample_weight_cal = sample_weight[calibration_mask.to_numpy(dtype=bool)]
    oof_prob = _build_oof_probabilities(
        X_cal,
        y_cal,
        n_splits=int(args.oof_splits),
        sample_weight=sample_weight_cal,
    )
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
    model.fit(X, y, sample_weight=sample_weight)

    raw_ev_series = target_net.loc[trade_mask.loc[target_net.index]] if int(trade_mask.loc[target_net.index].sum()) > 0 else target_net
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95]
    raw_ev_quantiles = {
        f"q{int(q * 100)}": float(np.quantile(raw_ev_series.to_numpy(dtype=float), q)) if len(raw_ev_series) else 0.0
        for q in quantiles
    }

    prob = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
    auc = _safe_auc(y.to_numpy(dtype=int), prob)
    oof_auc = _safe_auc(y_cal.to_numpy(dtype=int)[valid_oof], oof_prob[valid_oof]) if int(valid_oof.sum()) else float("nan")
    positive_oof_bins = [
        entry
        for entry in oof_curve.get("bins", [])
        if float(entry.get("mean_ret_net", 0.0)) > 0.0
    ]
    candidate_rows = int(trade_mask.sum())
    oof_rows = int(valid_oof.sum())
    effective_row_count = int(len(df)) if bool(args.candidate_only) else int(len(X))
    deploy_checks = {
        "min_rows": effective_row_count >= int(args.min_rows),
        "min_candidate_rows": candidate_rows >= int(args.min_candidate_rows),
        "min_oof_rows": oof_rows >= int(args.min_oof_rows),
        "min_positive_oof_bins": len(positive_oof_bins) >= int(args.min_positive_oof_bins),
    }
    deploy_reasons = [name for name, passed in deploy_checks.items() if not passed]

    payload: Dict[str, object] = {
        "feature_columns": FEATURE_COLUMNS,
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "threshold": float(args.threshold),
        "min_rows_required": int(args.min_rows),
        "deploy_ready": not deploy_reasons,
        "deploy_readiness": {
            "checks": deploy_checks,
            "failed_checks": deploy_reasons,
            "min_candidate_rows_required": int(args.min_candidate_rows),
            "min_oof_rows_required": int(args.min_oof_rows),
            "min_positive_oof_bins_required": int(args.min_positive_oof_bins),
            "positive_oof_bin_count": int(len(positive_oof_bins)),
        },
        "midband_focus": {
            "enabled": bool(args.midband_focus_enabled),
            "p_up_low": float(args.midband_focus_pup_low),
            "p_up_high": float(args.midband_focus_pup_high),
            "high_inclusive": bool(args.midband_focus_high_inclusive),
            "min_abs_ret_pred": float(args.midband_focus_min_abs_ret_pred),
            "max_abs_ret_pred": (
                None if args.midband_focus_max_abs_ret_pred is None else float(args.midband_focus_max_abs_ret_pred)
            ),
            "negative_weight": float(args.midband_focus_negative_weight),
            "positive_weight": float(args.midband_focus_positive_weight),
            "focused_rows": int((sample_weight > 1.0).sum()),
        },
        "reference_feature_controls": reference_feature_controls,
        "metrics": {
            "rows": int(len(X)),
            "full_rows": int(len(df)),
            "effective_row_count_for_deploy": effective_row_count,
            "candidate_rows": candidate_rows,
            "auc": auc,
            "oof_auc": oof_auc,
            "oof_rows": oof_rows,
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
