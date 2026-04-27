import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split
from src.data.labeling import (
    binary_direction_labels,
    binary_direction_labels_with_no_trade,
    triple_barrier_direction_labels,
)
from src.scripts.build_training_dataset import _apply_funding_rate_features
from src.scripts.build_training_dataset import _load_local_features as _load_hourly_local_features
from src.scripts.build_training_dataset import merge_intrahour_15m_features
from src.trading.feature_engineering import augment_hourly_price_features as _shared_augment_hourly_price_features
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    split_volatility_arrays,
)
from src.data.macro_loader import MACRO_FEATURE_COLUMNS
from src.data.onchain_loader import ONCHAIN_FEATURE_COLUMNS
from src.data.source_parity import (
    drop_unready_source_family_features,
    evaluate_source_family_readiness,
)


PROCESSED_PATHS = [
    Path("data/processed/technical/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/macro/daily_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
]

META_PATH = Path("artifacts/datasets/btc_features_1h_direction_meta.json")

TREND_IGNITION_LABEL = "trend_ignition_6h"
TREND_IGNITION_HORIZON = 6
TREND_IGNITION_THRESHOLD = 0.01

CORE_MODEL_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
    "close_delta_1h",
    "close_pct_change_1h",
    "volume_delta_1h",
    "volume_pct_change_1h",
    "close_zscore_7h",
    "close_zscore_24h",
    "funding_rate_zscore_24h",
    "cvd_ratio_6h",
    "cvd_zscore_6h",
    "liquidity_range_ratio_6h",
    "liquidity_close_position_ratio",
    "range_expansion_1h",
    "distance_from_session_high_8h",
    "distance_from_session_low_8h",
    "vwap_deviation_8h",
    "momentum_slope_2h",
    "momentum_slope_4h",
    "intrabar_realized_vol_15m",
    "intrabar_return_dispersion_15m",
    "intrabar_path_range",
    "intrabar_path_efficiency_1h",
    "intrabar_taker_imbalance_mean",
    "intrabar_taker_imbalance_persistence",
    "intrabar_taker_imbalance_early_late_delta",
    "intrabar_directional_persistence_1h",
    "intrabar_reversal_score_1h",
    "intrabar_wick_asymmetry_shift",
    "intrabar_vol_term_structure_6h_24h",
    "intrabar_volume_regime_zscore_24h",
    "intrabar_flow_acceleration_3h",
    "intrabar_breakout_failure_1h",
    "intrabar_return_dispersion_regime_3h",
    "intrabar_return_dispersion_regime_6h",
    *MACRO_FEATURE_COLUMNS,
    *ONCHAIN_FEATURE_COLUMNS,
]

ZERO_VARIANCE_CANDIDATES: set[str] = set()

EXCLUDED_FEATURES: set[str] = {
    "funding_rate_zscore_24h",
    "ret_1h",
}

EXTERNAL_SOURCE_PREFIXES = (
    "cq_",
    "funding_",
    "fut_",
    "macro_",
    "onchain_",
    "orderbook_",
    "depth_",
    "lob_",
    "slippage_",
)
EXTERNAL_SOURCE_COLUMNS = {
    "funding_rate",
    "funding_rate_annualized",
    "open_interest",
}
PRESERVED_EXTERNAL_COLUMNS = {
    "funding_rate_zscore_24h",
    *MACRO_FEATURE_COLUMNS,
    *ONCHAIN_FEATURE_COLUMNS,
}

TECHNICAL_PREFIXES = (
    "candle_",
    "close_lag_",
    "interaction_",
    "intrabar_",
    "log_",
    "mom_",
    "pattern_",
    "poly2_",
    "price_",
    "range_",
    "return_",
    "roll_",
    "time_",
    "trend_",
    "trades_",
    "volume_",
    "vol_",
)


def _filter_features_by_reliability(
    allowed_features: list[str],
    reliability_json: str | None,
    min_score: float,
    *,
    target_horizon: float | None = None,
) -> list[str]:
    if not reliability_json:
        return allowed_features
    payload_path = Path(reliability_json)
    if not payload_path.exists():
        print(f"Feature reliability file not found at {payload_path}; skipping reliability filter.")
        return allowed_features
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    horizon_key = None
    if target_horizon is not None:
        numeric = float(target_horizon)
        horizon_key = f"{int(numeric)}h" if numeric >= 1.0 and float(numeric).is_integer() else f"{int(round(numeric * 60))}m"

    if horizon_key:
        horizon_regime_scores = payload.get("horizon_regime_feature_scores", {}) if isinstance(payload, dict) else {}
        if isinstance(horizon_regime_scores, dict) and isinstance(horizon_regime_scores.get(horizon_key), dict):
            filtered_from_scores: list[str] = []
            for feature in allowed_features:
                best_score = None
                for regime_scores in horizon_regime_scores[horizon_key].values():
                    if not isinstance(regime_scores, dict):
                        continue
                    score_obj = regime_scores.get(feature)
                    if isinstance(score_obj, dict) and "score" in score_obj:
                        try:
                            score = float(score_obj["score"])
                        except Exception:
                            score = None
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score
                if best_score is not None and best_score >= float(min_score):
                    filtered_from_scores.append(feature)
            if filtered_from_scores:
                print(
                    f"Feature reliability horizon-regime score filter kept {len(filtered_from_scores)} / {len(allowed_features)} features for {horizon_key}.",
                )
                return filtered_from_scores

        horizon_regime = payload.get("accepted_features_by_horizon_regime", {}) if isinstance(payload, dict) else {}
        if isinstance(horizon_regime, dict) and isinstance(horizon_regime.get(horizon_key), dict):
            accepted_union: set[str] = set()
            for slice_features in horizon_regime[horizon_key].values():
                if isinstance(slice_features, list):
                    accepted_union.update(str(value) for value in slice_features)
            filtered = [feature for feature in allowed_features if feature in accepted_union]
            if filtered:
                print(
                    f"Feature reliability horizon-regime filter kept {len(filtered)} / {len(allowed_features)} features for {horizon_key}.",
                )
                return filtered

        horizon_scores = payload.get("horizon_feature_scores", {}) if isinstance(payload, dict) else {}
        if isinstance(horizon_scores, dict) and isinstance(horizon_scores.get(horizon_key), dict):
            filtered_from_scores = []
            for feature in allowed_features:
                score_obj = horizon_scores[horizon_key].get(feature)
                score = None
                if isinstance(score_obj, dict) and "score" in score_obj:
                    try:
                        score = float(score_obj["score"])
                    except Exception:
                        score = None
                if score is not None and score >= float(min_score):
                    filtered_from_scores.append(feature)
            if filtered_from_scores:
                print(
                    f"Feature reliability horizon score filter kept {len(filtered_from_scores)} / {len(allowed_features)} features for {horizon_key}.",
                )
                return filtered_from_scores

        accepted_by_horizon = payload.get("accepted_features_by_horizon", {}) if isinstance(payload, dict) else {}
        if isinstance(accepted_by_horizon, dict) and isinstance(accepted_by_horizon.get(horizon_key), list):
            accepted_set = {str(value) for value in accepted_by_horizon[horizon_key]}
            filtered = [feature for feature in allowed_features if feature in accepted_set]
            if filtered:
                print(
                    f"Feature reliability horizon filter kept {len(filtered)} / {len(allowed_features)} features for {horizon_key}.",
                )
                return filtered

    accepted = payload.get("accepted_features")
    feature_scores = payload.get("feature_scores", {}) if isinstance(payload, dict) else {}

    if isinstance(accepted, list):
        accepted_set = {str(v) for v in accepted}
        filtered = [feature for feature in allowed_features if feature in accepted_set]
        if filtered:
            print(f"Feature reliability accepted set kept {len(filtered)} / {len(allowed_features)} features.")
            return filtered
        print("Feature reliability accepted set matched no allowed features; falling back to score filter.")

    filtered: list[str] = []
    for feature in allowed_features:
        score_obj = feature_scores.get(feature) if isinstance(feature_scores, dict) else None
        score = None
        if isinstance(score_obj, dict) and "score" in score_obj:
            try:
                score = float(score_obj["score"])
            except Exception:
                score = None
        if score is not None and score >= float(min_score):
            filtered.append(feature)
    if filtered:
        print(f"Feature reliability filter kept {len(filtered)} / {len(allowed_features)} features.")
        return filtered
    print("Feature reliability filter removed all allowed features; using original set as fallback.")
    return allowed_features


def _drop_external_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = [
        column
        for column in df.columns
        if column not in PRESERVED_EXTERNAL_COLUMNS
        and (
            column in EXTERNAL_SOURCE_COLUMNS
            or any(column.startswith(prefix) for prefix in EXTERNAL_SOURCE_PREFIXES)
        )
    ]
    if to_remove:
        preview = ", ".join(sorted(to_remove)[:5])
        suffix = "..." if len(to_remove) > 5 else ""
        print(f"Dropped {len(to_remove)} non-Binance feature columns: {preview}{suffix}")
        df = df.drop(columns=sorted(set(to_remove)))
    return df


def _append_technical_feature_columns(df: pd.DataFrame, allowed: list[str]) -> list[str]:
    technical_columns = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in TECHNICAL_PREFIXES)
    ]
    for column in sorted(technical_columns):
        if column not in allowed:
            allowed.append(column)
    return allowed


def _drop_excluded_features(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = [col for col in EXCLUDED_FEATURES if col in df.columns]
    if to_remove:
        df = df.drop(columns=to_remove)
        preview = ", ".join(sorted(to_remove)[:5])
        suffix = "..." if len(to_remove) > 5 else ""
        print(f"Dropped {len(to_remove)} excluded features: {preview}{suffix}")
    return df


def _enforce_feature_coverage(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    if not required:
        return df

    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Cannot enforce coverage; missing columns: {missing_columns}")

    coverage = df[required].notna().all(axis=1)
    dropped = int((~coverage).sum())
    if dropped:
        missing_ts = df.loc[~coverage, "ts"].dropna()
        first_gap = missing_ts.min().isoformat() if not missing_ts.empty else "unknown"
        last_gap = missing_ts.max().isoformat() if not missing_ts.empty else "unknown"
        print(
            f"Dropped {dropped} rows lacking complete feature coverage between {first_gap} and {last_gap}.",
        )

    if not coverage.any():
        raise RuntimeError("No rows remain after enforcing feature coverage; check upstream merges.")

    return df.loc[coverage].reset_index(drop=True)


def _drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    removed: list[str] = []
    for column in ZERO_VARIANCE_CANDIDATES:
        if column not in df.columns:
            continue
        series = df[column]
        if series.dropna().empty or np.isclose(series.std(ddof=0), 0.0):
            df = df.drop(columns=column)
            removed.append(column)
    if removed:
        preview = ", ".join(removed[:5])
        suffix = "..." if len(removed) > 5 else ""
        print(f"Dropped {len(removed)} constant features: {preview}{suffix}")
    return df


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def _augment_price_features(df: pd.DataFrame) -> pd.DataFrame:
    return _shared_augment_hourly_price_features(df, strict_missing=True)


def _merge_processed_features(df: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise RuntimeError("Expected 'ts' column in curated features for feature alignment.")

    augmented = df.copy()
    augmented["ts"] = pd.to_datetime(augmented["ts"], utc=True, errors="coerce")
    augmented = augmented.dropna(subset=["ts"]).reset_index(drop=True)
    augmented["ts"] = augmented["ts"].dt.floor("h")
    augmented = augmented.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

    for path in paths:
        if not path.exists():
            print(f"Processed features not found at {path}; skipping.")
            continue

        extra = pd.read_parquet(path)
        if extra.empty:
            print(f"Processed features at {path} are empty; skipping.")
            continue

        if "timestamp" in extra.columns:
            extra = extra.rename(columns={"timestamp": "ts"})

        extra["ts"] = pd.to_datetime(extra["ts"], utc=True, errors="coerce")
        extra = extra.dropna(subset=["ts"]).reset_index(drop=True)
        extra["ts"] = extra["ts"].dt.floor("h")
        extra = extra.sort_values("ts").drop_duplicates(subset="ts", keep="last")

        columns_before = set(augmented.columns)
        merged = pd.merge_asof(
            augmented.sort_values("ts"),
            extra,
            on="ts",
            direction="backward",
            allow_exact_matches=True,
        )
        augmented = merged.sort_values("ts").reset_index(drop=True)
        new_columns = [col for col in augmented.columns if col not in columns_before]

        if new_columns:
            preview = ", ".join(new_columns[:5])
            suffix = "..." if len(new_columns) > 5 else ""
            print(f"Added {len(new_columns)} feature columns from {path}: {preview}{suffix}")
        else:
            print(f"No new columns merged from {path}; check schema overlap.")

    return augmented


def make_direction_labels(y_ret: pd.Series, threshold: float) -> pd.Series:
    return binary_direction_labels(y_ret, threshold)


def _load_local_features() -> pd.DataFrame:
    """Load local hourly features from the expanded Binance-only source stack."""
    return _load_hourly_local_features()


def prepare_direction_feature_frame(
    *,
    threshold: float,
    labeling_scheme: str = "binary",
    tb_horizon_steps: int = 1,
    tb_vol_window: int = 24,
    tb_upper_mult: float = 1.0,
    tb_lower_mult: float = 1.0,
    no_trade_abs_ret: float = 0.0,
    no_trade_vol_mult: float = 0.0,
    reliability_json: str | None = None,
    reliability_min_score: float = 0.55,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = _load_local_features()

    if df.empty:
        raise RuntimeError(
            "Loaded empty DataFrame from local; check data."
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df["ts"] = df["ts"].dt.floor("h")
    df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

    df = _merge_processed_features(df, PROCESSED_PATHS)
    df = merge_intrahour_15m_features(df)
    df = _apply_funding_rate_features(df)
    df = _drop_external_source_columns(df)
    df = _augment_price_features(df)
    df, volatility_columns = add_volatility_columns(
        df,
        realized_windows=DEFAULT_REALIZED_WINDOWS,
    )
    df = _drop_constant_features(df)
    df = _drop_excluded_features(df)
    df = df.sort_values("ts").reset_index(drop=True)

    # Compute 1h returns
    df["ret_1h"] = df["close"].pct_change().shift(-1)

    allowed_features = [feature for feature in CORE_MODEL_FEATURES if feature in df.columns]
    for column in volatility_columns:
        if column not in allowed_features:
            allowed_features.append(column)
    allowed_features = _append_technical_feature_columns(df, allowed_features)
    allowed_features = _filter_features_by_reliability(
        allowed_features,
        reliability_json=reliability_json,
        min_score=reliability_min_score,
        target_horizon=1.0,
    )
    allowed_features = [feature for feature in allowed_features if feature not in {col for col in allowed_features if col in df.columns and df[col].notna().sum() == 0}]
    source_family_parity = evaluate_source_family_readiness(df)
    allowed_features, dropped_source_family_features = drop_unready_source_family_features(
        allowed_features,
        source_family_parity,
    )

    if allowed_features:
        df = _enforce_feature_coverage(df, allowed_features)

    df.attrs["source_family_parity"] = source_family_parity
    df.attrs["dropped_source_family_features"] = dropped_source_family_features
    return df, allowed_features, sorted(volatility_columns)


def build_direction_splits(
    output_dir: str,
    threshold: float,
    *,
    labeling_scheme: str = "binary",
    tb_horizon_steps: int = 1,
    tb_vol_window: int = 24,
    tb_upper_mult: float = 1.0,
    tb_lower_mult: float = 1.0,
    no_trade_abs_ret: float = 0.0,
    no_trade_vol_mult: float = 0.0,
    reliability_json: str | None = None,
    reliability_min_score: float = 0.55,
    meta_path: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    df, allowed_features, volatility_columns = prepare_direction_feature_frame(
        threshold=threshold,
        labeling_scheme=labeling_scheme,
        tb_horizon_steps=tb_horizon_steps,
        tb_vol_window=tb_vol_window,
        tb_upper_mult=tb_upper_mult,
        tb_lower_mult=tb_lower_mult,
        no_trade_abs_ret=no_trade_abs_ret,
        no_trade_vol_mult=no_trade_vol_mult,
        reliability_json=reliability_json,
        reliability_min_score=reliability_min_score,
    )

    X, y_ret, ts_series = make_features_and_target(
        df,
        target_column="ret_1h",
        allowed_features=allowed_features if allowed_features else None,
        return_ts=True,
    )

    labeling_stats: dict[str, float] = {}
    if labeling_scheme == "triple_barrier":
        labels_raw, labeling_stats = triple_barrier_direction_labels(
            df.loc[X.index, "close"],
            horizon_steps=tb_horizon_steps,
            vol_window=tb_vol_window,
            upper_mult=tb_upper_mult,
            lower_mult=tb_lower_mult,
        )
        y_dir = labels_raw.dropna().astype(int)
        X = X.loc[y_dir.index]
    elif labeling_scheme == "binary_no_trade":
        y_labels, labeling_stats = binary_direction_labels_with_no_trade(
            y_ret,
            threshold=threshold,
            no_trade_abs_ret=no_trade_abs_ret,
            no_trade_vol_mult=no_trade_vol_mult,
            vol_window=24,
        )
        y_dir = y_labels.dropna().astype(int)
        X = X.loc[y_dir.index]
    else:
        y_dir = make_direction_labels(y_ret, threshold=threshold)
        X = X.loc[y_dir.index]

    y_dir = y_dir.loc[X.index]
    y_ret_selected = pd.to_numeric(y_ret.loc[X.index], errors="coerce").fillna(0.0)
    ts_selected = pd.to_datetime(ts_series.loc[X.index], utc=True, errors="coerce")
    aligned_df = df.loc[X.index].reset_index(drop=True)

    splits = time_series_train_val_test_split(X, y_dir)

    output_path = os.path.join(output_dir, "btc_features_1h_direction_splits.npz")
    ts_values = ts_selected.to_numpy(dtype="datetime64[ns]")
    n_train = splits.X_train.shape[0]
    n_val = splits.X_val.shape[0]
    y_ret_values = y_ret_selected.to_numpy(dtype=np.float32, copy=False)
    y_ret_train = y_ret_values[:n_train]
    y_ret_val = y_ret_values[n_train:n_train + n_val]
    y_ret_test = y_ret_values[n_train + n_val :]
    ts_train = ts_values[:n_train]
    ts_val = ts_values[n_train:n_train + n_val]
    ts_test = ts_values[n_train + n_val :]

    volatility_arrays = split_volatility_arrays(
        aligned_df,
        volatility_columns,
        n_train=n_train,
        n_val=n_val,
    )

    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        y_ret_train=y_ret_train,
        y_ret_val=y_ret_val,
        y_ret_test=y_ret_test,
        ts_train=ts_train,
        ts_val=ts_val,
        ts_test=ts_test,
        ts_all=ts_values,
        feature_names=np.array(splits.feature_names),
        threshold=np.array([threshold], dtype=float),
        scaler_mean=np.asarray(splits.scaler_mean, dtype=np.float32),
        scaler_scale=np.asarray(splits.scaler_scale, dtype=np.float32),
        **volatility_arrays,
    )
    print(f"Saved direction dataset splits to {output_path}")

    def _describe(ts_array: np.ndarray) -> dict[str, object]:
        if ts_array.size == 0:
            return {"rows": 0, "ts_min": None, "ts_max": None}
        series = pd.Series(pd.to_datetime(ts_array))
        if getattr(series.dt, "tz", None) is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
        return {
            "rows": int(ts_array.size),
            "ts_min": series.min().isoformat(),
            "ts_max": series.max().isoformat(),
        }

    meta_payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "row_count": int(ts_values.size),
        "feature_count": int(len(splits.feature_names)),
        "feature_names": list(splits.feature_names),
        "threshold": float(threshold),
        "labeling_scheme": labeling_scheme,
        "triple_barrier": {
            "horizon_steps": int(tb_horizon_steps),
            "vol_window": int(tb_vol_window),
            "upper_mult": float(tb_upper_mult),
            "lower_mult": float(tb_lower_mult),
            "stats": labeling_stats,
        },
        "no_trade_zone": {
            "no_trade_abs_ret": float(no_trade_abs_ret),
            "no_trade_vol_mult": float(no_trade_vol_mult),
        },
        "volatility": {
            "columns": sorted(volatility_columns),
            "realized_windows": list(DEFAULT_REALIZED_WINDOWS),
        },
        "scaler": {
            "available": bool(splits.scaler_mean is not None and splits.scaler_scale is not None),
            "mean_key": "scaler_mean",
            "scale_key": "scaler_scale",
        },
        "source_family_parity": df.attrs.get("source_family_parity", {}),
        "dropped_source_family_features": df.attrs.get("dropped_source_family_features", {}),
        "ts_range": _describe(ts_values),
        "splits": {
            "train": _describe(ts_train),
            "val": _describe(ts_val),
            "test": _describe(ts_test),
        },
    }
    resolved_meta_path = Path(meta_path) if meta_path else Path(output_dir) / "btc_features_1h_direction_meta.json"
    resolved_meta_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_meta_path.write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote direction dataset meta summary to {resolved_meta_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BTCUSDT 1h direction training dataset from BigQuery curated features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared direction dataset splits.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Direction threshold theta: label is 1 if ret_1h > theta, else 0.",
    )
    parser.add_argument(
        "--labeling-scheme",
        choices=("binary", "binary_no_trade", "triple_barrier"),
        default="binary",
        help="Direction label construction strategy.",
    )
    parser.add_argument("--tb-horizon-steps", type=int, default=1, help="Forward steps for triple-barrier labels.")
    parser.add_argument("--tb-vol-window", type=int, default=24, help="Rolling volatility window for barriers.")
    parser.add_argument("--tb-upper-mult", type=float, default=1.0, help="Upper barrier multiplier.")
    parser.add_argument("--tb-lower-mult", type=float, default=1.0, help="Lower barrier multiplier.")
    parser.add_argument("--no-trade-abs-ret", type=float, default=0.0, help="Absolute return no-trade band for binary labels.")
    parser.add_argument("--no-trade-vol-mult", type=float, default=0.0, help="Volatility-multiplier no-trade band for binary labels.")
    parser.add_argument("--feature-reliability-json", type=str, default=None, help="Optional feature reliability JSON with accepted_features.")
    parser.add_argument("--feature-reliability-min-score", type=float, default=0.55, help="Minimum feature score when reliability JSON provides per-feature scores.")
    parser.add_argument("--meta-path", type=str, default=None, help="Optional output path for direction dataset metadata JSON.")
    args = parser.parse_args()

    build_direction_splits(
        args.output_dir,
        args.threshold,
        labeling_scheme=args.labeling_scheme,
        tb_horizon_steps=args.tb_horizon_steps,
        tb_vol_window=args.tb_vol_window,
        tb_upper_mult=args.tb_upper_mult,
        tb_lower_mult=args.tb_lower_mult,
        no_trade_abs_ret=args.no_trade_abs_ret,
        no_trade_vol_mult=args.no_trade_vol_mult,
        reliability_json=args.feature_reliability_json,
        reliability_min_score=args.feature_reliability_min_score,
        meta_path=args.meta_path,
    )


if __name__ == "__main__":
    main()
