import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import (
    BQ_DATASET_CURATED,
    BQ_TABLE_FEATURES_1H,
    PROJECT_ID,
)
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import (
    enforce_unique_hourly_index,
    make_features_and_target,
    time_series_train_val_test_split,
)
from src.data.targets_multi_horizon import (
    RANGE_TARGET_HORIZONS,
    add_multi_horizon_targets,
)
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    split_volatility_arrays,
)
from src.scripts.build_training_dataset import _apply_funding_rate_features
from src.scripts.build_training_dataset import _load_local_features as _load_hourly_local_features
from src.trading.feature_engineering import augment_hourly_price_features as _shared_augment_hourly_price_features
from src.data.macro_loader import MACRO_FEATURE_COLUMNS
from src.data.onchain_loader import ONCHAIN_FEATURE_COLUMNS
from src.data.source_parity import (
    drop_unready_source_family_features,
    evaluate_source_family_readiness,
)
from src.scripts.build_training_dataset import _drop_uncovered_allowed_features
from src.scripts.build_training_dataset import merge_intrahour_15m_features


DEFAULT_HORIZONS: List[int] = [1, 4, 8, 12]
PROCESSED_PATHS = [
    Path("data/processed/technical/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/macro/daily_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
]

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
    "volatility_realized_24h",
    "volatility_realized_72h",
    "volatility_ewm_24h",
    "volatility_ewm_72h",
    "volatility_garch_like",
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
    "ret_1h",
    "ret_4h",
    "ret_8h",
    "ret_12h",
}

EXTERNAL_SOURCE_PREFIXES = (
    "cq_",
    "funding_",
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

FUTURES_FEATURE_PREFIXES = (
    "fut_",
)

FUTURES_FEATURE_COLUMNS = {
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "funding_rate_annualized",
}

RET_FEATURE_COLUMNS = (
    "ret_1h",
    "ret_4h",
    "ret_8h",
    "ret_12h",
)

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


def _is_futures_feature(column: str) -> bool:
    return column in FUTURES_FEATURE_COLUMNS or any(
        column.startswith(prefix) for prefix in FUTURES_FEATURE_PREFIXES
    )


def _append_futures_feature_columns(df: pd.DataFrame, allowed: list[str]) -> list[str]:
    futures_columns = [column for column in df.columns if _is_futures_feature(column)]
    for column in sorted(futures_columns):
        if column not in allowed:
            allowed.append(column)
    return allowed


def _append_return_feature_columns(df: pd.DataFrame, allowed: list[str]) -> list[str]:
    ret_columns = [column for column in df.columns if column in RET_FEATURE_COLUMNS]
    for column in sorted(ret_columns):
        if column not in allowed:
            allowed.append(column)
    return allowed


def _drop_external_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = [
        column
        for column in df.columns
        if column not in PRESERVED_EXTERNAL_COLUMNS
        and not _is_futures_feature(column)
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


def _drop_constant_features(df: pd.DataFrame, candidates: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    removed: List[str] = []
    for column in candidates:
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
    return df, removed


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
    return _shared_augment_hourly_price_features(df, strict_missing=False)

META_PATH = Path("artifacts/datasets/btc_features_multi_horizon_meta.json")


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
        extra = extra.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)

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

    augmented = augmented.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return augmented


def _split_array(values: np.ndarray, n_train: int, n_val: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        values[:n_train],
        values[n_train:n_train + n_val],
        values[n_train + n_val :],
    )

def build_multi_horizon_dataset(
    output_dir: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    features_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    if output_path:
        output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if features_path:
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features CSV not found at {features_path}")
        df = pd.read_csv(features_path, parse_dates=["ts"])
    else:
        df = _load_hourly_local_features()
        if df.empty:
            df = load_btc_features_1h(
                project_id=PROJECT_ID,
                dataset_id=BQ_DATASET_CURATED,
                table_id=BQ_TABLE_FEATURES_1H,
            )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from local or BigQuery 1h features; check historical inputs.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df, dup_count, gap_count = enforce_unique_hourly_index(
        df,
        label="curated_features",
        raise_on_gap=False,
        normalize_to_hour=True,
    )
    if dup_count == 0 and gap_count == 0:
        print("[curated_features] Hourly spacing verified; no duplicates detected.")
    elif gap_count:
        print(f"[curated_features] Logged {gap_count} non-hourly intervals; upstream gaps remain.")

    df = df.sort_values("ts").reset_index(drop=True)
    df = _merge_processed_features(df, PROCESSED_PATHS)
    df = merge_intrahour_15m_features(df)
    df = _apply_funding_rate_features(df)
    df = _drop_external_source_columns(df)
    df = _augment_price_features(df)
    df, volatility_columns = add_volatility_columns(
        df,
        realized_windows=DEFAULT_REALIZED_WINDOWS,
    )
    df, _ = _drop_constant_features(df, ZERO_VARIANCE_CANDIDATES)
    df = _drop_excluded_features(df)
    df, dup_after_merge, gap_after_merge = enforce_unique_hourly_index(
        df,
        label="curated_features_merged",
        raise_on_gap=False,
        normalize_to_hour=True,
    )
    if dup_after_merge:
        print(f"[curated_features_merged] Removed {dup_after_merge} duplicates introduced during merge.")
    if gap_after_merge:
        print(
            f"[curated_features_merged] Logged {gap_after_merge} non-hourly intervals after merge; "
            "downstream consumers should handle upstream gaps."
        )

    df_targets = add_multi_horizon_targets(df, horizons=horizons, price_col="close")

    ret_cols = [f"ret_{h}h" for h in horizons]
    df_targets = df_targets.dropna(subset=ret_cols)

    range_cols: list[str] = []
    for horizon in RANGE_TARGET_HORIZONS:
        for suffix in ("max", "min"):
            column = f"ret_{suffix}_{horizon}h"
            if column in df_targets.columns and horizon in horizons:
                range_cols.append(column)
    if range_cols:
        df_targets = df_targets.dropna(subset=range_cols)

    allowed_features = [feature for feature in CORE_MODEL_FEATURES if feature in df_targets.columns]
    for column in volatility_columns:
        if column in df_targets.columns and column not in allowed_features:
            allowed_features.append(column)
    allowed_features = _append_futures_feature_columns(df_targets, allowed_features)
    allowed_features = _append_technical_feature_columns(df_targets, allowed_features)
    allowed_features = _drop_uncovered_allowed_features(df_targets, allowed_features)
    source_family_parity = evaluate_source_family_readiness(df_targets)
    allowed_features, dropped_source_family_features = drop_unready_source_family_features(
        allowed_features,
        source_family_parity,
    )
    df_targets = _enforce_feature_coverage(df_targets, allowed_features)
    X, y_ret1h = make_features_and_target(
        df_targets,
        target_column="ret_1h",
        dropna=False,
        allowed_features=allowed_features,
    )
    splits = time_series_train_val_test_split(X, y_ret1h, train_frac=train_frac, val_frac=val_frac)

    n_train = splits.X_train.shape[0]
    n_val = splits.X_val.shape[0]
    n_total = len(df_targets)
    if n_train + n_val + splits.X_test.shape[0] != n_total:
        raise RuntimeError("Split sizes do not sum to dataset length; check split configuration.")

    data_ret = {h: df_targets[f"ret_{h}h"].to_numpy(dtype=np.float32) for h in horizons if h != 1}
    data_dir = {h: df_targets[f"dir_{h}h"].to_numpy(dtype=np.int8) for h in horizons}
    data_range: dict[tuple[int, str], np.ndarray] = {}
    for horizon in RANGE_TARGET_HORIZONS:
        if horizon not in horizons:
            continue
        max_col = f"ret_max_{horizon}h"
        min_col = f"ret_min_{horizon}h"
        if max_col in df_targets.columns:
            data_range[(horizon, "max")] = df_targets[max_col].to_numpy(dtype=np.float32)
        if min_col in df_targets.columns:
            data_range[(horizon, "min")] = df_targets[min_col].to_numpy(dtype=np.float32)
    ts_values = df_targets["ts"].to_numpy(dtype="datetime64[ns]")

    ts_train = ts_values[:n_train]
    ts_val = ts_values[n_train:n_train + n_val]
    ts_test = ts_values[n_train + n_val :]

    if output_path is None:
        output_path = os.path.join(output_dir, "btc_features_multi_horizon_splits.npz")

    volatility_arrays = split_volatility_arrays(
        df_targets,
        volatility_columns,
        n_train=n_train,
        n_val=n_val,
    )

    save_kwargs = {
        "X_train": splits.X_train,
        "y_train": splits.y_train,
        "X_val": splits.X_val,
        "y_val": splits.y_val,
        "X_test": splits.X_test,
        "y_test": splits.y_test,
        "ts_train": ts_train,
        "ts_val": ts_val,
        "ts_test": ts_test,
        "ts_all": ts_values,
        "feature_names": np.array(splits.feature_names),
        "scaler_mean": np.asarray(splits.scaler_mean, dtype=np.float32),
        "scaler_scale": np.asarray(splits.scaler_scale, dtype=np.float32),
        "horizons": np.array(sorted({int(h) for h in horizons}), dtype=np.int32),
        "direction_threshold": np.array([0.0], dtype=np.float32),
    }

    save_kwargs.update(volatility_arrays)

    for horizon, values in data_ret.items():
        train, val, test = _split_array(values, n_train, n_val)
        save_kwargs[f"y_ret{horizon}h_train"] = train
        save_kwargs[f"y_ret{horizon}h_val"] = val
        save_kwargs[f"y_ret{horizon}h_test"] = test

    for horizon, values in data_dir.items():
        train, val, test = _split_array(values, n_train, n_val)
        save_kwargs[f"y_dir{horizon}h_train"] = train
        save_kwargs[f"y_dir{horizon}h_val"] = val
        save_kwargs[f"y_dir{horizon}h_test"] = test

    for (horizon, kind), values in data_range.items():
        train, val, test = _split_array(values, n_train, n_val)
        if kind == "max":
            prefix = "y_retmax"
        else:
            prefix = "y_retmin"
        save_kwargs[f"{prefix}{horizon}h_train"] = train
        save_kwargs[f"{prefix}{horizon}h_val"] = val
        save_kwargs[f"{prefix}{horizon}h_test"] = test

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved multi-horizon dataset splits to {output_path}")
    print("Stored horizons:", save_kwargs["horizons"].tolist())

    def _describe_split(ts_array: np.ndarray) -> dict[str, object]:
        if ts_array.size == 0:
            return {"rows": 0, "ts_min": None, "ts_max": None}
        series = pd.to_datetime(ts_array).tz_localize("UTC")
        return {
            "rows": int(ts_array.size),
            "ts_min": series.min().isoformat(),
            "ts_max": series.max().isoformat(),
        }

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    meta_payload = {
        "generated_at": generated_at,
        "row_count": int(len(df_targets)),
        "feature_count": int(len(splits.feature_names)),
        "splits": {
            "train": _describe_split(ts_train),
            "val": _describe_split(ts_val),
            "test": _describe_split(ts_test),
        },
        "volatility": {
            "columns": volatility_columns,
            "realized_windows": list(DEFAULT_REALIZED_WINDOWS),
        },
        "scaler": {
            "available": bool(splits.scaler_mean is not None and splits.scaler_scale is not None),
            "mean_key": "scaler_mean",
            "scale_key": "scaler_scale",
        },
        "source_family_parity": source_family_parity,
        "dropped_source_family_features": dropped_source_family_features,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote dataset meta summary to {META_PATH}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-horizon BTC dataset (1h & 4h targets) from the curated BigQuery features. "
            "This keeps the legacy 1h dataset untouched and writes a separate NPZ with additional targets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit path for the NPZ file (overrides --output-dir filename).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Horizons (in hours) to include when computing targets (default: 1 4).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of samples allocated to the training split (default: 0.7).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of samples allocated to the validation split (default: 0.15).",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Optional CSV with curated 1h features to bypass BigQuery (expects ts column).",
    )
    args = parser.parse_args()

    build_multi_horizon_dataset(
        output_dir=args.output_dir,
        horizons=args.horizons,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        features_path=args.features_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
