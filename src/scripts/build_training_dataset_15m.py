import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_15M
from src.data.bq_loader import load_btc_features_15m
from src.data.dataset_preparation import (
    enforce_unique_hourly_index,
    make_features_and_target,
    repair_hourly_continuity,
    time_series_train_val_test_split,
)
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    split_volatility_arrays,
)
from src.scripts import build_training_dataset as hourly_builder


PROCESSED_PATHS = hourly_builder.PROCESSED_PATHS
META_PATH = Path("artifacts/datasets/btc_features_15m_meta.json")
OUTPUT_FILENAME = "btc_features_15m_splits.npz"
EXPECTED_FREQ = pd.Timedelta(minutes=15)
PERIODS_PER_HOUR = 4
VOLATILITY_WINDOW_PERIODS = tuple(window * PERIODS_PER_HOUR for window in DEFAULT_REALIZED_WINDOWS)

CORE_MODEL_FEATURES_15M = [
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
    "close_delta_15m",
    "close_pct_change_15m",
    "close_delta_1h",
    "close_pct_change_1h",
    "volume_delta_15m",
    "volume_pct_change_15m",
    "volume_delta_1h",
    "volume_pct_change_1h",
    "close_zscore_7h",
    "close_zscore_24h",
    "volatility_realized_24h",
    "volatility_realized_72h",
    "volatility_ewm_24h",
    "volatility_ewm_72h",
    "volatility_garch_like",
]

ZERO_VARIANCE_CANDIDATES = hourly_builder.ZERO_VARIANCE_CANDIDATES
EXCLUDED_FEATURES = hourly_builder.EXCLUDED_FEATURES


def _merge_processed_features_15m(df: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise RuntimeError("Expected 'ts' column in curated features for feature alignment.")

    augmented = df.copy()
    augmented["ts"] = pd.to_datetime(augmented["ts"], utc=True, errors="coerce")
    augmented["ts"] = augmented["ts"].dt.floor("15min")
    augmented = augmented.sort_values("ts").reset_index(drop=True)

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

        extra["ts"] = pd.to_datetime(extra["ts"], utc=True)
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

    augmented = augmented.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return augmented


def _augment_price_features_15m(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    def _safe_diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _safe_pct(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for base in ("close", "volume", "fut_close", "fut_volume"):
        if base not in result.columns:
            continue
        series = result[base]
        result[f"{base}_delta_15m"] = _safe_diff(series)
        result[f"{base}_pct_change_15m"] = _safe_pct(series)
        result[f"{base}_delta_1h"] = series.diff(periods=PERIODS_PER_HOUR).fillna(0.0)
        pct_1h = series.pct_change(periods=PERIODS_PER_HOUR).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result[f"{base}_pct_change_1h"] = pct_1h

    window_7h = 7 * PERIODS_PER_HOUR
    window_24h = 24 * PERIODS_PER_HOUR
    min_periods_7h = max(4, window_7h // 2)
    min_periods_24h = max(6, window_24h // 2)

    if "close" in result.columns:
        std_7 = result["close"].rolling(window=window_7h, min_periods=min_periods_7h).std(ddof=0)
        std_24 = result["close"].rolling(window=window_24h, min_periods=min_periods_24h).std(ddof=0)
        if "ma_close_7h" in result.columns:
            denom = std_7.replace(0.0, np.nan)
            result["close_zscore_7h"] = ((result["close"] - result["ma_close_7h"]) / denom).fillna(0.0)
        if "ma_close_24h" in result.columns:
            denom = std_24.replace(0.0, np.nan)
            result["close_zscore_24h"] = ((result["close"] - result["ma_close_24h"]) / denom).fillna(0.0)

    if "fut_close" in result.columns:
        rolling_mean = result["fut_close"].rolling(window=window_7h, min_periods=min_periods_7h).mean()
        rolling_std = (
            result["fut_close"].rolling(window=window_7h, min_periods=min_periods_7h).std(ddof=0).replace(0.0, np.nan)
        )
        result["fut_close_zscore_7h"] = ((result["fut_close"] - rolling_mean) / rolling_std).fillna(0.0)

    return result


def _recompute_return_targets_15m(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "close" not in result.columns:
        return result

    log_close = np.log(result["close"].astype(float))
    result["ret_15m"] = log_close.diff()
    result["ret_1h"] = log_close.diff(periods=PERIODS_PER_HOUR)
    result["ret_fwd_1h"] = log_close.shift(-PERIODS_PER_HOUR) - log_close
    result["ret_fwd_3h"] = log_close.shift(-3 * PERIODS_PER_HOUR) - log_close
    result["ret_4h"] = log_close.shift(-4 * PERIODS_PER_HOUR) - log_close
    return result


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_15m(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_15M,
    )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check that the 15m curated table has data.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df, dup_count, gap_count = enforce_unique_hourly_index(
        df,
        label="curated_features_15m",
        raise_on_gap=False,
        normalize_to_hour=False,
        expected_freq=EXPECTED_FREQ,
    )
    if dup_count == 0 and gap_count == 0:
        print("[curated_features_15m] 15m spacing verified; no duplicates detected.")
    elif gap_count:
        print(f"[curated_features_15m] Logged {gap_count} non-15m intervals; upstream gaps remain.")

    df, backfilled = repair_hourly_continuity(
        df,
        label="curated_features_15m",
        expected_freq=EXPECTED_FREQ,
    )
    if backfilled:
        print(f"[curated_features_15m] Reindexed with {backfilled} synthetic 15m rows via forward/back fill.")

    df = _merge_processed_features_15m(df, PROCESSED_PATHS)
    df = hourly_builder._drop_external_source_columns(df)
    df = _augment_price_features_15m(df)
    df, volatility_columns = add_volatility_columns(
        df,
        realized_windows=DEFAULT_REALIZED_WINDOWS,
        periods_per_hour=PERIODS_PER_HOUR,
    )
    df, _ = hourly_builder._drop_constant_features(df, ZERO_VARIANCE_CANDIDATES)
    df = hourly_builder._drop_excluded_features(df)
    df = _recompute_return_targets_15m(df)
    df, dup_after_merge, gap_after_merge = enforce_unique_hourly_index(
        df,
        label="curated_features_15m_merged",
        raise_on_gap=False,
        normalize_to_hour=False,
        expected_freq=EXPECTED_FREQ,
    )
    if dup_after_merge:
        print(f"[curated_features_15m_merged] Removed {dup_after_merge} duplicates introduced during merge.")
    if gap_after_merge:
        print(
            f"[curated_features_15m_merged] Logged {gap_after_merge} non-15m intervals after merge; "
            "downstream consumers should handle upstream gaps."
        )

    allowed_features = [feature for feature in CORE_MODEL_FEATURES_15M if feature in df.columns]
    for column in volatility_columns:
        if column not in allowed_features:
            allowed_features.append(column)
    allowed_features = hourly_builder._append_technical_feature_columns(df, allowed_features)
    df = hourly_builder._enforce_feature_coverage(df, allowed_features)

    # Align the downstream arrays with the exact rows used for model training so
    # auxiliary exports (e.g., close_all) stay in sync with the scaled feature splits.
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=["ret_15m"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("ret_15m target filtering removed all 15m rows; aborting dataset build.")

    close_snapshot = None
    if "close" in df.columns:
        close_snapshot = df["close"].to_numpy(dtype=np.float64, copy=False)

    X, y, ts_series = make_features_and_target(
        df,
        target_column="ret_15m",
        allowed_features=allowed_features,
        dropna=False,
        return_ts=True,
    )

    splits = time_series_train_val_test_split(X, y)

    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    ts_values = ts_series.to_numpy(dtype="datetime64[ns]")
    n_train = splits.X_train.shape[0]
    n_val = splits.X_val.shape[0]
    ts_train = ts_values[:n_train]
    ts_val = ts_values[n_train:n_train + n_val]
    ts_test = ts_values[n_train + n_val :]

    volatility_arrays = split_volatility_arrays(
        df,
        volatility_columns,
        n_train=n_train,
        n_val=n_val,
    )

    save_payload = {
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
        **volatility_arrays,
    }
    if close_snapshot is not None and close_snapshot.size == ts_values.size:
        save_payload["close_all"] = close_snapshot

    np.savez_compressed(output_path, **save_payload)
    print(f"Saved dataset splits to {output_path}")

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
        "target": "ret_15m",
        "ts_range": _describe(ts_values),
        "splits": {
            "train": _describe(ts_train),
            "val": _describe(ts_val),
            "test": _describe(ts_test),
        },
        "volatility": {
            "columns": volatility_columns,
            "realized_windows_hours": list(DEFAULT_REALIZED_WINDOWS),
            "realized_windows_periods": list(VOLATILITY_WINDOW_PERIODS),
        },
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote dataset meta summary to {META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BTCUSDT 15m training dataset from BigQuery curated features.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    args = parser.parse_args()
    main(args.output_dir)
