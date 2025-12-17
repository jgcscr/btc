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
from src.data.onchain_loader import fetch_onchain_metrics, load_onchain_cached, OnchainAPIError
from src.data.targets_multi_horizon import add_multi_horizon_targets


DEFAULT_HORIZONS: List[int] = [1, 4, 8, 12]
PROCESSED_PATHS = [
    Path("data/processed/macro/hourly_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/coinapi/market_hourly_features.parquet"),
    Path("data/processed/coinapi/funding_hourly_features.parquet"),
    Path("data/processed/cryptoquant/hourly_features.parquet"),
]

CORE_MODEL_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
    "macro_DXY_open",
    "macro_DXY_high",
    "macro_DXY_low",
    "macro_DXY_close",
    "macro_DXY_volume",
    "macro_DXY_close_realized_vol_1h",
    "macro_DXY_close_realized_vol_24h",
    "macro_US10Y_yield",
    "macro_VIX_open",
    "macro_VIX_high",
    "macro_VIX_low",
    "macro_VIX_close",
    "macro_VIX_volume",
    "macro_VIX_close_realized_vol_1h",
    "macro_VIX_close_realized_vol_24h",
]

META_PATH = Path("artifacts/datasets/btc_features_multi_horizon_meta.json")


def _add_cryptoquant_flags(df: pd.DataFrame) -> pd.DataFrame:
    cq_columns = [col for col in df.columns if col.startswith("cq_daily_")]
    if not cq_columns:
        df["cq_daily_fallback_active"] = 0
        df["cq_daily_fallback_complete"] = 0
        return df

    coverage = df[cq_columns].notna()
    df["cq_daily_fallback_active"] = coverage.any(axis=1).astype(int)
    df["cq_daily_fallback_complete"] = coverage.all(axis=1).astype(int)
    return df


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
    return _add_cryptoquant_flags(augmented)


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
    onchain_path: Optional[str] = None,
    fetch_onchain: bool = False,
    onchain_interval: str = "1h",
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
        df = load_btc_features_1h(
            project_id=PROJECT_ID,
            dataset_id=BQ_DATASET_CURATED,
            table_id=BQ_TABLE_FEATURES_1H,
        )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check curated features table content.")

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

    df_onchain = None
    if fetch_onchain:
        start_ts = df["ts"].iloc[0]
        end_ts = df["ts"].iloc[-1]
        try:
            df_onchain = fetch_onchain_metrics(start_ts=start_ts, end_ts=end_ts, interval=onchain_interval)
        except OnchainAPIError as exc:
            if onchain_path:
                print(f"Warning: API fetch failed ({exc}); falling back to cached CSV {onchain_path}.")
            else:
                raise

    if df_onchain is None and onchain_path:
        df_onchain = load_onchain_cached(onchain_path)

    if df_onchain is not None:
        df_onchain = df_onchain.set_index("ts").reindex(df["ts"]).ffill().bfill().reset_index()
        metric_cols = [col for col in df_onchain.columns if col != "ts"]
        print(f"Merging on-chain metrics: {metric_cols}")
        df = df.merge(df_onchain, on="ts", how="left")
    df_targets = add_multi_horizon_targets(df, horizons=horizons, price_col="close")

    ret_cols = [f"ret_{h}h" for h in horizons]
    df_targets = df_targets.dropna(subset=ret_cols)

    X, y_ret1h = make_features_and_target(
        df_targets,
        target_column="ret_1h",
        dropna=False,
        allowed_features=CORE_MODEL_FEATURES,
    )
    splits = time_series_train_val_test_split(X, y_ret1h, train_frac=train_frac, val_frac=val_frac)

    n_train = splits.X_train.shape[0]
    n_val = splits.X_val.shape[0]
    n_total = len(df_targets)
    if n_train + n_val + splits.X_test.shape[0] != n_total:
        raise RuntimeError("Split sizes do not sum to dataset length; check split configuration.")

    data_ret = {h: df_targets[f"ret_{h}h"].to_numpy(dtype=np.float32) for h in horizons if h != 1}
    data_dir = {h: df_targets[f"dir_{h}h"].to_numpy(dtype=np.int8) for h in horizons}
    ts_values = df_targets["ts"].to_numpy(dtype="datetime64[ns]")

    ts_train = ts_values[:n_train]
    ts_val = ts_values[n_train:n_train + n_val]
    ts_test = ts_values[n_train + n_val :]

    if output_path is None:
        output_path = os.path.join(output_dir, "btc_features_multi_horizon_splits.npz")

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
        "feature_names": np.array(splits.feature_names),
        "horizons": np.array(sorted({int(h) for h in horizons}), dtype=np.int32),
        "direction_threshold": np.array([0.0], dtype=np.float32),
    }

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
        "--onchain-path",
        type=str,
        default=None,
        help="Optional CSV containing cached on-chain metrics with ts column.",
    )
    parser.add_argument(
        "--fetch-onchain",
        action="store_true",
        help="Fetch on-chain metrics from the configured API instead of relying solely on cache.",
    )
    parser.add_argument(
        "--onchain-interval",
        type=str,
        default="1h",
        help="Interval for on-chain metrics when fetched via API (default: 1h).",
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
        onchain_path=args.onchain_path,
        fetch_onchain=args.fetch_onchain,
        onchain_interval=args.onchain_interval,
        features_path=args.features_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
