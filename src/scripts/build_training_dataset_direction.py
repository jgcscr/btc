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
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    split_volatility_arrays,
)


PROCESSED_PATHS = [
    Path("data/processed/technical/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
]

SPOT_KLINES_DIR = Path("data/spot_klines")
_BINANCE_SPOT_FEATURES: Optional[pd.DataFrame] = None

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
) -> list[str]:
    if not reliability_json:
        return allowed_features
    payload_path = Path(reliability_json)
    if not payload_path.exists():
        print(f"Feature reliability file not found at {payload_path}; skipping reliability filter.")
        return allowed_features
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    accepted = payload.get("accepted_features")
    if not isinstance(accepted, list):
        return allowed_features
    accepted_set = {str(v) for v in accepted}
    feature_scores = payload.get("feature_scores", {}) if isinstance(payload, dict) else {}
    filtered: list[str] = []
    for feature in allowed_features:
        if feature in accepted_set:
            filtered.append(feature)
            continue
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
    result = df.copy()

    def _safe_diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _safe_pct(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for base in ("close", "volume", "fut_close", "fut_volume"):
        if base not in result.columns:
            continue
        result[f"{base}_delta_1h"] = _safe_diff(result[base])
        result[f"{base}_pct_change_1h"] = _safe_pct(result[base])

    if "close" in result.columns:
        std_7 = result["close"].rolling(window=7, min_periods=3).std(ddof=0)
        std_24 = result["close"].rolling(window=24, min_periods=6).std(ddof=0)
        if "ma_close_7h" in result.columns:
            denom = std_7.replace(0.0, np.nan)
            result["close_zscore_7h"] = ((result["close"] - result["ma_close_7h"]) / denom).fillna(0.0)
        if "ma_close_24h" in result.columns:
            denom = std_24.replace(0.0, np.nan)
            result["close_zscore_24h"] = ((result["close"] - result["ma_close_24h"]) / denom).fillna(0.0)

    if "fut_close" in result.columns:
        rolling_mean = result["fut_close"].rolling(window=7, min_periods=3).mean()
        rolling_std = result["fut_close"].rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
        result["fut_close_zscore_7h"] = ((result["fut_close"] - rolling_mean) / rolling_std).fillna(0.0)

    required_cvd = {"volume", "taker_buy_base_volume"}
    if not required_cvd.issubset(result.columns):
        missing = ", ".join(sorted(required_cvd - set(result.columns)))
        raise ValueError(
            f"Cannot compute CVD features; missing columns: {missing}.",
        )
    total_volume = result["volume"].astype(float)
    taker_buy = result["taker_buy_base_volume"].astype(float)
    taker_sell = (total_volume - taker_buy).clip(lower=0.0)
    cvd_raw = taker_buy - taker_sell
    cvd_window = cvd_raw.rolling(window=6, min_periods=2).sum()
    vol_window = total_volume.rolling(window=6, min_periods=2).sum().replace(0.0, np.nan)
    ratio = (cvd_window / vol_window).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    result["cvd_ratio_6h"] = ratio
    cvd_mean = cvd_window.rolling(window=24, min_periods=6).mean()
    cvd_std = cvd_window.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
    zscore = ((cvd_window - cvd_mean) / cvd_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["cvd_zscore_6h"] = zscore.clip(-10.0, 10.0)

    required_liquidity = {"high", "low", "close"}
    if not required_liquidity.issubset(result.columns):
        missing = ", ".join(sorted(required_liquidity - set(result.columns)))
        raise ValueError(f"Cannot compute liquidity features; missing OHLC columns: {missing}.")
    true_range = _true_range(result["high"].astype(float), result["low"].astype(float), result["close"].astype(float))
    atr_6h = true_range.rolling(window=6, min_periods=2).mean().replace(0.0, np.nan)
    range_span = (result["high"].astype(float) - result["low"].astype(float)).abs()
    liquidity_ratio = (range_span / atr_6h).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["liquidity_range_ratio_6h"] = liquidity_ratio.clip(0.0, 10.0)

    mid_price = (result["high"].astype(float) + result["low"].astype(float)) / 2.0
    half_range = (result["high"].astype(float) - result["low"].astype(float)).replace(0.0, np.nan) / 2.0
    close_position = ((result["close"].astype(float) - mid_price) / half_range)
    close_position = close_position.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    result["liquidity_close_position_ratio"] = close_position

    return result


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
    """Load features from local files instead of BigQuery."""
    # Load spot klines
    spot_files = list(SPOT_KLINES_DIR.glob("*.parquet"))
    if not spot_files:
        raise FileNotFoundError(f"No spot klines found in {SPOT_KLINES_DIR}")
    
    spot_df = pd.concat([pd.read_parquet(f) for f in spot_files], ignore_index=True)
    # ts is already datetime
    spot_df = spot_df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]]
    
    # Merge with processed features
    df = _merge_processed_features(spot_df, PROCESSED_PATHS)
    return df


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
) -> str:
    os.makedirs(output_dir, exist_ok=True)

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
    )

    if allowed_features:
        df = _enforce_feature_coverage(df, allowed_features)

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
    else:
        if no_trade_abs_ret > 0.0 or no_trade_vol_mult > 0.0:
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

    splits = time_series_train_val_test_split(X, y_dir)

    output_path = os.path.join(output_dir, "btc_features_1h_direction_splits.npz")
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

    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        ts_train=ts_train,
        ts_val=ts_val,
        ts_test=ts_test,
        ts_all=ts_values,
        feature_names=np.array(splits.feature_names),
        threshold=np.array([threshold], dtype=float),
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
        "ts_range": _describe(ts_values),
        "splits": {
            "train": _describe(ts_train),
            "val": _describe(ts_val),
            "test": _describe(ts_test),
        },
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote direction dataset meta summary to {META_PATH}")
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
        choices=("binary", "triple_barrier"),
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
    )


if __name__ == "__main__":
    main()
