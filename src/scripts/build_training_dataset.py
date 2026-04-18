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
from src.data.dataset_preparation import (
    enforce_unique_hourly_index,
    make_features_and_target,
    repair_hourly_continuity,
    time_series_train_val_test_split,
)
from src.data.targets_multi_horizon import add_trend_ignition_label
from src.trading.volatility import (
    DEFAULT_REALIZED_WINDOWS,
    add_volatility_columns,
    split_volatility_arrays,
)
from src.trading.feature_engineering import (
    apply_funding_rate_features as _shared_apply_funding_rate_features,
    augment_hourly_price_features as _shared_augment_hourly_price_features,
)
from src.data.macro_loader import MACRO_FEATURE_COLUMNS
from src.data.onchain_loader import ONCHAIN_FEATURE_COLUMNS
from src.data.source_parity import (
    drop_unready_source_family_features,
    evaluate_source_family_readiness,
)
from src.trading.intrabar_features import compute_hourly_intrabar_features


def _load_local_features() -> pd.DataFrame:
    """Load features from local files instead of BigQuery.

    Uses the union of `data/spot_klines` and raw Binance tidy parquet history
    when available, which materially increases 1h training window coverage.
    """

    def _load_hourly_from_spot_klines() -> pd.DataFrame:
        spot_files = sorted(SPOT_KLINES_DIR.glob("*.parquet"))
        if not spot_files:
            return pd.DataFrame()
        frames = []
        required = {
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "num_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        }
        for path in spot_files:
            frame = pd.read_parquet(path)
            if not required.issubset(set(frame.columns)):
                continue
            frames.append(frame.loc[:, sorted(required)].copy())
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.floor("h")
        out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset="ts", keep="last")
        return out.reset_index(drop=True)

    spot_df = _load_hourly_from_spot_klines()
    raw_df = _load_spot_ohlcv_from_raw("1h")
    if spot_df.empty and raw_df.empty:
        raise FileNotFoundError(
            f"No local Binance hourly source found in {SPOT_KLINES_DIR} or {RAW_SPOT_METRICS_DIR}",
        )

    if not spot_df.empty and not raw_df.empty:
        merged = pd.concat([spot_df, raw_df], ignore_index=True)
        merged = merged.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
        source_df = merged
    else:
        source_df = raw_df if not raw_df.empty else spot_df

    if not source_df.empty:
        print(
            "Local 1h source coverage: "
            f"rows={len(source_df)}, ts_min={source_df['ts'].min()}, ts_max={source_df['ts'].max()}",
        )

    return source_df


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    suffix = interval[-1]
    amount = int(interval[:-1])
    if suffix == "m":
        return pd.Timedelta(minutes=amount)
    if suffix == "h":
        return pd.Timedelta(hours=amount)
    if suffix == "d":
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported interval: {interval}")


def _infer_raw_tidy_interval(frame: pd.DataFrame) -> Optional[pd.Timedelta]:
    ts = pd.to_datetime(frame.get("ts"), utc=True, errors="coerce")
    ts = pd.Series(ts).dropna().drop_duplicates().sort_values()
    if ts.size < 2:
        return None
    deltas = ts.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return deltas.mode().iloc[0]


def _pivot_raw_spot_tidy_to_ohlcv(tidy: pd.DataFrame) -> pd.DataFrame:
    tidy = tidy.sort_values("ts").drop_duplicates(subset=["ts", "metric"], keep="last")
    wide = tidy.pivot(index="ts", columns="metric", values="value").reset_index()
    rename_map = {
        "spot_open": "open",
        "spot_high": "high",
        "spot_low": "low",
        "spot_close": "close",
        "spot_volume": "volume",
        "spot_quote_volume": "quote_volume",
        "spot_num_trades": "num_trades",
        "spot_taker_buy_base_volume": "taker_buy_base_volume",
        "spot_taker_buy_quote_volume": "taker_buy_quote_volume",
    }
    wide = wide.rename(columns=rename_map)
    required_cols = {
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    if not required_cols.issubset(set(wide.columns)):
        return pd.DataFrame()
    wide = wide.loc[:, sorted(required_cols)].copy()
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True, errors="coerce")
    wide = wide.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset="ts", keep="last")
    return wide.reset_index(drop=True)


def _load_spot_ohlcv_from_raw(interval: str, raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_SPOT_METRICS_DIR
    if not raw_dir.exists():
        return pd.DataFrame()

    target_delta = _interval_to_timedelta(interval)
    paths = sorted(p for p in raw_dir.glob("*.parquet") if p.is_file())
    if not paths:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    needed = {"ts", "metric", "value"}
    matched_paths = 0
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=["ts", "metric", "value"])
        except Exception as exc:
            print(f"Warning: failed to load raw spot parquet {path}: {exc}; skipping.")
            continue
        if not needed.issubset(set(frame.columns)):
            continue
        inferred = _infer_raw_tidy_interval(frame)
        if inferred is None or inferred != target_delta:
            continue
        slim = frame.loc[:, ["ts", "metric", "value"]].copy()
        slim["ts"] = pd.to_datetime(slim["ts"], utc=True, errors="coerce")
        slim = slim.dropna(subset=["ts", "metric"])
        if slim.empty:
            continue
        rows.append(slim)
        matched_paths += 1

    if not rows:
        return pd.DataFrame()

    tidy = pd.concat(rows, ignore_index=True)
    wide = _pivot_raw_spot_tidy_to_ohlcv(tidy)
    if wide.empty:
        return wide

    print(
        f"Loaded raw Binance spot {interval} history from {matched_paths} parquet files; "
        f"coverage spans {wide['ts'].min()} -> {wide['ts'].max()}."
    )
    return wide


PROCESSED_PATHS = [
    Path("data/processed/technical/hourly_features.parquet"),
    Path("data/processed/funding/hourly_features.parquet"),
    Path("data/processed/macro/daily_features.parquet"),
    Path("data/processed/onchain/hourly_features.parquet"),
]

SPOT_KLINES_DIR = Path("data/spot_klines")
RAW_SPOT_METRICS_DIR = Path("data/raw/market/binanceus/entity=spot/symbol=BINANCEUS_SPOT_BTC_USDT")
_BINANCE_SPOT_FEATURES: Optional[pd.DataFrame] = None
_BINANCE_INTRAHOUR_FEATURES: Optional[pd.DataFrame] = None

META_PATH = Path("artifacts/datasets/btc_features_1h_meta.json")

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
    "intrabar_directional_persistence_1h",
    "intrabar_vol_term_structure_6h_24h",
    "intrabar_volume_regime_zscore_24h",
    "intrabar_flow_acceleration_3h",
    *MACRO_FEATURE_COLUMNS,
    *ONCHAIN_FEATURE_COLUMNS,
]

ZERO_VARIANCE_CANDIDATES: set[str] = set()

EXCLUDED_FEATURES = {
    "funding_rate_zscore_24h",
    "ret_1h",
    "ret_4h",
    "ret_8h",
    "ret_12h",
}

HOURLY_REGRESSION_RELIABILITY_DEFAULT_MIN_SCORE = 0.60

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

NON_BINANCE_BREAKOUT_PREFIXES = (
    "cq_",
    "orderbook_",
    "depth_",
    "lob_",
    "slippage_",
)

FUNDING_SOURCE_CANDIDATES = (
    "funding_rate",
    "funding_BTCUSDT_funding_rate",
    "fut_funding_rate",
)

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

FUNDING_FALLBACKS = {
    "funding_rate": (
        "funding_BTCUSDT_funding_rate",
        "fut_funding_rate",
    ),
    "funding_rate_annualized": (
        "funding_BTCUSDT_funding_rate_annualized",
        "fut_funding_rate_annualized",
    ),
    "open_interest": (
        "fut_open_interest",
        "open_interest_deriv",
    ),
}


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


def _filter_allowed_features_by_reliability(
    allowed_features: list[str],
    *,
    reliability_json: str | None,
    min_score: float,
    target_horizon: float | None,
) -> list[str]:
    if not reliability_json:
        return allowed_features

    from src.scripts.build_training_dataset_direction import _filter_features_by_reliability

    return _filter_features_by_reliability(
        allowed_features,
        reliability_json=reliability_json,
        min_score=min_score,
        target_horizon=target_horizon,
    )


def _drop_uncovered_allowed_features(df: pd.DataFrame, allowed_features: list[str]) -> list[str]:
    dropped = [feature for feature in allowed_features if feature in df.columns and df[feature].notna().sum() == 0]
    if dropped:
        preview = ", ".join(dropped[:5])
        suffix = "..." if len(dropped) > 5 else ""
        print(f"Dropped {len(dropped)} all-null allowed features: {preview}{suffix}")
    return [feature for feature in allowed_features if feature not in set(dropped)]


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
    removed: list[str] = []
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


def _reconcile_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical funding/open interest columns exist with sensible fallbacks."""

    result = df.copy()
    for target, sources in FUNDING_FALLBACKS.items():
        if target not in result.columns:
            for source in sources:
                if source in result.columns:
                    result[target] = result[source]
                    break
        if target in result.columns:
            series = result[target]
            for source in sources:
                if source in result.columns:
                    series = series.fillna(result[source])
            result[target] = series
        else:
            result[target] = 0.0
    return result


def _drop_non_binance_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strip non-Binance breakout feeds before downstream feature engineering."""

    drop_candidates = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in NON_BINANCE_BREAKOUT_PREFIXES)
    ]
    if drop_candidates:
        preview = ", ".join(sorted(drop_candidates)[:5])
        suffix = "..." if len(drop_candidates) > 5 else ""
        print(
            f"Removed {len(drop_candidates)} non-Binance breakout columns: {preview}{suffix}",
        )
        df = df.drop(columns=sorted(set(drop_candidates)))
    return df


def _apply_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    return _shared_apply_funding_rate_features(df, strict_missing=False)


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
    return _shared_augment_hourly_price_features(df, strict_missing=True)


def _recompute_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "close" not in result.columns:
        return result

    log_close = np.log(result["close"].astype(float))
    result["ret_1h"] = log_close.diff()
    result["ret_fwd_3h"] = log_close.shift(-3) - log_close
    result["ret_4h"] = log_close.shift(-4) - log_close
    result["ret_8h"] = log_close.shift(-8) - log_close
    result["ret_12h"] = log_close.shift(-12) - log_close
    return result


def _merge_processed_features(df: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise RuntimeError("Expected 'ts' column in curated features for feature alignment.")

    augmented = df.copy()
    augmented["ts"] = pd.to_datetime(augmented["ts"], utc=True)
    augmented["ts"] = augmented["ts"].dt.floor("h")
    augmented = augmented.sort_values("ts").reset_index(drop=True)
    binance_spot = _load_binance_spot_features()
    if binance_spot is not None:
        columns_before = set(augmented.columns)
        merge_columns = ["ts", *[col for col in binance_spot.columns if col != "ts" and col not in columns_before]]
        if len(merge_columns) > 1:
            augmented = augmented.merge(binance_spot.loc[:, merge_columns], on="ts", how="left")
        new_columns = [col for col in augmented.columns if col not in columns_before]
        if new_columns:
            preview = ", ".join(new_columns[:5])
            suffix = "..." if len(new_columns) > 5 else ""
            print(
                f"Added {len(new_columns)} Binance spot columns from {SPOT_KLINES_DIR}: {preview}{suffix}",
            )

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


def _sign_persistence(series: pd.Series) -> float:
    signs = np.sign(pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float))
    if signs.size <= 1:
        return float("nan")
    return float(np.mean(signs[1:] == signs[:-1]))


def _load_binance_intrahour_features(directory: Path = RAW_SPOT_METRICS_DIR) -> Optional[pd.DataFrame]:
    global _BINANCE_INTRAHOUR_FEATURES
    if _BINANCE_INTRAHOUR_FEATURES is not None:
        return _BINANCE_INTRAHOUR_FEATURES

    if not directory.exists():
        return None

    parquet_paths = sorted(p for p in directory.glob("*.parquet") if p.is_file())
    if not parquet_paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in parquet_paths:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - defensive against partial files
            print(f"Warning: failed to load intrahour parquet {path}: {exc}; skipping.")
            continue
        needed = {"ts", "metric", "value"}
        if not needed.issubset(frame.columns):
            continue
        slim = frame.loc[:, ["ts", "metric", "value"]].copy()
        slim["ts"] = pd.to_datetime(slim["ts"], utc=True, errors="coerce")
        slim = slim.dropna(subset=["ts", "metric"]).reset_index(drop=True)
        if slim.empty:
            continue
        frames.append(slim)

    if not frames:
        return None

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values("ts").drop_duplicates(subset=["ts", "metric"], keep="last")
    wide = tidy.pivot(index="ts", columns="metric", values="value").reset_index()
    required = {
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_close",
        "spot_volume",
        "spot_quote_volume",
        "spot_num_trades",
        "spot_taker_buy_base_volume",
    }
    if not required.issubset(set(wide.columns)):
        return None

    wide = wide.sort_values("ts").reset_index(drop=True)
    deltas = wide["ts"].diff().dropna()
    if deltas.empty:
        return None
    median_minutes = float(deltas.dt.total_seconds().median() / 60.0)
    if median_minutes > 20.0:
        # 1h-only source does not contain true intrahour paths.
        print("Intrahour 15m features unavailable: raw spot cadence is hourly.")
        return None

    renamed = wide.rename(
        columns={
            "spot_open": "open",
            "spot_high": "high",
            "spot_low": "low",
            "spot_close": "close",
            "spot_volume": "volume",
            "spot_quote_volume": "quote_volume",
            "spot_num_trades": "num_trades",
            "spot_taker_buy_base_volume": "taker_buy_base_volume",
            "spot_taker_buy_quote_volume": "taker_buy_quote_volume",
        }
    )
    output = compute_hourly_intrabar_features(renamed)
    if output.empty:
        return None
    intrabar_cols = [col for col in output.columns if col != "ts"]
    _BINANCE_INTRAHOUR_FEATURES = output.copy()
    print(
        f"Built {len(intrabar_cols)} intrahour Binance features from {len(parquet_paths)} raw parquet files.",
    )
    return _BINANCE_INTRAHOUR_FEATURES


def merge_intrahour_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    intrahour = _load_binance_intrahour_features()
    if intrahour is None or intrahour.empty:
        return df
    merged = df.merge(intrahour, on="ts", how="left")
    intrabar_cols = [col for col in intrahour.columns if col != "ts"]
    for col in intrabar_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def _load_binance_spot_features(directory: Path = SPOT_KLINES_DIR) -> Optional[pd.DataFrame]:
    global _BINANCE_SPOT_FEATURES
    if _BINANCE_SPOT_FEATURES is not None:
        return _BINANCE_SPOT_FEATURES

    if not directory.exists():
        return None

    parquet_paths = sorted(p for p in directory.glob("*1h*.parquet") if p.is_file())
    if not parquet_paths:
        return None

    frames: list[pd.DataFrame] = []
    required_cols = {"ts", "taker_buy_base_volume", "taker_buy_quote_volume"}
    for path in parquet_paths:
        try:
            extra = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - defensive against partial files
            print(f"Warning: failed to load {path}: {exc}; skipping.")
            continue
        if not required_cols.issubset(extra.columns):
            continue
        frame = extra.loc[:, list(required_cols)].copy()
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.floor("h")
        frames.append(frame.dropna(subset=["ts"]))

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    _BINANCE_SPOT_FEATURES = combined
    print(
        f"Loaded Binance spot klines from {len(parquet_paths)} parquet files; "
        f"taker coverage spans {combined['ts'].min()} -> {combined['ts'].max()}."
    )
    return combined


def main(
    output_dir: str,
    *,
    feature_reliability_json: str | None = None,
    feature_reliability_min_score: float = HOURLY_REGRESSION_RELIABILITY_DEFAULT_MIN_SCORE,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load from local instead of BigQuery
    df = _load_local_features()

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from local files; check data.")

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

    df, backfilled = repair_hourly_continuity(
        df,
        label="curated_features",
        expected_freq=pd.Timedelta(hours=1),
    )
    if backfilled:
        print(f"[curated_features] Reindexed with {backfilled} synthetic hourly rows via forward/back fill.")

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
    df = _recompute_return_targets(df)
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

    df = add_trend_ignition_label(
        df,
        horizon_hours=TREND_IGNITION_HORIZON,
        threshold=TREND_IGNITION_THRESHOLD,
        price_col="close",
        label_col=TREND_IGNITION_LABEL,
    )
    df = df.dropna(subset=[TREND_IGNITION_LABEL]).reset_index(drop=True)

    allowed_features = [feature for feature in CORE_MODEL_FEATURES if feature in df.columns]
    for column in volatility_columns:
        if column not in allowed_features:
            allowed_features.append(column)
    allowed_features = _append_futures_feature_columns(df, allowed_features)
    allowed_features = _append_technical_feature_columns(df, allowed_features)
    allowed_features = [feature for feature in allowed_features if feature != TREND_IGNITION_LABEL]
    allowed_features = _filter_allowed_features_by_reliability(
        allowed_features,
        reliability_json=feature_reliability_json,
        min_score=feature_reliability_min_score,
        target_horizon=1.0,
    )
    allowed_features = _drop_uncovered_allowed_features(df, allowed_features)
    source_family_parity = evaluate_source_family_readiness(df)
    allowed_features, dropped_source_family_features = drop_unready_source_family_features(
        allowed_features,
        source_family_parity,
    )
    df = _enforce_feature_coverage(df, allowed_features)
    df = df.sort_values("ts").reset_index(drop=True)
    label_series = df[TREND_IGNITION_LABEL].astype(int)
    X, y, ts_series = make_features_and_target(
        df,
        target_column="ret_1h",
        allowed_features=allowed_features,
        return_ts=True,
        dropna=False,
    )

    mask = (~y.isna()) & label_series.notna()
    if not mask.any():
        raise RuntimeError("No rows remain after aligning trend ignition labels with targets.")

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    ts_series = ts_series.loc[mask].reset_index(drop=True)
    label_series = label_series.loc[mask].reset_index(drop=True).astype(int)

    splits = time_series_train_val_test_split(X, y)

    output_path = os.path.join(output_dir, "btc_features_1h_splits.npz")
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

    ignition_values = label_series.to_numpy(dtype=np.int8, copy=False)
    y_ignition_train = ignition_values[:n_train]
    y_ignition_val = ignition_values[n_train:n_train + n_val]
    y_ignition_test = ignition_values[n_train + n_val :]

    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        y_ignition_train=y_ignition_train,
        y_ignition_val=y_ignition_val,
        y_ignition_test=y_ignition_test,
        ts_train=ts_train,
        ts_val=ts_val,
        ts_test=ts_test,
        ts_all=ts_values,
        feature_names=np.array(splits.feature_names),
        scaler_mean=np.asarray(splits.scaler_mean, dtype=np.float32),
        scaler_scale=np.asarray(splits.scaler_scale, dtype=np.float32),
        trend_ignition_label=np.array([TREND_IGNITION_LABEL]),
        trend_ignition_threshold=np.array([TREND_IGNITION_THRESHOLD], dtype=np.float32),
        trend_ignition_horizon=np.array([TREND_IGNITION_HORIZON], dtype=np.float32),
        **volatility_arrays,
    )
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
        "ts_range": _describe(ts_values),
        "splits": {
            "train": _describe(ts_train),
            "val": _describe(ts_val),
            "test": _describe(ts_test),
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
        "trend_ignition": {
            "label": TREND_IGNITION_LABEL,
            "threshold": TREND_IGNITION_THRESHOLD,
            "horizon_hours": TREND_IGNITION_HORIZON,
            "positive_rate": float(label_series.mean()),
        },
        "source_family_parity": source_family_parity,
        "dropped_source_family_features": dropped_source_family_features,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta_payload, indent=2))
    print(f"Wrote dataset meta summary to {META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BTCUSDT 1h training dataset from BigQuery curated features.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    parser.add_argument(
        "--feature-reliability-json",
        type=str,
        default=None,
        help="Optional reliability JSON used to prune the 1h regression feature set.",
    )
    parser.add_argument(
        "--feature-reliability-min-score",
        type=float,
        default=HOURLY_REGRESSION_RELIABILITY_DEFAULT_MIN_SCORE,
        help="Minimum reliability score when filtering 1h regression features.",
    )
    args = parser.parse_args()
    main(
        args.output_dir,
        feature_reliability_json=args.feature_reliability_json,
        feature_reliability_min_score=args.feature_reliability_min_score,
    )
