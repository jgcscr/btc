from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.macro_loader import MACRO_FEATURE_COLUMNS
from src.data.onchain_loader import ONCHAIN_FEATURE_COLUMNS
from src.trading.data_quality import DataQualityError
from src.trading.feature_engineering import (
    apply_funding_rate_features as _shared_apply_funding_rate_features,
    augment_hourly_price_features as _shared_augment_hourly_price_features,
)
from src.trading.intrabar_features import compute_hourly_intrabar_features
from src.trading.signals import PreparedData, format_ts_iso, prepare_data_for_signals_from_ohlcv
from src.trading.volatility import DEFAULT_REALIZED_WINDOWS, add_volatility_columns


def pivot_tidy_spot_ohlcv(path: Path) -> pd.DataFrame:
    tidy = pd.read_parquet(path)
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
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True, errors="coerce")
    wide = wide.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return wide


def compute_intrabar_features_from_15m(path_15m_tidy: Path) -> pd.DataFrame:
    frame = pivot_tidy_spot_ohlcv(path_15m_tidy)
    intrabar = compute_hourly_intrabar_features(frame)
    if intrabar.empty:
        raise RuntimeError("15m frame did not produce any intrabar features; check source cadence and required columns.")
    return intrabar


def build_ohlcv_frame_from_tidy(df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "spot_open": "open",
        "spot_high": "high",
        "spot_low": "low",
        "spot_close": "close",
        "spot_volume": "volume",
    }
    subset = df[df["metric"].isin(metric_map.keys())].copy()
    if subset.empty:
        raise DataQualityError("No OHLCV metrics found in ingestion output")
    subset["metric"] = subset["metric"].map(metric_map)
    return subset.pivot(index="ts", columns="metric", values="value").reset_index()


def read_timeseries_frame(path: str, label: str) -> pd.DataFrame:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} override not found at {resolved}")

    ext = resolved.suffix.lower()
    if ext in {".csv", ".tsv"}:
        df = pd.read_csv(resolved)
    else:
        try:
            df = pd.read_parquet(resolved)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {label} override at {resolved}: {exc}") from exc

    if "ts" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError(f"{label} override at {resolved} must include a 'ts' column")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].dt.floor("h")
    df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last").reset_index(drop=True)
    return df


def summarize_frame(df: pd.DataFrame, label: str, path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": str(Path(path).expanduser()),
        "row_count": int(len(df)),
        "columns": int(len(df.columns)),
        "label": label,
    }
    if not df.empty and "ts" in df.columns:
        summary["ts_start"] = str(df["ts"].min().isoformat())
        summary["ts_end"] = str(df["ts"].max().isoformat())
    return summary


def merge_override_features(base: pd.DataFrame, extra: pd.DataFrame, label: str) -> tuple[pd.DataFrame, List[str]]:
    if extra.empty:
        print(f"Override '{label}' is empty; skipping merge.")
        return base, []

    columns_before = set(base.columns)
    merged = pd.merge_asof(
        base.sort_values("ts"),
        extra.sort_values("ts"),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("ts").reset_index(drop=True)
    new_columns = [col for col in merged.columns if col not in columns_before]
    if new_columns:
        preview = ", ".join(new_columns[:5])
        suffix = "..." if len(new_columns) > 5 else ""
        print(f"Merged {len(new_columns)} '{label}' columns: {preview}{suffix}")
    else:
        print(f"Override '{label}' did not contribute new columns; check schema overlap.")
    return merged, new_columns


def load_training_feature_names(
    dataset_multi_path: Path,
    dataset_1h_path: Path,
    *,
    stderr_write: Any = None,
) -> List[str] | None:
    if stderr_write is None:
        stderr_write = sys.stderr.write
    dataset_path = dataset_multi_path if dataset_multi_path.exists() else dataset_1h_path
    if not dataset_path.exists():
        stderr_write("Warning: dataset NPZ missing; falling back to local feature column order.\n")
        return None

    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        if "feature_names" not in dataset_npz.files:
            stderr_write(f"Warning: {dataset_path} missing feature_names; using local column order.\n")
            return None
        data = dataset_npz["feature_names"].tolist()

    return [str(name) for name in data]


def enrich_local_features_for_model(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    required = set(required_columns)
    if not required:
        return frame, []

    enriched = frame.copy()
    added: List[str] = []

    def _record_added(column: str) -> None:
        if column in required and column not in added:
            added.append(column)

    def _add_numeric_column(column: str, values: pd.Series) -> None:
        if column not in required or column in enriched.columns:
            return
        series = pd.to_numeric(values, errors="coerce")
        enriched[column] = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        _record_added(column)

    close = pd.to_numeric(enriched["close"], errors="coerce") if "close" in enriched.columns else None
    volume = pd.to_numeric(enriched["volume"], errors="coerce") if "volume" in enriched.columns else None

    if close is not None:
        ma_7 = close.rolling(window=7, min_periods=3).mean()
        ma_24 = close.rolling(window=24, min_periods=6).mean()
        _add_numeric_column("ma_close_7h", ma_7)
        _add_numeric_column("ma_close_24h", ma_24)
        if "ma_ratio_7_24" in required and "ma_ratio_7_24" not in enriched.columns:
            denom = ma_24.replace(0.0, np.nan)
            ratio = (ma_7 / denom).replace([np.inf, -np.inf], np.nan)
            _add_numeric_column("ma_ratio_7_24", ratio)

        if "vol_24h" in required and "vol_24h" not in enriched.columns:
            _add_numeric_column("vol_24h", close.rolling(window=24, min_periods=6).std(ddof=0))

        _add_numeric_column("close_delta_1h", close.diff())
        _add_numeric_column("close_pct_change_1h", close.pct_change())

        if "close_zscore_7h" in required and "close_zscore_7h" not in enriched.columns:
            std_7 = close.rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
            if "ma_close_7h" in enriched.columns:
                z_7 = (close - pd.to_numeric(enriched["ma_close_7h"], errors="coerce")) / std_7
                _add_numeric_column("close_zscore_7h", z_7)
        if "close_zscore_24h" in required and "close_zscore_24h" not in enriched.columns:
            std_24 = close.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
            if "ma_close_24h" in enriched.columns:
                z_24 = (close - pd.to_numeric(enriched["ma_close_24h"], errors="coerce")) / std_24
                _add_numeric_column("close_zscore_24h", z_24)

        ret_columns = [
            column
            for column in required
            if column.startswith("ret_")
            and column.endswith("h")
            and column not in enriched.columns
            and column not in {"ret_max_4h", "ret_min_4h", "ret_max_8h", "ret_min_8h", "ret_max_12h", "ret_min_12h"}
        ]
        for column in sorted(ret_columns):
            horizon_raw = column[4:-1]
            try:
                periods = int(round(float(horizon_raw)))
            except ValueError:
                continue
            if periods <= 0:
                continue
            _add_numeric_column(column, close.pct_change(periods=periods))

        required_volatility = {
            "volatility_realized_24h",
            "volatility_realized_72h",
            "volatility_ewm_24h",
            "volatility_ewm_72h",
            "volatility_garch_like",
        }
        if any(column in required and column not in enriched.columns for column in required_volatility):
            enriched, computed_volatility = add_volatility_columns(
                enriched,
                realized_windows=DEFAULT_REALIZED_WINDOWS,
            )
            for column in computed_volatility:
                _record_added(column)

    if volume is not None:
        _add_numeric_column("volume_delta_1h", volume.diff())
        _add_numeric_column("volume_pct_change_1h", volume.pct_change())

    taker_buy_base = None
    taker_buy_quote = None
    if "taker_buy_base_volume" in enriched.columns:
        taker_buy_base = pd.to_numeric(enriched["taker_buy_base_volume"], errors="coerce")
    elif volume is not None:
        taker_buy_base = volume * 0.5
    if taker_buy_base is not None:
        _add_numeric_column("taker_buy_base_volume", taker_buy_base)

    if "taker_buy_quote_volume" in enriched.columns:
        taker_buy_quote = pd.to_numeric(enriched["taker_buy_quote_volume"], errors="coerce")
    elif taker_buy_base is not None and close is not None:
        taker_buy_quote = taker_buy_base * close
    if taker_buy_quote is not None:
        _add_numeric_column("taker_buy_quote_volume", taker_buy_quote)

    if taker_buy_base is not None and volume is not None:
        taker_sell = (volume - taker_buy_base).clip(lower=0.0)
        cvd_raw = taker_buy_base - taker_sell
        cvd_window = cvd_raw.rolling(window=6, min_periods=2).sum()
        vol_window = volume.rolling(window=6, min_periods=2).sum().replace(0.0, np.nan)
        cvd_ratio = (cvd_window / vol_window).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0)
        _add_numeric_column("cvd_ratio_6h", cvd_ratio)
        cvd_mean = cvd_window.rolling(window=24, min_periods=6).mean()
        cvd_std = cvd_window.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
        cvd_zscore = ((cvd_window - cvd_mean) / cvd_std).replace([np.inf, -np.inf], np.nan).clip(lower=-10.0, upper=10.0)
        _add_numeric_column("cvd_zscore_6h", cvd_zscore)

    if "funding_rate_zscore_24h" in required and "funding_rate_zscore_24h" not in enriched.columns:
        funding_rate = pd.to_numeric(enriched["funding_rate"], errors="coerce") if "funding_rate" in enriched.columns else None
        if funding_rate is None:
            funding_zscore = pd.Series(0.0, index=enriched.index)
        else:
            mean_24 = funding_rate.rolling(window=24, min_periods=6).mean()
            std_24 = funding_rate.rolling(window=24, min_periods=6).std(ddof=0).replace(0.0, np.nan)
            funding_zscore = (funding_rate - mean_24) / std_24
        _add_numeric_column("funding_rate_zscore_24h", funding_zscore)

    if "trend_ignition_6h" in required and "trend_ignition_6h" not in enriched.columns and close is not None:
        momentum = close.pct_change().fillna(0.0)
        ignition_score = (momentum > 0.0).astype(float).rolling(window=6, min_periods=1).mean()
        _add_numeric_column("trend_ignition_6h", ignition_score)

    if {"high", "low", "close"}.issubset(enriched.columns):
        high = pd.to_numeric(enriched["high"], errors="coerce")
        low = pd.to_numeric(enriched["low"], errors="coerce")
        close_for_liquidity = pd.to_numeric(enriched["close"], errors="coerce")
        prev_close = close_for_liquidity.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        atr_6h = true_range.rolling(window=6, min_periods=2).mean().replace(0.0, np.nan)
        range_span = (high - low).abs()
        liquidity_ratio = (range_span / atr_6h).replace([np.inf, -np.inf], np.nan)
        _add_numeric_column("liquidity_range_ratio_6h", liquidity_ratio.clip(lower=0.0, upper=10.0))

        mid_price = (high + low) / 2.0
        half_range = (high - low).replace(0.0, np.nan) / 2.0
        close_position = ((close_for_liquidity - mid_price) / half_range).replace([np.inf, -np.inf], np.nan)
        _add_numeric_column("liquidity_close_position_ratio", close_position.clip(lower=-1.0, upper=1.0))

        atr_24h = true_range.rolling(window=24, min_periods=6).mean().replace(0.0, np.nan)
        _add_numeric_column(
            "range_expansion_1h",
            (true_range / atr_24h).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=10.0),
        )

        session_high = high.rolling(window=8, min_periods=2).max().replace(0.0, np.nan)
        session_low = low.rolling(window=8, min_periods=2).min().replace(0.0, np.nan)
        _add_numeric_column(
            "distance_from_session_high_8h",
            ((close_for_liquidity / session_high) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
        )
        _add_numeric_column(
            "distance_from_session_low_8h",
            ((close_for_liquidity / session_low) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
        )

        if volume is not None:
            typical_price = (high + low + close_for_liquidity) / 3.0
            rolling_notional = (typical_price * volume).rolling(window=8, min_periods=2).sum()
            rolling_volume = volume.rolling(window=8, min_periods=2).sum().replace(0.0, np.nan)
            rolling_vwap = (rolling_notional / rolling_volume).replace([np.inf, -np.inf], np.nan)
            _add_numeric_column(
                "vwap_deviation_8h",
                ((close_for_liquidity / rolling_vwap) - 1.0).replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=1.0),
            )

        _add_numeric_column("momentum_slope_2h", close_for_liquidity.pct_change(periods=2))
        _add_numeric_column("momentum_slope_4h", close_for_liquidity.pct_change(periods=4))

    before_shared = set(enriched.columns)
    enriched = _shared_apply_funding_rate_features(enriched, strict_missing=False)
    enriched = _shared_augment_hourly_price_features(enriched, strict_missing=False)
    for column in required:
        if column in enriched.columns and column not in before_shared:
            _record_added(column)

    return enriched, added


def prepare_local_feature_bundle(
    *,
    features_path: str,
    hours: int,
    optional_sources: Mapping[str, str] | None = None,
    dataset_multi_path: Path,
    dataset_1h_path: Path,
    local_feature_required_columns: Mapping[str, Sequence[str]],
    stderr_write: Any = None,
) -> tuple[tuple[PreparedData, int, float, str], Dict[str, Any]]:
    if stderr_write is None:
        stderr_write = sys.stderr.write

    base_df = read_timeseries_frame(features_path, "features")
    metadata: Dict[str, Any] = {
        "features": summarize_frame(base_df, "features", features_path),
    }

    if optional_sources:
        for label, path in optional_sources.items():
            try:
                frame = read_timeseries_frame(path, label)
            except Exception as exc:
                stderr_write(f"Warning: failed to load local override '{label}' at {path}: {exc}\n")
                continue
            base_df, added_columns = merge_override_features(base_df, frame, label)
            summary = summarize_frame(frame, label, path)
            summary["added_columns"] = added_columns
            required = local_feature_required_columns.get(label, tuple())
            if required:
                missing = [col for col in required if col not in base_df.columns]
                summary["required_columns"] = list(required)
                summary["missing_required_columns"] = missing
                if missing:
                    stderr_write(
                        f"Warning: override '{label}' missing columns {missing}; breakout features may stay zeroed.\n"
                    )
            metadata[label] = summary

    feature_ts_end = None
    if not base_df.empty and "ts" in base_df.columns:
        feature_ts_end = pd.to_datetime(base_df["ts"], utc=True, errors="coerce").max()
    source_freshness: Dict[str, Any] = {}
    if feature_ts_end is not None:
        for label, payload in metadata.items():
            if not isinstance(payload, Mapping):
                continue
            ts_end_raw = payload.get("ts_end")
            if not isinstance(ts_end_raw, str) or not ts_end_raw.strip():
                continue
            try:
                source_ts_end = pd.to_datetime(ts_end_raw, utc=True)
            except Exception:
                continue
            lag_hours = max((feature_ts_end - source_ts_end).total_seconds() / 3600.0, 0.0)
            source_freshness[str(label)] = {
                "ts_end": source_ts_end.isoformat(),
                "lag_hours": float(lag_hours),
            }

    if hours > 0 and len(base_df) > hours:
        base_df = base_df.iloc[-hours:].reset_index(drop=True)

    feature_names = load_training_feature_names(
        dataset_multi_path,
        dataset_1h_path,
        stderr_write=stderr_write,
    )
    if feature_names:
        supplemental_feature_names = [
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "cvd_ratio_6h",
            "cvd_zscore_6h",
            "funding_rate_zscore_24h",
            "trend_ignition_6h",
            "range_expansion_1h",
            "distance_from_session_high_8h",
            "distance_from_session_low_8h",
            "vwap_deviation_8h",
            "momentum_slope_2h",
            "momentum_slope_4h",
            *MACRO_FEATURE_COLUMNS,
            *ONCHAIN_FEATURE_COLUMNS,
        ]
        for column in supplemental_feature_names:
            if column not in feature_names:
                feature_names.append(column)

        base_df, synthesized_columns = enrich_local_features_for_model(
            base_df,
            required_columns=feature_names,
        )
        missing = [col for col in feature_names if col not in base_df.columns]
        if missing:
            unresolved_futures = [
                col
                for col in missing
                if col.startswith("fut_") or col in {"funding_rate", "funding_rate_annualized", "open_interest"}
            ]
            stderr_write(
                "Warning: local feature alignment still missing "
                f"{len(missing)} model columns after synthesizing {len(synthesized_columns)} columns; "
                f"imputing zeros ({len(unresolved_futures)} futures/funding/open-interest columns).\n"
            )
            for column in missing:
                base_df[column] = 0.0
        elif synthesized_columns:
            print(f"Info: synthesized {len(synthesized_columns)} local model columns from OHLCV context.")

        numeric_features = base_df[feature_names].apply(pd.to_numeric, errors="coerce")
        base_df[feature_names] = numeric_features.ffill().bfill().fillna(0.0)

        metadata["feature_alignment"] = {
            "required_columns": len(feature_names),
            "synthesized_columns": synthesized_columns,
            "imputed_zero_columns": missing,
        }
    else:
        feature_names = [col for col in base_df.columns if col != "ts"]

    metadata["source_freshness"] = source_freshness

    prepared = prepare_data_for_signals_from_ohlcv(
        base_df,
        feature_names=feature_names,
        train_frac=0.7,
    )

    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Local feature overrides produced an empty dataframe.")
    if "close" not in prepared.df_all.columns:
        raise RuntimeError("Local feature overrides must include a 'close' column for predictions.")

    close = float(prepared.df_all["close"].iloc[index])
    ts_iso = format_ts_iso(prepared.df_all["ts"].iloc[index])
    return (prepared, index, close, ts_iso), metadata