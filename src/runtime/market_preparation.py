from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.runtime.models import RuntimeRunPaths
from src.runtime.persistence import RuntimeStateStore
from src.runtime.local_feature_support import (
    build_ohlcv_frame_from_tidy as runtime_build_ohlcv_frame_from_tidy,
    compute_intrabar_features_from_15m as runtime_compute_intrabar_features_from_15m,
    prepare_local_feature_bundle as runtime_prepare_local_feature_bundle,
    read_timeseries_frame as runtime_read_timeseries_frame,
)
from src.runtime.dataset_profile_support import DatasetCandidate, DatasetProfile
from src.runtime.local_feature_defaults import LOCAL_FEATURE_OPTIONAL_PATHS, LOCAL_FEATURE_REQUIRED_COLUMNS
from src.runtime.prediction_paths import DATASET_15M_PATH, DATASET_1H_PATH, DATASET_MULTI_PATH, DATA_QUALITY_MONITOR_PATH
from src.runtime.refresh_stage_support import rebuild_datasets, run_feature_builders, run_ingestion
from src.runtime.quality_support import (
    evaluate_data_quality as runtime_evaluate_data_quality,
    evaluate_feature_coverage as runtime_evaluate_feature_coverage,
    resolve_data_quality_policy as runtime_resolve_data_quality_policy,
    resolve_feature_coverage_policy as runtime_resolve_feature_coverage_policy,
    write_data_quality_payload as runtime_write_data_quality_payload,
)
from src.trading.data_quality import DataQualityError, DataQualityPolicy, evaluate_ohlcv_quality
from src.trading.signals import PreparedData, format_ts_iso, prepare_data_for_signals, prepare_data_for_signals_from_ohlcv


PreparedOverride = tuple[PreparedData, int, float, str] | None


@dataclass(frozen=True)
class MarketPreparationResult:
    prepared_override: PreparedOverride
    latest_close: float | None


def prepare_market_data(
    args: argparse.Namespace,
    run_paths: RuntimeRunPaths,
    store: RuntimeStateStore,
) -> MarketPreparationResult:
    store.append_event(run_paths, stage="data_preparation", status="started")

    if getattr(args, "use_local_features", False):
        prepared_override = _load_local_feature_override(args)
        store.append_event(run_paths, stage="data_preparation", status="completed", details={"source": "local_features"})
        return MarketPreparationResult(prepared_override=prepared_override, latest_close=None)

    if getattr(args, "dry_run", False):
        store.append_event(run_paths, stage="data_preparation", status="completed", details={"source": "cached_datasets"})
        return MarketPreparationResult(prepared_override=None, latest_close=None)

    latest_close = _run_refresh_stages(args)
    store.append_event(run_paths, stage="data_preparation", status="completed", details={"source": "fresh_refresh"})
    return MarketPreparationResult(prepared_override=getattr(args, "_prepared_override", None), latest_close=latest_close)


def apply_replay_override(
    args: argparse.Namespace,
    run_paths: RuntimeRunPaths,
    store: RuntimeStateStore,
    *,
    prepared_override: PreparedOverride,
    latest_close: float | None,
) -> MarketPreparationResult:
    from src.runtime.refresh_support import (
        dataset_profile_for_horizon as runtime_dataset_profile_for_horizon,
        load_prepared as runtime_load_prepared,
        load_prepared_offline as runtime_load_prepared_offline,
        select_dataset_candidate as runtime_select_dataset_candidate,
    )

    replay_offset_bars = int(getattr(args, "replay_offset_bars", 0) or 0)
    if replay_offset_bars <= 0:
        return MarketPreparationResult(prepared_override=prepared_override, latest_close=latest_close)

    store.append_event(run_paths, stage="replay_override", status="started", details={"offset_bars": replay_offset_bars})
    replay_profile = runtime_dataset_profile_for_horizon(
        1.0,
        dataset_multi_path=DATASET_MULTI_PATH,
        dataset_1h_path=DATASET_1H_PATH,
        dataset_15m_path=DATASET_15M_PATH,
        dataset_candidate_type=DatasetCandidate,
        dataset_profile_type=DatasetProfile,
    )
    replay_candidate, used_fallback = runtime_select_dataset_candidate(replay_profile)
    prepared, replay_latest_index, _close_snapshot, _ts_snapshot = runtime_load_prepared(
        replay_candidate.path,
        target_column=replay_candidate.target_column,
        offline=True,
        load_prepared_offline_fn=lambda dataset_path, *, base_horizon: runtime_load_prepared_offline(
            dataset_path,
            base_horizon=base_horizon,
            prepare_data_for_signals_from_ohlcv_fn=prepare_data_for_signals_from_ohlcv,
            format_ts_iso_fn=format_ts_iso,
            stderr_write=sys.stderr.write,
        ),
        prepare_data_for_signals_fn=prepare_data_for_signals,
        format_ts_iso_fn=format_ts_iso,
    )
    replay_index = replay_latest_index - replay_offset_bars
    if replay_index < 0:
        raise ValueError(
            f"Replay offset {replay_offset_bars} exceeds prepared dataset length {replay_latest_index + 1}."
        )
    replay_close = float(prepared.df_all["close"].iloc[replay_index])
    replay_ts = format_ts_iso(prepared.df_all["ts"].iloc[replay_index])
    prepared_override = (prepared, replay_index, replay_close, replay_ts)
    fallback_msg = " (fallback dataset)" if used_fallback else ""
    print(
        "Replay mode enabled: using hourly cached dataset "
        f"{replay_candidate.path.name}{fallback_msg} at index offset {replay_offset_bars} "
        f"(timestamp={replay_ts})."
    )
    store.append_event(run_paths, stage="replay_override", status="completed", details={"timestamp": replay_ts})
    return MarketPreparationResult(prepared_override=prepared_override, latest_close=replay_close)


def _load_local_feature_override(args: argparse.Namespace) -> PreparedOverride:
    optional_sources = {
        label: getattr(args, attr)
        for attr, label in LOCAL_FEATURE_OPTIONAL_PATHS
        if getattr(args, attr, None)
    }
    prepared_override, metadata = runtime_prepare_local_feature_bundle(
        features_path=args.features_path,
        hours=args.hours,
        optional_sources=optional_sources,
        dataset_multi_path=DATASET_MULTI_PATH,
        dataset_1h_path=DATASET_1H_PATH,
        local_feature_required_columns=LOCAL_FEATURE_REQUIRED_COLUMNS,
        stderr_write=sys.stderr.write,
    )
    args.local_feature_metadata = metadata
    coverage_policy = runtime_resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
    coverage_payload = runtime_evaluate_feature_coverage(metadata, coverage_policy)
    args.local_feature_metadata["feature_coverage"] = coverage_payload
    if coverage_policy.get("enabled") and not coverage_payload.get("ok", False) and coverage_payload.get("block_on_violation", True):
        raise RuntimeError(
            "Feature coverage gate blocked prediction run: " + ", ".join(coverage_payload.get("failed_checks", []))
        )
    quality_policy = runtime_resolve_data_quality_policy(getattr(args, "data_quality", None))
    if quality_policy.get("enabled"):
        quality_frame = runtime_read_timeseries_frame(args.features_path, "features")
        quality_payload = runtime_evaluate_data_quality(
            quality_frame,
            quality_policy,
            data_quality_policy_type=DataQualityPolicy,
            evaluate_ohlcv_quality=evaluate_ohlcv_quality,
            data_quality_error_type=DataQualityError,
            write_data_quality_payload=lambda payload: runtime_write_data_quality_payload(
                payload,
                DATA_QUALITY_MONITOR_PATH,
            ),
        )
        if not quality_payload.get("ok", False):
            raise RuntimeError(
                f"Data quality gate blocked prediction run: {quality_payload.get('error', 'unknown data quality failure')}"
            )
    return prepared_override


def _run_refresh_stages(args: argparse.Namespace) -> float | None:
    output_path = run_ingestion(hours=args.hours, provider=args.spot_provider)
    intrabar_features_path = None
    if getattr(args, "_intrabar_enabled", False):
        intrabar_cfg = getattr(args, "_intrabar_cfg")
        intrabar_interval = str(intrabar_cfg.get("interval") or "15m")
        hours_mult = max(int(intrabar_cfg.get("hours_multiplier") or 4), 1)
        max_rows = max(int(intrabar_cfg.get("max_rows") or 4000), 1)
        intrabar_limit = min(max_rows, max(args.hours * hours_mult, args.hours))
        intrabar_tidy_path = run_ingestion(
            hours=intrabar_limit,
            interval=intrabar_interval,
            provider=args.spot_provider,
        )
        intrabar_df = runtime_compute_intrabar_features_from_15m(intrabar_tidy_path)
        intrabar_output = Path("data/processed/technical") / "intrabar_features_15m_to_1h.parquet"
        intrabar_output.parent.mkdir(parents=True, exist_ok=True)
        intrabar_df.to_parquet(intrabar_output, index=False)
        intrabar_features_path = str(intrabar_output)
        print(
            "Saved intrabar aggregated features to "
            f"{intrabar_output} (rows={len(intrabar_df)}, interval={intrabar_interval}).",
        )

    latest_spot_features_path, latest_close = _persist_latest_spot_features(args, output_path)
    feature_build_results = run_feature_builders(price_source=output_path)
    rebuild_datasets(args.targets)

    technical_features_path = feature_build_results.get("technical")
    if latest_spot_features_path:
        try:
            prepared_override, metadata = runtime_prepare_local_feature_bundle(
                features_path=latest_spot_features_path,
                hours=args.hours,
                optional_sources={
                    key: value
                    for key, value in {
                        "technical": technical_features_path,
                        "funding": feature_build_results.get("funding"),
                        "macro": feature_build_results.get("macro"),
                        "onchain": feature_build_results.get("onchain"),
                        "intrabar": intrabar_features_path,
                    }.items()
                    if value
                }
                or None,
                dataset_multi_path=DATASET_MULTI_PATH,
                dataset_1h_path=DATASET_1H_PATH,
                local_feature_required_columns=LOCAL_FEATURE_REQUIRED_COLUMNS,
                stderr_write=sys.stderr.write,
            )
            args.local_feature_metadata = metadata
            coverage_policy = runtime_resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
            coverage_payload = runtime_evaluate_feature_coverage(metadata, coverage_policy)
            args.local_feature_metadata["feature_coverage"] = coverage_payload
            if coverage_policy.get("enabled") and not coverage_payload.get("ok", False) and coverage_payload.get("block_on_violation", True):
                raise RuntimeError(
                    "feature coverage gate failed: " + ", ".join(coverage_payload.get("failed_checks", []))
                )
            print(
                "Using freshly rebuilt local feature bundle for live inference "
                f"({latest_spot_features_path}).",
            )
            args._prepared_override = prepared_override
        except Exception as exc:
            coverage_policy = runtime_resolve_feature_coverage_policy(getattr(args, "feature_coverage_policy", None))
            if coverage_policy.get("enabled") and coverage_policy.get("block_on_violation", True):
                raise RuntimeError(
                    "Fresh local inference bundle preparation failed and coverage blocking is enabled: "
                    f"{exc}"
                ) from exc
            print(
                "Warning: failed to prepare fresh local inference bundle; "
                f"falling back to dataset-based inference ({exc}).",
                file=sys.stderr,
            )
    elif technical_features_path:
        print(
            "Warning: fresh spot feature file unavailable; local inference override disabled.",
            file=sys.stderr,
        )
    return latest_close


def _persist_latest_spot_features(args: argparse.Namespace, output_path: Any) -> tuple[str | None, float | None]:
    latest_spot_features_path: str | None = None
    latest_close: float | None = None
    if output_path and output_path.exists():
        df = pd.read_parquet(output_path)
        quality_policy = runtime_resolve_data_quality_policy(getattr(args, "data_quality", None))
        if quality_policy.get("enabled"):
            quality_frame = runtime_build_ohlcv_frame_from_tidy(df)
            quality_payload = runtime_evaluate_data_quality(
                quality_frame,
                quality_policy,
                data_quality_policy_type=DataQualityPolicy,
                evaluate_ohlcv_quality=evaluate_ohlcv_quality,
                data_quality_error_type=DataQualityError,
                write_data_quality_payload=lambda payload: runtime_write_data_quality_payload(
                    payload,
                    DATA_QUALITY_MONITOR_PATH,
                ),
            )
            if not quality_payload.get("ok", False):
                raise RuntimeError(
                    "Data quality gate blocked prediction run: "
                    f"{quality_payload.get('error', 'unknown data quality failure')}"
                )
        wide_df = df.pivot(index="ts", columns="metric", values="value").reset_index()
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
        wide_df = wide_df.rename(columns=rename_map)
        wide_df["interval"] = "1h"
        today = datetime.now().strftime("%Y-%m-%d")
        spot_path = Path("data/spot_klines") / f"btcusdt_spot_1h_{today}.parquet"
        wide_df.to_parquet(spot_path, index=False)
        latest_spot_features_path = str(spot_path)
        print(f"Saved latest price data to {spot_path}")
        close_df = df[df.metric == "spot_close"]
        if not close_df.empty:
            latest_close = float(close_df.value.iloc[-1])
    return latest_spot_features_path, latest_close