from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Sequence

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.processed.compute_technical_features import process_technical_features
from src.data.derivatives_loader import (
    DEFAULT_DERIVATIVES_METADATA_PATH,
    DEFAULT_DERIVATIVES_OUTPUT_PATH,
    build_derivatives_feature_frame,
    build_derivatives_source_manifest,
    load_derivatives_features,
    resolve_incremental_start_timestamp as resolve_derivatives_incremental_start_timestamp,
    write_derivatives_source_manifest,
)
from src.data.macro_loader import (
    DEFAULT_MACRO_METADATA_PATH,
    DEFAULT_MACRO_OUTPUT_PATH,
    DEFAULT_MACRO_START_DATE,
    build_macro_feature_frame,
    build_source_manifest as build_macro_source_manifest,
    load_macro_features,
    resolve_incremental_start_date,
)
from src.data.onchain_loader import (
    DEFAULT_ONCHAIN_METADATA_PATH,
    DEFAULT_ONCHAIN_OUTPUT_PATH,
    DEFAULT_ONCHAIN_START_DATE,
    OnchainAPIError,
    build_onchain_feature_frame,
    build_onchain_source_manifest,
    load_onchain_features,
    resolve_incremental_start_timestamp,
    write_onchain_source_manifest,
)
from src.runtime.prediction_paths import DATASET_DIR
from src.scripts.build_training_dataset import main as build_1h_dataset
from src.scripts.build_training_dataset_15m import main as build_15m_dataset
from src.scripts.build_training_dataset_multi_horizon import build_multi_horizon_dataset


def run_ingestion(
    hours: int,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    provider: str = "binanceus",
) -> Path:
    if provider != "binanceus":
        raise ValueError(f"Unsupported provider '{provider}'. Binance-only mode requires --spot-provider binanceus.")

    limit = max(hours, 1)
    print(f"Fetching {limit} {interval} klines from Binance US for {symbol}...")
    output_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=limit)
    print(f"Saved spot tidy parquet to {output_path}")
    return output_path


def run_feature_builders(price_source: Path | None = None) -> Dict[str, str]:
    results: Dict[str, str] = {}
    print("Recomputing technical indicator features...")
    technical_path = process_technical_features(price_source=price_source, include_history=True)
    results["technical"] = str(technical_path)

    try:
        existing_derivatives = (
            load_derivatives_features(DEFAULT_DERIVATIVES_OUTPUT_PATH)
            if DEFAULT_DERIVATIVES_OUTPUT_PATH.exists()
            else None
        )
        derivatives_start = resolve_derivatives_incremental_start_timestamp(existing_derivatives)
        derivatives_frame = build_derivatives_feature_frame(
            start_ts=derivatives_start,
            existing=existing_derivatives,
        )
        if derivatives_frame.empty:
            raise RuntimeError("Binance futures refresh returned no usable rows.")
        DEFAULT_DERIVATIVES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        derivatives_frame.to_parquet(DEFAULT_DERIVATIVES_OUTPUT_PATH, index=False)
        derivatives_manifest = build_derivatives_source_manifest(
            derivatives_frame,
            start_ts=derivatives_start,
        )
        write_derivatives_source_manifest(DEFAULT_DERIVATIVES_METADATA_PATH, derivatives_manifest)
        results["funding"] = str(DEFAULT_DERIVATIVES_OUTPUT_PATH)
        print(f"Refreshed derivatives features at {DEFAULT_DERIVATIVES_OUTPUT_PATH}")
    except Exception as exc:
        print(f"Warning: derivatives feature refresh failed: {exc}", file=sys.stderr)

    try:
        existing_macro = load_macro_features(DEFAULT_MACRO_OUTPUT_PATH) if DEFAULT_MACRO_OUTPUT_PATH.exists() else None
        macro_start = resolve_incremental_start_date(
            existing_macro,
            default_start_date=DEFAULT_MACRO_START_DATE,
        )
        macro_frame = build_macro_feature_frame(start_date=macro_start, existing=existing_macro)
        DEFAULT_MACRO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        macro_frame.to_parquet(DEFAULT_MACRO_OUTPUT_PATH, index=False)
        macro_manifest = build_macro_source_manifest()
        macro_manifest["row_count"] = int(len(macro_frame))
        macro_manifest["ts_start"] = macro_frame["ts"].min().isoformat() if not macro_frame.empty else None
        macro_manifest["ts_end"] = macro_frame["ts"].max().isoformat() if not macro_frame.empty else None
        macro_manifest["refresh"] = {
            "requested_start_date": macro_start,
            "output_path": str(DEFAULT_MACRO_OUTPUT_PATH),
        }
        DEFAULT_MACRO_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_MACRO_METADATA_PATH.write_text(json.dumps(macro_manifest, indent=2), encoding="utf-8")
        results["macro"] = str(DEFAULT_MACRO_OUTPUT_PATH)
        print(f"Refreshed macro features at {DEFAULT_MACRO_OUTPUT_PATH}")
    except Exception as exc:
        print(f"Warning: macro feature refresh failed: {exc}", file=sys.stderr)

    try:
        existing_onchain = load_onchain_features(DEFAULT_ONCHAIN_OUTPUT_PATH) if DEFAULT_ONCHAIN_OUTPUT_PATH.exists() else None
        onchain_start = resolve_incremental_start_timestamp(
            existing_onchain,
            default_start=DEFAULT_ONCHAIN_START_DATE,
        )
        onchain_frame = build_onchain_feature_frame(start_ts=onchain_start, existing=existing_onchain)
        DEFAULT_ONCHAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        onchain_frame.to_parquet(DEFAULT_ONCHAIN_OUTPUT_PATH, index=False)
        onchain_manifest = build_onchain_source_manifest()
        onchain_manifest["row_count"] = int(len(onchain_frame))
        onchain_manifest["ts_start"] = onchain_frame["ts"].min().isoformat() if not onchain_frame.empty else None
        onchain_manifest["ts_end"] = onchain_frame["ts"].max().isoformat() if not onchain_frame.empty else None
        onchain_manifest["refresh"] = {
            "requested_start_ts": onchain_start,
            "output_path": str(DEFAULT_ONCHAIN_OUTPUT_PATH),
        }
        write_onchain_source_manifest(DEFAULT_ONCHAIN_METADATA_PATH, onchain_manifest)
        results["onchain"] = str(DEFAULT_ONCHAIN_OUTPUT_PATH)
        print(f"Refreshed on-chain features at {DEFAULT_ONCHAIN_OUTPUT_PATH}")
    except OnchainAPIError as exc:
        print(f"Warning: on-chain feature refresh skipped: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: on-chain feature refresh failed: {exc}", file=sys.stderr)

    return results


def rebuild_datasets(horizons: Sequence[float]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 1h dataset splits...")
    build_1h_dataset(str(DATASET_DIR))

    hourly_targets = {int(round(h)) for h in horizons if h >= 1.0}
    expanded_horizons = sorted(hourly_targets | {1, 4})
    print(f"Building multi-horizon dataset for horizons {expanded_horizons}...")
    build_multi_horizon_dataset(
        output_dir=str(DATASET_DIR),
        horizons=expanded_horizons,
        train_frac=0.7,
        val_frac=0.15,
    )

    if any(h < 1.0 for h in horizons):
        print("Detected sub-hourly targets; refreshing 15m dataset splits...")
        build_15m_dataset(str(DATASET_DIR))
