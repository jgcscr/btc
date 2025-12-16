"""Refresh local Binance US-driven features and emit multi-horizon signals."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.processed.compute_coinapi_features import process_coinapi_features
from data.processed.compute_cryptoquant_resampled import process_cryptoquant_resampled
from data.processed.compute_funding_features import process_funding_features
from data.processed.compute_macro_features import process_macro_features
from data.processed.compute_onchain_features import process_onchain_features
from src.scripts.build_training_dataset import main as build_1h_dataset
from src.scripts.build_training_dataset_multi_horizon import build_multi_horizon_dataset
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    format_ts_iso,
    load_models,
    prepare_data_for_signals,
    prepare_data_for_signals_from_ohlcv,
)

DEFAULT_HOURS = 360
DEFAULT_TARGETS = (1, 4, 8, 12)
DEFAULT_P_UP_MIN = 0.45
DEFAULT_RET_MIN = 0.0
MODEL_ROOT = Path("artifacts/models")
DATASET_DIR = Path("artifacts/datasets")
LATEST_PREDICTION_PATH = Path("artifacts/predictions/latest.json")
DATASET_1H_PATH = DATASET_DIR / "btc_features_1h_splits.npz"
DATASET_MULTI_PATH = DATASET_DIR / "btc_features_multi_horizon_splits.npz"
HISTORY_PREDICTION_PATH = Path("artifacts/predictions/history.json")


def parse_targets(value: str) -> List[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one horizon must be provided.")
    targets: List[int] = []
    for part in parts:
        try:
            horizon = int(part)
        except ValueError as exc:  # pragma: no cover - CLI validation guard
            raise argparse.ArgumentTypeError(f"Invalid horizon: {part}") from exc
        if horizon <= 0:
            raise argparse.ArgumentTypeError("Horizons must be positive integers.")
        targets.append(horizon)
    return targets


def _bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_stub_summary(
    targets: Iterable[int],
    p_up_min: float,
    ret_min: float,
    close: float = 0.0,
    ts_iso: str | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    generated_ts = ts_iso or datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Dict[str, float | str | int]] = {}
    for horizon in sorted({int(h) for h in targets}):
        summary[f"{horizon}h"] = {
            "timestamp": generated_ts,
            "horizon_hours": horizon,
            "close": close,
            "p_up": 0.5,
            "ret_pred": 0.0,
            "projected_price": close,
            "signal_ensemble": 0,
            "signal_dir_only": 0,
            "thresholds": {
                "p_up_min": p_up_min,
                "ret_min": ret_min,
            },
        }
    return summary


def run_ingestion(hours: int, symbol: str = "BTCUSDT", interval: str = "1h") -> Path:
    limit = max(hours, 1)
    print(f"Fetching {limit} {interval} klines from Binance US for {symbol}...")
    output_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=limit)
    print(f"Saved spot tidy parquet to {output_path}")
    return output_path


def run_feature_builders(run_without_funding: bool) -> Dict[str, str]:
    results: Dict[str, str] = {}

    print("Recomputing CoinAPI-derived market features...")
    market_path, _ = process_coinapi_features()
    results["coinapi_market"] = str(market_path)

    print("Recomputing macro features...")
    macro_path = process_macro_features()
    results["macro"] = str(macro_path)

    print("Recomputing CryptoQuant hourly fallback features...")
    cq_path = process_cryptoquant_resampled()
    results["cryptoquant"] = str(cq_path)

    print("Recomputing on-chain features...")
    onchain_path = process_onchain_features()
    results["onchain"] = str(onchain_path)

    print("Recomputing funding features...")
    funding_path = process_funding_features(
        pair="BTCUSDT",
        live_fetch=False,
        live_limit=1000,
        allow_missing=run_without_funding,
    )
    results["funding"] = str(funding_path)

    return results


def rebuild_datasets(horizons: Sequence[int]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 1h dataset splits...")
    build_1h_dataset(str(DATASET_DIR))

    expanded_horizons = sorted(set(horizons) | {1, 4})
    print(f"Building multi-horizon dataset for horizons {expanded_horizons}...")
    build_multi_horizon_dataset(
        output_dir=str(DATASET_DIR),
        horizons=expanded_horizons,
        train_frac=0.7,
        val_frac=0.15,
        onchain_path=None,
        fetch_onchain=False,
        onchain_interval="1h",
        features_path=None,
        output_path=None,
    )


def _model_paths_for_horizon(horizon: int) -> tuple[Path, Path]:
    reg_dir = MODEL_ROOT / f"xgb_ret{horizon}h_v1"
    dir_dir = MODEL_ROOT / f"xgb_dir{horizon}h_v1"
    reg_path = reg_dir / f"xgb_ret{horizon}h_model.json"
    dir_path = dir_dir / f"xgb_dir{horizon}h_model.json"
    return reg_path, dir_path


def _load_prepared(dataset_path: Path, offline: bool = False) -> tuple:
    if offline:
        return _load_prepared_offline(dataset_path)

    prepared = prepare_data_for_signals(str(dataset_path), target_column="ret_1h")
    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Prepared dataset has no rows.")
    ts_value = prepared.df_all["ts"].iloc[index]
    close = float(prepared.df_all["close"].iloc[index])
    ts_iso = format_ts_iso(ts_value)
    return prepared, index, close, ts_iso


def _load_prepared_offline(dataset_path: Path) -> tuple[PreparedData, int, float, str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found for offline preparation: {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        if "feature_names" not in dataset_npz.files:
            raise KeyError("Dataset NPZ missing feature_names for offline preparation.")
        feature_names = dataset_npz["feature_names"].tolist()
        arrays = [dataset_npz[key] for key in ("X_train", "X_val", "X_test") if key in dataset_npz.files]

    if not arrays:
        raise RuntimeError("Dataset NPZ does not contain any feature splits for offline preparation.")

    X_all = np.concatenate(arrays, axis=0)
    if X_all.size == 0:
        raise RuntimeError("Dataset NPZ is empty after concatenation; cannot build offline prepared data.")

    df_features = pd.DataFrame(X_all, columns=feature_names)
    if "close" not in df_features.columns:
        raise RuntimeError("Offline dataset must include a 'close' feature column.")

    periods = len(df_features)
    ts_index = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq="H")
    df_features.insert(0, "ts", ts_index)

    prepared = prepare_data_for_signals_from_ohlcv(
        df_features,
        feature_names=feature_names,
        train_frac=0.7,
    )

    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Offline prepared dataset has no rows.")

    ts_value = prepared.df_all["ts"].iloc[index]
    close = float(prepared.df_all["close"].iloc[index])
    ts_iso = format_ts_iso(ts_value)
    return prepared, index, close, ts_iso


def _project_price(close: float, log_return: float) -> float:
    return close * math.exp(log_return)


def run_predictions(
    targets: Iterable[int],
    p_up_min: float,
    ret_min: float,
    offline: bool = False,
) -> Dict[str, Dict[str, float | str | int]]:
    dataset_path = DATASET_MULTI_PATH if DATASET_MULTI_PATH.exists() else DATASET_1H_PATH
    if not dataset_path.exists():
        if offline:
            print("Dry run: dataset not found, emitting stub predictions.")
            return _build_stub_summary(targets, p_up_min, ret_min)
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    prepared, index, close, ts_iso = _load_prepared(dataset_path, offline=offline)

    summary: Dict[str, Dict[str, float | str | int]] = {}
    for horizon in sorted(set(targets)):
        reg_path, dir_path = _model_paths_for_horizon(horizon)
        if not reg_path.exists() or not dir_path.exists():
            print(
                f"Warning: skipping {horizon}h horizon because model files are missing",
                file=sys.stderr,
            )
            continue

        models = load_models(str(reg_path), str(dir_path))
        signal = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=p_up_min,
            ret_min=ret_min,
        )

        ret_pred = float(signal.get("ret_pred", 0.0))
        p_up = float(signal.get("p_up", 0.0))
        signal_ts = str(signal.get("ts", ts_iso))
        result = {
            "timestamp": signal_ts,
            "horizon_hours": horizon,
            "close": close,
            "p_up": p_up,
            "ret_pred": ret_pred,
            "projected_price": _project_price(close, ret_pred),
            "signal_ensemble": int(signal.get("signal_ensemble", 0)),
            "signal_dir_only": int(signal.get("signal_dir_only", 0)),
            "thresholds": {
                "p_up_min": p_up_min,
                "ret_min": ret_min,
            },
        }
        summary[f"{horizon}h"] = result
    if not summary:
        if offline:
            print("Dry run: model artifacts missing, emitting stub predictions.")
            return _build_stub_summary(targets, p_up_min, ret_min, close=close, ts_iso=ts_iso)
        raise RuntimeError("No predictions were produced; ensure model artifacts exist.")
    return summary


def write_summary(summary: Dict[str, Dict[str, float | str | int]]) -> None:
    LATEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    json_payload = {
        "generated_at": generated_at,
        "predictions": summary,
    }
    LATEST_PREDICTION_PATH.write_text(json.dumps(json_payload, indent=2))
    print(json.dumps(json_payload, indent=2))

    history_entry = {
        "generated_at": generated_at,
        "predictions": summary,
    }
    history: List[Dict[str, object]] = []
    if HISTORY_PREDICTION_PATH.exists():
        try:
            history = json.loads(HISTORY_PREDICTION_PATH.read_text())
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    history.append(history_entry)
    HISTORY_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PREDICTION_PATH.write_text(json.dumps(history, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Binance US spot data, rebuild local features/datasets, and emit multi-horizon predictions."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Number of hourly candles to fetch from Binance US (default: 360).",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=list(DEFAULT_TARGETS),
        help="Comma-separated prediction horizons in hours (default: 1,4,8,12).",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Probability threshold for ensemble activation (default: 0.45).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Return threshold for ensemble activation (default: 0.0).",
    )
    parser.add_argument(
        "--run-without-funding",
        action="store_true",
        help="Force RUN_WITHOUT_FUNDING=1 for the duration of this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip network-dependent steps and reuse cached datasets/models for smoke testing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.run_without_funding or _bool_env(os.getenv("RUN_WITHOUT_FUNDING")):
        os.environ["RUN_WITHOUT_FUNDING"] = "1"
        run_without_funding = True
    else:
        run_without_funding = False

    if args.dry_run:
        print("Dry run enabled: using cached datasets and skipping ingestion, feature rebuild, and dataset regeneration.")
    else:
        try:
            run_ingestion(hours=args.hours)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            run_feature_builders(run_without_funding=run_without_funding)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Feature rebuild failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            rebuild_datasets(args.targets)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Dataset build failed: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        summary = run_predictions(
            args.targets,
            args.p_up_min,
            args.ret_min,
            offline=args.dry_run,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Prediction step failed: {exc}", file=sys.stderr)
        sys.exit(1)

    write_summary(summary)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
