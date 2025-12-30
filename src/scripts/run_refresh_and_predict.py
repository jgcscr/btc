"""Refresh local Binance US-driven features and emit multi-horizon signals."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.ingestors.tiingo_spot import ingest_tiingo_spot
from data.processed.compute_cryptoquant_resampled import process_cryptoquant_resampled
from data.processed.compute_funding_features import process_funding_features
from data.processed.compute_macro_features import process_macro_features
from data.processed.compute_onchain_features import process_onchain_features
from src.scripts.build_training_dataset import main as build_1h_dataset
from src.scripts.build_training_dataset_multi_horizon import build_multi_horizon_dataset
from src.scripts.build_signal_baseline import (
    DEFAULT_COLUMNS as BASELINE_DEFAULT_COLUMNS,
    _append_detected_meta_columns,
    baseline_to_dataframe,
    compute_baseline,
    load_dataframe,
)
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    format_ts_iso,
    load_residual_std_from_dataset,
    load_models,
    populate_sequence_cache_from_prepared,
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
MACRO_FEATURES_PATH = Path("data/processed/macro/hourly_features.parquet")
ONCHAIN_FEATURES_PATH = Path("data/processed/onchain/hourly_features.parquet")
TRADE_READY_MONITOR_PATH = Path("artifacts/monitoring/trade_ready_summary.json")
META_BASELINE_JSON_PATH = Path("artifacts/monitoring/meta_baseline.json")
META_BASELINE_PARQUET_PATH = Path("artifacts/monitoring/meta_baseline.parquet")
META_BASELINE_SOURCE_CSV = Path("artifacts/backtests/backtest_signals_meta_ensemble.csv")


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
            "p_up_components": {},
            "stop_loss": close,
            "take_profit": close,
            "expected_value": 0.0,
            "thresholds": {
                "p_up_min": p_up_min,
                "ret_min": ret_min,
            },
        }
    return summary


def run_ingestion(
    hours: int,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    provider: str = "binanceus",
) -> Path:
    if provider == "tiingo":
        lookback_days = max(math.ceil(hours / 24), 1)
        print(f"Fetching {lookback_days} day(s) of {interval} candles from Tiingo for BTCUSD...")
        output_path = ingest_tiingo_spot(lookback_days=lookback_days)
        print(f"Saved Tiingo spot tidy parquet to {output_path}")
        return output_path

    limit = max(hours, 1)
    print(f"Fetching {limit} {interval} klines from Binance US for {symbol}...")
    output_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=limit)
    print(f"Saved spot tidy parquet to {output_path}")
    return output_path


def run_feature_builders(
    run_without_funding: bool,
    macro_source: str,
    onchain_source: str,
    funding_provider: str,
) -> Dict[str, str]:
    results: Dict[str, str] = {}

    if macro_source == "fallback":
        print(
            "Macro source set to fallback; reusing synthesized parquet at"
            f" {MACRO_FEATURES_PATH.as_posix()}.",
        )
        results["macro"] = str(MACRO_FEATURES_PATH)
    else:
        print("Recomputing macro features...")
        macro_path = process_macro_features()
        results["macro"] = str(macro_path)

    print("Recomputing CryptoQuant hourly fallback features...")
    cq_path = process_cryptoquant_resampled()
    results["cryptoquant"] = str(cq_path)

    if onchain_source == "fallback":
        print(
            "On-chain source set to fallback; reusing synthesized parquet at"
            f" {ONCHAIN_FEATURES_PATH.as_posix()}.",
        )
        results["onchain"] = str(ONCHAIN_FEATURES_PATH)
    else:
        print("Recomputing on-chain features...")
        onchain_path = process_onchain_features()
        results["onchain"] = str(onchain_path)

    print("Recomputing funding features...")
    funding_path = process_funding_features(
        pair="BTCUSDT",
        live_fetch=False,
        live_limit=1000,
        allow_missing=run_without_funding,
        provider=funding_provider,
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
    dir_lstm_path: str | None = None,
    dir_transformer_path: str | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    dataset_path = DATASET_MULTI_PATH if DATASET_MULTI_PATH.exists() else DATASET_1H_PATH
    if not dataset_path.exists():
        if offline:
            print("Dry run: dataset not found, emitting stub predictions.")
            return _build_stub_summary(targets, p_up_min, ret_min)
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    prepared, index, close, ts_iso = _load_prepared(dataset_path, offline=offline)
    residual_std_by_horizon = load_residual_std_from_dataset(str(dataset_path), targets)

    summary: Dict[str, Dict[str, float | str | int]] = {}
    for horizon in sorted(set(targets)):
        reg_path, dir_path = _model_paths_for_horizon(horizon)
        if not reg_path.exists() or not dir_path.exists():
            print(
                f"Warning: skipping {horizon}h horizon because model files are missing",
                file=sys.stderr,
            )
            continue

        models = load_models(
            str(reg_path),
            str(dir_path),
            lstm_model_dir=dir_lstm_path,
            transformer_model_dir=dir_transformer_path,
        )
        populate_sequence_cache_from_prepared(prepared, models)
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
        residual_std = float(residual_std_by_horizon.get(horizon, 0.01))
        stop_loss_price = _project_price(close, ret_pred - residual_std)
        take_profit_price = _project_price(close, ret_pred + residual_std)
        expected_value = p_up * ret_pred - (1 - p_up) * residual_std
        result = {
            "timestamp": signal_ts,
            "horizon_hours": horizon,
            "close": close,
            "p_up": p_up,
            "ret_pred": ret_pred,
            "projected_price": _project_price(close, ret_pred),
            "signal_ensemble": int(signal.get("signal_ensemble", 0)),
            "signal_dir_only": int(signal.get("signal_dir_only", 0)),
            "p_up_components": signal.get("p_up_components", {}),
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "expected_value": expected_value,
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


def write_summary(summary: Dict[str, Dict[str, float | str | int]]) -> dict[str, Any]:
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
    return json_payload


def _build_trade_ready_monitoring_payload(predictions_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    predictions = predictions_payload.get("predictions", {})
    horizons: list[dict[str, Any]] = []
    for horizon_key in sorted(
        predictions.keys(),
        key=lambda item: int(item.rstrip("h")) if item.rstrip("h").isdigit() else item,
    ):
        entry = predictions[horizon_key]
        if isinstance(entry, dict):
            horizons.append(entry)
    request = {
        "targets": args.targets,
        "spot_provider": args.spot_provider,
        "macro_source": args.macro_source,
        "onchain_source": args.onchain_source,
        "funding_provider": args.funding_provider,
        "hours": args.hours,
        "dry_run": bool(args.dry_run),
    }
    return {
        "generated_at": predictions_payload.get("generated_at"),
        "source": "run_refresh_and_predict",
        "request": request,
        "horizons": horizons,
    }


def _write_trade_ready_monitoring(predictions_payload: dict[str, Any], args: argparse.Namespace) -> None:
    payload = _build_trade_ready_monitoring_payload(predictions_payload, args)
    TRADE_READY_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADE_READY_MONITOR_PATH.write_text(json.dumps(payload, indent=2))


def _refresh_meta_baseline() -> None:
    if not META_BASELINE_SOURCE_CSV.exists():
        print(
            f"Meta baseline CSV not found at {META_BASELINE_SOURCE_CSV.as_posix()}; skipping baseline refresh.",
            file=sys.stderr,
        )
        return
    df = load_dataframe(META_BASELINE_SOURCE_CSV, limit=0)
    if df.empty:
        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": 0,
            "columns": {},
            "column_order": list(BASELINE_DEFAULT_COLUMNS),
        }
    else:
        columns = _append_detected_meta_columns(df, BASELINE_DEFAULT_COLUMNS)
        baseline = compute_baseline(df, columns)
    META_BASELINE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_BASELINE_JSON_PATH.write_text(json.dumps(baseline, indent=2))
    META_BASELINE_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_df = baseline_to_dataframe(baseline)
    baseline_df.to_parquet(META_BASELINE_PARQUET_PATH, index=False)


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
    parser.add_argument(
        "--spot-provider",
        choices=("binanceus", "tiingo"),
        default="binanceus",
        help="Spot ingestion provider for hourly candles (default: binanceus).",
    )
    parser.add_argument(
        "--macro-source",
        choices=("vendor", "fallback"),
        default="vendor",
        help="Select whether macro features should come from vendor data or the fallback builder.",
    )
    parser.add_argument(
        "--onchain-source",
        choices=("cryptocompare", "fallback"),
        default="cryptocompare",
        help="Choose the on-chain feed used when rebuilding features (default: cryptocompare).",
    )
    parser.add_argument(
        "--funding-provider",
        choices=("binance", "cryptocompare"),
        default="binance",
        help="Funding provider to use during feature rebuilds (default: binance).",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Update monitoring artifacts (trade_ready_summary + meta baseline) after predictions complete.",
    )
    parser.add_argument(
        "--dir-lstm-path",
        type=str,
        default=None,
        help="Optional directory containing the LSTM direction model ensemble.",
    )
    parser.add_argument(
        "--dir-transformer-path",
        type=str,
        default=None,
        help="Optional directory containing the transformer direction model ensemble.",
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
            run_ingestion(hours=args.hours, provider=args.spot_provider)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Ingestion failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            run_feature_builders(
                run_without_funding=run_without_funding,
                macro_source=args.macro_source,
                onchain_source=args.onchain_source,
                funding_provider=args.funding_provider,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Feature rebuild failed: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            rebuild_datasets(args.targets)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Dataset build failed: {exc}", file=sys.stderr)
            sys.exit(1)

    env_dir_lstm = os.getenv("DIR_LSTM_PATH") or args.dir_lstm_path
    env_dir_transformer = os.getenv("DIR_TRANSFORMER_PATH") or args.dir_transformer_path
    if env_dir_lstm or env_dir_transformer:
        print(
            "Sequence ensemble directories:"
            f" LSTM={env_dir_lstm or 'None'}"
            f", transformer={env_dir_transformer or 'None'}",
        )

    try:
        summary = run_predictions(
            args.targets,
            args.p_up_min,
            args.ret_min,
            offline=args.dry_run,
            dir_lstm_path=env_dir_lstm,
            dir_transformer_path=env_dir_transformer,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Prediction step failed: {exc}", file=sys.stderr)
        sys.exit(1)

    predictions_payload = write_summary(summary)

    if args.write_artifacts:
        _write_trade_ready_monitoring(predictions_payload, args)
        _refresh_meta_baseline()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
