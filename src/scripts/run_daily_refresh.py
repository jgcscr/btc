"""Daily Binance US spot and feature refresh automation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # Google Cloud optional, skip gracefully if unavailable
    from google.cloud import bigquery
    from google.api_core.exceptions import GoogleAPIError
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:  # pragma: no cover - environment without BigQuery support
    bigquery = None  # type: ignore
    GoogleAPIError = Exception  # type: ignore
    google = None  # type: ignore
    DefaultCredentialsError = Exception  # type: ignore

from data.ingestors.binance_futures_metrics import ingest_futures_day_to_gcs
from data.ingestors.binance_spot_klines import ingest_spot_day_to_gcs
from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.ingestors.alpha_vantage_macro import (
    AlphaVantageIngestionError,
    AlphaVantageInvalidKeyError,
    AlphaVantageRateLimitError,
    ingest_catalog as ingest_alpha_vantage_catalog,
)
from data.processed.compute_coinapi_features import process_coinapi_features
from data.processed.compute_cryptoquant_resampled import process_cryptoquant_resampled
from data.processed.compute_funding_features import (
    FundingProcessingError,
    process_funding_features,
)
from data.processed.compute_macro_features import process_macro_features
from data.processed.compute_onchain_features import process_onchain_features
from src.config import BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H, PROJECT_ID
from src.data.bq_loader import load_btc_features_1h
from src.scripts.build_training_dataset import main as build_1h_dataset
from src.scripts.build_training_dataset_multi_horizon import build_multi_horizon_dataset

DATASET_DIR = Path("artifacts/datasets")
DATASET_1H_PATH = DATASET_DIR / "btc_features_1h_splits.npz"
DATASET_MULTI_PATH = DATASET_DIR / "btc_features_multi_horizon_splits.npz"
SQL_REFRESH_PATH = Path("sql/create_btc_features_1h.sql")
DEFAULT_FUTURES_DAYS = 2


def _date_range(days: int, anchor: Optional[date] = None) -> List[date]:
    if days <= 0:
        return []
    end_date = anchor or datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days - 1)
    return [start_date + timedelta(days=offset) for offset in range(days)]


def _spot_ingest_dates(hours: int) -> List[date]:
    now = datetime.now(timezone.utc)
    end_date = now.date()
    start_date = (now - timedelta(hours=hours)).date()
    total_days = (end_date - start_date).days + 1
    total_days = max(total_days, 1)
    return _date_range(total_days, anchor=end_date)


def run_futures_ingest(
    bucket: Optional[str],
    interval: str,
    pair: str,
    days: int,
    market: Optional[str] = None,
    instrument: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    log_event(
        "futures_ingest.start",
        bucket=bucket,
        interval=interval,
        pair=pair,
        days=days,
        market=market,
        instrument=instrument,
    )
    if not bucket:
        log_event("futures_ingest.skipped", note="Futures bucket not provided; skipping upload")
        return {}
    target_dates = _date_range(days)
    uploads: List[Dict[str, Any]] = []
    for target_date in target_dates:
        try:
            blob_path = ingest_futures_day_to_gcs(
                pair=pair,
                interval=interval,
                date=target_date,
                bucket_name=bucket,
                market=market,
                instrument=instrument,
                api_key=api_key,
            )
            uploads.append({
                "date": target_date.isoformat(),
                "blob_path": blob_path,
            })
            log_event(
                "futures_ingest.uploaded",
                date=target_date,
                blob_path=blob_path,
                bucket=bucket,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            log_event(
                "futures_ingest.error",
                date=target_date,
                error=str(exc),
            )
            raise
    summary = {
        "bucket": bucket,
        "interval": interval,
        "pair": pair,
        "dates": [entry["date"] for entry in uploads],
        "upload_count": len(uploads),
    }
    log_event("futures_ingest.complete", **summary)
    return summary


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(message: str, **fields: Any) -> None:
    payload = {"timestamp": iso_now(), "message": message}
    if fields:
        payload.update({key: _stringify(value) for key, value in fields.items()})
    print(json.dumps(payload))


def _stringify(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    return value


def _summarize_parquet(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "latest": None}
    frame = pd.read_parquet(path)
    timestamp_col = "timestamp" if "timestamp" in frame.columns else "ts" if "ts" in frame.columns else None
    latest = None
    if timestamp_col:
        ts_series = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        latest = ts_series.max().isoformat() if ts_series.notna().any() else None
    return {
        "rows": int(len(frame)),
        "latest": latest,
    }


def _summarize_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"rows_total": 0, "feature_count": 0}
    with np.load(path, allow_pickle=True) as data:
        splits = {}
        total = 0
        for split_name in ("X_train", "X_val", "X_test"):
            if split_name in data:
                split_rows = int(data[split_name].shape[0])
                splits[split_name] = split_rows
                total += split_rows
        feature_count = int(len(data["feature_names"])) if "feature_names" in data else 0
    return {"rows_total": total, "feature_count": feature_count, "splits": splits}


def upload_spot_history_to_gcs(
    bucket: str,
    symbol: str,
    interval: str,
    hours: int,
) -> Dict[str, Any]:
    log_event(
        "spot_ingest.start",
        bucket=bucket,
        symbol=symbol,
        interval=interval,
        hours=hours,
    )
    target_dates = _spot_ingest_dates(hours)
    if not target_dates:
        summary = {
            "bucket": bucket,
            "symbol": symbol,
            "interval": interval,
            "dates": [],
            "upload_count": 0,
        }
        log_event("spot_ingest.complete", **summary)
        return summary

    uploads: List[Dict[str, Any]] = []
    for target_date in target_dates:
        try:
            blob_path = ingest_spot_day_to_gcs(
                symbol=symbol,
                interval=interval,
                date=target_date,
                bucket_name=bucket,
            )
            uploads.append({
                "date": target_date.isoformat(),
                "blob_path": blob_path,
            })
            log_event(
                "spot_ingest.uploaded",
                date=target_date,
                blob_path=blob_path,
                bucket=bucket,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            log_event(
                "spot_ingest.error",
                date=target_date,
                error=str(exc),
            )
            raise

    summary = {
        "bucket": bucket,
        "symbol": symbol,
        "interval": interval,
        "dates": [entry["date"] for entry in uploads],
        "upload_count": len(uploads),
    }
    log_event("spot_ingest.complete", **summary)
    return summary


def run_binance_ingest(hours: int, spot_bucket: Optional[str] = None, symbol: str = "BTCUSDT", interval: str = "1h") -> Dict[str, Any]:
    log_event("binance_ingest.start", hours=hours, symbol=symbol, interval=interval)
    parquet_path = ingest_binance_us_spot(symbol=symbol, interval=interval, limit=hours)
    tidy = pd.read_parquet(parquet_path)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True, errors="coerce")
    row_count = int(len(tidy))
    unique_hours = int(tidy["ts"].nunique())
    latest_ts = tidy["ts"].max().isoformat() if tidy["ts"].notna().any() else None
    gcs_uploads = None
    if spot_bucket:
        try:
            gcs_uploads = upload_spot_history_to_gcs(
                bucket=spot_bucket,
                symbol=symbol,
                interval=interval,
                hours=hours,
            )
        except Exception:
            raise
    else:
        log_event("spot_ingest.skipped", note="Spot bucket not provided; skipping GCS upload")
    log_event(
        "binance_ingest.complete",
        path=parquet_path,
        rows=row_count,
        unique_hours=unique_hours,
        latest=latest_ts,
    )
    return {
        "path": parquet_path,
        "rows": row_count,
        "unique_hours": unique_hours,
        "latest": latest_ts,
        "gcs_uploads": gcs_uploads,
    }


def run_feature_builders(run_without_funding: bool) -> Dict[str, Dict[str, Any]]:
    log_event("feature_builders.start", run_without_funding=run_without_funding)
    outputs: Dict[str, Dict[str, Any]] = {}

    alpha_catalog_result: List[Dict[str, Any]] | None = None
    try:
        alpha_catalog_result = ingest_alpha_vantage_catalog()
    except AlphaVantageInvalidKeyError as exc:
        log_event(
            "feature_builders.error",
            note="Alpha Vantage API keys invalid; skipping catalog",
            error=str(exc),
        )
    except AlphaVantageRateLimitError as exc:
        log_event(
            "feature_builders.warning",
            note="Alpha Vantage rate limit encountered; catalog ingestion deferred",
            error=str(exc),
            wait_seconds=getattr(exc, "wait_seconds", None),
        )
    except AlphaVantageIngestionError as exc:
        log_event(
            "feature_builders.warning",
            note="Alpha Vantage catalog ingestion skipped",
            error=str(exc),
        )

    if alpha_catalog_result:
        outputs["alpha_vantage"] = {
            "calls": len(alpha_catalog_result),
            "success": sum(1 for row in alpha_catalog_result if row.get("status") == "success"),
            "latest_timestamps": {
                f"{row['symbol']}_{row['function']}": row.get("latest_timestamp")
                for row in alpha_catalog_result
                if row.get("status") == "success"
            },
        }

    market_path, funding_optional = process_coinapi_features()
    outputs["coinapi_market"] = {"path": market_path, **_summarize_parquet(market_path)}
    if funding_optional:
        outputs["coinapi_funding"] = {"path": funding_optional, **_summarize_parquet(Path(funding_optional))}
    else:
        log_event("feature_builders.warning", note="CoinAPI funding parquet missing; continuing with market-only data")

    macro_path = process_macro_features()
    outputs["macro"] = {"path": macro_path, **_summarize_parquet(macro_path)}

    cq_path = process_cryptoquant_resampled()
    outputs["cryptoquant"] = {"path": cq_path, **_summarize_parquet(cq_path)}

    onchain_path = process_onchain_features()
    outputs["onchain"] = {"path": onchain_path, **_summarize_parquet(onchain_path)}

    try:
        funding_path = process_funding_features(
            pair="BTCUSDT",
            live_fetch=False,
            live_limit=1000,
            allow_missing=run_without_funding,
        )
    except FundingProcessingError as exc:
        log_event(
            "feature_builders.warning",
            note="Funding features missing; rerunning with allow_missing",
            error=str(exc),
        )
        funding_path = process_funding_features(
            pair="BTCUSDT",
            live_fetch=False,
            live_limit=1000,
            allow_missing=True,
        )
    outputs["funding"] = {"path": funding_path, **_summarize_parquet(funding_path)}

    log_event("feature_builders.complete", outputs=outputs)
    return outputs


def _bigquery_available() -> bool:
    if bigquery is None or google is None:
        return False
    try:
        google.auth.default()
    except DefaultCredentialsError:
        return False
    return True


def run_bigquery_refresh(sql_path: Path) -> Optional[Dict[str, Any]]:
    log_event("bigquery_refresh.start", sql_path=sql_path)
    if bigquery is None or google is None:
        log_event("bigquery_refresh.warning", note="google-cloud-bigquery not installed; skipping")
        return None
    if not sql_path.exists():
        log_event("bigquery_refresh.warning", note="SQL file missing; skipping reload")
        return None
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except DefaultCredentialsError as exc:
        log_event("bigquery_refresh.warning", note="Credentials not available", error=str(exc))
        return None
    try:
        sql_text = sql_path.read_text()
        job = client.query(sql_text)
        job.result()
        table_ref = f"{PROJECT_ID}.{BQ_DATASET_CURATED}.{BQ_TABLE_FEATURES_1H}"
        table = client.get_table(table_ref)
        row_count = int(table.num_rows)
        latest_result = client.query(
            f"SELECT ts, close FROM `{table_ref}` ORDER BY ts DESC LIMIT 1",
        ).result()
        latest_row = next(iter(latest_result), None)
        latest_ts = latest_close = None
        if latest_row is not None:
            latest_ts = latest_row.ts
            latest_close = float(latest_row.close)
        summary = {
            "table": table_ref,
            "row_count": row_count,
            "latest_timestamp": latest_ts,
            "latest_close": latest_close,
        }
        log_event("bigquery_refresh.complete", **summary)
        return summary
    except GoogleAPIError as exc:
        log_event("bigquery_refresh.error", error=str(exc))
        raise


def rebuild_datasets(horizons: Iterable[int]) -> Dict[str, Dict[str, Any]]:
    log_event("dataset_build.start", horizons=list(horizons))
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    build_1h_dataset(str(DATASET_DIR))
    summary_1h = _summarize_npz(DATASET_1H_PATH)

    multi_path = build_multi_horizon_dataset(
        output_dir=str(DATASET_DIR),
        horizons=horizons,
        train_frac=0.7,
        val_frac=0.15,
        onchain_path=None,
        fetch_onchain=False,
        onchain_interval="1h",
        features_path=None,
        output_path=str(DATASET_MULTI_PATH),
    )
    summary_multi = _summarize_npz(Path(multi_path))

    payload = {
        "dataset_1h": {"path": DATASET_1H_PATH, **summary_1h},
        "dataset_multi": {"path": multi_path, **summary_multi},
    }
    log_event("dataset_build.complete", **payload)
    return payload


def load_curated_latest(window_hours: int = 720) -> Optional[Dict[str, Any]]:
    where_clause = f"ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_hours} HOUR)"
    try:
            df = load_btc_features_1h()
    except Exception as exc:  # pragma: no cover - guard against missing credentials
        log_event("curated_latest.warning", note="Failed to load curated table", error=str(exc))
        return None
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "close"])
    if df.empty:
        return None
    min_ts = datetime.now(timezone.utc) - pd.Timedelta(hours=window_hours)
    df = df[df["ts"] >= min_ts]
    if df.empty:
        return None
    latest_row = df.sort_values("ts").iloc[-1]
    return {
        "timestamp": latest_row["ts"],
        "close": float(latest_row["close"]),
        "row_count": int(len(df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate Binance US spot + feature refresh workflow.")
    parser.add_argument(
        "--hours",
        type=int,
        default=720,
        help="Number of hourly candles to fetch from Binance US (default: 720).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 4, 8, 12],
        help="Prediction horizons to include when rebuilding datasets (default: 1 4 8 12).",
    )
    parser.add_argument(
        "--run-without-funding",
        action="store_true",
        help="Force RUN_WITHOUT_FUNDING=1 for this refresh run.",
    )
    parser.add_argument(
        "--skip-futures",
        action="store_true",
        help="Disable Binance futures ingestion step.",
    )
    parser.add_argument(
        "--spot-bucket",
        default=os.getenv("SPOT_GCS_BUCKET"),
        help="GCS bucket for spot klines uploads (default: SPOT_GCS_BUCKET env).",
    )
    parser.add_argument(
        "--futures-bucket",
        default=os.getenv("FUTURES_GCS_BUCKET"),
        help="GCS bucket for futures parquet uploads (default: FUTURES_GCS_BUCKET env).",
    )
    parser.add_argument(
        "--futures-interval",
        default="1h",
        help="Futures interval to ingest (default: 1h).",
    )
    parser.add_argument(
        "--futures-days",
        type=int,
        default=DEFAULT_FUTURES_DAYS,
        help="Number of UTC days (including today) to ingest for futures metrics (default: 2).",
    )
    parser.add_argument(
        "--futures-pair",
        default="BTCUSDT",
        help="Perpetual futures pair symbol (default: BTCUSDT).",
    )
    parser.add_argument(
        "--futures-market",
        default=os.getenv("CRYPTOCOMPARE_MARKET"),
        help="Optional CryptoCompare market override (default: env CRYPTOCOMPARE_MARKET).",
    )
    parser.add_argument(
        "--futures-instrument",
        default=os.getenv("CRYPTOCOMPARE_INSTRUMENT"),
        help="Optional CryptoCompare instrument override (default: env CRYPTOCOMPARE_INSTRUMENT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_without_funding = args.run_without_funding or os.getenv("RUN_WITHOUT_FUNDING") in {"1", "true", "yes", "on"}
    if resolved_without_funding:
        os.environ["RUN_WITHOUT_FUNDING"] = "1"
    log_event("refresh.start", hours=args.hours, horizons=args.horizons, run_without_funding=resolved_without_funding)

    futures_info = None
    if not args.skip_futures:
        futures_info = run_futures_ingest(
            bucket=args.futures_bucket,
            interval=args.futures_interval,
            pair=args.futures_pair,
            days=args.futures_days,
            market=args.futures_market,
            instrument=args.futures_instrument,
            api_key=os.getenv("CRYPTOCOMPARE_API_KEY"),
        )

    ingest_info = run_binance_ingest(args.hours, spot_bucket=args.spot_bucket)
    feature_info = run_feature_builders(run_without_funding=resolved_without_funding)

    bq_info = run_bigquery_refresh(SQL_REFRESH_PATH)
    dataset_info = rebuild_datasets(args.horizons)

    latest_curated = load_curated_latest()

    final_summary = {
        "latest_curated": _stringify(latest_curated) if latest_curated else None,
        "dataset_artifacts": {
            "dataset_1h": str(DATASET_1H_PATH),
            "dataset_multi": str(DATASET_MULTI_PATH),
        },
        "ingest_path": str(ingest_info["path"]),
        "spot_uploads": ingest_info.get("gcs_uploads"),
        "futures": futures_info,
    }
    log_event(
        "refresh.complete",
        latest_curated=final_summary["latest_curated"],
        dataset_artifacts=final_summary["dataset_artifacts"],
        ingest_path=final_summary["ingest_path"],
        spot_uploads=_stringify(ingest_info.get("gcs_uploads")) if ingest_info.get("gcs_uploads") else None,
        bigquery_refreshed=bool(bq_info),
        feature_outputs=feature_info,
    )

    if latest_curated:
        latest_ts = latest_curated["timestamp"].astimezone(timezone.utc).isoformat() if isinstance(latest_curated["timestamp"], datetime) else latest_curated["timestamp"]
        latest_close = latest_curated["close"]
        print(
            f"Latest curated ts: {latest_ts} | close: {latest_close} | datasets: {DATASET_1H_PATH}, {DATASET_MULTI_PATH}",
        )
    else:
        print(
            f"Datasets refreshed at {iso_now()}; latest curated metrics unavailable. Datasets: {DATASET_1H_PATH}, {DATASET_MULTI_PATH}",
        )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
