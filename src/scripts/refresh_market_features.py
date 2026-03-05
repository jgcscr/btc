from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from data.processed.compute_funding_features import (
    SUMMARY_PATH as FUNDING_SUMMARY_PATH,
    FundingProcessingError,
    process_funding_features,
)
from data.ingestors.cryptocompare_onchain import (
    CryptoCompareIngestionError,
    ingest_metrics as ingest_cryptocompare_metrics,
)
from data.processed.compute_onchain_features import (
    SUMMARY_PATH as ONCHAIN_SUMMARY_PATH,
    process_onchain_features,
)
from data.processed.compute_macro_features import (
    OUTPUT_PATH as MACRO_FEATURES_PATH,
    SUMMARY_PATH as MACRO_SUMMARY_PATH,
    process_macro_features,
)
from data.ingestors.tiingo_spot import ingest_tiingo_spot
from data.processed.compute_technical_features import (
    SUMMARY_PATH as TECHNICAL_SUMMARY_PATH,
    process_technical_features,
    resolve_price_source_spec,
)
from src.data.binance_klines import BinanceAPIError
from src.config import ONCHAIN_METRICS
from src.scripts.build_onchain_fallback import build_onchain_fallback
from src.scripts.build_macro_fallback import build_macro_fallback

ONCHAIN_RAW_ROOT = Path("data/raw/onchain/cryptocompare")
FUNDING_RAW_ROOT = Path("data/raw/funding/binance")
FUNDING_OUTPUT_PATH = Path("data/processed/funding/hourly_features.parquet")
MACRO_CHAIN_PATH = Path("artifacts/monitoring/macro_chain_comparison.json")
TECHNICAL_HISTORY_LIMIT = 5000
DEFAULT_ONCHAIN_LIMIT = 720
DEFAULT_FUNDING_LIMIT = 1000
FUNDING_PAIR = "BTCUSDT"
DEFAULT_FUNDING_PROVIDER = "binance"
DEFAULT_TIINGO_LOOKBACK_DAYS = 7
DEFAULT_ONCHAIN_SOURCE = "cryptocompare"
SAMPLE_ROOT = Path("tmp/kaiko_dry_run")
KAIKO_OHLCV_SAMPLE = SAMPLE_ROOT / "kaiko_ohlcv.parquet"
TWELVEDATA_PREMIUM_SAMPLE = SAMPLE_ROOT / "twelvedata_premium.parquet"


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh funding, on-chain, and technical feature artifacts.",
    )
    parser.add_argument(
        "--skip-funding",
        action="store_true",
        help="Skip fetching and processing funding rate features.",
    )
    parser.add_argument(
        "--skip-onchain",
        action="store_true",
        help="Skip ingesting on-chain metrics and rebuilding on-chain features.",
    )
    parser.add_argument(
        "--onchain-source",
        choices=("cryptocompare", "fallback"),
        default=DEFAULT_ONCHAIN_SOURCE,
        help=(
            "Choose the on-chain feed: 'cryptocompare' pulls vendor data, 'fallback' synthesizes metrics using "
            "spot/funding proxies when the upstream feed lags (default: cryptocompare)."
        ),
    )
    parser.add_argument(
        "--onchain-limit",
        type=int,
        default=DEFAULT_ONCHAIN_LIMIT,
        help=f"Number of recent observations to request from CryptoCompare (default: {DEFAULT_ONCHAIN_LIMIT}).",
    )
    parser.add_argument(
        "--funding-limit",
        type=int,
        default=DEFAULT_FUNDING_LIMIT,
        help=f"Maximum funding records to fetch from Binance (default: {DEFAULT_FUNDING_LIMIT}).",
    )
    parser.add_argument(
        "--funding-provider",
        choices=("binance", "cryptocompare"),
        default=DEFAULT_FUNDING_PROVIDER,
        help="Funding data provider to use (default: binance).",
    )
    parser.add_argument(
        "--technical-price-source",
        default="curated",
        help=(
            "Path or keyword describing the technical price source (curated, binanceus, tiingo, kaiko). "
            "When set to tiingo, the script ingests fresh Tiingo spot candles; when set to kaiko (or kaiko_sample) "
            "it expects tmp/kaiko_dry_run/kaiko_ohlcv.parquet from the sample ingestor."
        ),
    )
    parser.add_argument(
        "--tiingo-lookback-days",
        type=int,
        default=DEFAULT_TIINGO_LOOKBACK_DAYS,
        help="Lookback window (days) for Tiingo spot ingestion when requested (default: 7).",
    )
    parser.add_argument(
        "--macro-source",
        choices=("vendor", "fallback", "kaiko", "kaiko_sample", "twelvedata", "twelvedata_sample"),
        default="vendor",
        help=(
            "Select the macro feature feed: vendor (Alpha/Twelve/FRED blend), fallback (synthetic), "
            "kaiko (_sample) to reuse tmp/kaiko_dry_run/kaiko_ohlcv.parquet, or twelvedata (_sample) "
            "to reuse tmp/kaiko_dry_run/twelvedata_premium.parquet."
        ),
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _latest_timestamp(summary: Dict[str, Any]) -> Any:
    for key in ("latest_timestamp", "latest", "max_timestamp"):
        if key in summary:
            return summary[key]
    return None

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _annotate_summary(path: Path, summary: Dict[str, Any] | None, provider: str) -> Dict[str, Any]:
    payload = dict(summary or {})
    payload["provider"] = provider
    payload.setdefault("source", provider)
    _write_json(path, payload)
    return payload


def _ensure_sample_parquet(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Run the Kaiko/Twelve sample ingestors before selecting this provider."
        )
    return path


def _infer_latest_timestamp_from_parquet(path: Path) -> str | None:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.empty:
        return None
    timestamp_column = None
    for candidate in ("timestamp", "ts", "time"):
        if candidate in frame.columns:
            timestamp_column = candidate
            break
    if timestamp_column is None:
        return None
    series = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").dropna()
    if series.empty:
        return None
    return series.max().isoformat()


def _normalize_macro_choice(choice: str) -> str:
    lowered = choice.lower()
    if lowered.startswith("kaiko"):
        return "kaiko"
    if lowered.startswith("twelvedata"):
        return "twelvedata"
    return lowered


def _macro_from_sample(provider: str) -> Dict[str, Any]:
    sample_map = {
        "kaiko": KAIKO_OHLCV_SAMPLE,
        "twelvedata": TWELVEDATA_PREMIUM_SAMPLE,
    }
    sample_path = _ensure_sample_parquet(sample_map[provider], f"{provider.title()} sample parquet")
    MACRO_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample_path, MACRO_FEATURES_PATH)
    latest_ts = _infer_latest_timestamp_from_parquet(MACRO_FEATURES_PATH)
    summary = {
        "provider": provider,
        "source": provider,
        "sample_path": str(sample_path),
        "latest_timestamp": latest_ts,
    }
    _write_json(MACRO_SUMMARY_PATH, summary)
    return {
        "source": provider,
        "provider": provider,
        "features_path": str(MACRO_FEATURES_PATH),
        "summary_path": str(MACRO_SUMMARY_PATH),
        "latest_timestamp": latest_ts,
        "summary": summary,
    }


def _refresh_onchain(limit: int, source: str) -> Dict[str, Any]:
    def _onchain_fallback(reason: str | None = None) -> Dict[str, Any]:
        fallback = build_onchain_fallback(
            history_hours=limit,
            summary_path=ONCHAIN_SUMMARY_PATH,
            output_path=Path("data/processed/onchain/hourly_features.parquet"),
        )
        summary = dict(fallback["summary"])
        if reason:
            summary["fallback_reason"] = reason
            _write_json(Path(fallback["summary_path"]), summary)
        return {
            "source": "fallback",
            "features_path": fallback["features_path"],
            "summary_path": fallback["summary_path"],
            "latest_timestamp": fallback["latest_timestamp"],
            "summary": summary,
            "metrics": list(ONCHAIN_METRICS),
            "raw_paths": [],
        }

    if source == "fallback":
        return _onchain_fallback()

    metrics = list(ONCHAIN_METRICS)
    try:
        raw_paths = [
            str(path)
            for path in ingest_cryptocompare_metrics(
                metrics=metrics,
                limit=limit,
                output_root=ONCHAIN_RAW_ROOT,
                api_key=None,
            )
        ]
    except CryptoCompareIngestionError as exc:
        message = str(exc)
        logger.warning(
            "refresh_market_features: CryptoCompare ingest failed (%s); falling back to synthesized on-chain metrics.",
            message,
        )
        print(
            "refresh_market_features: on-chain ingest failed; falling back to synthesized metrics: " f"{message}",
            file=sys.stderr,
        )
        return _onchain_fallback(reason=message)

    features_path = process_onchain_features()
    summary = _load_json(ONCHAIN_SUMMARY_PATH)
    return {
        "metrics": metrics,
        "limit": limit,
        "raw_paths": raw_paths,
        "features_path": str(features_path),
        "summary_path": str(ONCHAIN_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
    }


def _refresh_macro(source: str) -> Dict[str, Any]:
    normalized = _normalize_macro_choice(source)
    if normalized in {"kaiko", "twelvedata"}:
        return _macro_from_sample(normalized)

    if normalized == "fallback":
        result = build_macro_fallback(
            promote_features_path=MACRO_FEATURES_PATH,
            promote_summary_path=MACRO_SUMMARY_PATH,
            macro_chain_path=MACRO_CHAIN_PATH,
        )
        summary = _annotate_summary(MACRO_SUMMARY_PATH, result.get("summary"), "fallback")
        result["summary"] = summary
        result["provider"] = "fallback"
        return {
            "source": "fallback",
            "provider": "fallback",
            "features_path": str(MACRO_FEATURES_PATH),
            "summary_path": str(MACRO_SUMMARY_PATH),
            "latest_timestamp": result["latest_timestamp"],
            "summary": summary,
        }

    features_path = process_macro_features()
    summary = _annotate_summary(MACRO_SUMMARY_PATH, _load_json(MACRO_SUMMARY_PATH), "vendor")
    return {
        "source": "vendor",
        "provider": "vendor",
        "features_path": str(features_path),
        "summary_path": str(MACRO_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
    }


def _refresh_funding(limit: int, provider: str) -> Dict[str, Any]:
    try:
        return _run_funding_job(limit, allow_missing=False, live_fetch=True, provider=provider)
    except FundingProcessingError as exc:
        return _funding_fallback(limit, exc, provider)
    except BinanceAPIError as exc:
        message = str(exc)
        if not _is_http_client_error(message):
            raise
        return _funding_fallback(limit, exc, provider)


def _run_funding_job(limit: int, *, allow_missing: bool, live_fetch: bool, provider: str) -> Dict[str, Any]:
    features_path = process_funding_features(
        pair=FUNDING_PAIR,
        live_fetch=live_fetch,
        live_limit=limit,
        allow_missing=allow_missing,
        provider=provider,
        raw_root=FUNDING_RAW_ROOT,
        output_path=FUNDING_OUTPUT_PATH,
    )
    summary = _load_json(FUNDING_SUMMARY_PATH)
    status = "fallback" if allow_missing else "ok"
    payload: Dict[str, Any] = {
        "pair": FUNDING_PAIR,
        "live_limit": limit,
        "provider": provider,
        "features_path": str(features_path),
        "summary_path": str(FUNDING_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
        "status": status,
    }
    return payload


def _funding_fallback(limit: int, error: Exception, provider: str) -> Dict[str, Any]:
    message = str(error)
    logger.warning(
        "refresh_market_features: funding refresh encountered an error (%s); falling back to allow-missing mode.",
        message,
    )
    print(
        f"refresh_market_features: funding refresh encountered an error ({message}); falling back to allow-missing mode.",
        file=sys.stderr,
    )
    payload = _run_funding_job(limit, allow_missing=True, live_fetch=False, provider=provider)
    payload["error"] = message
    return payload


def _is_http_client_error(message: str) -> bool:
    return bool(re.search(r"\b4\d{2}\b", message))


def _normalize_spec(spec: str | Path | None) -> str:
    if spec is None:
        return "curated"
    if isinstance(spec, Path):
        return str(spec)
    text = spec.strip()
    return text or "curated"


def _refresh_technical(price_source_spec: str | Path | None, tiingo_lookback_days: int) -> Dict[str, Any]:
    ingestion_path: Path | None = None
    resolved_price_path: Path | None = None
    raw_spec = _normalize_spec(price_source_spec)
    normalized = raw_spec.lower()
    provider_label = "curated"
    try:
        if normalized in {"tiingo", "tiingo_spot"}:
            tiingo_path = ingest_tiingo_spot(lookback_days=tiingo_lookback_days)
            ingestion_path = Path(tiingo_path)
            resolved_price_path = ingestion_path
            provider_label = "tiingo"
        elif normalized in {"kaiko", "kaiko_sample"}:
            resolved_price_path = _ensure_sample_parquet(KAIKO_OHLCV_SAMPLE, "Kaiko OHLCV sample")
            provider_label = "kaiko"
        else:
            resolved_price_path = resolve_price_source_spec(price_source_spec)
            if normalized in {"binance", "binanceus", "binance_us"}:
                provider_label = "binanceus"
            elif normalized not in {"curated", "default", "auto"}:
                provider_label = "custom"
    except (ValueError, FileNotFoundError) as exc:
        raise RuntimeError(f"Failed to resolve price source '{price_source_spec}': {exc}") from exc

    features_path = process_technical_features(
        price_source=resolved_price_path,
        include_history=True,
        history_limit=TECHNICAL_HISTORY_LIMIT,
    )
    summary = _annotate_summary(TECHNICAL_SUMMARY_PATH, _load_json(TECHNICAL_SUMMARY_PATH), provider_label)
    return {
        "include_history": True,
        "history_limit": TECHNICAL_HISTORY_LIMIT,
        "features_path": str(features_path),
        "summary_path": str(TECHNICAL_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
        "price_source": _normalize_spec(price_source_spec),
        "price_path": str(resolved_price_path) if resolved_price_path else None,
        "tiingo_ingest_path": str(ingestion_path) if ingestion_path else None,
        "provider": provider_label,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "parameters": {
            "skip_onchain": bool(args.skip_onchain),
            "skip_funding": bool(args.skip_funding),
            "onchain_limit": int(args.onchain_limit),
            "onchain_source": args.onchain_source,
            "funding_limit": int(args.funding_limit),
            "funding_provider": args.funding_provider,
            "technical_price_source": args.technical_price_source,
            "tiingo_lookback_days": int(args.tiingo_lookback_days),
            "macro_source": args.macro_source,
        },
    }

    try:
        # Helper routines may print diagnostics; keep stdout reserved for the final JSON report.
        with redirect_stdout(sys.stderr):
            if not args.skip_onchain:
                report["onchain"] = _refresh_onchain(args.onchain_limit, args.onchain_source)
            if not args.skip_funding:
                report["funding"] = _refresh_funding(args.funding_limit, args.funding_provider)
            report["macro"] = _refresh_macro(args.macro_source)
            report["technical"] = _refresh_technical(
                price_source_spec=args.technical_price_source,
                tiingo_lookback_days=int(args.tiingo_lookback_days),
            )
    except Exception as exc:  # pragma: no cover - surfaced via tests
        print(f"refresh_market_features failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
