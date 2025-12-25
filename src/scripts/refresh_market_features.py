from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from data.processed.compute_funding_features import (
    SUMMARY_PATH as FUNDING_SUMMARY_PATH,
    process_funding_features,
)
from data.ingestors.cryptocompare_onchain import ingest_metrics as ingest_cryptocompare_metrics
from data.processed.compute_onchain_features import (
    SUMMARY_PATH as ONCHAIN_SUMMARY_PATH,
    process_onchain_features,
)
from data.processed.compute_technical_features import (
    SUMMARY_PATH as TECHNICAL_SUMMARY_PATH,
    process_technical_features,
)
from src.config import ONCHAIN_METRICS

ONCHAIN_RAW_ROOT = Path("data/raw/onchain/cryptocompare")
FUNDING_RAW_ROOT = Path("data/raw/funding/binance")
FUNDING_OUTPUT_PATH = Path("data/processed/funding/hourly_features.parquet")
TECHNICAL_HISTORY_LIMIT = 5000
DEFAULT_ONCHAIN_LIMIT = 720
DEFAULT_FUNDING_LIMIT = 1000
FUNDING_PAIR = "BTCUSDT"


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


def _refresh_onchain(limit: int) -> Dict[str, Any]:
    metrics = list(ONCHAIN_METRICS)
    raw_paths = [
        str(path)
        for path in ingest_cryptocompare_metrics(
            metrics=metrics,
            limit=limit,
            output_root=ONCHAIN_RAW_ROOT,
            api_key=None,
        )
    ]

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


def _refresh_funding(limit: int) -> Dict[str, Any]:
    features_path = process_funding_features(
        pair=FUNDING_PAIR,
        live_fetch=True,
        live_limit=limit,
        allow_missing=False,
        raw_root=FUNDING_RAW_ROOT,
        output_path=FUNDING_OUTPUT_PATH,
    )
    summary = _load_json(FUNDING_SUMMARY_PATH)
    return {
        "pair": FUNDING_PAIR,
        "live_limit": limit,
        "features_path": str(features_path),
        "summary_path": str(FUNDING_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
    }


def _refresh_technical() -> Dict[str, Any]:
    features_path = process_technical_features(include_history=True, history_limit=TECHNICAL_HISTORY_LIMIT)
    summary = _load_json(TECHNICAL_SUMMARY_PATH)
    return {
        "include_history": True,
        "history_limit": TECHNICAL_HISTORY_LIMIT,
        "features_path": str(features_path),
        "summary_path": str(TECHNICAL_SUMMARY_PATH),
        "latest_timestamp": _latest_timestamp(summary),
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "parameters": {
            "skip_onchain": bool(args.skip_onchain),
            "skip_funding": bool(args.skip_funding),
            "onchain_limit": int(args.onchain_limit),
            "funding_limit": int(args.funding_limit),
        },
    }

    try:
        if not args.skip_onchain:
            report["onchain"] = _refresh_onchain(args.onchain_limit)
        if not args.skip_funding:
            report["funding"] = _refresh_funding(args.funding_limit)
        report["technical"] = _refresh_technical()
    except Exception as exc:  # pragma: no cover - surfaced via tests
        print(f"refresh_market_features failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
