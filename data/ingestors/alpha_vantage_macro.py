"""Alpha Vantage macro/market data ingestion helper."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

ALPHA_URL = "https://www.alphavantage.co/query"
RAW_ROOT = Path("data/raw/macro/alpha_vantage")
SUPPORTED_FUNCTIONS = {
    "TIME_SERIES_DAILY": {"default_interval": None},
    "TIME_SERIES_INTRADAY": {"default_interval": "60min"},
}


class AlphaVantageIngestionError(RuntimeError):
    """Raised when the Alpha Vantage API call fails."""


def _require_api_key() -> str:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        raise AlphaVantageIngestionError(
            "ALPHA_VANTAGE_API_KEY is not set; export the key before running the loader.",
        )
    return api_key


def _build_params(
    function: str,
    symbol: str,
    api_key: str,
    interval: str | None,
) -> Dict[str, str]:
    params: Dict[str, str] = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "datatype": "json",
    }
    if interval:
        params["interval"] = interval
    return params


def _flatten_time_series(symbol: str, payload: dict) -> pd.DataFrame:
    time_series = None
    for key, value in payload.items():
        if key.startswith("Time Series"):
            time_series = value
            break
    if not isinstance(time_series, dict):
        raise AlphaVantageIngestionError("Response did not contain a time series block.")

    rows: list[dict[str, object]] = []
    for timestamp_str, fields in time_series.items():
        ts = pd.Timestamp(timestamp_str, tz="UTC")
        if isinstance(fields, dict):
            for field, value in fields.items():
                clean_field = field.split(". ", maxsplit=1)[-1]
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = None
                rows.append({
                    "ts": ts,
                    "metric": f"{symbol}_{clean_field}",
                    "value": numeric_value,
                    "source": "alpha_vantage",
                })

    if not rows:
        raise AlphaVantageIngestionError("Flattened Alpha Vantage response was empty.")

    frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return frame


def ingest_series(
    function: str,
    symbol: str,
    interval: str | None = None,
    output_root: Path = RAW_ROOT,
) -> Path:
    if function not in SUPPORTED_FUNCTIONS:
        supported = ", ".join(SUPPORTED_FUNCTIONS)
        raise AlphaVantageIngestionError(f"Unsupported function '{function}'. Supported: {supported}")

    api_key = _require_api_key()
    interval = interval or SUPPORTED_FUNCTIONS[function]["default_interval"]
    params = _build_params(function, symbol, api_key, interval)

    try:
        response = requests.get(ALPHA_URL, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise AlphaVantageIngestionError(f"Alpha Vantage request failed: {exc}") from exc

    if response.status_code != 200:
        raise AlphaVantageIngestionError(
            f"Alpha Vantage request returned status {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()
    if payload.get("Error Message"):
        raise AlphaVantageIngestionError(payload["Error Message"])
    if payload.get("Note"):
        raise AlphaVantageIngestionError(payload["Note"])

    frame = _flatten_time_series(symbol, payload)

    remaining = response.headers.get("X-RateLimit-Remaining")
    limit = response.headers.get("X-RateLimit-Limit")
    if remaining or limit:
        print(f"Alpha Vantage quota remaining {remaining}/{limit} (function={function}, interval={interval})")
    else:
        print("Alpha Vantage quota headers unavailable; premium key in use.")
    sample = frame.head(5)
    print("Alpha Vantage sample rows:\n", sample)

    output_dir = output_root / f"symbol={symbol}" / function.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"alpha_{symbol}_{function.lower()}_{timestamp}.parquet"
    frame.to_parquet(output_path, index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage data into Parquet.")
    parser.add_argument("function", choices=sorted(SUPPORTED_FUNCTIONS), help="Alpha Vantage function")
    parser.add_argument("symbol", help="Ticker symbol (e.g. SPY, DXY).")
    parser.add_argument(
        "--interval",
        default=None,
        help="Interval for intraday data (e.g. 60min). Required only for TIME_SERIES_INTRADAY.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw Parquet output (default: data/raw/macro/alpha_vantage).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = ingest_series(
        function=args.function,
        symbol=args.symbol,
        interval=args.interval,
        output_root=args.output_root,
    )
    print(f"Saved Alpha Vantage {args.function} {args.symbol} data to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
