"""CoinAPI loader for BTCUSDT spot, futures, and funding metrics."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import quote

import pandas as pd
import requests

COINAPI_URL = "https://rest.coinapi.io/v1"
RAW_MARKET_ROOT = Path("data/raw/market/coinapi")
RAW_FUNDING_ROOT = Path("data/raw/funding/coinapi")
FUNDING_FAILURE_PATH = Path("artifacts/monitoring/coinapi_funding_failure.json")
SPOT_SYMBOL_DEFAULT = "BINANCE_SPOT_BTC_USDT"
FUTURES_SYMBOL_DEFAULT = "BINANCEFTS_PERP_BTC_USDT"
FUNDING_SYMBOL_DEFAULT = "BINANCEFTS_PERP_BTC_USDT"


class CoinAPIIngestionError(RuntimeError):
    """Raised when a CoinAPI request fails."""


def _require_api_key() -> str:
    api_key = os.getenv("COINAPI_KEY", "").strip()
    if not api_key:
        raise CoinAPIIngestionError("COINAPI_KEY is not set; export the key before running the loader.")
    return api_key


def _headers(api_key: str) -> dict[str, str]:
    return {"X-CoinAPI-Key": api_key}


def _fetch_ohlcv(symbol_id: str, api_key: str, period_id: str, limit: int) -> List[dict]:
    url = f"{COINAPI_URL}/ohlcv/{symbol_id}/history"
    params = {"period_id": period_id, "limit": limit}
    try:
        response = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise CoinAPIIngestionError(f"CoinAPI ohlcv request failed: {exc}") from exc

    if response.status_code != 200:
        raise CoinAPIIngestionError(
            f"CoinAPI ohlcv request returned status {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise CoinAPIIngestionError("Unexpected OHLCV payload from CoinAPI.")
    return payload


def _discover_funding_symbol(symbol_hint: str, api_key: str) -> Tuple[str, List[dict]]:
    url = f"{COINAPI_URL}/symbols"
    params = {"filter_symbol_id": symbol_hint}
    try:
        response = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise CoinAPIIngestionError(f"CoinAPI symbol discovery failed: {exc}") from exc

    if response.status_code != 200:
        raise CoinAPIIngestionError(
            f"CoinAPI symbol discovery returned status {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise CoinAPIIngestionError("CoinAPI symbol discovery returned no entries; cannot resolve funding symbol.")

    normalized_hint = symbol_hint.lower()
    canonical = None
    for entry in payload:
        symbol_id = str(entry.get("symbol_id", ""))
        if symbol_id.lower() == normalized_hint:
            canonical = symbol_id
            break
        if canonical is None and symbol_id:
            canonical = symbol_id

    if not canonical:
        raise CoinAPIIngestionError("Unable to resolve canonical CoinAPI symbol id.")

    print(f"Resolved CoinAPI funding symbol {symbol_hint} -> {canonical}")
    return canonical, payload


def _record_funding_failure(
    symbol_hint: str,
    canonical_symbol: str,
    request_ts: str,
    status_code: int,
    response_text: str,
    symbols_payload: List[dict],
) -> None:
    diagnostics = {
        "requested_symbol_hint": symbol_hint,
        "resolved_symbol_id": canonical_symbol,
        "request_timestamp_utc": request_ts,
        "status_code": status_code,
        "response_text": response_text,
        "symbols_payload": symbols_payload,
    }
    FUNDING_FAILURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FUNDING_FAILURE_PATH.write_text(json.dumps(diagnostics, indent=2))
    print(
        "CoinAPI funding request failed; wrote diagnostics to "
        f"{FUNDING_FAILURE_PATH} for support follow-up.",
    )


def _fetch_funding(symbol_id_hint: str, api_key: str, limit: int) -> Tuple[str, List[dict]]:
    canonical_symbol, symbols_payload = _discover_funding_symbol(symbol_id_hint, api_key)
    encoded_symbol = quote(canonical_symbol, safe="")
    url = f"{COINAPI_URL}/futures/funding_rates/{encoded_symbol}"
    params = {"limit": limit}
    request_ts = datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        response = requests.get(url, headers=_headers(api_key), params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise CoinAPIIngestionError(f"CoinAPI funding request failed: {exc}") from exc

    raw_text = response.text
    print(
        "CoinAPI funding request",
        {
            "symbol_id": canonical_symbol,
            "encoded_path": encoded_symbol,
            "status": response.status_code,
        },
    )

    if response.status_code == 404:
        _record_funding_failure(symbol_id_hint, canonical_symbol, request_ts, response.status_code, raw_text, symbols_payload)
        return canonical_symbol, []
    if response.status_code != 200:
        raise CoinAPIIngestionError(
            f"CoinAPI funding request returned status {response.status_code}: {raw_text[:256]}",
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise CoinAPIIngestionError("Unexpected funding payload from CoinAPI.")
    return canonical_symbol, payload


def _ohlcv_to_tidy(symbol_id: str, kind: str, records: Iterable[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        ts = record.get("time_period_end") or record.get("time_period_start")
        if not ts:
            continue
        try:
            ts_parsed = pd.Timestamp(ts, tz="UTC")
        except ValueError:
            continue
        metrics = {
            f"{kind}_open": record.get("price_open"),
            f"{kind}_high": record.get("price_high"),
            f"{kind}_low": record.get("price_low"),
            f"{kind}_close": record.get("price_close"),
            f"{kind}_volume": record.get("volume_traded"),
        }
        for metric, value in metrics.items():
            rows.append({
                "ts": ts_parsed,
                "metric": metric,
                "value": float(value) if value is not None else None,
                "source": "coinapi",
                "symbol_id": symbol_id,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise CoinAPIIngestionError(f"No OHLCV observations returned for {symbol_id}.")
    return frame


def _funding_to_tidy(symbol_id: str, records: Iterable[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        ts = record.get("time") or record.get("time_exchange")
        if not ts:
            continue
        try:
            ts_parsed = pd.Timestamp(ts, tz="UTC")
        except ValueError:
            continue
        rate = record.get("funding_rate")
        rows.append({
            "ts": ts_parsed,
            "metric": "funding_rate",
            "value": float(rate) if rate is not None else None,
            "source": "coinapi",
            "symbol_id": symbol_id,
        })
    return pd.DataFrame(rows)


def _write_parquet(frame: pd.DataFrame, output_root: Path, entity: str, symbol_id: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"entity={entity}" / f"symbol={symbol_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coinapi_{entity}_{symbol_id}_{timestamp}.parquet"
    frame.to_parquet(output_path, index=False)
    return output_path


def ingest_coinapi(
    period_id: str,
    limit: int,
    spot_symbol: str = SPOT_SYMBOL_DEFAULT,
    futures_symbol: str = FUTURES_SYMBOL_DEFAULT,
    funding_symbol: str = FUNDING_SYMBOL_DEFAULT,
) -> Tuple[List[Path], List[Path]]:
    api_key = _require_api_key()
    market_paths: List[Path] = []
    funding_paths: List[Path] = []

    spot_records = _fetch_ohlcv(spot_symbol, api_key, period_id, limit)
    spot_frame = _ohlcv_to_tidy(spot_symbol, "spot", spot_records)
    market_paths.append(_write_parquet(spot_frame, RAW_MARKET_ROOT, "spot", spot_symbol))
    print(f"Saved {len(spot_frame)} spot metric rows to {market_paths[-1]}")

    futures_records = _fetch_ohlcv(futures_symbol, api_key, period_id, limit)
    futures_frame = _ohlcv_to_tidy(futures_symbol, "futures", futures_records)
    market_paths.append(_write_parquet(futures_frame, RAW_MARKET_ROOT, "futures", futures_symbol))
    print(f"Saved {len(futures_frame)} futures metric rows to {market_paths[-1]}")

    resolved_symbol, funding_records = _fetch_funding(funding_symbol, api_key, limit)
    if funding_records:
        funding_frame = _funding_to_tidy(resolved_symbol, funding_records)
        if not funding_frame.empty:
            funding_paths.append(_write_parquet(funding_frame, RAW_FUNDING_ROOT, "funding", resolved_symbol))
            print(f"Saved {len(funding_frame)} funding records to {funding_paths[-1]}")
        else:
            print("CoinAPI funding payload empty; nothing written.")

    return market_paths, funding_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch BTCUSDT market and funding data from CoinAPI.")
    parser.add_argument("--period-id", default="1HRS", help="Period identifier (default: 1HRS).")
    parser.add_argument("--limit", type=int, default=720, help="Number of rows to fetch (default: 720).")
    parser.add_argument("--spot-symbol", default=SPOT_SYMBOL_DEFAULT, help="CoinAPI symbol ID for spot data.")
    parser.add_argument("--futures-symbol", default=FUTURES_SYMBOL_DEFAULT, help="CoinAPI symbol ID for futures data.")
    parser.add_argument("--funding-symbol", default=FUNDING_SYMBOL_DEFAULT, help="CoinAPI symbol ID for funding data.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ingest_coinapi(
        period_id=args.period_id,
        limit=args.limit,
        spot_symbol=args.spot_symbol,
        futures_symbol=args.futures_symbol,
        funding_symbol=args.funding_symbol,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
