"""CryptoQuant daily metrics loader using CQ_TOKEN."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests

CRYPTOQUANT_API_URL = "https://api.cryptoquant.com/v2/data/metrics"
RAW_ROOT = Path("data/raw/cryptoquant_daily")
DEFAULT_SYMBOL = "btc"
DEFAULT_INTERVAL = "day"
DEFAULT_LIMIT = 730  # ~2 years

# Map friendly metric keys to CryptoQuant metric endpoints.
DEFAULT_METRICS: Dict[str, str] = {
    "exchange_reserve": "btc-exchange-reserve",
    "exchange_netflow": "btc-exchange-netflow",
    "stablecoin_reserve": "stablecoin-total-exchange-reserve",
    "miner_flow": "btc-miners-outflow",
    "whale_count": "btc-whale-address-count",
}


class CryptoQuantIngestionError(RuntimeError):
    """Raised when the CryptoQuant API call fails."""


def _require_token() -> str:
    token = os.getenv("CQ_TOKEN", "").strip()
    if not token:
        raise CryptoQuantIngestionError("CQ_TOKEN environment variable is not set; export the API token.")
    return token


def _build_url(metric_endpoint: str) -> str:
    return f"{CRYPTOQUANT_API_URL}/{metric_endpoint}"


def _fetch_metric(
    metric_key: str,
    endpoint: str,
    token: str,
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    headers = {
        "Authorization": f"Bearer {token}",
    }
    url = _build_url(endpoint)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
        raise CryptoQuantIngestionError(f"CryptoQuant request failed for {metric_key}: {exc}") from exc

    if response.status_code != 200:
        raise CryptoQuantIngestionError(
            f"CryptoQuant request for {metric_key} returned {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()
    data: Iterable[dict] | None = None
    # CryptoQuant responses typically place data under "result" -> "data".
    if isinstance(payload, dict):
        if isinstance(payload.get("result"), dict):
            data = payload["result"].get("data")
        if data is None:
            data = payload.get("data")
    if data is None:
        raise CryptoQuantIngestionError(f"Unexpected payload structure for {metric_key}: {payload}")

    rows: List[dict[str, object]] = []
    for row in data:
        ts = row.get("time") or row.get("timestamp") or row.get("date")
        value = row.get("value")
        if ts is None:
            continue
        try:
            ts_parsed = pd.Timestamp(ts, tz="UTC")
        except ValueError:
            continue
        rows.append(
            {
                "ts": ts_parsed,
                "metric": metric_key,
                "endpoint": endpoint,
                "symbol": symbol,
                "interval": interval,
                "value": float(value) if value is not None else None,
                "source": "cryptoquant",
            },
        )

    frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise CryptoQuantIngestionError(f"No data returned for {metric_key}; check metric availability.")
    return frame


def _write_outputs(frame: pd.DataFrame, metric_key: str, output_root: Path) -> Dict[str, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"metric={metric_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"cryptoquant_{metric_key}_{timestamp}.parquet"
    csv_path = output_dir / f"cryptoquant_{metric_key}_{timestamp}.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    return {"parquet": parquet_path, "csv": csv_path}


def ingest_cryptoquant_daily(
    metrics: Dict[str, str],
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    output_root: Path = RAW_ROOT,
) -> Dict[str, Dict[str, Path]]:
    token = _require_token()
    outputs: Dict[str, Dict[str, Path]] = {}
    for key, endpoint in metrics.items():
        frame = _fetch_metric(key, endpoint, token, symbol, interval, limit)
        outputs[key] = _write_outputs(frame, key, output_root)
        print(
            f"Saved CryptoQuant daily metric {key} ({len(frame)} rows, "
            f"latest {frame['ts'].max().isoformat()}) to {outputs[key]['parquet']}",
        )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CryptoQuant daily metrics and store raw outputs.")
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Asset symbol supported by CryptoQuant (default: btc).",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="CryptoQuant interval (default: day).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of samples to request (default: 730).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS.keys()),
        help="Metric keys to fetch (defaults cover exchange reserve/netflow, stablecoins, miners, whales).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw outputs (default: data/raw/cryptoquant_daily).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_metrics = {key: DEFAULT_METRICS[key] for key in args.metrics if key in DEFAULT_METRICS}
    missing = set(args.metrics) - set(selected_metrics)
    if missing:
        print(f"Warning: unknown metric keys requested and skipped: {sorted(missing)}")
    ingest_cryptoquant_daily(
        metrics=selected_metrics,
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        output_root=args.output_root,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
