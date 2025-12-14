"""CryptoQuant daily metrics loader using CQ_TOKEN."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests

CRYPTOQUANT_API_URL = "https://api.cryptoquant.com/v2/data/metrics"
CRYPTOQUANT_API_URL_V1 = "https://api.cryptoquant.com/v1"
RAW_ROOT = Path("data/raw/cryptoquant_daily")
CATALOG_PATH = Path("artifacts/monitoring/cryptoquant_daily_catalog.json")
DEFAULT_SYMBOL = "btc"
DEFAULT_INTERVAL = "day"
DEFAULT_LIMIT = 730  # ~2 years
SUPPORTED_DAILY_TOKENS = {
    "day",
    "1d",
    "1day",
    "daily",
    "d",
    "24h",
    "24hr",
    "24hrs",
    "24hour",
    "per_day",
}


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for a CryptoQuant metric request."""

    key: str
    endpoint: str
    api_version: str = "v2"
    origin: str = "static"
    category: Optional[str] = None
    subcategory: Optional[str] = None
    frequency: Optional[str] = None
    windows: Tuple[str, ...] = field(default_factory=tuple)
    extra_params: Mapping[str, str] = field(default_factory=dict)
    source_key: Optional[str] = None
    display_name: Optional[str] = None
    required_params: Tuple[str, ...] = field(default_factory=tuple)

    def build_request(self, symbol: str, interval: str, limit: int) -> Tuple[str, Dict[str, object]]:
        effective_limit = limit if limit and limit > 0 else None
        if self.api_version == "v2":
            url = f"{CRYPTOQUANT_API_URL}/{self.endpoint}"
            params: Dict[str, object] = {
                "symbol": symbol,
                "interval": interval,
            }
            if effective_limit is not None:
                params["limit"] = effective_limit
        elif self.api_version == "v1":
            endpoint_path = self.endpoint.lstrip("/")
            normalized_symbol = _normalise_symbol(symbol)
            if endpoint_path.startswith(f"{normalized_symbol}/") or endpoint_path.startswith("exchange/"):
                url = f"{CRYPTOQUANT_API_URL_V1}/{endpoint_path}"
            else:
                url = f"{CRYPTOQUANT_API_URL_V1}/{normalized_symbol}/{endpoint_path}"
            params = {
                "window": interval,
            }
            if effective_limit is not None:
                params["limit"] = effective_limit
        else:  # pragma: no cover - defensive safeguard
            raise CryptoQuantIngestionError(f"Unsupported API version {self.api_version} for {self.key}")

        params.update(self.extra_params)
        return url, params


def _normalise_symbol(symbol: str) -> str:
    return symbol.strip().lower()


class CryptoQuantIngestionError(RuntimeError):
    """Raised when the CryptoQuant API call fails."""


def _require_token() -> str:
    token = os.getenv("CQ_TOKEN", "").strip()
    if not token:
        raise CryptoQuantIngestionError("CQ_TOKEN environment variable is not set; export the API token.")
    return token


def _extract_timeseries(payload: object, metric_key: str) -> Tuple[List[MutableMapping[str, object]], Optional[str]]:
    data: Optional[Iterable[MutableMapping[str, object]]] = None
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, Mapping) and isinstance(result.get("data"), Iterable):
            data = result["data"]  # type: ignore[assignment]
        elif isinstance(payload.get("data"), Iterable):
            data = payload["data"]  # type: ignore[assignment]
    if data is None:
        raise CryptoQuantIngestionError(f"Unexpected payload structure for {metric_key}: {payload}")

    time_fields = {"time", "timestamp", "date", "datetime", "windowTimestamp", "window_end_time", "window"}
    value_field: Optional[str] = None
    rows: List[MutableMapping[str, object]] = []
    for row in data:
        if not isinstance(row, Mapping):
            continue
        ts_value = None
        for field in time_fields:
            if field in row:
                ts_value = row[field]
                break
        if ts_value is None:
            continue
        try:
            ts_parsed = pd.Timestamp(ts_value, tz="UTC")
        except (ValueError, TypeError):  # pragma: no cover - defensive
            continue

        current_value: Optional[float] = None
        current_field: Optional[str] = None
        for field, raw_value in row.items():
            if field in time_fields:
                continue
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            current_value = numeric_value
            current_field = field
            break

        if current_value is None:
            continue

        if value_field is None:
            value_field = current_field

        rows.append(
            {
                "ts": ts_parsed,
                "value": current_value,
            },
        )

    if not rows:
        raise CryptoQuantIngestionError(f"No data rows parsed for {metric_key}; check metric availability.")

    return rows, value_field


def _fetch_metric(
    config: MetricConfig,
    token: str,
    symbol: str,
    interval: str,
    limit: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    url, params = config.build_request(symbol=symbol, interval=interval, limit=limit)
    headers = {
        "Authorization": f"Bearer {token}",
    }

    max_retries = 4
    attempt = 0
    while True:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
            raise CryptoQuantIngestionError(f"CryptoQuant request failed for {config.key}: {exc}") from exc

        full_request_url = response.url
        status_code = response.status_code
        print(
            f"Request {config.key}: GET {full_request_url} -> {status_code}",
        )

        if status_code == 429 and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after)
            except (TypeError, ValueError):
                wait_seconds = min(2 ** attempt, 60)
            wait_seconds = max(wait_seconds, 1.0)
            print(f"Rate limited on {config.key}; sleeping {wait_seconds:.1f}s before retry {attempt + 1}.")
            time.sleep(wait_seconds)
            attempt += 1
            continue

        break

    if response.status_code != 200:
        raise CryptoQuantIngestionError(
            f"CryptoQuant request for {config.key} returned {response.status_code}: {response.text[:256]}",
        )

    payload = response.json()

    rows, value_field = _extract_timeseries(payload, config.key)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("ts").reset_index(drop=True)
    frame["metric"] = config.key
    frame["endpoint"] = config.endpoint
    frame["symbol"] = symbol
    frame["interval"] = interval
    frame["source"] = "cryptoquant"
    frame["api_version"] = config.api_version

    if frame.empty:
        raise CryptoQuantIngestionError(f"No data returned for {config.key}; check metric availability.")

    metadata = {
        "metric_key": config.key,
        "endpoint": config.endpoint,
        "api_version": config.api_version,
        "origin": config.origin,
        "category": config.category,
        "subcategory": config.subcategory,
        "frequency": config.frequency,
        "windows": list(config.windows),
        "source_key": config.source_key,
        "display_name": config.display_name,
        "required_params": list(config.required_params),
        "symbol": symbol,
        "interval": interval,
        "limit": limit if limit and limit > 0 else None,
        "value_field": value_field,
        "rows": len(frame),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "request_url": url,
        "request_params": params,
    }
    return frame, metadata


def _write_outputs(
    frame: pd.DataFrame,
    metric_key: str,
    output_root: Path,
    metadata: Mapping[str, object],
) -> Dict[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"metric={metric_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"cryptoquant_{metric_key}_{timestamp}.parquet"
    csv_path = output_dir / f"cryptoquant_{metric_key}_{timestamp}.csv"
    metadata_path = output_dir / f"cryptoquant_{metric_key}_{timestamp}_metadata.json"

    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
        handle.write("\n")

    return {"parquet": parquet_path, "csv": csv_path, "metadata": metadata_path}


def _windows_include_daily(windows: Iterable[str]) -> bool:
    for window in windows:
        token = str(window).strip().lower()
        if not token:
            continue
        if token in SUPPORTED_DAILY_TOKENS or "day" in token:
            return True
    return False


def _load_discovery_payload(token: str) -> Mapping[str, object]:
    url = f"{CRYPTOQUANT_API_URL_V1}/discovery/endpoints"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise CryptoQuantIngestionError(f"CryptoQuant discovery request failed: {exc}") from exc

    if response.status_code != 200:
        raise CryptoQuantIngestionError(
            "CryptoQuant discovery request returned "
            f"{response.status_code}: {response.text[:256]}"
        )

    payload = response.json()
    if not isinstance(payload, Mapping):
        raise CryptoQuantIngestionError("Unexpected discovery payload structure; expected mapping")
    return payload


def _extract_discovery_records(payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        data = result.get("data")
        if isinstance(data, Iterable):
            return data  # type: ignore[return-value]
    if isinstance(payload.get("data"), Iterable):
        return payload["data"]  # type: ignore[return-value]
    return []


def _discover_metric_configs(token: str, symbol: str, interval: str) -> Tuple[Dict[str, MetricConfig], Mapping[str, object]]:
    payload = _load_discovery_payload(token)
    records = _extract_discovery_records(payload)

    configs: Dict[str, MetricConfig] = {}
    target_symbol = _normalise_symbol(symbol)

    for record in records:
        if not isinstance(record, Mapping):
            continue

        record_symbol = _normalise_symbol(str(record.get("symbol") or record.get("asset") or ""))
        if record_symbol and record_symbol != target_symbol:
            continue

        windows_raw = record.get("windows") or record.get("window") or record.get("intervals") or []
        if isinstance(windows_raw, str):
            windows_raw = [windows_raw]
        if not windows_raw and isinstance(record.get("parameters"), Mapping):
            window_param = record["parameters"].get("window")  # type: ignore[index]
            if isinstance(window_param, str):
                windows_raw = [window_param]
            elif isinstance(window_param, Iterable):
                windows_raw = list(window_param)
        windows_tuple: Tuple[str, ...] = tuple(
            str(window).strip().lower() for window in windows_raw if isinstance(window, (str, bytes))
        )

        frequency = record.get("frequency") or record.get("interval") or record.get("period")
        frequency_str = str(frequency).strip().lower() if frequency else None

        has_daily_support = False
        if windows_tuple:
            has_daily_support = _windows_include_daily(windows_tuple)
        elif frequency_str:
            has_daily_support = frequency_str in SUPPORTED_DAILY_TOKENS or "day" in frequency_str

        if not has_daily_support:
            continue

        required_parameters: Tuple[str, ...] = tuple(
            str(param) for param in record.get("required_parameters", []) if isinstance(param, (str, bytes))
        )
        if required_parameters:
            continue

        path = record.get("path") or record.get("endpoint") or record.get("metric")
        if not path:
            continue
        endpoint = str(path).strip("/")
        if endpoint.lower().startswith("v1/"):
            endpoint = endpoint.split("/", 1)[1]
        if not endpoint:
            continue

        endpoint_parts = endpoint.split("/")
        endpoint_symbol = _normalise_symbol(endpoint_parts[0]) if endpoint_parts else ""
        if endpoint_symbol and endpoint_symbol != target_symbol:
            continue

        key_base = endpoint.replace("/", "_").replace("-", "_")
        default_key = str(record.get("key") or key_base).strip().lower()
        key = default_key or key_base.lower()
        original_key = key
        suffix = 2
        while key in configs:
            key = f"{original_key}_{suffix}"
            suffix += 1

        category = str(record.get("category") or record.get("group") or record.get("family") or "").strip() or None
        subcategory = str(
            record.get("subcategory")
            or record.get("subGroup")
            or record.get("sub_category")
            or record.get("subgroup")
            or ""
        ).strip() or None
        display_name = str(
            record.get("title")
            or record.get("displayName")
            or record.get("name")
            or record.get("metric")
            or ""
        ).strip() or None

        configs[key] = MetricConfig(
            key=key,
            endpoint=endpoint,
            api_version="v1",
            origin="discovery",
            category=category,
            subcategory=subcategory,
            frequency=frequency_str,
            windows=windows_tuple,
            source_key=str(record.get("key") or None),
            display_name=display_name,
            required_params=required_parameters,
        )

    return configs, payload


def _write_metric_catalog(
    configs: Mapping[str, MetricConfig],
    discovery_payload: Mapping[str, object],
    output_path: Path = CATALOG_PATH,
) -> None:
    catalog = []
    for key in sorted(configs):
        config = configs[key]
        catalog.append(
            {
                "key": config.key,
                "endpoint": config.endpoint,
                "api_version": config.api_version,
                "origin": config.origin,
                "category": config.category,
                "subcategory": config.subcategory,
                "frequency": config.frequency,
                "windows": list(config.windows),
                "source_key": config.source_key,
                "display_name": config.display_name,
                "extra_params": dict(config.extra_params),
                "required_params": list(config.required_params),
            },
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(catalog),
        "metrics": catalog,
        "discovery_raw": discovery_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Cataloged {len(catalog)} CryptoQuant daily metrics to {output_path}")


def _resolve_metric_keys(
    metrics: Sequence[str],
    discovered_configs: Mapping[str, MetricConfig],
    symbol: str,
) -> Tuple[List[str], List[str]]:
    if not metrics:
        return [], []

    lower_key_map = {key.lower(): key for key in discovered_configs}
    endpoint_map: Dict[str, str] = {}
    for key, config in discovered_configs.items():
        endpoint_lower = config.endpoint.lower()
        endpoint_map[endpoint_lower] = key
        if "/" in endpoint_lower:
            _, stripped = endpoint_lower.split("/", 1)
            endpoint_map[stripped] = key
        normalized = endpoint_lower.replace("/", "_").replace("-", "_")
        endpoint_map[normalized] = key

    symbol_prefix = f"{_normalise_symbol(symbol)}/"

    resolved: List[str] = []
    missing: List[str] = []
    for metric in metrics:
        candidate = metric.strip().lower()
        if not candidate:
            continue

        if candidate in lower_key_map:
            resolved.append(lower_key_map[candidate])
            continue

        normalized_candidate = candidate.replace("/", "_").replace("-", "_")
        if normalized_candidate in lower_key_map:
            resolved.append(lower_key_map[normalized_candidate])
            continue

        endpoint_candidate = candidate
        if endpoint_candidate.startswith(symbol_prefix):
            endpoint_candidate = endpoint_candidate[len(symbol_prefix) :]

        if candidate in endpoint_map:
            resolved.append(endpoint_map[candidate])
            continue
        if endpoint_candidate in endpoint_map:
            resolved.append(endpoint_map[endpoint_candidate])
            continue
        if normalized_candidate in endpoint_map:
            resolved.append(endpoint_map[normalized_candidate])
            continue

        missing.append(metric)

    return resolved, missing


def ingest_cryptoquant_daily(
    metrics: Sequence[str] | None,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    output_root: Path = RAW_ROOT,
    probe: bool = False,
    save_raw: bool = True,
) -> Dict[str, Dict[str, Path]]:
    token = _require_token()
    discovered_configs, discovery_payload = _discover_metric_configs(token=token, symbol=symbol, interval=interval)

    if not discovered_configs:
        raise CryptoQuantIngestionError("Discovery did not return any daily BTC metrics for this token scope.")

    _write_metric_catalog(discovered_configs, discovery_payload)

    if metrics is None or "all" in metrics:
        selected_keys = sorted(discovered_configs.keys())
        missing: List[str] = []
    else:
        selected_keys, missing = _resolve_metric_keys(metrics, discovered_configs, symbol)
        if missing:
            print(f"Skipping undiscovered metric keys: {sorted(set(missing))}")

    if probe and not selected_keys:
        print("Probe mode requested but no metrics resolved; nothing to do.")
        return {}

    outputs: Dict[str, Dict[str, Path]] = {}
    successes: List[str] = []
    failures: Dict[str, str] = {}
    for key in selected_keys:
        config = discovered_configs[key]
        try:
            frame, metadata = _fetch_metric(config, token, symbol, interval, limit)
        except CryptoQuantIngestionError as exc:
            failures[key] = str(exc)
            print(f"Skipped {key} due to error: {exc}")
            continue
        metadata = dict(metadata)
        metadata["discovered"] = True
        metadata["value_column"] = metadata.get("value_field")
        if save_raw:
            outputs[key] = _write_outputs(frame, key, output_root, metadata)
            successes.append(key)
            print(
                f"Saved CryptoQuant daily metric {key} ({len(frame)} rows, "
                f"latest {frame['ts'].max().isoformat()}) to {outputs[key]['parquet']}"
            )
        else:
            successes.append(key)
            print(
                f"Probe success {key} ({len(frame)} rows, latest {frame['ts'].max().isoformat()}); "
                "raw outputs not persisted."
            )

    print(
        "Ingestion summary: {success} succeeded, {failed} failed.".format(
            success=len(successes),
            failed=len(failures),
        )
    )
    if failures:
        for key, message in failures.items():
            print(f"  - {key}: {message}")
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
        default=["all"],
        help="Metric keys to fetch (default: all discovered + static metrics).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw outputs (default: data/raw/cryptoquant_daily).",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Exercise discovery endpoints without persisting outputs (unless --save-raw).",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Persist raw files when used with --probe; standard runs always persist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    save_raw = args.save_raw or not args.probe
    ingest_cryptoquant_daily(
        metrics=args.metrics,
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        output_root=args.output_root,
        probe=args.probe,
        save_raw=save_raw,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
