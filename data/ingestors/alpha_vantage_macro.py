"""Alpha Vantage macro/market data ingestion helper."""
from __future__ import annotations

import argparse
import json
import os
import time
import io
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

ALPHA_URL = "https://www.alphavantage.co/query"
RAW_ROOT = Path("data/raw/macro/alpha_vantage")
KEY_USAGE_PATH = Path("artifacts/monitoring/alpha_vantage_key_usage.json")
CATALOG_SUMMARY_PATH = Path("artifacts/monitoring/alpha_vantage_catalog.json")
DEFAULT_BACKOFF_SECONDS = 30.0
MAX_BACKOFF_SECONDS = 900.0
DEFAULT_SLEEP_SECONDS = 5.0
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_RAW_ROOT = Path("data/raw/macro/twelvedata")
TWELVE_DATA_SYMBOL_MAP: Dict[str, List[str]] = {
    "DXY": ["UUP", "DXY"],
    "VIX": ["VIXY", "VIX"],
}
TWELVE_DATA_API_KEY_ENV = "TWELVE_DATA_API_KEY"
_PROVIDER_ALIASES: Dict[str, Sequence[str]] = {
    "alpha": ("alpha", "alphavantage", "alpha_vantage"),
    "twelve": ("twelve", "twelvedata", "twelve_data"),
}
SUPPORTED_PROVIDERS = frozenset(_PROVIDER_ALIASES.keys())
SUPPORTED_PROVIDER_CHOICES = tuple(sorted(SUPPORTED_PROVIDERS))
DEFAULT_PROVIDER = "alpha"
DEFAULT_CATALOG: List[Dict[str, Any]] = [
    {
        "symbol": "SPY",
        "functions": [
            {"function": "TIME_SERIES_INTRADAY", "params": {"interval": "60min"}},
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "QQQ",
        "functions": [
            {"function": "TIME_SERIES_INTRADAY", "params": {"interval": "60min"}},
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "DXY",
        "functions": [
            {"function": "TIME_SERIES_INTRADAY", "params": {"interval": "60min"}},
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "GLD",
        "functions": [
            {
                "function": "TIME_SERIES_INTRADAY_EXTENDED",
                "params": {"interval": "60min", "slices": ["year1month1", "year1month2"]},
            },
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "HYG",
        "functions": [
            {
                "function": "TIME_SERIES_INTRADAY_EXTENDED",
                "params": {"interval": "60min", "slices": ["year1month1", "year1month2"]},
            },
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "USO",
        "functions": [
            {
                "function": "TIME_SERIES_INTRADAY_EXTENDED",
                "params": {"interval": "60min", "slices": ["year1month1", "year1month2"]},
            },
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "VIX",
        "functions": [
            {"function": "TIME_SERIES_INTRADAY", "params": {"interval": "60min"}},
            {"function": "TIME_SERIES_DAILY"},
        ],
    },
    {
        "symbol": "US10Y",
        "functions": [
            {
                "function": "TREASURY_YIELD",
                "params": {
                    "interval": "daily",
                    "maturity": "10year",
                },
            },
        ],
    },
]
DEFAULT_ALIASES: Dict[str, List[str]] = {
    "DXY": ["USDX", "UUP", "FX_DXY", "DX-Y.NYB"],
    "US10Y": ["^TNX", "US10Y"],
    "VIX": ["VIXY", "VXX", "VIXM", "^VIX", "VIX"],
    "^TNX": ["US10Y"],
}
SUPPORTED_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "TIME_SERIES_INTRADAY": {
        "response_type": "time_series",
        "default_params": {"interval": "60min"},
        "requires_symbol": True,
    },
    "TIME_SERIES_DAILY": {
        "response_type": "time_series",
        "default_params": {},
        "requires_symbol": True,
    },
    "TIME_SERIES_INTRADAY_EXTENDED": {
        "response_type": "intraday_extended",
        "default_params": {"interval": "60min", "slices": ["year1month1"]},
        "requires_symbol": True,
    },
    "TREASURY_YIELD": {
        "response_type": "treasury_yield",
        "default_params": {"interval": "daily", "maturity": "10year"},
        "requires_symbol": False,
    },
}


class AlphaVantageIngestionError(RuntimeError):
    """Raised when the Alpha Vantage API call fails."""


class AlphaVantageRateLimitError(AlphaVantageIngestionError):
    """Raised when all keys are temporarily rate-limited."""

    def __init__(self, message: str, wait_seconds: float) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds


class AlphaVantageInvalidKeyError(AlphaVantageIngestionError):
    """Raised when no valid Alpha Vantage API keys remain."""


class TwelveDataIngestionError(AlphaVantageIngestionError):
    """Raised when Twelve Data ingestion fails."""


def _normalize_provider(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    for canonical, aliases in _PROVIDER_ALIASES.items():
        if cleaned == canonical or cleaned in aliases:
            return canonical
    return cleaned


def _resolve_provider(provider: Optional[str]) -> str:
    normalized = _normalize_provider(provider)
    if normalized:
        if normalized not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise AlphaVantageIngestionError(f"Unsupported provider '{provider}'. Supported: {choices}")
        return normalized
    env_value = _normalize_provider(os.getenv("MACRO_PROVIDER"))
    if env_value:
        if env_value not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise AlphaVantageIngestionError(
                f"Environment variable MACRO_PROVIDER specifies unsupported provider '{env_value}'. Supported: {choices}",
            )
        return env_value
    return DEFAULT_PROVIDER


def _require_twelve_api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        key = explicit.strip()
        if key:
            return key
    env_value = os.getenv(TWELVE_DATA_API_KEY_ENV, "").strip()
    if not env_value:
        raise TwelveDataIngestionError(
            f"Twelve Data ingestion requires {TWELVE_DATA_API_KEY_ENV} environment variable to be set.",
        )
    return env_value


@dataclass
class _KeyState:
    next_retry: datetime
    attempts: int = 0
    invalid: bool = False


class AlphaVantageKeyManager:
    """Manages Alpha Vantage API keys with rotation and backoff."""

    def __init__(
        self,
        keys: Sequence[str],
        usage_path: Path = KEY_USAGE_PATH,
        base_backoff: float = DEFAULT_BACKOFF_SECONDS,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        unique = [key.strip() for key in keys if key and key.strip()]
        if not unique:
            raise AlphaVantageInvalidKeyError(
                "ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_KEYS must provide at least one key.",
            )
        self._keys = deque(dict.fromkeys(unique))
        initial_time = datetime.min.replace(tzinfo=timezone.utc)
        self._states: Dict[str, _KeyState] = {
            key: _KeyState(next_retry=initial_time) for key in self._keys
        }
        self._usage_path = usage_path
        self._usage: Dict[str, Dict[str, Dict[str, Any]]] = self._load_usage()
        self._base_backoff = max(base_backoff, 1.0)
        self._now: Callable[[], datetime] = now_fn or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_env(cls) -> "AlphaVantageKeyManager":
        keys_env = os.getenv("ALPHA_VANTAGE_KEYS", "")
        if keys_env.strip():
            keys = [part.strip() for part in keys_env.split(",") if part.strip()]
        else:
            fallback = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
            keys = [fallback] if fallback else []
        base_backoff = float(os.getenv("ALPHA_VANTAGE_BACKOFF_SECONDS", str(DEFAULT_BACKOFF_SECONDS)))
        return cls(keys=keys, base_backoff=base_backoff)

    def acquire(self) -> str:
        now = self._now()
        for _ in range(len(self._keys)):
            key = self._keys[0]
            self._keys.rotate(-1)
            state = self._states[key]
            if state.invalid:
                continue
            if now >= state.next_retry:
                return key
        valid_keys = [key for key, state in self._states.items() if not state.invalid]
        if not valid_keys:
            raise AlphaVantageInvalidKeyError("All Alpha Vantage API keys are marked invalid.")
        wait_seconds = min(
            max((self._states[key].next_retry - now).total_seconds(), 0.0)
            for key in valid_keys
        )
        raise AlphaVantageRateLimitError(
            f"All Alpha Vantage keys are backoff-limited for {wait_seconds:.1f} seconds.",
            wait_seconds,
        )

    def mark_success(self, key: str) -> None:
        state = self._states.get(key)
        if not state:
            return
        state.attempts = 0
        state.next_retry = self._now()
        self._increment_usage(key, "calls")

    def mark_rate_limit(self, key: str, retry_after: Optional[float]) -> None:
        state = self._states.get(key)
        if not state or state.invalid:
            return
        state.attempts += 1
        if retry_after is None:
            backoff_seconds = min(self._base_backoff * (2 ** (state.attempts - 1)), MAX_BACKOFF_SECONDS)
        else:
            backoff_seconds = min(max(retry_after, self._base_backoff), MAX_BACKOFF_SECONDS)
        state.next_retry = self._now() + timedelta(seconds=backoff_seconds)
        self._increment_usage(key, "rate_limit_hits")

    def mark_invalid(self, key: str) -> None:
        state = self._states.get(key)
        if not state:
            return
        state.invalid = True
        state.next_retry = datetime.max.replace(tzinfo=timezone.utc)

    def _increment_usage(self, key: str, field: str) -> None:
        today = self._now().date().isoformat()
        day_entry = self._usage.setdefault(today, {})
        key_entry = day_entry.setdefault(key, {"calls": 0, "rate_limit_hits": 0})
        key_entry[field] = key_entry.get(field, 0) + 1
        key_entry["last_updated"] = self._now().isoformat()
        self._persist_usage()

    def _load_usage(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self._usage_path.exists():
            return {}
        try:
            with self._usage_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _persist_usage(self) -> None:
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._usage_path.open("w", encoding="utf-8") as handle:
            json.dump(self._usage, handle, indent=2)


def _twelve_attempt_symbols(symbol: str, aliases: Sequence[str]) -> List[str]:
    attempts: List[str] = []
    preferred = TWELVE_DATA_SYMBOL_MAP.get(symbol, [symbol])
    for candidate in preferred:
        if candidate and candidate not in attempts:
            attempts.append(candidate)
    for alias in aliases:
        mapped = TWELVE_DATA_SYMBOL_MAP.get(alias, [alias])
        for candidate in mapped:
            if candidate and candidate not in attempts:
                attempts.append(candidate)
    return attempts or [symbol]


def _map_twelve_interval(function: str, interval: Optional[str]) -> str:
    if function == "TIME_SERIES_DAILY":
        return "1day"
    mapping = {
        "60min": "1h",
        "30min": "30min",
        "15min": "15min",
        "5min": "5min",
        "1min": "1min",
    }
    if not interval:
        return "1h"
    cleaned = interval.strip().lower()
    return mapping.get(cleaned, cleaned)


def _fetch_twelve_time_series(
    api_key: str,
    request_symbol: str,
    canonical_symbol: str,
    interval: str,
    outputsize: str = "5000",
    timezone_value: str = "UTC",
) -> pd.DataFrame:
    params = {
        "symbol": request_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": timezone_value,
        "apikey": api_key,
    }
    try:
        response = requests.get(TWELVE_DATA_URL, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - defensive guard
        raise TwelveDataIngestionError(f"Twelve Data request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise TwelveDataIngestionError(f"Failed to parse Twelve Data response: {exc}") from exc

    if response.status_code != 200 or payload.get("status") == "error":
        message = payload.get("message") if isinstance(payload, dict) else response.text
        raise TwelveDataIngestionError(f"Twelve Data returned error for {request_symbol}: {message}")

    values = payload.get("values")
    if not values:
        raise TwelveDataIngestionError(f"Twelve Data returned no values for {request_symbol} ({canonical_symbol}).")

    rows: List[Dict[str, Any]] = []
    for entry in values:
        timestamp = entry.get("datetime")
        if not timestamp:
            continue
        ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        for field in ("open", "high", "low", "close", "volume"):
            value = entry.get(field)
            try:
                numeric_value = float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                numeric_value = None
            rows.append(
                {
                    "ts": ts,
                    "metric": f"{canonical_symbol}_{field}",
                    "value": numeric_value,
                    "source": "twelve_data",
                },
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise TwelveDataIngestionError(f"Twelve Data parsing produced no rows for {request_symbol}.")
    frame = frame.sort_values("ts").reset_index(drop=True)
    return frame


def _persist_provider_frame(
    frame: pd.DataFrame,
    provider: str,
    output_root: Path,
    canonical_symbol: str,
    attempt_symbol: str,
    function: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    safe_canonical = canonical_symbol.replace("/", "_").replace(":", "_")
    output_dir = output_root / f"symbol={safe_canonical}" / function.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_label = attempt_symbol.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{provider}_{attempt_label}_{function.lower()}_{timestamp}.parquet"
    frame.to_parquet(output_path, index=False)
    summary = {
        "rows": int(len(frame)),
        "first_timestamp": frame["ts"].min().isoformat() if not frame.empty else None,
        "latest_timestamp": frame["ts"].max().isoformat() if not frame.empty else None,
        "interval": params.get("interval") if params else None,
        "params": params or {},
        "provider": provider,
    }
    return output_path, summary

def _build_params(function: str, api_key: str, params: Optional[Dict[str, Any]]) -> Dict[str, str]:
    request: Dict[str, str] = {
        "function": function,
        "apikey": api_key,
    }
    if params:
        for key, value in params.items():
            if value is None:
                continue
            request[str(key)] = str(value)
    request.setdefault("datatype", "json")
    return request


def _classify_error_from_text(text: str) -> str:
    lowered = text.lower()
    if "invalid api key" in lowered:
        return "invalid_key"
    if "premium endpoint" in lowered or ("premium" in lowered and "instantly unlock" in lowered):
        return "invalid_key"
    if "call frequency" in lowered:
        return "rate_limit"
    if "premium" in lowered:
        return "rate_limit"
    return "other"


def _classify_response_error(response: requests.Response, payload: Optional[Dict[str, Any]]) -> str:
    if response.status_code in {401, 403}:
        return "invalid_key"
    if response.status_code == 429:
        return "rate_limit"
    if payload:
        note = str(payload.get("Note", ""))
        error_message = str(payload.get("Error Message", ""))
        classification = _classify_error_from_text(note or error_message)
        if classification != "other":
            return classification
    return _classify_error_from_text(response.text)


def _classify_payload_status(payload: Dict[str, Any]) -> str:
    note = str(payload.get("Note", ""))
    error_message = str(payload.get("Error Message", ""))
    classification = _classify_error_from_text(note or error_message)
    if classification != "other":
        return classification
    information = str(payload.get("Information", ""))
    if information:
        info_classification = _classify_error_from_text(information)
        if info_classification != "other":
            return info_classification
        return "other_error"
    if error_message:
        return "other_error"
    if note:
        # Alpha Vantage uses Note for both quota and informational messages.
        # If it wasn't classified as rate limit above, treat as generic error.
        return "other_error"
    return "ok"


def _retry_after_seconds(response: requests.Response, payload: Optional[Dict[str, Any]]) -> Optional[float]:
    header_value = response.headers.get("Retry-After")
    if header_value:
        try:
            return float(header_value)
        except ValueError:
            pass
    if payload:
        note = str(payload.get("Note", "")).lower()
        for token in ("seconds", "second", "minutes", "minute"):
            if token in note:
                numbers = [word for word in note.replace("-", " ").split() if word.isdigit()]
                if numbers:
                    value = float(numbers[0])
                    if "minute" in token:
                        value *= 60.0
                    return value
    return None


def _flatten_time_series(symbol: str, payload: dict, canonical_symbol: Optional[str] = None) -> pd.DataFrame:
    time_series = None
    for key, value in payload.items():
        if key.startswith("Time Series"):
            time_series = value
            break
    if not isinstance(time_series, dict):
        raise AlphaVantageIngestionError("Response did not contain a time series block.")

    rows: list[dict[str, object]] = []
    metric_symbol = canonical_symbol or symbol
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
                    "metric": f"{metric_symbol}_{clean_field}",
                    "value": numeric_value,
                    "source": "alpha_vantage",
                })

    if not rows:
        raise AlphaVantageIngestionError("Flattened Alpha Vantage response was empty.")

    frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return frame


def _parse_intraday_extended_csv(
    symbol: str,
    csv_text: str,
    canonical_symbol: Optional[str] = None,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.StringIO(csv_text))
    except pd.errors.EmptyDataError as exc:  # pragma: no cover - defensive guard
        raise AlphaVantageIngestionError("Alpha Vantage extended intraday returned empty CSV.") from exc
    if frame.empty:
        raise AlphaVantageIngestionError("Alpha Vantage extended intraday returned no rows.")
    time_column = None
    if "time" in frame.columns:
        time_column = "time"
    elif "timestamp" in frame.columns:
        time_column = "timestamp"
    if time_column is None:
        raise AlphaVantageIngestionError("Extended intraday CSV missing time column.")

    expected_columns = {"open", "high", "low", "close", "volume"}
    if not expected_columns.issubset(set(frame.columns)):
        raise AlphaVantageIngestionError("Extended intraday CSV missing expected OHLCV columns.")

    frame["ts"] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"])
    value_columns = ["open", "high", "low", "close", "volume"]
    frame[value_columns] = frame[value_columns].apply(pd.to_numeric, errors="coerce")

    metric_symbol = canonical_symbol or symbol
    melted = frame.melt(
        id_vars=["ts"],
        value_vars=value_columns,
        var_name="component",
        value_name="value",
    )
    melted["metric"] = melted["component"].apply(lambda comp: f"{metric_symbol}_{comp}")
    melted["source"] = "alpha_vantage"
    return melted[["ts", "metric", "value", "source"]].sort_values("ts").reset_index(drop=True)


def _fetch_intraday_extended_slice(
    manager: AlphaVantageKeyManager,
    function: str,
    request_params: Dict[str, Any],
    canonical_symbol: str,
    attempt_symbol: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    last_error: Optional[str] = None
    while True:
        try:
            api_key = manager.acquire()
        except AlphaVantageRateLimitError as exc:
            raise exc

        prepared_params = dict(request_params)
        prepared_params.setdefault("symbol", attempt_symbol)
        prepared_params.setdefault("datatype", "csv")
        prepared_params["adjusted"] = str(prepared_params.get("adjusted", "false")).lower()
        api_function = "TIME_SERIES_INTRADAY" if function == "TIME_SERIES_INTRADAY_EXTENDED" else function
        params_for_request = _build_params(api_function, api_key, prepared_params)

        try:
            response = requests.get(ALPHA_URL, params=params_for_request, timeout=30)
        except requests.RequestException as exc:  # pragma: no cover - network safeguard
            raise AlphaVantageIngestionError(f"Alpha Vantage request failed: {exc}") from exc

        payload: Optional[Dict[str, Any]] = None
        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            classification = _classify_response_error(response, payload)
            if classification == "invalid_key":
                manager.mark_invalid(api_key)
                last_error = f"Invalid API key response ({response.status_code})"
                continue
            if classification == "rate_limit":
                manager.mark_rate_limit(api_key, _retry_after_seconds(response, payload))
                last_error = f"Rate limit response ({response.status_code})"
                continue
            raise AlphaVantageIngestionError(
                f"Alpha Vantage extended request returned status {response.status_code}: {response.text[:256]}",
            )

        text_body = response.text
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if payload is not None:
            classification = _classify_payload_status(payload)
            if classification == "invalid_key":
                manager.mark_invalid(api_key)
                last_error = "Invalid API key response payload"
                continue
            if classification == "rate_limit":
                manager.mark_rate_limit(api_key, _retry_after_seconds(response, payload))
                last_error = "Alpha Vantage rate limit escalation"
                continue
            if classification == "other_error":
                message = payload.get("Error Message") or payload.get("Note") or last_error or "Unknown Alpha Vantage error"
                raise AlphaVantageIngestionError(str(message))
            frame = _flatten_time_series(attempt_symbol, payload, canonical_symbol=canonical_symbol)
            manager.mark_success(api_key)
            return frame, dict(response.headers)

        frame = _parse_intraday_extended_csv(attempt_symbol, text_body, canonical_symbol=canonical_symbol)
        manager.mark_success(api_key)
        return frame, dict(response.headers)


def _ingest_intraday_extended(
    manager: AlphaVantageKeyManager,
    function: str,
    params: Dict[str, Any],
    canonical_symbol: str,
    attempt_symbol: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    slices_param = params.get("slices")
    if isinstance(slices_param, str):
        slices = [slices_param]
    elif isinstance(slices_param, Iterable):
        slices = [str(value) for value in slices_param if value]
    else:
        slices = []
    if not slices:
        slices = ["year1month1"]

    base_params = {key: value for key, value in params.items() if key != "slices"}
    base_params.setdefault("interval", params.get("interval", "60min"))
    frames: List[pd.DataFrame] = []
    last_headers: Dict[str, Any] = {}

    for slice_name in slices:
        slice_params = dict(base_params)
        slice_params["slice"] = slice_name
        frame, headers = _fetch_intraday_extended_slice(
            manager=manager,
            function=function,
            request_params=slice_params,
            canonical_symbol=canonical_symbol,
            attempt_symbol=attempt_symbol,
        )
        frames.append(frame)
        last_headers = headers

    if not frames:
        raise AlphaVantageIngestionError("Alpha Vantage extended intraday produced no data slices.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ts", "metric"]).drop_duplicates(subset=["ts", "metric"], keep="last").reset_index(drop=True)
    return combined, last_headers


def _flatten_treasury_yield(payload: dict, canonical_symbol: str) -> pd.DataFrame:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise AlphaVantageIngestionError("Treasury yield response did not contain data rows.")

    rows: list[dict[str, object]] = []
    for entry in data:
        date_str = entry.get("date")
        value = entry.get("value")
        if not date_str:
            continue
        ts = pd.Timestamp(date_str).tz_localize("UTC")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = None
        rows.append({
            "ts": ts,
            "metric": f"{canonical_symbol}_yield",
            "value": numeric_value,
            "source": "alpha_vantage",
        })

    if not rows:
        raise AlphaVantageIngestionError("Treasury yield response parsing produced no rows.")

    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def ingest_series(
    function: str,
    params: Dict[str, Any],
    output_root: Path = RAW_ROOT,
    return_summary: bool = False,
    canonical_symbol: Optional[str] = None,
    attempt_symbol: Optional[str] = None,
    key_manager: Optional[AlphaVantageKeyManager] = None,
    provider: Optional[str] = None,
    aliases: Optional[Sequence[str]] = None,
    twelve_api_key: Optional[str] = None,
) -> Path | Tuple[Path, Dict[str, Any]]:
    if function not in SUPPORTED_FUNCTIONS:
        supported = ", ".join(SUPPORTED_FUNCTIONS)
        raise AlphaVantageIngestionError(f"Unsupported function '{function}'. Supported: {supported}")
    params = {str(key): value for key, value in params.items()}
    resolved_provider = _resolve_provider(provider)
    canonical = canonical_symbol or attempt_symbol or params.get("symbol")
    if not canonical:
        raise AlphaVantageIngestionError("Canonical symbol is required to label metrics.")
    canonical = str(canonical)
    attempt_value = str(attempt_symbol or canonical)
    alias_values = [alias for alias in (aliases or []) if alias]
    target_output_root = output_root
    if resolved_provider == "twelve" and target_output_root == RAW_ROOT:
        target_output_root = TWELVE_DATA_RAW_ROOT

    if resolved_provider == "twelve":
        if SUPPORTED_FUNCTIONS[function]["response_type"] != "time_series":
            raise TwelveDataIngestionError(
                f"Twelve Data provider currently supports only time series functions; received {function}.",
            )
        interval = _map_twelve_interval(function, params.get("interval"))
        timezone_value = str(params.get("timezone", "UTC"))
        outputsize = str(params.get("outputsize", "5000"))
        request_candidates = _twelve_attempt_symbols(attempt_value, alias_values)
        api_key = _require_twelve_api_key(twelve_api_key)
        last_error: Optional[Exception] = None
        frame: Optional[pd.DataFrame] = None
        output_path: Optional[Path] = None
        summary: Dict[str, Any] = {}
        selected_symbol: Optional[str] = None
        for request_symbol in request_candidates:
            try:
                frame = _fetch_twelve_time_series(
                    api_key=api_key,
                    request_symbol=request_symbol,
                    canonical_symbol=canonical,
                    interval=interval,
                    outputsize=outputsize,
                    timezone_value=timezone_value,
                )
                output_path, summary = _persist_provider_frame(
                    frame=frame,
                    provider="twelvedata",
                    output_root=target_output_root,
                    canonical_symbol=canonical,
                    attempt_symbol=request_symbol,
                    function=function,
                    params={"interval": interval, "outputsize": outputsize, "timezone": timezone_value},
                )
                selected_symbol = request_symbol
                break
            except TwelveDataIngestionError as exc:
                last_error = exc
                print(f"Warning: Twelve Data attempt failed for {canonical} via {request_symbol}: {exc}")
                continue
        if frame is None or output_path is None:
            raise TwelveDataIngestionError(
                f"Twelve Data ingestion failed for {canonical}. Last error: {last_error}",
            )

        sample = frame.head(5)
        print("Twelve Data sample rows:\n", sample)
        metadata_params = dict(params)
        if selected_symbol:
            metadata_params.setdefault("provider_symbol", selected_symbol)
        summary.update({
            "provider": resolved_provider,
            "params": metadata_params,
        })
        if return_summary:
            return output_path, summary
        return output_path

    manager = key_manager or AlphaVantageKeyManager.from_env()

    response_type = SUPPORTED_FUNCTIONS[function]["response_type"]
    interval = params.get("interval")
    quota_headers: Dict[str, Any] = {}

    if response_type == "intraday_extended":
        frame, quota_headers = _ingest_intraday_extended(
            manager=manager,
            function=function,
            params=params,
            canonical_symbol=canonical,
            attempt_symbol=attempt_value,
        )
    else:
        last_error: Optional[str] = None
        while True:
            try:
                api_key = manager.acquire()
            except AlphaVantageRateLimitError as exc:
                raise exc

            request_params = _build_params(function, api_key, params)

            try:
                response = requests.get(ALPHA_URL, params=request_params, timeout=30)
            except requests.RequestException as exc:  # pragma: no cover - network failure safeguard
                raise AlphaVantageIngestionError(f"Alpha Vantage request failed: {exc}") from exc

            payload: Optional[Dict[str, Any]] = None
            if response.status_code != 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                classification = _classify_response_error(response, payload)
                if classification == "invalid_key":
                    manager.mark_invalid(api_key)
                    last_error = f"Invalid API key response ({response.status_code})"
                    continue
                if classification == "rate_limit":
                    manager.mark_rate_limit(api_key, _retry_after_seconds(response, payload))
                    last_error = f"Rate limit response ({response.status_code})"
                    continue
                raise AlphaVantageIngestionError(
                    f"Alpha Vantage request returned status {response.status_code}: {response.text[:256]}",
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise AlphaVantageIngestionError(f"Failed to parse Alpha Vantage response: {exc}") from exc

            classification = _classify_payload_status(payload)
            if classification == "invalid_key":
                manager.mark_invalid(api_key)
                last_error = "Invalid API key response payload"
                continue
            if classification == "rate_limit":
                manager.mark_rate_limit(api_key, _retry_after_seconds(response, payload))
                last_error = "Alpha Vantage rate limit escalation"
                continue
            if classification == "other_error":
                message = payload.get("Error Message") or payload.get("Note") or last_error or "Unknown Alpha Vantage error"
                raise AlphaVantageIngestionError(str(message))

            if response_type == "time_series":
                frame = _flatten_time_series(attempt_value, payload, canonical_symbol=canonical)
            elif response_type == "treasury_yield":
                frame = _flatten_treasury_yield(payload, canonical_symbol=canonical)
            else:  # pragma: no cover - defensive branch for future extension
                raise AlphaVantageIngestionError(f"Unhandled response type '{response_type}' for function {function}.")

            quota_headers = dict(response.headers)
            manager.mark_success(api_key)
            break

    remaining = quota_headers.get("X-RateLimit-Remaining") if quota_headers else None
    limit = quota_headers.get("X-RateLimit-Limit") if quota_headers else None
    if remaining or limit:
        print(f"Alpha Vantage quota remaining {remaining}/{limit} (function={function}, interval={interval})")
    else:
        print("Alpha Vantage quota headers unavailable; premium key in use.")
    sample = frame.head(5)
    print("Alpha Vantage sample rows:\n", sample)

    safe_canonical = canonical.replace("/", "_").replace(":", "_")
    output_dir = target_output_root / f"symbol={safe_canonical}" / function.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_label = attempt_value.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"alpha_{attempt_label}_{function.lower()}_{timestamp}.parquet"
    frame.to_parquet(output_path, index=False)

    summary = {
        "rows": int(len(frame)),
        "first_timestamp": frame["ts"].min().isoformat() if not frame.empty else None,
        "latest_timestamp": frame["ts"].max().isoformat() if not frame.empty else None,
        "interval": params.get("interval"),
        "params": params,
        "provider": resolved_provider,
    }
    if return_summary:
        return output_path, summary
    return output_path


def _load_catalog(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return DEFAULT_CATALOG
    if not path.exists():
        raise AlphaVantageIngestionError(f"Catalog file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - validation guard
        raise AlphaVantageIngestionError(f"Failed to parse catalog JSON: {exc}") from exc
    if not isinstance(data, list):
        raise AlphaVantageIngestionError("Catalog JSON must be a list of symbol entries.")
    return data


def _validate_catalog(catalog: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    for entry in catalog:
        symbol = entry.get("symbol")
        functions = entry.get("functions")
        if not symbol or not isinstance(functions, list):
            raise AlphaVantageIngestionError("Catalog entries must include 'symbol' and 'functions' list.")
        entry_provider_raw = entry.get("provider")
        entry_provider: Optional[str] = None
        if entry_provider_raw is not None:
            entry_provider = _normalize_provider(str(entry_provider_raw))
            if entry_provider not in SUPPORTED_PROVIDERS:
                choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
                raise AlphaVantageIngestionError(
                    f"Catalog entry for {symbol} references unsupported provider '{entry_provider_raw}'. Supported: {choices}",
                )
        parsed_funcs: List[Dict[str, Any]] = []
        for spec in functions:
            fn = spec.get("function")
            if fn not in SUPPORTED_FUNCTIONS:
                supported = ", ".join(sorted(SUPPORTED_FUNCTIONS))
                raise AlphaVantageIngestionError(
                    f"Catalog entry for {symbol} references unsupported function '{fn}'. Supported: {supported}",
                )
            fn_config = SUPPORTED_FUNCTIONS[fn]
            params: Dict[str, Any] = dict(fn_config.get("default_params", {}))
            if "interval" in spec:
                params["interval"] = spec.get("interval")
            extra_params = spec.get("params", {})
            if extra_params:
                if not isinstance(extra_params, dict):
                    raise AlphaVantageIngestionError(
                        f"Catalog entry for {symbol} function {fn} must provide params as an object.",
                    )
                for key, value in extra_params.items():
                    params[str(key)] = value
            spec_provider_raw = spec.get("provider")
            spec_provider: Optional[str] = None
            if spec_provider_raw is not None:
                spec_provider = _normalize_provider(str(spec_provider_raw))
                if spec_provider not in SUPPORTED_PROVIDERS:
                    choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
                    raise AlphaVantageIngestionError(
                        f"Catalog entry for {symbol} function {fn} references unsupported provider '{spec_provider_raw}'. Supported: {choices}",
                    )
            parsed_funcs.append({"function": fn, "params": params, "provider": spec_provider})
        validated.append({
            "symbol": str(symbol),
            "name": entry.get("name", str(symbol)),
            "functions": parsed_funcs,
            "aliases": [str(alias) for alias in entry.get("aliases", DEFAULT_ALIASES.get(str(symbol), []))],
            "provider": entry_provider,
        })
    return validated


def _sleep_between_calls(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def ingest_catalog(
    catalog: Optional[Sequence[Dict[str, Any]]] = None,
    sleep_seconds: Optional[float] = None,
    output_root: Path = RAW_ROOT,
    provider: Optional[str] = None,
    twelve_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    catalog_path_env = os.getenv("ALPHA_VANTAGE_CATALOG_PATH")
    resolved_catalog = catalog
    if resolved_catalog is None:
        path = Path(catalog_path_env) if catalog_path_env else None
        resolved_catalog = _load_catalog(path)
    validated_catalog = _validate_catalog(resolved_catalog)

    if sleep_seconds is None:
        sleep_env = os.getenv("ALPHA_VANTAGE_SLEEP_SECONDS")
        sleep_seconds = float(sleep_env) if sleep_env else DEFAULT_SLEEP_SECONDS

    total_requests = sum(len(entry["functions"]) for entry in validated_catalog)
    alpha_manager: Optional[AlphaVantageKeyManager] = None
    results: List[Dict[str, Any]] = []
    completed = 0

    for entry in validated_catalog:
        symbol = entry["symbol"]
        aliases = [alias for alias in entry.get("aliases", []) if alias]
        entry_provider = entry.get("provider") or provider
        for spec in entry["functions"]:
            fn = spec["function"]
            fn_config = SUPPORTED_FUNCTIONS[fn]
            base_params = dict(spec.get("params", {}))
            requires_symbol = fn_config.get("requires_symbol", True)
            attempt_symbols = [symbol] + [alias for alias in aliases if alias not in {symbol}]
            if not requires_symbol:
                attempt_symbols = [symbol]
            last_error: Optional[str] = None
            resolved_symbol: Optional[str] = None
            output_path: Optional[Path] = None
            summary: Dict[str, Any] = {}
            spec_provider = spec.get("provider") or entry_provider
            for attempt in attempt_symbols:
                try:
                    params = dict(base_params)
                    if requires_symbol:
                        params["symbol"] = attempt
                    resolved_provider = _resolve_provider(spec_provider)
                    manager_to_use: Optional[AlphaVantageKeyManager] = None
                    if resolved_provider == "alpha":
                        if alpha_manager is None:
                            alpha_manager = AlphaVantageKeyManager.from_env()
                        manager_to_use = alpha_manager
                    path, attempt_summary = ingest_series(
                        function=fn,
                        params=params,
                        output_root=output_root,
                        return_summary=True,
                        canonical_symbol=symbol,
                        attempt_symbol=attempt if requires_symbol else symbol,
                        key_manager=manager_to_use,
                        provider=resolved_provider,
                        aliases=aliases,
                        twelve_api_key=twelve_api_key,
                    )
                    resolved_symbol = attempt
                    output_path = Path(path)
                    summary = attempt_summary
                    last_error = None
                    break
                except AlphaVantageIngestionError as exc:
                    last_error = str(exc)
                    print(f"Warning: failed to ingest {symbol} via {attempt} {fn}: {exc}")
                    _sleep_between_calls(sleep_seconds)
            if resolved_symbol is not None and output_path is not None:
                result = {
                    "symbol": symbol,
                    "function": fn,
                    "interval": summary.get("interval"),
                    "rows": summary.get("rows"),
                    "latest_timestamp": summary.get("latest_timestamp"),
                    "path": str(output_path),
                    "status": "success",
                    "resolved_symbol": resolved_symbol,
                    "params": summary.get("params"),
                    "provider": summary.get("provider", spec_provider),
                }
            else:
                result = {
                    "symbol": symbol,
                    "function": fn,
                    "interval": base_params.get("interval"),
                    "rows": 0,
                    "latest_timestamp": None,
                    "path": None,
                    "status": "error",
                    "resolved_symbol": None,
                    "error": last_error,
                    "params": base_params,
                    "provider": _normalize_provider(spec_provider) if spec_provider else None,
                }
            results.append(result)
            completed += 1
            if completed < total_requests:
                _sleep_between_calls(sleep_seconds)

    success = sum(1 for row in results if row["status"] == "success")
    errors = [row for row in results if row["status"] != "success"]
    providers_run = sorted({row.get("provider") for row in results if row.get("provider")})
    providers_label = ", ".join(providers_run) if providers_run else "unknown"
    print(
        f"Macro catalog ingestion finished ({providers_label}): {success}/{len(results)} calls succeeded. "
        f"sleep_seconds={sleep_seconds}.",
    )
    if errors:
        print("Failures detected:")
        for row in errors:
            print(
                f"  - {row['symbol']} {row['function']} ({row.get('interval')}) [{row.get('provider')}] -> {row.get('error')}",
            )
    CATALOG_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def _infer_frequency(ts: pd.Series) -> str:
    if ts.empty:
        return "unknown"
    ordered = ts.sort_values().dropna()
    if len(ordered) < 2:
        return "unknown"
    deltas = ordered.diff().dropna()
    if deltas.empty:
        return "unknown"
    median_delta = deltas.median()
    minutes = int(median_delta.total_seconds() // 60)
    if minutes == 0:
        return "unknown"
    if minutes % (24 * 60) == 0:
        days = minutes // (24 * 60)
        return f"{days}d"
    return f"{minutes}min"


def audit_outputs(output_root: Path = RAW_ROOT) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not output_root.exists():
        print(f"No macro outputs found under {output_root}.")
        return rows

    for parquet_path in output_root.rglob("*.parquet"):
        parts = parquet_path.relative_to(output_root).parts
        if len(parts) < 3:
            continue
        symbol_part = parts[0]
        function_part = parts[1]
        symbol = symbol_part.split("=")[-1]
        function_name = function_part.upper()
        df = pd.read_parquet(parquet_path, columns=["ts"])
        freq = _infer_frequency(df["ts"])
        latest_ts = df["ts"].max()
        latest_iso = latest_ts.isoformat() if pd.notna(latest_ts) else None
        rows.append(
            {
                "symbol": symbol,
                "function": function_name,
                "frequency": freq,
                "rows": int(len(df)),
                "latest_timestamp": latest_iso,
                "path": str(parquet_path),
            },
        )

    if not rows:
        print(f"No macro parquet files detected under {output_root}.")
        return rows

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["symbol", "function", "latest_timestamp"], ascending=[True, True, False])
    latest_per_series = summary.groupby(["symbol", "function"], as_index=False).first()
    print(latest_per_series[["symbol", "function", "frequency", "rows", "latest_timestamp"]])
    return latest_per_series.to_dict(orient="records")


def _parse_cli_params(pairs: Iterable[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise AlphaVantageIngestionError(
                f"Invalid --param value '{pair}'. Use key=value format.",
            )
        key, value = pair.split("=", maxsplit=1)
        params[key.strip()] = value.strip()
    return params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage data into Parquet.")
    parser.add_argument("function", nargs="?", choices=sorted(SUPPORTED_FUNCTIONS), help="Alpha Vantage function")
    parser.add_argument("symbol", nargs="?", help="Ticker symbol (e.g. SPY, DXY).")
    parser.add_argument(
        "--interval",
        default=None,
        help="Interval for intraday data (e.g. 60min). Required only for TIME_SERIES_INTRADAY.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional query parameter in key=value form (repeat for multiple values).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_ROOT,
        help="Root directory for raw Parquet output (default: data/raw/macro/alpha_vantage).",
    )
    parser.add_argument(
        "--run-catalog",
        action="store_true",
        help="Ingest the default (or configured) Alpha Vantage catalog instead of a single series.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Optional path to a JSON catalog overriding the built-in symbol list.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Seconds to wait between catalog requests (default: ALPHA_VANTAGE_SLEEP_SECONDS env or 5.0).",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDER_CHOICES,
        default=None,
        help="Data provider to use (default resolves from MACRO_PROVIDER env or alpha).",
    )
    parser.add_argument(
        "--twelve-api-key",
        default=None,
        help=f"Override Twelve Data API key (default uses {TWELVE_DATA_API_KEY_ENV}).",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print a summary of the most recent macro ingestions and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_provider = _resolve_provider(args.provider)
    output_root = args.output_root
    if resolved_provider == "twelve" and output_root == RAW_ROOT:
        output_root = TWELVE_DATA_RAW_ROOT
    if args.audit:
        audit_outputs(output_root)
        return

    if args.run_catalog:
        catalog = _load_catalog(args.catalog_path) if args.catalog_path else None
        ingest_catalog(
            catalog=catalog,
            sleep_seconds=args.sleep_seconds,
            output_root=output_root,
            provider=resolved_provider,
            twelve_api_key=args.twelve_api_key,
        )
        return

    if not args.function or not args.symbol:
        raise AlphaVantageIngestionError(
            "Function and symbol arguments are required unless --run-catalog or --audit is provided.",
        )

    params: Dict[str, Any] = {"symbol": args.symbol}
    if args.interval:
        params["interval"] = args.interval
    extra_params = _parse_cli_params(args.param)
    params.update(extra_params)

    manager: Optional[AlphaVantageKeyManager] = None
    if resolved_provider == "alpha":
        manager = AlphaVantageKeyManager.from_env()
    ingest_series(
        function=args.function,
        params=params,
        output_root=output_root,
        canonical_symbol=args.symbol,
        attempt_symbol=args.symbol,
        key_manager=manager,
        provider=resolved_provider,
        twelve_api_key=args.twelve_api_key,
    )


if __name__ == "__main__":
    try:
        main()
    except AlphaVantageRateLimitError as exc:
        print(f"Alpha Vantage rate limit exhausted: {exc}")
        raise SystemExit(1)
    except AlphaVantageIngestionError as exc:
        print(f"Alpha Vantage ingestion failed: {exc}")
        raise SystemExit(1)
