"""Simple Alpha Vantage quota monitor for Phase 2 automation groundwork."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

DEFAULT_USAGE_PATH = "artifacts/monitoring/alpha_vantage_key_usage.json"

def _load_usage(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"Usage file not found at {path}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"Usage file {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not isinstance(data, dict):
        print(f"Usage file {path} is not a JSON object", file=sys.stderr)
        raise SystemExit(1)
    return data

def _resolve_usage_path() -> Path:
    env_path = os.getenv("ALPHA_VANTAGE_USAGE_PATH")
    return Path(env_path) if env_path else Path(DEFAULT_USAGE_PATH)

def _resolve_target_date() -> str:
    override = os.getenv("ALPHA_VANTAGE_ALERT_DATE")
    if override:
        return override
    return datetime.now(timezone.utc).date().isoformat()

def _resolve_threshold() -> float:
    raw = os.getenv("ALPHA_VANTAGE_ALERT_THRESHOLD")
    if not raw:
        return 180.0
    try:
        return float(raw)
    except ValueError:
        print(
            "ALPHA_VANTAGE_ALERT_THRESHOLD must be numeric; received"
            f" '{raw}'.",
            file=sys.stderr,
        )
        raise SystemExit(1)

def run_monitor() -> int:
    usage_path = _resolve_usage_path()
    usage = _load_usage(usage_path)
    if not usage:
        return 0

    threshold = _resolve_threshold()
    target_date = _resolve_target_date()
    day_usage = usage.get(target_date, {})

    if not day_usage:
        print(
            json.dumps(
                {
                    "date": target_date,
                    "threshold": threshold,
                    "message": "No usage entries found for target date.",
                }
            )
        )
        return 0

    alerts = []
    summary: Dict[str, Dict[str, Any]] = {}
    for key, stats in day_usage.items():
        calls = float(stats.get("calls", 0))
        rate_hits = int(stats.get("rate_limit_hits", 0))
        last_updated = stats.get("last_updated")
        remaining = threshold - calls
        summary[key] = {
            "calls": calls,
            "rate_limit_hits": rate_hits,
            "remaining": remaining,
            "last_updated": last_updated,
        }
        if calls > threshold:
            alerts.append(
                {
                    "date": target_date,
                    "key": key,
                    "calls": calls,
                    "threshold": threshold,
                    "exceeded_by": calls - threshold,
                    "rate_limit_hits": rate_hits,
                    "last_updated": last_updated,
                }
            )

    if alerts:
        for payload in alerts:
            print(json.dumps(payload))
        return 0

    print(
        json.dumps(
            {
                "date": target_date,
                "threshold": threshold,
                "keys": summary,
                "message": "All keys remain under threshold.",
            }
        )
    )
    return 0

def main() -> None:
    run_monitor()

if __name__ == "__main__":
    main()
