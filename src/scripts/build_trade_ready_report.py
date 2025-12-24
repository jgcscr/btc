"""Compile trade-ready workflow outputs into a structured report.

Reads the captured responses from the dataset refresh and signal steps, extracts
summary metadata, and writes a consolidated JSON document plus a text file
containing the destination Cloud Storage URI.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_refresh_summary(refresh: Dict[str, Any]) -> Dict[str, Any]:
    messages = []
    for line in refresh.get("stdout", "").splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return next((msg for msg in messages if msg.get("message") == "refresh.complete"), {})


def _parse_signal_payload(signal: Dict[str, Any]) -> Dict[str, Any]:
    stdout = signal.get("stdout", "")
    start = stdout.find('{\n  "generated_at"')
    if start == -1:
        return {}

    brace_depth = 0
    for index, char in enumerate(stdout[start:], start=start):
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                snippet = stdout[start : index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return {}
    return {}


def _resolve_report_uri(bucket: str, payload: Dict[str, Any]) -> str:
    generated = payload.get("generated_at") if payload else None
    if generated:
        timestamp = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    else:
        timestamp = datetime.now(timezone.utc)
    return f"{bucket.rstrip('/')}/reports/trade_ready/{timestamp:%Y%m%d}/{timestamp:%H}.json"


def main() -> None:
    bucket = os.environ.get("REPORT_BUCKET")
    if not bucket:
        raise RuntimeError("REPORT_BUCKET environment variable is required")

    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    refresh = _read_json(workspace / "run_dataset_refresh.json")
    signal = _read_json(workspace / "run_signal.json")

    refresh_summary = _parse_refresh_summary(refresh)
    signal_payload = _parse_signal_payload(signal)

    report = {
        "run_dataset_refresh": {
            "returncode": refresh.get("returncode"),
            "duration_seconds": refresh.get("duration_seconds"),
            "refresh_complete": refresh_summary,
        },
        "run_signal": {
            "returncode": signal.get("returncode"),
            "duration_seconds": signal.get("duration_seconds"),
            "payload": signal_payload,
        },
    }

    report_path = workspace / "trade_ready_report.json"
    uri_path = workspace / "report_uri.txt"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    report_uri = _resolve_report_uri(bucket, signal_payload)
    uri_path.write_text(report_uri, encoding="utf-8")
    print(f"Report URI: {report_uri}")


if __name__ == "__main__":
    main()
