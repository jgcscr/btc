"""Send monitoring alert payloads to a webhook (e.g., Slack)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _build_message(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "UNKNOWN")).upper()
    run_meta = payload.get("run", {}) if isinstance(payload.get("run"), dict) else {}
    job_id = run_meta.get("job_id")
    duration = run_meta.get("duration_seconds")
    parts = [f"[{status}] pipeline health"]
    if job_id:
        parts.append(f"job={job_id}")
    if duration is not None:
        parts.append(f"duration={float(duration):.2f}s")
    header = " ".join(parts)

    issue_lines: list[str] = []
    issues = payload.get("issues", [])
    if isinstance(issues, list):
        for issue in issues[:10]:
            if not isinstance(issue, dict):
                continue
            sev = str(issue.get("severity", "unknown")).upper()
            artifact = issue.get("artifact") or issue.get("path") or "artifact"
            details = issue.get("issues")
            detail_text = details[0] if isinstance(details, list) and details else str(details)
            issue_lines.append(f"• {sev} {artifact}: {detail_text}")
        if len(issues) > 10:
            issue_lines.append(f"• +{len(issues) - 10} more …")

    body_lines = [header]
    body_lines.extend(issue_lines)
    return "\n".join(body_lines)


def _post_with_retry(url: str, body: bytes, *, timeout: float, max_retries: int, initial_backoff: float) -> None:
    backoff = max(initial_backoff, 1.0)
    attempts = max(max_retries, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout):  # nosec - controlled destination
                return
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:  # pragma: no cover - network side effects
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(backoff)
            backoff *= 2
    if last_error:
        raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post pipeline health alerts to a webhook.")
    parser.add_argument("--alert-path", required=True, help="Path to the alert JSON emitted by check_pipeline_health.")
    parser.add_argument("--webhook-url", required=True, help="Destination webhook URL (Slack, Teams, etc.).")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum webhook retries (default: 3).")
    parser.add_argument("--initial-backoff", type=float, default=5.0, help="Initial backoff seconds before retry (default: 5).")
    parser.add_argument("--timeout", type=float, default=10.0, help="Webhook request timeout (seconds, default: 10).")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload instead of sending it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alert_path = Path(args.alert_path)
    if not alert_path.exists() or alert_path.stat().st_size == 0:
        print(f"Alert file {alert_path} missing or empty; skipping webhook post.")
        return 0

    try:
        payload = json.loads(alert_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Alert file {alert_path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    message_text = _build_message(payload)
    body = json.dumps({"text": message_text, "attachments": [{"text": json.dumps(payload, indent=2)}]}).encode("utf-8")

    if args.dry_run:
        print(body.decode("utf-8"))
        return 0

    try:
        _post_with_retry(
            args.webhook_url,
            body,
            timeout=float(args.timeout),
            max_retries=int(args.max_retries),
            initial_backoff=float(args.initial_backoff),
        )
    except Exception as exc:  # pragma: no cover - network call
        print(f"Failed to post alert: {exc}", file=sys.stderr)
        return 1

    print("Alert delivered to webhook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
