"""Validate monitoring artifacts for freshness and data completeness."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:  # Optional dependency for YAML configs
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional feature
    yaml = None

TIMESTAMP_PRIORITY: tuple[str, ...] = (
    "generated_at",
    "latest_timestamp",
    "last_updated",
    "updated_at",
    "timestamp",
    "ts",
)


@dataclass
class TimestampInfo:
    path: str
    value: datetime


@dataclass
class MissingRatioInfo:
    path: str
    value: float


DEGRADED_STATES = frozenset({"degraded", "outage", "maintenance", "partial"})


@dataclass
class VendorStatus:
    state: str
    reason: str | None = None
    expected_recovery: str | None = None
    manual_override: str | None = None
    updated_at: str | None = None

    def is_degraded(self) -> bool:
        return self.state.lower() in DEGRADED_STATES

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "state": self.state,
            "reason": self.reason,
            "expected_recovery": self.expected_recovery,
            "manual_override": self.manual_override,
            "updated_at": self.updated_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class ArtifactPolicy:
    name: str
    path: str
    staleness_hours: float | None = None
    max_missing_ratio: float | None = None
    vendor_status: VendorStatus | None = None
    notes: str | None = None
    resolved_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.resolved_path = Path(self.path).resolve()


class MonitoringConfig:
    def __init__(self, policies: list[ArtifactPolicy]) -> None:
        self._policies = policies
        self._index: dict[str, ArtifactPolicy] = {}
        for policy in policies:
            keys = {
                policy.name,
                policy.path,
                policy.resolved_path.as_posix(),
                policy.resolved_path.name,
            }
            for key in keys:
                self._index[key] = policy

    def resolve(self, artifact_path: Path) -> ArtifactPolicy | None:
        candidates = {
            artifact_path.name,
            artifact_path.as_posix(),
            artifact_path.resolve().as_posix(),
        }
        try:
            candidates.add(artifact_path.relative_to(Path.cwd()).as_posix())
        except ValueError:
            pass
        for candidate in candidates:
            policy = self._index.get(candidate)
            if policy:
                return policy
        # Fall back to suffix match for relative paths not covered above.
        normalized = artifact_path.as_posix()
        for policy in self._policies:
            if normalized.endswith(policy.path):
                return policy
        return None


@dataclass
class AlertOptions:
    emit_json: bool = False
    output_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline health monitor for trade-ready datasets.")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/monitoring",
        help="Directory containing monitoring JSON artifacts (default: artifacts/monitoring)",
    )
    parser.add_argument(
        "--staleness-hours",
        type=float,
        default=2.0,
        help="Maximum allowed age (in hours) for any artifact timestamp (default: 2)",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.05,
        help="Maximum allowed missing_ratio value within an artifact (default: 0.05)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config describing per-artifact SLA overrides and vendor status.",
    )
    parser.add_argument(
        "--alert-output",
        type=Path,
        default=None,
        help="Write the JSON alert payload to this path when critical failures are detected.",
    )
    parser.add_argument(
        "--emit-alert-json",
        action="store_true",
        help="Print the JSON alert payload to stdout when critical failures are detected.",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Optional scheduler/build job identifier recorded inside alert payloads.",
    )
    return parser.parse_args()


def _walk_payload(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            yield from _walk_payload(value, child_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _walk_payload(value, child_path)
    else:
        yield path, node


def _path_leaf(path: str) -> str:
    if not path:
        return ""
    leaf = path.split(".")[-1]
    bracket = leaf.find("[")
    if bracket != -1:
        leaf = leaf[:bracket]
    return leaf


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_timestamp_info(payload: Any) -> TimestampInfo | None:
    by_leaf: dict[str, TimestampInfo] = {}
    fallback: TimestampInfo | None = None

    for path, value in _walk_payload(payload):
        if not isinstance(value, str):
            continue
        parsed = _parse_timestamp(value)
        if not parsed:
            continue
        info = TimestampInfo(path=path, value=parsed)
        leaf = _path_leaf(path)
        prev = by_leaf.get(leaf)
        if prev is None or parsed > prev.value:
            by_leaf[leaf] = info
        if fallback is None or parsed > fallback.value:
            fallback = info

    for field in TIMESTAMP_PRIORITY:
        info = by_leaf.get(field)
        if info:
            return info
    return fallback


def collect_missing_ratios(payload: Any) -> list[MissingRatioInfo]:
    ratios: list[MissingRatioInfo] = []
    for path, value in _walk_payload(payload):
        if _path_leaf(path) != "missing_ratio":
            continue
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(ratio):
            continue
        ratios.append(MissingRatioInfo(path=path, value=ratio))
    return ratios


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_vendor_status(raw: Any) -> VendorStatus | None:
    if not isinstance(raw, Mapping):
        return None
    state = raw.get("state")
    if not state:
        return None
    return VendorStatus(
        state=str(state),
        reason=_stringify(raw.get("reason")),
        expected_recovery=_stringify(raw.get("expected_recovery")),
        manual_override=_stringify(raw.get("manual_override")),
        updated_at=_stringify(raw.get("updated_at")),
    )


def _load_mapping_from_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs. Install pyyaml or supply JSON.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a JSON/YAML object.")
    return data


def load_monitoring_config(path: Path) -> MonitoringConfig:
    raw = _load_mapping_from_file(path)
    artifacts_section = raw.get("artifacts")
    if not isinstance(artifacts_section, Mapping):
        raise ValueError("Config file must contain an 'artifacts' mapping.")
    policies: list[ArtifactPolicy] = []
    for name, entry in artifacts_section.items():
        if not isinstance(entry, Mapping):
            continue
        resolved_path = str(entry.get("path") or f"artifacts/monitoring/{name}.json")
        policy = ArtifactPolicy(
            name=str(name),
            path=resolved_path,
            staleness_hours=_coerce_float(entry.get("staleness_hours")),
            max_missing_ratio=_coerce_float(entry.get("max_missing_ratio")),
            vendor_status=_parse_vendor_status(entry.get("vendor_status")),
            notes=_stringify(entry.get("notes")),
        )
        policies.append(policy)
    return MonitoringConfig(policies)


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _format_issue_message(relative: str, issues: list[str], vendor_status: VendorStatus | None) -> str:
    vendor_note = ""
    if vendor_status:
        details = [vendor_status.state]
        if vendor_status.reason:
            details.append(vendor_status.reason)
        if vendor_status.expected_recovery:
            details.append(f"eta {vendor_status.expected_recovery}")
        vendor_note = f" [{'; '.join(details)}]"
    joined = "; ".join(issues)
    return f"{relative}{vendor_note}: {joined}"


def _emit_alert_payload(
    options: AlertOptions | None,
    payload: dict[str, Any],
) -> None:
    if not options:
        return
    text = json.dumps(payload, indent=2)
    if options.emit_json:
        print(text)
    if options.output_path:
        options.output_path.parent.mkdir(parents=True, exist_ok=True)
        options.output_path.write_text(text, encoding="utf-8")
        print(f"Wrote alert payload to {options.output_path}")


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):  # pragma: no cover - git optional
        return None
    return result.stdout.strip() or None


def _git_metadata() -> dict[str, str | None]:
    return {
        "commit_sha": _git_output("git", "rev-parse", "HEAD"),
        "commit_describe": _git_output("git", "describe", "--tags", "--always"),
        "branch": _git_output("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _git_output("git", "config", "--get", "remote.origin.url"),
    }


def _detect_job_id(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_candidates = [
        os.getenv("JOB_ID"),
        os.getenv("CLOUD_SCHEDULER_JOB"),
        os.getenv("SCHEDULER_JOB_NAME"),
        os.getenv("BUILD_ID"),
    ]
    for value in env_candidates:
        if value:
            return value
    return None


def collect_run_metadata(job_id: str | None = None, started_at: datetime | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if started_at:
        metadata["started_at"] = started_at.isoformat()
    resolved_job = _detect_job_id(job_id)
    if resolved_job:
        metadata["job_id"] = resolved_job
    hostname = socket.gethostname()
    if hostname:
        metadata["host"] = hostname
    build_id = os.getenv("CLOUD_BUILD_BUILD_ID")
    if build_id:
        metadata["cloud_build_id"] = build_id
    trigger_id = os.getenv("CLOUD_BUILD_TRIGGER_ID")
    if trigger_id:
        metadata["cloud_build_trigger"] = trigger_id
    run_env = os.getenv("GOOGLE_CLOUD_PROJECT")
    if run_env:
        metadata["project_id"] = run_env
    git_meta = _git_metadata()
    for key, value in git_meta.items():
        if value:
            metadata[key] = value
    return metadata


def load_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_artifact(
    path: Path,
    *,
    now: datetime,
    staleness_hours: float,
    missing_ratio_limit: float,
) -> list[str]:
    relative = path.name
    issues: list[str] = []

    try:
        payload = load_payload(path)
    except FileNotFoundError:
        return [f"{relative}: file not found"]
    except json.JSONDecodeError as exc:
        return [f"{relative}: invalid JSON ({exc})"]

    timestamp_info = find_timestamp_info(payload)
    if not timestamp_info:
        issues.append(f"{relative}: no timestamp fields detected")
    else:
        age_hours = max((now - timestamp_info.value).total_seconds() / 3600.0, 0.0)
        if age_hours > staleness_hours:
            issues.append(
                (
                    f"{relative}: stale by {age_hours:.2f}h (field {timestamp_info.path}, "
                    f"limit {staleness_hours:.2f}h)"
                )
            )

    ratios = collect_missing_ratios(payload)
    if ratios:
        offenders = [ratio for ratio in ratios if ratio.value > missing_ratio_limit]
        if offenders:
            offenders.sort(key=lambda info: info.value, reverse=True)
            preview = ", ".join(
                f"{item.path}={item.value:.4f}" for item in offenders[:3]
            )
            extra = len(offenders) - 3
            if extra > 0:
                preview += f", +{extra} more"
            issues.append(
                (
                    f"{relative}: missing_ratio over {missing_ratio_limit:.4f} ({preview})"
                )
            )

    return issues


def run_check(
    artifact_root: Path,
    staleness_hours: float,
    missing_ratio_limit: float,
    *,
    config: MonitoringConfig | None = None,
    alert: AlertOptions | None = None,
    run_metadata: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> int:
    artifact_root = artifact_root.resolve()
    json_paths = sorted(p for p in artifact_root.rglob("*.json") if p.is_file())
    if not json_paths:
        print(f"No JSON artifacts found under {artifact_root}")
        return 1

    now = datetime.now(timezone.utc)
    metadata = dict(run_metadata) if run_metadata else {}
    metadata.setdefault("checked_at", now.isoformat())
    if started_at and "started_at" not in metadata:
        metadata["started_at"] = started_at.isoformat()
    if started_at:
        metadata["duration_seconds"] = max((now - started_at).total_seconds(), 0.0)
    critical_messages: list[str] = []
    warning_messages: list[str] = []
    alert_entries: list[dict[str, Any]] = []

    for path in json_paths:
        policy = config.resolve(path) if config else None
        staleness_limit = (
            policy.staleness_hours
            if (policy and policy.staleness_hours is not None)
            else staleness_hours
        )
        missing_limit = (
            policy.max_missing_ratio
            if (policy and policy.max_missing_ratio is not None)
            else missing_ratio_limit
        )

        entry_issues = analyze_artifact(
            path,
            now=now,
            staleness_hours=staleness_limit,
            missing_ratio_limit=missing_limit,
        )
        if not entry_issues:
            continue

        vendor_status = policy.vendor_status if policy else None
        relative = _relative_label(path, artifact_root)
        message = _format_issue_message(relative, entry_issues, vendor_status)
        is_degraded = bool(vendor_status and vendor_status.is_degraded())

        alert_entry: dict[str, Any] = {
            "artifact": relative,
            "path": str(path),
            "issues": entry_issues,
            "severity": "warning" if is_degraded else "critical",
        }
        if vendor_status:
            alert_entry["vendor_status"] = vendor_status.as_dict()
        if policy and policy.notes:
            alert_entry["notes"] = policy.notes
        alert_entries.append(alert_entry)

        if is_degraded:
            warning_messages.append(message)
        else:
            critical_messages.append(message)

    summary = (
        f"Checked {len(json_paths)} artifact(s) with staleness <= {staleness_hours:.2f}h "
        f"and missing_ratio <= {missing_ratio_limit:.4f}."
    )
    print(summary)

    overall_status = "critical" if critical_messages else ("warning" if warning_messages else "ok")

    if alert_entries:
        alert_payload = {
            "generated_at": now.isoformat(),
            "artifact_root": str(artifact_root),
            "status": overall_status,
            "issues": alert_entries,
        }
        if metadata:
            alert_payload["run"] = metadata
        _emit_alert_payload(alert, alert_payload)

    if critical_messages:
        print("Detected issues:")
        for message in critical_messages:
            print(f"- {message}")
        if warning_messages:
            print("Degraded artifacts (vendor-reported outages):")
            for message in warning_messages:
                print(f"- {message}")
        print("Pipeline health check failed.")
        return 1

    if warning_messages:
        print("Degraded artifacts (vendor-reported outages):")
        for message in warning_messages:
            print(f"- {message}")
        print("All critical artifacts are healthy; degraded feeds align with vendor outage notices.")
        return 0

    print("All monitored artifacts are within thresholds.")
    return 0


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)

    monitoring_config: MonitoringConfig | None = None
    if args.config:
        try:
            monitoring_config = load_monitoring_config(Path(args.config))
        except Exception as exc:  # pragma: no cover - CLI safeguard
            print(f"Failed to load config {args.config}: {exc}", file=sys.stderr)
            sys.exit(1)

    alert_options: AlertOptions | None = None
    if args.emit_alert_json or args.alert_output:
        output_path = Path(args.alert_output).resolve() if args.alert_output else None
        alert_options = AlertOptions(emit_json=bool(args.emit_alert_json), output_path=output_path)

    started_at = datetime.now(timezone.utc)
    run_metadata = collect_run_metadata(job_id=args.job_id, started_at=started_at)

    exit_code = run_check(
        artifact_root=artifact_root,
        staleness_hours=float(args.staleness_hours),
        missing_ratio_limit=float(args.max_missing_ratio),
        config=monitoring_config,
        alert=alert_options,
        run_metadata=run_metadata,
        started_at=started_at,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
