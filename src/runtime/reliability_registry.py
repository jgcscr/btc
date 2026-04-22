from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def infer_reliability_run_id(manifest: Mapping[str, Any]) -> str | None:
    explicit = str(manifest.get("run_id") or "").strip()
    if explicit:
        return explicit
    run_dir = str(manifest.get("run_dir") or "").strip()
    if run_dir:
        return Path(run_dir).name
    return None


class ReliabilityRunRegistry:
    def __init__(self, artifacts_root: Path | str = Path("artifacts")) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.registry_root = self.artifacts_root / "reliability" / "registry"

    def record_workflow_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
        run_id = infer_reliability_run_id(manifest)
        if not run_id:
            return None
        summary_dir = self.artifacts_root / "reliability" / run_id / "summary"
        edge_path = summary_dir / "edge_trustworthiness.json"
        edge_payload = _load_json_dict(edge_path)
        payload = {
            "run_id": run_id,
            "run_dir": str(manifest.get("run_dir") or ""),
            "summary_dir": summary_dir.as_posix(),
            "workflow_manifest_path": (summary_dir / "workflow_manifest.json").as_posix(),
            "edge_trustworthiness_path": edge_path.as_posix(),
            "edge_trustworthy": bool(edge_payload.get("edge_trustworthy", False)),
        }
        self._write_json(self.registry_root / "latest.json", payload)
        if payload["edge_trustworthy"]:
            self._write_json(self.registry_root / "latest_trustworthy.json", payload)
        return payload

    def resolve_latest_trustworthy_run_id(self, *, allow_scan_fallback: bool = True) -> str | None:
        latest_payload = _load_json_dict(self.registry_root / "latest_trustworthy.json")
        latest_run_id = str(latest_payload.get("run_id") or "").strip()
        if latest_run_id:
            return latest_run_id
        if allow_scan_fallback:
            return _scan_latest_trustworthy_run_id(self.artifacts_root / "reliability")
        return None

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scan_latest_trustworthy_run_id(run_root: Path) -> str | None:
    if not run_root.exists():
        return None
    for run_dir in sorted((path for path in run_root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        edge_path = run_dir / "summary" / "edge_trustworthiness.json"
        thresholds_path = run_dir / "summary" / "calibrated_thresholds.json"
        platt_path = run_dir / "summary" / "platt_calibration.json"
        if not edge_path.exists() or not thresholds_path.exists() or not platt_path.exists():
            continue
        payload = _load_json_dict(edge_path)
        if bool(payload.get("edge_trustworthy", False)):
            return run_dir.name
    return None


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}