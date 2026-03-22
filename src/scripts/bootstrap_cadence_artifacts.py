from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.utils.cloud_io import is_gcs_uri, join_uri, resolve_to_local


def _normalize_source_uri(source_path: str, artifacts_root_uri: str) -> str:
    source = str(source_path or "").strip()
    if not source:
        raise ValueError("Source path must be non-empty")
    if is_gcs_uri(source):
        return source

    source_path_obj = Path(source)
    if source_path_obj.is_absolute():
        return str(source_path_obj)

    normalized = source.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("artifacts/"):
        normalized = normalized[len("artifacts/") :]
    return join_uri(artifacts_root_uri, normalized)


def _copy_from_uri(source_uri: str, target_path: Path) -> None:
    local_source, cleanup = resolve_to_local(source_uri)
    try:
        source_path = Path(local_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact missing: {source_uri}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
    finally:
        if cleanup is not None:
            cleanup()


def _copy_local_directory(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory missing: {source_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _load_json_from_uri(source_uri: str) -> Dict[str, Any]:
    local_source, cleanup = resolve_to_local(source_uri)
    try:
        payload = json.loads(Path(local_source).read_text(encoding="utf-8"))
    finally:
        if cleanup is not None:
            cleanup()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {source_uri}")
    return payload


def _default_manifest_uri(artifacts_root_uri: str) -> str:
    return join_uri(artifacts_root_uri, "monitoring/reliability_promotion_deploy_manifest.json")


def _resolve_manifest_uri(artifacts_root_uri: str, manifest_uri: Optional[str]) -> str:
    return manifest_uri or _default_manifest_uri(artifacts_root_uri)


def _required_summary_targets(run_id: str, repo_root: Path) -> Dict[str, Path]:
    summary_dir = repo_root / "artifacts" / "reliability" / run_id / "summary"
    return {
        "edge_trustworthiness": summary_dir / "edge_trustworthiness.json",
        "calibrated_thresholds": summary_dir / "calibrated_thresholds.json",
        "platt_calibration": summary_dir / "platt_calibration.json",
    }


def validate_cadence_artifacts(
    *,
    artifacts_root_uri: str,
    repo_root: Path,
    manifest_uri: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_manifest_uri = _resolve_manifest_uri(artifacts_root_uri, manifest_uri)
    manifest_payload = _load_json_from_uri(resolved_manifest_uri)
    run_id = str(manifest_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"Deploy manifest at {resolved_manifest_uri} does not contain run_id")

    required_sources = {
        "manifest": resolved_manifest_uri,
        "models_root": join_uri(artifacts_root_uri, "models"),
        "edge_trustworthiness": join_uri(
            artifacts_root_uri,
            f"reliability/{run_id}/summary/edge_trustworthiness.json",
        ),
        "calibrated_thresholds": join_uri(
            artifacts_root_uri,
            f"reliability/{run_id}/summary/calibrated_thresholds.json",
        ),
        "platt_calibration": join_uri(
            artifacts_root_uri,
            f"reliability/{run_id}/summary/platt_calibration.json",
        ),
    }

    deployed_files = manifest_payload.get("deployed_files")
    if isinstance(deployed_files, dict):
        for name, spec in deployed_files.items():
            if not isinstance(spec, dict):
                continue
            source = spec.get("source")
            if not source:
                continue
            required_sources[f"deployed:{name}"] = _normalize_source_uri(str(source), artifacts_root_uri)

    missing: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    for name, source_uri in required_sources.items():
        source_path = Path(source_uri)
        exists = source_path.exists()
        checked.append({"name": name, "path": source_uri, "exists": str(exists).lower()})
        if not exists:
            missing.append({"name": name, "path": source_uri})

    return {
        "artifacts_root_uri": artifacts_root_uri,
        "manifest_uri": resolved_manifest_uri,
        "run_id": run_id,
        "repo_root": str(repo_root),
        "ok": not missing,
        "checked": checked,
        "missing": missing,
    }


def _iter_manifest_restore_pairs(
    *, manifest_payload: Dict[str, Any], artifacts_root_uri: str, repo_root: Path
) -> Iterable[tuple[str, Path]]:
    deployed_files = manifest_payload.get("deployed_files")
    if not isinstance(deployed_files, dict):
        return

    for spec in deployed_files.values():
        if not isinstance(spec, dict):
            continue
        source = spec.get("source")
        target = spec.get("target")
        if not source or not target:
            continue
        yield _normalize_source_uri(str(source), artifacts_root_uri), repo_root / Path(str(target))


def bootstrap_cadence_artifacts(
    *,
    artifacts_root_uri: str,
    repo_root: Path,
    manifest_uri: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_manifest_uri = _resolve_manifest_uri(artifacts_root_uri, manifest_uri)
    manifest_payload = _load_json_from_uri(resolved_manifest_uri)
    run_id = str(manifest_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"Deploy manifest at {resolved_manifest_uri} does not contain run_id")

    restored: list[str] = []
    manifest_target = repo_root / "artifacts" / "monitoring" / "reliability_promotion_deploy_manifest.json"
    _copy_from_uri(resolved_manifest_uri, manifest_target)
    restored.append(str(manifest_target))

    models_source_dir = Path(join_uri(artifacts_root_uri, "models"))
    models_target_dir = repo_root / "artifacts" / "models"
    _copy_local_directory(models_source_dir, models_target_dir)
    restored.append(str(models_target_dir))

    for _, target in _required_summary_targets(run_id, repo_root).items():
        relative = target.relative_to(repo_root / "artifacts").as_posix()
        source_uri = join_uri(artifacts_root_uri, relative)
        _copy_from_uri(source_uri, target)
        restored.append(str(target))

    seen_targets = {str(manifest_target), *(str(path) for path in _required_summary_targets(run_id, repo_root).values())}
    for source_uri, target_path in _iter_manifest_restore_pairs(
        manifest_payload=manifest_payload,
        artifacts_root_uri=artifacts_root_uri,
        repo_root=repo_root,
    ):
        target_str = str(target_path)
        if target_str in seen_targets:
            continue
        _copy_from_uri(source_uri, target_path)
        restored.append(target_str)
        seen_targets.add(target_str)

    return {
        "manifest_uri": resolved_manifest_uri,
        "artifacts_root_uri": artifacts_root_uri,
        "run_id": run_id,
        "restored_files": restored,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the deployed reliability bundle and required cadence inputs "
            "from an external artifacts root into the local workspace."
        )
    )
    parser.add_argument(
        "--artifacts-root-uri",
        required=True,
        help=(
            "Base URI for the external artifacts tree. Supports a local directory or gs:// URI "
            "that mirrors the repository's artifacts/ layout."
        ),
    )
    parser.add_argument(
        "--manifest-uri",
        default=None,
        help=(
            "Optional explicit URI for reliability_promotion_deploy_manifest.json. "
            "Defaults to <artifacts-root-uri>/monitoring/reliability_promotion_deploy_manifest.json."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root into which restored artifacts should be copied.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check that the local artifact root contains all required cadence inputs without copying files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_uri = str(args.manifest_uri) if args.manifest_uri else None
    if args.validate_only:
        result = validate_cadence_artifacts(
            artifacts_root_uri=str(args.artifacts_root_uri),
            repo_root=repo_root,
            manifest_uri=manifest_uri,
        )
        print(json.dumps(result, indent=2))
        if not bool(result.get("ok", False)):
            raise SystemExit(1)
        return

    result = bootstrap_cadence_artifacts(
        artifacts_root_uri=str(args.artifacts_root_uri),
        repo_root=repo_root,
        manifest_uri=manifest_uri,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()