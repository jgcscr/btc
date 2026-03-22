from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _ensure_local_path(path_value: str, *, argument_name: str) -> str:
    normalized = str(path_value or "").strip()
    if not normalized:
        raise ValueError(f"{argument_name} must be non-empty")
    if "://" in normalized:
        raise ValueError(f"{argument_name} must be a local filesystem path, not a remote URI")
    return normalized


def _normalize_source_path(source_path: str, artifacts_root_path: str) -> str:
    source = _ensure_local_path(source_path, argument_name="source path")
    if not source:
        raise ValueError("Source path must be non-empty")

    source_path_obj = Path(source)
    if source_path_obj.is_absolute():
        return str(source_path_obj)

    normalized = source.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("artifacts/"):
        normalized = normalized[len("artifacts/") :]
    return str(Path(artifacts_root_path) / normalized)


def _copy_from_path(source_path_str: str, target_path: Path) -> None:
    source_path = Path(_ensure_local_path(source_path_str, argument_name="source path"))
    if not source_path.exists():
        raise FileNotFoundError(f"Source artifact missing: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def _copy_local_directory(source_dir: Path, target_dir: Path, *, replace: bool = False) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory missing: {source_dir}")
    if replace and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _load_json_from_path(source_path_str: str) -> Dict[str, Any]:
    source_path = Path(_ensure_local_path(source_path_str, argument_name="manifest path"))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {source_path}")
    return payload


def _default_manifest_path(artifacts_root_path: str) -> str:
    return str(Path(artifacts_root_path) / "monitoring" / "reliability_promotion_deploy_manifest.json")


def _resolve_manifest_path(artifacts_root_path: str, manifest_path: Optional[str]) -> str:
    return _ensure_local_path(
        manifest_path or _default_manifest_path(artifacts_root_path),
        argument_name="manifest path",
    )


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
    artifacts_root_path = _ensure_local_path(artifacts_root_uri, argument_name="artifacts root")
    resolved_manifest_path = _resolve_manifest_path(artifacts_root_path, manifest_uri)
    manifest_payload = _load_json_from_path(resolved_manifest_path)
    run_id = str(manifest_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"Deploy manifest at {resolved_manifest_path} does not contain run_id")

    required_sources = {
        "manifest": resolved_manifest_path,
        "models_root": str(Path(artifacts_root_path) / "models"),
        "edge_trustworthiness": str(
            Path(artifacts_root_path) / "reliability" / run_id / "summary" / "edge_trustworthiness.json"
        ),
        "calibrated_thresholds": str(
            Path(artifacts_root_path) / "reliability" / run_id / "summary" / "calibrated_thresholds.json"
        ),
        "platt_calibration": str(
            Path(artifacts_root_path) / "reliability" / run_id / "summary" / "platt_calibration.json"
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
            required_sources[f"deployed:{name}"] = _normalize_source_path(str(source), artifacts_root_path)

    missing: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    for name, source_uri in required_sources.items():
        source_path = Path(source_uri)
        exists = source_path.exists()
        checked.append({"name": name, "path": source_uri, "exists": str(exists).lower()})
        if not exists:
            missing.append({"name": name, "path": source_uri})

    return {
        "artifacts_root_uri": artifacts_root_path,
        "manifest_uri": resolved_manifest_path,
        "run_id": run_id,
        "repo_root": str(repo_root),
        "ok": not missing,
        "checked": checked,
        "missing": missing,
    }


def _iter_manifest_restore_pairs(
    *, manifest_payload: Dict[str, Any], artifacts_root_path: str, repo_root: Path
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
        yield _normalize_source_path(str(source), artifacts_root_path), repo_root / Path(str(target))


def bootstrap_cadence_artifacts(
    *,
    artifacts_root_uri: str,
    repo_root: Path,
    manifest_uri: Optional[str] = None,
) -> Dict[str, Any]:
    artifacts_root_path = _ensure_local_path(artifacts_root_uri, argument_name="artifacts root")
    resolved_manifest_path = _resolve_manifest_path(artifacts_root_path, manifest_uri)
    manifest_payload = _load_json_from_path(resolved_manifest_path)
    run_id = str(manifest_payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"Deploy manifest at {resolved_manifest_path} does not contain run_id")

    restored: list[str] = []
    manifest_target = repo_root / "artifacts" / "monitoring" / "reliability_promotion_deploy_manifest.json"
    _copy_from_path(resolved_manifest_path, manifest_target)
    restored.append(str(manifest_target))

    models_source_dir = Path(artifacts_root_path) / "models"
    models_target_dir = repo_root / "artifacts" / "models"
    _copy_local_directory(models_source_dir, models_target_dir, replace=True)
    restored.append(str(models_target_dir))

    for _, target in _required_summary_targets(run_id, repo_root).items():
        relative = target.relative_to(repo_root / "artifacts").as_posix()
        source_path = Path(artifacts_root_path) / relative
        _copy_from_path(str(source_path), target)
        restored.append(str(target))

    seen_targets = {str(manifest_target), *(str(path) for path in _required_summary_targets(run_id, repo_root).values())}
    for source_uri, target_path in _iter_manifest_restore_pairs(
        manifest_payload=manifest_payload,
        artifacts_root_path=artifacts_root_path,
        repo_root=repo_root,
    ):
        target_str = str(target_path)
        if target_str in seen_targets:
            continue
        _copy_from_path(source_uri, target_path)
        restored.append(target_str)
        seen_targets.add(target_str)

    return {
        "manifest_uri": resolved_manifest_path,
        "artifacts_root_uri": artifacts_root_path,
        "run_id": run_id,
        "restored_files": restored,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the deployed reliability bundle and required cadence inputs "
            "from a local artifacts root into the local workspace."
        )
    )
    parser.add_argument(
        "--artifacts-root-uri",
        required=True,
        help=(
            "Base local filesystem path for the external artifacts tree that mirrors "
            "the repository's artifacts/ layout."
        ),
    )
    parser.add_argument(
        "--manifest-uri",
        default=None,
        help=(
            "Optional explicit local path for reliability_promotion_deploy_manifest.json. "
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