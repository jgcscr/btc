from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _optional_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _resolve_snapshot_path(summary_dir: Path) -> Path:
    snapshot = summary_dir / "btc_features_1h_direction_splits.snapshot.npz"
    if snapshot.exists():
        return snapshot

    overlap_meta_path = summary_dir / "walkforward_labeled_overlap_meta.json"
    if overlap_meta_path.exists():
        overlap_meta = _load_json(overlap_meta_path)
        source_dataset = overlap_meta.get("source_dataset")
        if source_dataset:
            candidate = Path(str(source_dataset))
            if candidate.exists():
                return candidate

    raise FileNotFoundError(snapshot)


def _snapshot_meta_path(snapshot: Path) -> Path | None:
    snapshot_name = snapshot.name
    if snapshot_name.endswith("_splits.snapshot.npz"):
        candidate_name = snapshot_name.replace("_splits.snapshot.npz", "_meta.snapshot.json")
        candidate = snapshot.with_name(candidate_name)
        if candidate.exists():
            return candidate
    return None


def _paired_raw_snapshot_path(snapshot: Path, overlap: bool = False) -> Path | None:
    suffix = "direction_features_raw.labeled_overlap.csv" if overlap else "direction_features_raw.snapshot.csv"
    candidate = snapshot.parent / suffix
    if candidate.exists():
        return candidate
    return None


def _paired_raw_snapshot_meta_path(snapshot: Path, overlap: bool = False) -> Path | None:
    suffix = (
        "direction_features_raw.labeled_overlap_meta.json"
        if overlap
        else "direction_features_raw.snapshot_meta.json"
    )
    candidate = snapshot.parent / suffix
    if candidate.exists():
        return candidate
    return None


def _collect_compare_details(compare_summary_path: Path) -> Dict[str, Any] | None:
    if not compare_summary_path.exists():
        return None
    payload = _load_json(compare_summary_path)
    rows_obj = payload.get("rows", [])
    rows = rows_obj if isinstance(rows_obj, list) else []
    detail_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        detail_rows.append(
            {
                "model_kind": str(row.get("model_kind", "")),
                "path": str(row.get("path", "")) or None,
                "detail_path": str(row.get("detail_path", "")) or None,
            }
        )
    return {
        "summary_path": str(compare_summary_path),
        "selected_model_kind": str(payload.get("selected_model_kind", "")) or None,
        "selection_policy": str(payload.get("selection_policy", "")) or None,
        "detail_rows": detail_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a run-local trusted baseline pack manifest for deterministic replay and future drift checks.",
    )
    parser.add_argument("--run-id", required=True, help="Reliability run id to package.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Reliability workflow run root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the baseline pack manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.run_root / args.run_id / "summary"
    if not summary_dir.exists():
        raise FileNotFoundError(summary_dir)

    snapshot = _resolve_snapshot_path(summary_dir)
    labeled_csv = summary_dir / "labeled_backtest.snapshot.csv"
    overlap_dataset = summary_dir / "btc_features_1h_direction_splits.labeled_overlap.npz"
    edge_path = summary_dir / "edge_trustworthiness.json"
    if not labeled_csv.exists():
        raise FileNotFoundError(labeled_csv)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    edge_payload = _load_json(edge_path)
    snapshot_meta = _snapshot_meta_path(snapshot)
    raw_feature_snapshot = summary_dir / "direction_features_raw.snapshot.csv"
    raw_feature_snapshot_meta = summary_dir / "direction_features_raw.snapshot_meta.json"
    if not raw_feature_snapshot.exists():
        paired = _paired_raw_snapshot_path(snapshot, overlap=False)
        if paired is not None:
            raw_feature_snapshot = paired
    if not raw_feature_snapshot_meta.exists():
        paired_meta = _paired_raw_snapshot_meta_path(snapshot, overlap=False)
        if paired_meta is not None:
            raw_feature_snapshot_meta = paired_meta
    replay_inputs = {
        "snapshot": str(snapshot),
        "snapshot_meta": str(snapshot_meta) if snapshot_meta is not None else None,
        "labeled_csv": str(labeled_csv),
        "labeled_meta": _optional_path(summary_dir / "labeled_backtest_meta.snapshot.json")
        or _optional_path(summary_dir / "labeled_backtest_meta.json"),
        "overlap_dataset": _optional_path(overlap_dataset),
        "overlap_meta": _optional_path(summary_dir / "walkforward_labeled_overlap_meta.json"),
        "raw_feature_snapshot": str(raw_feature_snapshot) if raw_feature_snapshot.exists() else None,
        "raw_feature_snapshot_meta": str(raw_feature_snapshot_meta) if raw_feature_snapshot_meta.exists() else None,
        "raw_feature_overlap_snapshot": _optional_path(summary_dir / "direction_features_raw.labeled_overlap.csv"),
        "raw_feature_overlap_snapshot_meta": _optional_path(summary_dir / "direction_features_raw.labeled_overlap_meta.json"),
    }
    compare_artifacts = {
        "full": _collect_compare_details(summary_dir / "walkforward_model_compare.json"),
        "overlap": _collect_compare_details(summary_dir / "walkforward_model_compare_labeled_overlap.json"),
    }
    trust_artifacts = {
        "edge_trustworthiness": str(edge_path),
        "walkforward_labeled_reconciliation": _optional_path(summary_dir / "walkforward_labeled_reconciliation.json"),
        "overlap_trust_stability": _optional_path(summary_dir / "overlap_trust_stability.json"),
        "canonical_dataset_consistency": _optional_path(summary_dir / "canonical_dataset_consistency.json"),
    }
    workflow_manifest_path = summary_dir / "workflow_manifest.json"
    workflow_manifest = _load_json(workflow_manifest_path) if workflow_manifest_path.exists() else {}
    payload = {
        "generated_at": _utc_now(),
        "run_id": args.run_id,
        "run_root": str(args.run_root),
        "summary_dir": str(summary_dir),
        "edge_trustworthy": bool(edge_payload.get("edge_trustworthy", False)),
        "profile": workflow_manifest.get("profile") if isinstance(workflow_manifest.get("profile"), dict) else None,
        "replay_inputs": replay_inputs,
        "compare_artifacts": compare_artifacts,
        "trust_artifacts": trust_artifacts,
    }

    output_path = args.output or (summary_dir / "trusted_baseline_pack.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
