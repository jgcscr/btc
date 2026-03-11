from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _list_run_dirs(run_root: Path) -> Set[str]:
    if not run_root.exists():
        return set()
    return {p.name for p in run_root.iterdir() if p.is_dir()}


def _profile_id_for_run(run_root: Path, run_id: str) -> str:
    manifest_path = run_root / run_id / "summary" / "workflow_manifest.json"
    if not manifest_path.exists():
        return ""
    manifest = _load_json(manifest_path)
    profile = manifest.get("profile") if isinstance(manifest.get("profile"), dict) else {}
    return str(profile.get("id", "")).strip()


def _find_new_run_id(
    *,
    run_root: Path,
    before: Set[str],
    expected_profile_id: str,
) -> str:
    after = _list_run_dirs(run_root)
    new_ids = sorted(after.difference(before))
    if not new_ids:
        raise RuntimeError("No new reliability run directory detected after workflow execution.")

    profile_matches = [
        run_id
        for run_id in new_ids
        if _profile_id_for_run(run_root, run_id) == expected_profile_id
    ]
    if len(profile_matches) == 1:
        return profile_matches[0]
    if len(profile_matches) > 1:
        return sorted(profile_matches)[-1]

    # Fallback: take latest new run if profile metadata is temporarily unavailable.
    return sorted(new_ids)[-1]


def _run_workflow(
    *,
    config: Path,
    run_root: Path,
    continue_on_promotion_fail: bool,
) -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.scripts.run_reliability_workflow",
        "--config",
        str(config),
        "--run-root",
        str(run_root),
    ]
    if continue_on_promotion_fail:
        cmd.append("--continue-on-promotion-fail")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _write_default_config_with_pinned_snapshot(
    *,
    base_config: Path,
    pinned_snapshot: Path,
    pinned_snapshot_meta: Path | None,
    pinned_labeled_csv: Path | None,
    pinned_labeled_meta: Path | None,
) -> Path:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML config at {base_config}")

    quality_obj = payload.get("quality")
    quality = quality_obj if isinstance(quality_obj, dict) else {}
    canonical_obj = quality.get("canonical_direction_dataset")
    canonical = canonical_obj if isinstance(canonical_obj, dict) else {}
    canonical["pinned_dataset_path"] = str(pinned_snapshot)
    if pinned_snapshot_meta is not None:
        canonical["pinned_meta_path"] = str(pinned_snapshot_meta)
    else:
        canonical.pop("pinned_meta_path", None)
    quality["canonical_direction_dataset"] = canonical
    if pinned_labeled_csv is not None:
        quality["pinned_labeled_csv_path"] = str(pinned_labeled_csv)
    else:
        quality.pop("pinned_labeled_csv_path", None)
    if pinned_labeled_meta is not None:
        quality["pinned_labeled_meta_path"] = str(pinned_labeled_meta)
    else:
        quality.pop("pinned_labeled_meta_path", None)
    payload["quality"] = quality

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="reliability_workflow.default.matched.",
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return Path(handle.name)


def _write_midband_config_with_walkforward_dataset(
    *,
    base_config: Path,
    walkforward_dataset: Path,
    pinned_labeled_csv: Path | None,
    pinned_labeled_meta: Path | None,
) -> Path:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML config at {base_config}")

    quality_obj = payload.get("quality")
    quality = quality_obj if isinstance(quality_obj, dict) else {}
    quality["walkforward_dataset"] = str(walkforward_dataset)
    if pinned_labeled_csv is not None:
        quality["pinned_labeled_csv_path"] = str(pinned_labeled_csv)
    else:
        quality.pop("pinned_labeled_csv_path", None)
    if pinned_labeled_meta is not None:
        quality["pinned_labeled_meta_path"] = str(pinned_labeled_meta)
    else:
        quality.pop("pinned_labeled_meta_path", None)
    payload["quality"] = quality

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="reliability_workflow.midband.matched.",
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return Path(handle.name)


def _read_watchlist(watchlist_path: Path) -> Dict[str, Any]:
    payload = _load_json(watchlist_path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    triggers = payload.get("triggers") if isinstance(payload.get("triggers"), dict) else {}

    return {
        "recommendation_status": str(payload.get("recommendation_status", "unknown")),
        "matched_pair_count": int(summary.get("total_matched_pairs", 0) or 0),
        "additional_pairs_needed_for_formal_reassessment": int(
            summary.get("additional_pairs_needed_for_formal_reassessment", 0) or 0
        ),
        "early_reassessment_trigger_active": bool(
            triggers.get("early_operational_streak_triggered", False)
            or triggers.get("early_actionable_asymmetry_streak_triggered", False)
        ),
    }


def _resolve_replay_inputs_from_run_id(run_root: Path, run_id: str) -> Dict[str, Path | None]:
    summary_dir = run_root / run_id / "summary"
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory for pinned run id not found: {summary_dir}")

    baseline_pack_path = summary_dir / "trusted_baseline_pack.json"
    if baseline_pack_path.exists():
        baseline_pack = _load_json(baseline_pack_path)
        replay_inputs = baseline_pack.get("replay_inputs") if isinstance(baseline_pack.get("replay_inputs"), dict) else {}
        snapshot = Path(str(replay_inputs.get("snapshot", "")))
        if snapshot.exists():
            snapshot_meta_value = replay_inputs.get("snapshot_meta")
            labeled_csv_value = replay_inputs.get("labeled_csv")
            labeled_meta_value = replay_inputs.get("labeled_meta")
            labeled_csv = Path(str(labeled_csv_value)) if labeled_csv_value else None
            if labeled_csv is None or not labeled_csv.exists():
                raise FileNotFoundError(
                    "Pinned run baseline pack is missing a valid run-local labeled backtest snapshot: "
                    f"{labeled_csv_value}"
                )
            snapshot_meta = Path(str(snapshot_meta_value)) if snapshot_meta_value else None
            labeled_meta = Path(str(labeled_meta_value)) if labeled_meta_value else None
            return {
                "snapshot": snapshot,
                "snapshot_meta": snapshot_meta if snapshot_meta is not None and snapshot_meta.exists() else None,
                "labeled_csv": labeled_csv,
                "labeled_meta": labeled_meta if labeled_meta is not None and labeled_meta.exists() else None,
            }

    snapshot = summary_dir / "btc_features_1h_direction_splits.snapshot.npz"
    if not snapshot.exists():
        raise FileNotFoundError(f"Pinned run is missing snapshot dataset: {snapshot}")

    snapshot_meta = summary_dir / "btc_features_1h_direction_meta.snapshot.json"
    labeled_csv = summary_dir / "labeled_backtest.snapshot.csv"
    if not labeled_csv.exists():
        raise FileNotFoundError(
            "Pinned run is missing run-local labeled backtest snapshot: "
            f"{labeled_csv}. Recreate the source run with the newer replay-support workflow first."
        )

    labeled_meta_candidates = [
        summary_dir / "labeled_backtest_meta.snapshot.json",
        summary_dir / "labeled_backtest_meta.json",
    ]
    labeled_meta = next((path for path in labeled_meta_candidates if path.exists()), None)

    return {
        "snapshot": snapshot,
        "snapshot_meta": snapshot_meta if snapshot_meta.exists() else None,
        "labeled_csv": labeled_csv,
        "labeled_meta": labeled_meta,
    }


def _resolve_replay_inputs_from_cycle_id(run_root: Path, cycle_id: str) -> Dict[str, Any]:
    cycle_path = run_root / "cycles" / f"{cycle_id}.json"
    if not cycle_path.exists():
        raise FileNotFoundError(f"Pinned cycle artifact not found: {cycle_path}")

    payload = _load_json(cycle_path)
    default_run_id = str(payload.get("default_run_id", "")).strip()
    midband_run_id = str(payload.get("midband_run_id", "")).strip()
    if not default_run_id:
        raise ValueError(f"Pinned cycle artifact is missing default_run_id: {cycle_path}")
    resolved_inputs = _resolve_replay_inputs_from_run_id(run_root, default_run_id)
    resolved_inputs["default_run_id"] = default_run_id
    resolved_inputs["midband_run_id"] = midband_run_id or None
    resolved_inputs["cycle_path"] = cycle_path
    return resolved_inputs


def _pair_matched_in_longitudinal(
    *,
    longitudinal_path: Path,
    default_run_id: str,
    midband_run_id: str,
) -> bool:
    if not longitudinal_path.exists():
        return False
    payload = _load_json(longitudinal_path)
    pairs_obj = payload.get("pairs", [])
    if not isinstance(pairs_obj, list):
        return False

    for row in pairs_obj:
        if not isinstance(row, dict):
            continue
        if str(row.get("default_run_id", "")) == default_run_id and str(
            row.get("midband_run_id", "")
        ) == midband_run_id:
            return True
    return False


def _read_trust_status(run_root: Path, run_id: str) -> bool | None:
    path = run_root / run_id / "summary" / "edge_trustworthiness.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    return bool(payload.get("edge_trustworthy", False))


def _read_overlap_source_dataset(run_root: Path, run_id: str) -> str | None:
    path = run_root / run_id / "summary" / "walkforward_labeled_overlap_meta.json"
    if not path.exists():
        return None
    payload = _load_json(path)
    source_dataset = payload.get("source_dataset")
    if source_dataset is None:
        return None
    return str(source_dataset)


def _read_promotion_gate(run_root: Path, run_id: str) -> Dict[str, Any] | None:
    path = run_root / run_id / "summary" / "promotion_gate.json"
    if not path.exists():
        return None
    return _load_json(path)


def _promotion_unchanged(gate: Dict[str, Any] | None) -> bool | None:
    if gate is None:
        return None
    reason = str(gate.get("reason", "")).strip()
    promote = gate.get("promote")
    if promote is False and reason in {
        "champion_challenger_blocked",
        "not_evaluated_incumbent_quality_missing",
        "not_evaluated_candidate_quality_missing",
    }:
        return True
    return False


def _promotion_status(gate: Dict[str, Any] | None) -> Dict[str, Any]:
    if gate is None:
        return {
            "artifact_exists": False,
            "promote": None,
            "reason": None,
            "evaluated": None,
        }
    return {
        "artifact_exists": True,
        "promote": gate.get("promote"),
        "reason": gate.get("reason"),
        "evaluated": gate.get("evaluated", True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic default->midband matched reliability cycle and write a concise cycle summary.",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=Path("configs/reliability_workflow.default.yaml"),
        help="Default runtime reliability workflow config.",
    )
    parser.add_argument(
        "--midband-config",
        type=Path,
        default=Path("configs/reliability_workflow.midband_paper.yaml"),
        help="Midband paper reliability workflow config.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Reliability workflow run root.",
    )
    parser.add_argument(
        "--default-pinned-snapshot",
        type=Path,
        default=None,
        help="Optional fixed default 1h direction snapshot to replay instead of rebuilding the latest canonical dataset.",
    )
    parser.add_argument(
        "--default-pinned-snapshot-meta",
        type=Path,
        default=None,
        help="Optional metadata JSON paired with --default-pinned-snapshot.",
    )
    parser.add_argument(
        "--default-pinned-labeled-csv",
        type=Path,
        default=None,
        help="Optional run-local labeled backtest CSV to pin for both default and midband replay runs.",
    )
    parser.add_argument(
        "--default-pinned-labeled-meta",
        type=Path,
        default=None,
        help="Optional metadata JSON paired with --default-pinned-labeled-csv.",
    )
    parser.add_argument(
        "--default-pinned-run-id",
        type=str,
        default=None,
        help="Replay a prior run id by resolving its run-local snapshot and labeled backtest artifacts automatically.",
    )
    parser.add_argument(
        "--default-pinned-cycle-id",
        type=str,
        default=None,
        help="Replay from a prior matched-cycle id by resolving the cycle's default-side run-local snapshot and labeled artifacts automatically.",
    )
    parser.add_argument(
        "--continue-on-promotion-fail",
        action="store_true",
        help="Pass through continue-on-promotion-fail to each workflow invocation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reliability/default_midband_matched_cycle_latest.json"),
        help="Canonical cycle summary artifact path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.default_config.exists():
        raise FileNotFoundError(args.default_config)
    if not args.midband_config.exists():
        raise FileNotFoundError(args.midband_config)
    pinned_resolver_count = sum(
        1
        for value in (args.default_pinned_run_id, args.default_pinned_cycle_id)
        if value is not None
    )
    if pinned_resolver_count > 1:
        raise ValueError("Specify only one of --default-pinned-run-id or --default-pinned-cycle-id.")
    if pinned_resolver_count and any(
        value is not None
        for value in (
            args.default_pinned_snapshot,
            args.default_pinned_snapshot_meta,
            args.default_pinned_labeled_csv,
            args.default_pinned_labeled_meta,
        )
    ):
        raise ValueError(
            "Resolver-based replay arguments cannot be combined with manual --default-pinned-* path arguments."
        )

    resolved_pinned_cycle_id = args.default_pinned_cycle_id
    resolved_pinned_run_id = args.default_pinned_run_id
    resolved_pinned_snapshot = args.default_pinned_snapshot
    resolved_pinned_snapshot_meta = args.default_pinned_snapshot_meta
    resolved_pinned_labeled_csv = args.default_pinned_labeled_csv
    resolved_pinned_labeled_meta = args.default_pinned_labeled_meta

    if resolved_pinned_cycle_id:
        resolved_inputs = _resolve_replay_inputs_from_cycle_id(args.run_root, resolved_pinned_cycle_id)
        resolved_pinned_run_id = str(resolved_inputs["default_run_id"])
        resolved_pinned_snapshot = resolved_inputs["snapshot"]
        resolved_pinned_snapshot_meta = resolved_inputs["snapshot_meta"]
        resolved_pinned_labeled_csv = resolved_inputs["labeled_csv"]
        resolved_pinned_labeled_meta = resolved_inputs["labeled_meta"]
    elif resolved_pinned_run_id:
        resolved_inputs = _resolve_replay_inputs_from_run_id(args.run_root, resolved_pinned_run_id)
        resolved_pinned_snapshot = resolved_inputs["snapshot"]
        resolved_pinned_snapshot_meta = resolved_inputs["snapshot_meta"]
        resolved_pinned_labeled_csv = resolved_inputs["labeled_csv"]
        resolved_pinned_labeled_meta = resolved_inputs["labeled_meta"]
    if resolved_pinned_snapshot is not None and not resolved_pinned_snapshot.exists():
        raise FileNotFoundError(resolved_pinned_snapshot)
    if resolved_pinned_snapshot_meta is not None and not resolved_pinned_snapshot_meta.exists():
        raise FileNotFoundError(resolved_pinned_snapshot_meta)
    if resolved_pinned_labeled_csv is not None and not resolved_pinned_labeled_csv.exists():
        raise FileNotFoundError(resolved_pinned_labeled_csv)
    if resolved_pinned_labeled_meta is not None and not resolved_pinned_labeled_meta.exists():
        raise FileNotFoundError(resolved_pinned_labeled_meta)

    args.run_root.mkdir(parents=True, exist_ok=True)

    default_config_for_cycle = args.default_config
    if resolved_pinned_snapshot is not None:
        default_config_for_cycle = _write_default_config_with_pinned_snapshot(
            base_config=args.default_config,
            pinned_snapshot=resolved_pinned_snapshot,
            pinned_snapshot_meta=resolved_pinned_snapshot_meta,
            pinned_labeled_csv=resolved_pinned_labeled_csv,
            pinned_labeled_meta=resolved_pinned_labeled_meta,
        )

    before_default = _list_run_dirs(args.run_root)
    try:
        _run_workflow(
            config=default_config_for_cycle,
            run_root=args.run_root,
            continue_on_promotion_fail=bool(args.continue_on_promotion_fail),
        )
    finally:
        if default_config_for_cycle != args.default_config:
            default_config_for_cycle.unlink(missing_ok=True)
    default_run_id = _find_new_run_id(
        run_root=args.run_root,
        before=before_default,
        expected_profile_id="default_runtime",
    )

    default_snapshot_dataset = (
        args.run_root / default_run_id / "summary" / "btc_features_1h_direction_splits.snapshot.npz"
    )
    if not default_snapshot_dataset.exists():
        raise FileNotFoundError(
            f"Expected default snapshot dataset for matched cycle not found: {default_snapshot_dataset}"
        )

    midband_config_for_cycle = _write_midband_config_with_walkforward_dataset(
        base_config=args.midband_config,
        walkforward_dataset=default_snapshot_dataset,
        pinned_labeled_csv=resolved_pinned_labeled_csv,
        pinned_labeled_meta=resolved_pinned_labeled_meta,
    )

    before_midband = _list_run_dirs(args.run_root)
    try:
        _run_workflow(
            config=midband_config_for_cycle,
            run_root=args.run_root,
            continue_on_promotion_fail=bool(args.continue_on_promotion_fail),
        )
    finally:
        midband_config_for_cycle.unlink(missing_ok=True)
    midband_run_id = _find_new_run_id(
        run_root=args.run_root,
        before=before_midband,
        expected_profile_id="midband_paper_evaluation",
    )

    midband_summary_dir = args.run_root / midband_run_id / "summary"
    longitudinal_path = midband_summary_dir / "default_vs_midband_paper_live_longitudinal.json"
    watchlist_canonical_path = args.run_root / "default_vs_midband_paper_live_watchlist.json"
    watchlist_run_scoped_path = midband_summary_dir / "default_vs_midband_paper_live_watchlist.json"
    watchlist_path = watchlist_canonical_path if watchlist_canonical_path.exists() else watchlist_run_scoped_path
    if not watchlist_path.exists():
        raise FileNotFoundError(
            f"Expected watchlist artifact not found at {watchlist_canonical_path} or {watchlist_run_scoped_path}"
        )

    watchlist = _read_watchlist(watchlist_path)
    pair_matched = _pair_matched_in_longitudinal(
        longitudinal_path=longitudinal_path,
        default_run_id=default_run_id,
        midband_run_id=midband_run_id,
    )

    default_promotion_gate = _read_promotion_gate(args.run_root, default_run_id)
    midband_promotion_gate = _read_promotion_gate(args.run_root, midband_run_id)
    trust_default = _read_trust_status(args.run_root, default_run_id)
    trust_midband = _read_trust_status(args.run_root, midband_run_id)
    default_overlap_source = _read_overlap_source_dataset(args.run_root, default_run_id)
    midband_overlap_source = _read_overlap_source_dataset(args.run_root, midband_run_id)
    if default_overlap_source is None or midband_overlap_source is None:
        raise RuntimeError(
            "Matched cycle is missing walkforward_labeled_overlap_meta.json on one or both runs, so overlap lineage cannot be verified."
        )
    if default_overlap_source != midband_overlap_source:
        raise RuntimeError(
            "Matched cycle overlap lineage mismatch: "
            f"default uses '{default_overlap_source}' while midband uses '{midband_overlap_source}'."
        )

    cycle_id = f"{default_run_id}__{midband_run_id}"
    cycle_payload = {
        "generated_at": _utc_now(),
        "cycle_id": cycle_id,
        "default_run_id": default_run_id,
        "midband_run_id": midband_run_id,
        "pair_successfully_matched": bool(pair_matched),
        "watchlist": watchlist,
        "validation": {
            "trust_default": trust_default,
            "trust_midband": trust_midband,
            "overlap_lineage": {
                "default_source_dataset": default_overlap_source,
                "midband_source_dataset": midband_overlap_source,
                "paths_match": True,
            },
            "trust_artifacts_exist": {
                "default": trust_default is not None,
                "midband": trust_midband is not None,
            },
            "promotion": {
                "default": _promotion_status(default_promotion_gate),
                "midband": _promotion_status(midband_promotion_gate),
            },
            "promotion_gate_artifacts_exist": {
                "default": default_promotion_gate is not None,
                "midband": midband_promotion_gate is not None,
            },
            "default_promotion_unchanged": _promotion_unchanged(default_promotion_gate),
        },
        "paths": {
            "run_root": str(args.run_root),
            "midband_longitudinal": str(longitudinal_path),
            "watchlist": str(watchlist_path),
            "default_pinned_cycle_id": resolved_pinned_cycle_id,
            "default_pinned_run_id": resolved_pinned_run_id,
            "default_pinned_snapshot": str(resolved_pinned_snapshot) if resolved_pinned_snapshot else None,
            "default_pinned_snapshot_meta": str(resolved_pinned_snapshot_meta) if resolved_pinned_snapshot_meta else None,
            "default_pinned_labeled_csv": str(resolved_pinned_labeled_csv) if resolved_pinned_labeled_csv else None,
            "default_pinned_labeled_meta": str(resolved_pinned_labeled_meta) if resolved_pinned_labeled_meta else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cycle_payload, indent=2), encoding="utf-8")

    cycle_history_dir = args.run_root / "cycles"
    cycle_history_dir.mkdir(parents=True, exist_ok=True)
    cycle_history_path = cycle_history_dir / f"{cycle_id}.json"
    cycle_history_path.write_text(json.dumps(cycle_payload, indent=2), encoding="utf-8")

    print(json.dumps(cycle_payload, indent=2))


if __name__ == "__main__":
    main()