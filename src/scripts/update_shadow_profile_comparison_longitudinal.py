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


def _infer_source_reliability_run_id(manifest_payload: Dict[str, Any]) -> str | None:
    explicit = str(manifest_payload.get("source_reliability_run_id") or "").strip()
    if explicit:
        return explicit
    thresholds_path = manifest_payload.get("thresholds_json")
    if not thresholds_path:
        return None
    path = Path(str(thresholds_path))
    summary_dir = path.parent
    run_dir = summary_dir.parent if summary_dir.name == "summary" else summary_dir
    run_id = str(run_dir.name).strip()
    return run_id or None


def _parse_generated_at(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _run_sort_key(record: Dict[str, Any]) -> tuple[datetime, str]:
    return (_parse_generated_at(record.get("generated_at")), str(record.get("run_id") or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update consolidated longitudinal shadow profile comparison artifact."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _extract_run_record(
    manifest_payload: Dict[str, Any],
    comparison_payload: Dict[str, Any],
    *,
    manifest_path: Path,
    comparison_path: Path,
) -> Dict[str, Any]:
    profiles = manifest_payload.get("profiles") if isinstance(manifest_payload.get("profiles"), dict) else {}
    profile_labels = [str(label) for label in profiles.keys()]
    if len(profile_labels) < 2:
        raise ValueError("Shadow comparison manifest must contain two profiles.")
    lhs_label = profile_labels[0]
    rhs_label = profile_labels[1]
    pair_key = f"{lhs_label}_vs_{rhs_label}"

    summary = comparison_payload.get("overall_summary") if isinstance(comparison_payload.get("overall_summary"), dict) else {}
    per_horizon = comparison_payload.get("per_horizon") if isinstance(comparison_payload.get("per_horizon"), dict) else {}

    lhs_actionable_horizons: List[str] = []
    rhs_actionable_horizons: List[str] = []
    differing_horizons: List[str] = []
    for horizon, payload in per_horizon.items():
        if not isinstance(payload, dict):
            continue
        flags = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
        if bool(flags.get(f"{lhs_label}_actionable")):
            lhs_actionable_horizons.append(str(horizon))
        if bool(flags.get(f"{rhs_label}_actionable")):
            rhs_actionable_horizons.append(str(horizon))
        if any(
            bool(flags.get(name))
            for name in ("differs_operationally", "differs_decision_state", "differs_score_level")
        ):
            differing_horizons.append(str(horizon))

    return {
        "run_id": str(manifest_payload.get("run_id") or ""),
        "generated_at": str(manifest_payload.get("generated_at") or _utc_now()),
        "source_reliability_run_id": _infer_source_reliability_run_id(manifest_payload),
        "manifest_path": str(manifest_path),
        "comparison_path": str(comparison_path),
        "profile_pair": pair_key,
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "restore_latest_to": manifest_payload.get("restore_latest_to"),
        "targets": manifest_payload.get("targets"),
        "thresholds_json": manifest_payload.get("thresholds_json"),
        "platt_calibration": manifest_payload.get("platt_calibration"),
        "profiles_differ": bool(summary.get("profiles_differ", False)),
        "difference_only_probabilities_or_scores": bool(
            summary.get("difference_only_probabilities_or_scores", False)
        ),
        "either_profile_actionable": bool(summary.get("either_profile_actionable", False)),
        "both_resolve_to_hold": bool(summary.get("both_resolve_to_hold", False)),
        "operationally_meaningful_difference": bool(summary.get("operationally_meaningful_difference", False)),
        "operational_diff_horizons": [str(value) for value in (summary.get("operational_diff_horizons") or [])],
        "decision_state_only_diff_horizons": [
            str(value) for value in (summary.get("decision_state_only_diff_horizons") or [])
        ],
        "score_only_diff_horizons": [str(value) for value in (summary.get("score_only_diff_horizons") or [])],
        "differing_horizons": differing_horizons,
        "lhs_actionable_horizons": lhs_actionable_horizons,
        "rhs_actionable_horizons": rhs_actionable_horizons,
    }


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if not args.comparison.exists():
        raise FileNotFoundError(args.comparison)

    manifest_payload = _load_json(args.manifest)
    comparison_payload = _load_json(args.comparison)
    run_record = _extract_run_record(
        manifest_payload,
        comparison_payload,
        manifest_path=args.manifest,
        comparison_path=args.comparison,
    )

    longitudinal_payload: Dict[str, Any]
    if args.output.exists():
        existing = _load_json(args.output)
        longitudinal_payload = existing if isinstance(existing, dict) else {}
    else:
        longitudinal_payload = {}

    tracks_obj = longitudinal_payload.get("tracks")
    tracks = tracks_obj if isinstance(tracks_obj, dict) else {}
    pair_key = str(run_record.get("profile_pair"))
    existing_track = tracks.get(pair_key)
    track = existing_track if isinstance(existing_track, dict) else {}
    existing_runs = track.get("runs")
    runs = existing_runs if isinstance(existing_runs, list) else []

    filtered_runs = [
        record
        for record in runs
        if isinstance(record, dict) and str(record.get("run_id")) != str(run_record.get("run_id"))
    ]
    filtered_runs.append(run_record)
    filtered_runs.sort(key=_run_sort_key)
    latest_record = filtered_runs[-1] if filtered_runs else run_record

    tracks[pair_key] = {
        "lhs_label": run_record.get("lhs_label"),
        "rhs_label": run_record.get("rhs_label"),
        "runs": filtered_runs,
        "latest_run_id": latest_record.get("run_id"),
        "latest": latest_record,
    }

    longitudinal_payload = {
        "updated_at": _utc_now(),
        "tracked_comparison": "shadow_profile_comparison",
        "tracks": tracks,
        "legacy": {
            "active_track": pair_key,
            "latest_run_id": latest_record.get("run_id"),
            "latest": latest_record,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(longitudinal_payload, indent=2), encoding="utf-8")
    print(json.dumps(longitudinal_payload, indent=2))


if __name__ == "__main__":
    main()