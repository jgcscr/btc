from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update consolidated longitudinal default-vs-midband shadow artifact.")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--track",
        type=str,
        default="shadow_retrospective",
        choices=["shadow_retrospective", "paper_profile"],
    )
    return parser.parse_args()


def _extract_run_record(run_id: str, comparison_payload: Dict[str, Any], comparison_path: Path) -> Dict[str, Any]:
    aggregate = comparison_payload.get("aggregate_summary", {})
    metrics = comparison_payload.get("metrics", {})
    has_aggregate = isinstance(aggregate, dict) and bool(aggregate)
    has_metrics = isinstance(metrics, dict) and bool(metrics)
    run_record = {
        "run_id": str(run_id),
        "generated_at": _utc_now(),
        "comparison_path": str(comparison_path),
        "run_level_verdict": comparison_payload.get(
            "run_level_verdict",
            "midband better"
            if has_metrics and float(metrics.get("mean_diff", 0.0) or 0.0) > 0.0
            else "inconclusive",
        ),
        "number_of_evaluated_windows": int(aggregate.get("number_of_evaluated_windows", 0) or 0),
        "windows_improved_by_mean_diff": int(aggregate.get("windows_improved_by_mean_diff", 0) or 0),
        "windows_improved_by_candidate_net_return": int(aggregate.get("windows_improved_by_candidate_net_return", 0) or 0),
        "clearly_harmed_window_ids": [
            int(window.get("window_id", 0))
            for window in (aggregate.get("clearly_harmed_windows", []) or [])
            if isinstance(window, dict)
        ],
        "aggregate_delta_candidate_net_return_total": float(
            aggregate.get("aggregate_delta_candidate_net_return_total", float("nan"))
        ),
        "aggregate_delta_mean_diff": float(aggregate.get("aggregate_delta_mean_diff", float("nan"))),
        "median_vetoed_rows": float(aggregate.get("median_vetoed_rows", float("nan"))),
        "mean_vetoed_rows": float(aggregate.get("mean_vetoed_rows", float("nan"))),
    }
    if has_metrics:
        run_record["profile_metrics"] = {
            "candidate_trade_count": int(metrics.get("candidate_trade_count", 0) or 0),
            "incumbent_trade_count": int(metrics.get("incumbent_trade_count", 0) or 0),
            "candidate_net_return_total": float(metrics.get("candidate_net_return_total", float("nan"))),
            "incumbent_net_return_total": float(metrics.get("incumbent_net_return_total", float("nan"))),
            "mean_diff": float(metrics.get("mean_diff", float("nan"))),
            "pvalue_one_sided": float(metrics.get("pvalue_one_sided", float("nan"))),
            "nonzero_paired_rows": int(metrics.get("nonzero_paired_rows", 0) or 0),
            "std_diff": float(metrics.get("std_diff", float("nan"))),
            "vetoed_row_count": metrics.get("vetoed_row_count"),
        }
    return run_record


def main() -> None:
    args = parse_args()
    if not args.comparison.exists():
        raise FileNotFoundError(args.comparison)

    comparison_payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    run_record = _extract_run_record(args.run_id, comparison_payload, args.comparison)

    longitudinal_payload: Dict[str, Any]
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            longitudinal_payload = existing
        else:
            longitudinal_payload = {}
    else:
        longitudinal_payload = {}

    runs = longitudinal_payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    # Backward compatibility: migrate legacy flat runs into shadow_retrospective track.
    tracks_obj = longitudinal_payload.get("tracks", {})
    tracks = tracks_obj if isinstance(tracks_obj, dict) else {}
    for key in ("shadow_retrospective", "paper_profile"):
        existing_track = tracks.get(key, {})
        if not isinstance(existing_track, dict):
            existing_track = {}
        existing_track_runs = existing_track.get("runs", [])
        if not isinstance(existing_track_runs, list):
            existing_track_runs = []
        tracks[key] = {
            "runs": existing_track_runs,
            "latest_run_id": existing_track.get("latest_run_id"),
            "latest": existing_track.get("latest"),
        }
    if runs:
        tracks["shadow_retrospective"]["runs"] = [
            record for record in tracks["shadow_retrospective"]["runs"] if isinstance(record, dict)
        ] + [record for record in runs if isinstance(record, dict)]

    selected_track = str(args.track)
    selected_runs = tracks[selected_track].get("runs", [])
    if not isinstance(selected_runs, list):
        selected_runs = []

    filtered_runs: List[Dict[str, Any]] = [
        record for record in selected_runs if isinstance(record, dict) and str(record.get("run_id")) != str(args.run_id)
    ]
    filtered_runs.append(run_record)
    filtered_runs.sort(key=lambda record: str(record.get("run_id", "")))

    tracks[selected_track]["runs"] = filtered_runs
    tracks[selected_track]["latest_run_id"] = str(args.run_id)
    tracks[selected_track]["latest"] = run_record

    longitudinal_payload = {
        "updated_at": _utc_now(),
        "tracked_comparison": "default_vs_midband",
        "tracks": tracks,
        "legacy": {
            "active_track": selected_track,
            "latest_run_id": str(args.run_id),
            "latest": run_record,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(longitudinal_payload, indent=2), encoding="utf-8")
    print(json.dumps(longitudinal_payload, indent=2))


if __name__ == "__main__":
    main()
