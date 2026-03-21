from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _counter_payload(values: Iterable[str]) -> Dict[str, int]:
    counter = Counter(str(value) for value in values if str(value))
    return {key: int(counter[key]) for key in sorted(counter.keys())}


def _infer_source_reliability_run_id(row: Mapping[str, Any]) -> str | None:
    explicit = str(row.get("source_reliability_run_id") or "").strip()
    if explicit:
        return explicit
    thresholds_path = row.get("thresholds_json")
    if not thresholds_path:
        return None
    path = Path(str(thresholds_path))
    summary_dir = path.parent
    run_dir = summary_dir.parent if summary_dir.name == "summary" else summary_dir
    run_id = str(run_dir.name).strip()
    return run_id or None


def _coerce_runs(track: Mapping[str, Any] | Dict[str, Any]) -> List[Dict[str, Any]]:
    runs_obj = track.get("runs") if isinstance(track, dict) else []
    return [row for row in runs_obj if isinstance(row, dict)] if isinstance(runs_obj, list) else []


def _format_list(values: Iterable[Any]) -> str:
    rendered = [str(value) for value in values if str(value)]
    return ", ".join(rendered) if rendered else "none"


def _select_track(payload: Dict[str, Any], track_key: str | None) -> tuple[str, Dict[str, Any]]:
    tracks_obj = payload.get("tracks")
    tracks = tracks_obj if isinstance(tracks_obj, dict) else {}
    if not tracks:
        raise ValueError("Longitudinal payload does not contain any tracks.")

    if track_key:
        selected = tracks.get(track_key)
        if not isinstance(selected, dict):
            raise KeyError(f"Track not found: {track_key}")
        return track_key, selected

    legacy = payload.get("legacy") if isinstance(payload.get("legacy"), dict) else {}
    active_track = str(legacy.get("active_track") or "")
    if active_track and isinstance(tracks.get(active_track), dict):
        return active_track, tracks[active_track]

    first_track_key = sorted(str(key) for key in tracks.keys())[0]
    return first_track_key, tracks[first_track_key]


def build_summary(longitudinal_payload: Dict[str, Any], *, track_key: str | None = None) -> Dict[str, Any]:
    selected_track_key, track = _select_track(longitudinal_payload, track_key)
    runs = _coerce_runs(track)
    latest = track.get("latest") if isinstance(track.get("latest"), dict) else {}

    lhs_label = str(track.get("lhs_label") or latest.get("lhs_label") or "lhs")
    rhs_label = str(track.get("rhs_label") or latest.get("rhs_label") or "rhs")

    source_reliability_counts = _counter_payload(
        str(_infer_source_reliability_run_id(row) or "") for row in runs
    )
    operational_diff_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("operational_diff_horizons") or [])
    )
    decision_state_only_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("decision_state_only_diff_horizons") or [])
    )
    score_only_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("score_only_diff_horizons") or [])
    )
    differing_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("differing_horizons") or [])
    )
    lhs_actionable_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("lhs_actionable_horizons") or [])
    )
    rhs_actionable_counts = _counter_payload(
        horizon for row in runs for horizon in (row.get("rhs_actionable_horizons") or [])
    )

    return {
        "generated_at": _utc_now(),
        "tracked_comparison": str(longitudinal_payload.get("tracked_comparison") or "shadow_profile_comparison"),
        "track_key": selected_track_key,
        "input_updated_at": longitudinal_payload.get("updated_at"),
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "total_runs": len(runs),
        "latest_run_id": latest.get("run_id"),
        "latest_generated_at": latest.get("generated_at"),
        "latest_source_reliability_run_id": _infer_source_reliability_run_id(latest),
        "latest_summary": {
            "profiles_differ": bool(latest.get("profiles_differ", False)),
            "difference_only_probabilities_or_scores": bool(
                latest.get("difference_only_probabilities_or_scores", False)
            ),
            "either_profile_actionable": bool(latest.get("either_profile_actionable", False)),
            "both_resolve_to_hold": bool(latest.get("both_resolve_to_hold", False)),
            "operationally_meaningful_difference": bool(
                latest.get("operationally_meaningful_difference", False)
            ),
            "operational_diff_horizons": [str(value) for value in (latest.get("operational_diff_horizons") or [])],
            "decision_state_only_diff_horizons": [
                str(value) for value in (latest.get("decision_state_only_diff_horizons") or [])
            ],
            "score_only_diff_horizons": [str(value) for value in (latest.get("score_only_diff_horizons") or [])],
            "lhs_actionable_horizons": [str(value) for value in (latest.get("lhs_actionable_horizons") or [])],
            "rhs_actionable_horizons": [str(value) for value in (latest.get("rhs_actionable_horizons") or [])],
        },
        "aggregate_counts": {
            "profiles_differ_runs": sum(bool(row.get("profiles_differ", False)) for row in runs),
            "difference_only_probabilities_or_scores_runs": sum(
                bool(row.get("difference_only_probabilities_or_scores", False)) for row in runs
            ),
            "operationally_meaningful_difference_runs": sum(
                bool(row.get("operationally_meaningful_difference", False)) for row in runs
            ),
            "either_profile_actionable_runs": sum(bool(row.get("either_profile_actionable", False)) for row in runs),
            "both_resolve_to_hold_runs": sum(bool(row.get("both_resolve_to_hold", False)) for row in runs),
            f"{lhs_label}_actionable_runs": sum(bool(row.get("lhs_actionable_horizons")) for row in runs),
            f"{rhs_label}_actionable_runs": sum(bool(row.get("rhs_actionable_horizons")) for row in runs),
        },
        "horizon_counts": {
            "operational_diff": operational_diff_counts,
            "decision_state_only_diff": decision_state_only_counts,
            "score_only_diff": score_only_counts,
            "any_difference": differing_counts,
            lhs_label: lhs_actionable_counts,
            rhs_label: rhs_actionable_counts,
        },
        "source_reliability_run_counts": source_reliability_counts,
    }


def build_run_rows(longitudinal_payload: Dict[str, Any], *, track_key: str | None = None) -> List[Dict[str, Any]]:
    _selected_track_key, track = _select_track(longitudinal_payload, track_key)
    runs = _coerce_runs(track)
    lhs_label = str(track.get("lhs_label") or "lhs")
    rhs_label = str(track.get("rhs_label") or "rhs")

    rows: List[Dict[str, Any]] = []
    for row in runs:
        rows.append(
            {
                "run_id": str(row.get("run_id") or ""),
                "generated_at": str(row.get("generated_at") or ""),
                "source_reliability_run_id": str(_infer_source_reliability_run_id(row) or ""),
                "profiles_differ": bool(row.get("profiles_differ", False)),
                "difference_only_probabilities_or_scores": bool(
                    row.get("difference_only_probabilities_or_scores", False)
                ),
                "either_profile_actionable": bool(row.get("either_profile_actionable", False)),
                "both_resolve_to_hold": bool(row.get("both_resolve_to_hold", False)),
                "operationally_meaningful_difference": bool(
                    row.get("operationally_meaningful_difference", False)
                ),
                "operational_diff_horizons": _format_list(row.get("operational_diff_horizons") or []),
                "decision_state_only_diff_horizons": _format_list(
                    row.get("decision_state_only_diff_horizons") or []
                ),
                "score_only_diff_horizons": _format_list(row.get("score_only_diff_horizons") or []),
                f"{lhs_label}_actionable_horizons": _format_list(row.get("lhs_actionable_horizons") or []),
                f"{rhs_label}_actionable_horizons": _format_list(row.get("rhs_actionable_horizons") or []),
            }
        )
    return rows


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    latest_summary = summary.get("latest_summary") if isinstance(summary.get("latest_summary"), Mapping) else {}
    aggregate_counts = summary.get("aggregate_counts") if isinstance(summary.get("aggregate_counts"), Mapping) else {}
    horizon_counts = summary.get("horizon_counts") if isinstance(summary.get("horizon_counts"), Mapping) else {}
    source_counts = summary.get("source_reliability_run_counts") if isinstance(summary.get("source_reliability_run_counts"), Mapping) else {}

    lhs_label = str(summary.get("lhs_label") or "lhs")
    rhs_label = str(summary.get("rhs_label") or "rhs")

    lines = [
        "# Shadow Profile Comparison Summary",
        "",
        f"- Generated at: {summary.get('generated_at')}",
        f"- Track: {summary.get('track_key')}",
        f"- Profiles: {lhs_label} vs {rhs_label}",
        f"- Total runs: {summary.get('total_runs')}",
        f"- Latest run: {summary.get('latest_run_id')} ({summary.get('latest_generated_at')})",
        f"- Latest source reliability run: {summary.get('latest_source_reliability_run_id') or 'unknown'}",
        "",
        "## Latest Outcome",
        "",
        f"- Profiles differ: {bool(latest_summary.get('profiles_differ', False))}",
        f"- Difference only probabilities or scores: {bool(latest_summary.get('difference_only_probabilities_or_scores', False))}",
        f"- Either profile actionable: {bool(latest_summary.get('either_profile_actionable', False))}",
        f"- Both resolve to hold: {bool(latest_summary.get('both_resolve_to_hold', False))}",
        f"- Operationally meaningful difference: {bool(latest_summary.get('operationally_meaningful_difference', False))}",
        f"- Operational diff horizons: {_format_list(latest_summary.get('operational_diff_horizons') or [])}",
        f"- Decision-state-only diff horizons: {_format_list(latest_summary.get('decision_state_only_diff_horizons') or [])}",
        f"- Score-only diff horizons: {_format_list(latest_summary.get('score_only_diff_horizons') or [])}",
        f"- {lhs_label} actionable horizons: {_format_list(latest_summary.get('lhs_actionable_horizons') or [])}",
        f"- {rhs_label} actionable horizons: {_format_list(latest_summary.get('rhs_actionable_horizons') or [])}",
        "",
        "## Aggregate Counts",
        "",
    ]
    for key in sorted(str(name) for name in aggregate_counts.keys()):
        lines.append(f"- {key}: {aggregate_counts[key]}")

    lines.extend(["", "## Horizon Counts", ""])
    for section_name in sorted(str(name) for name in horizon_counts.keys()):
        section = horizon_counts.get(section_name)
        if isinstance(section, Mapping):
            rendered = ", ".join(f"{key}={section[key]}" for key in sorted(str(name) for name in section.keys()))
            lines.append(f"- {section_name}: {rendered or 'none'}")

    lines.extend(["", "## Source Reliability Runs", ""])
    rendered_sources = ", ".join(f"{key}={source_counts[key]}" for key in sorted(str(name) for name in source_counts.keys()))
    lines.append(f"- counts: {rendered_sources or 'none'}")
    lines.append("")
    return "\n".join(lines)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact summary from the shadow profile longitudinal comparison artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.md"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv"),
    )
    parser.add_argument("--track-key", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    payload = _load_json(args.input)
    summary = build_summary(payload, track_key=args.track_key)
    run_rows = build_run_rows(payload, track_key=args.track_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(build_markdown_summary(summary), encoding="utf-8")
    _write_rows_csv(args.csv_output, run_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()