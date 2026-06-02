from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/downtrend_bias_remediation")
DEFAULT_AUDIT_HORIZONS = ("1h", "4h", "8h", "12h")
DEFAULT_RECALIBRATION_HORIZONS = ("4h", "8h", "12h")


def _run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    subprocess.run(command, check=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_calibration_gate_summary(
    labeled_meta: dict,
    *,
    recalibration_horizons: Sequence[str],
    min_rows_per_horizon: int,
) -> dict:
    rows_by_horizon = labeled_meta.get("rows_by_horizon") if isinstance(labeled_meta.get("rows_by_horizon"), dict) else {}
    horizon_rows = {str(horizon): int(rows_by_horizon.get(str(horizon), 0) or 0) for horizon in recalibration_horizons}
    insufficient = {
        horizon: rows
        for horizon, rows in horizon_rows.items()
        if rows < int(min_rows_per_horizon)
    }
    return {
        "enabled": True,
        "min_rows_per_horizon": int(min_rows_per_horizon),
        "rows_by_horizon": horizon_rows,
        "eligible": not bool(insufficient),
        "insufficient_horizons": insufficient,
        "reason": "ready" if not insufficient else "insufficient_rows_per_horizon",
    }


def _filter_calibration_candidate_by_horizon(
    payload: dict,
    *,
    allowed_horizons: Sequence[str],
) -> tuple[dict, dict]:
    allowed = {str(value).strip() for value in allowed_horizons if str(value).strip()}
    if not allowed:
        return dict(payload), {"allowed_horizons": [], "removed_keys": [], "retained_keys": sorted(payload.keys())}

    filtered: dict = {}
    removed_keys: list[str] = []
    for raw_key, raw_value in payload.items():
        key = str(raw_key)
        horizon = key.split("@", 1)[0]
        if horizon in allowed:
            filtered[key] = raw_value
        else:
            removed_keys.append(key)

    summary = {
        "allowed_horizons": sorted(allowed),
        "removed_keys": sorted(removed_keys),
        "retained_keys": sorted(filtered.keys()),
    }
    return filtered, summary


def _merge_calibration_candidates(
    base_payload: dict,
    overlay_payload: dict,
) -> dict:
    merged = dict(base_payload)
    merged.update(overlay_payload)
    return merged


def _candidate_lookback_hours(
    initial_lookback_hours: int,
    *,
    max_lookback_hours: int,
    step_hours: int,
) -> list[int]:
    initial = max(int(initial_lookback_hours), 1)
    maximum = max(int(max_lookback_hours), initial)
    step = max(int(step_hours), 1)
    values: list[int] = []
    current = initial
    while current < maximum:
        values.append(current)
        current += step
    if not values or values[-1] != maximum:
        values.append(maximum)
    return values


def _resolve_effective_lookback_hours(
    *,
    initial_lookback_hours: int,
    max_lookback_hours: int,
    step_hours: int,
    evaluate_lookback: Callable[[int], dict],
) -> dict:
    attempts: list[dict[str, object]] = []
    chosen_payload: dict | None = None
    for lookback_hours in _candidate_lookback_hours(
        initial_lookback_hours,
        max_lookback_hours=max_lookback_hours,
        step_hours=step_hours,
    ):
        payload = evaluate_lookback(int(lookback_hours))
        attempts.append(
            {
                "lookback_hours": int(lookback_hours),
                "calibration_gate": payload["calibration_gate"],
                "rows_by_horizon": payload["calibration_gate"].get("rows_by_horizon", {}),
            }
        )
        chosen_payload = payload
        if payload["calibration_gate"].get("eligible"):
            break
    if chosen_payload is None:
        raise ValueError("evaluate_lookback produced no payloads")
    return {
        "requested_lookback_hours": int(initial_lookback_hours),
        "effective_lookback_hours": int(chosen_payload["lookback_hours"]),
        "auto_expanded": int(chosen_payload["lookback_hours"]) != int(initial_lookback_hours),
        "attempts": attempts,
        "calibration_gate": chosen_payload["calibration_gate"],
        "labeled_meta": chosen_payload["labeled_meta"],
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lookback_resolution = payload.get("lookback_resolution") if isinstance(payload.get("lookback_resolution"), dict) else {}
    lines = [
        "# Downtrend Bias Remediation",
        "",
        "## Inputs",
        f"- history_path: {payload['inputs']['history_path']}",
        f"- spot_ohlcv_path: {payload['inputs']['spot_ohlcv_path']}",
        f"- requested_lookback_hours: {payload['inputs']['lookback_hours']}",
        f"- effective_lookback_hours: {payload['inputs']['effective_lookback_hours']}",
        f"- recent_window: {payload['inputs']['recent_window']}",
        f"- audit_horizons: {', '.join(payload['inputs']['audit_horizons'])}",
        f"- recalibration_horizons: {', '.join(payload['inputs']['recalibration_horizons'])}",
        f"- auto_expand_lookback: {payload['inputs']['auto_expand_lookback']}",
        f"- lookback_step_hours: {payload['inputs']['lookback_step_hours']}",
        f"- max_lookback_hours: {payload['inputs']['max_lookback_hours']}",
        f"- include_runtime_runs: {payload['inputs']['include_runtime_runs']}",
        f"- candidate_allowed_horizons: {', '.join(payload['inputs']['candidate_allowed_horizons'])}",
        f"- base_calibration_path: {payload['inputs']['base_calibration_path']}",
        "",
        "## Outputs",
        f"- probability_branch_alignment: {payload['outputs']['probability_branch_alignment']}",
        f"- labeled_backtest_csv: {payload['outputs']['labeled_backtest_csv']}",
        f"- labeled_backtest_meta: {payload['outputs']['labeled_backtest_meta']}",
        f"- calibration_candidate: {payload['outputs']['calibration_candidate']}",
        f"- calibration_coverage: {payload['outputs']['calibration_coverage']}",
        "",
        "## Marginal Audits",
    ]
    for row in payload["outputs"]["marginal_audits"]:
        lines.append(f"- {row['horizon']}: {row['report_path']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This workflow builds a recent-history candidate calibration artifact; it does not auto-promote it into the live default config.")
    lines.append("- Use the emitted candidate calibration in a shadow or replay config first, then compare against fresh live-style refresh output before promotion.")
    calibration_gate = payload.get("calibration_gate") if isinstance(payload.get("calibration_gate"), dict) else {}
    if calibration_gate:
        lines.append(
            "- Calibration gate: "
            + ("ready" if calibration_gate.get("eligible") else f"skipped ({calibration_gate.get('reason')})")
        )
    if lookback_resolution:
        lines.append(
            "- Lookback resolution: "
            + (
                f"expanded from {lookback_resolution.get('requested_lookback_hours')}h to {lookback_resolution.get('effective_lookback_hours')}h"
                if lookback_resolution.get("auto_expanded")
                else f"kept requested window at {lookback_resolution.get('effective_lookback_hours')}h"
            )
        )
    candidate_filter = payload.get("candidate_filter") if isinstance(payload.get("candidate_filter"), dict) else {}
    if candidate_filter:
        lines.append(
            "- Candidate filter: "
            + (
                f"removed {', '.join(candidate_filter.get('removed_keys', []))}"
                if candidate_filter.get("removed_keys")
                else "no keys removed"
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recent-history downtrend-bias remediation workflow: audit probability alignment, "
            "audit marginal calibration, build a labeled recent-history backtest, and fit a higher-horizon calibration candidate."
        )
    )
    parser.add_argument("--history-path", type=Path, default=Path("artifacts/predictions/history.json"))
    parser.add_argument("--spot-ohlcv-path", type=Path, default=Path("data/spot_klines"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback-hours", type=int, default=24 * 45)
    parser.add_argument("--recent-window", type=int, default=50)
    parser.add_argument("--audit-horizons", nargs="+", default=list(DEFAULT_AUDIT_HORIZONS))
    parser.add_argument("--recalibration-horizons", nargs="+", default=list(DEFAULT_RECALIBRATION_HORIZONS))
    parser.add_argument("--calibration-method", choices=("platt", "isotonic", "beta"), default="isotonic")
    parser.add_argument("--min-regime-rows", type=int, default=3)
    parser.add_argument("--min-labeled-rows", type=int, default=1)
    parser.add_argument("--min-calibration-rows-per-horizon", type=int, default=25)
    parser.add_argument("--auto-expand-lookback", action="store_true")
    parser.add_argument("--lookback-step-hours", type=int, default=24 * 15)
    parser.add_argument("--max-lookback-hours", type=int, default=24 * 180)
    parser.add_argument("--include-reliability-snapshots", action="store_true")
    parser.add_argument("--no-include-runtime-runs", dest="include_runtime_runs", action="store_false")
    parser.add_argument("--candidate-allowed-horizons", nargs="+", default=["4h", "8h"])
    parser.add_argument("--base-calibration-path", type=Path, default=Path("artifacts/models/platt_calibration.json"))
    parser.set_defaults(include_runtime_runs=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_csv = output_dir / "labeled_backtest_recent_multi.csv"
    labeled_meta = output_dir / "labeled_backtest_recent_multi_meta.json"

    def build_labeled_payload(lookback_hours: int) -> dict:
        labeled_command = [
            "--history-path",
            str(args.history_path),
            "--spot-ohlcv-path",
            str(args.spot_ohlcv_path),
            "--horizons",
            *[str(value) for value in args.recalibration_horizons],
            "--lookback-rows",
            "0",
            "--lookback-hours",
            str(int(lookback_hours)),
            "--min-rows",
            str(max(int(args.min_labeled_rows), 1)),
            "--output",
            str(labeled_csv),
            "--meta-output",
            str(labeled_meta),
        ]
        if args.include_reliability_snapshots:
            labeled_command.append("--include-reliability-snapshots")
        if args.include_runtime_runs:
            labeled_command.append("--include-runtime-runs")
        _run_module("src.scripts.build_labeled_backtest_from_history", *labeled_command)
        labeled_meta_payload = _read_json(labeled_meta)
        calibration_gate = _build_calibration_gate_summary(
            labeled_meta_payload,
            recalibration_horizons=[str(value) for value in args.recalibration_horizons],
            min_rows_per_horizon=max(int(args.min_calibration_rows_per_horizon), 1),
        )
        return {
            "lookback_hours": int(lookback_hours),
            "labeled_meta": labeled_meta_payload,
            "calibration_gate": calibration_gate,
        }

    if args.auto_expand_lookback:
        lookback_resolution = _resolve_effective_lookback_hours(
            initial_lookback_hours=int(args.lookback_hours),
            max_lookback_hours=int(args.max_lookback_hours),
            step_hours=int(args.lookback_step_hours),
            evaluate_lookback=build_labeled_payload,
        )
    else:
        single_payload = build_labeled_payload(int(args.lookback_hours))
        lookback_resolution = {
            "requested_lookback_hours": int(args.lookback_hours),
            "effective_lookback_hours": int(args.lookback_hours),
            "auto_expanded": False,
            "attempts": [
                {
                    "lookback_hours": int(args.lookback_hours),
                    "calibration_gate": single_payload["calibration_gate"],
                    "rows_by_horizon": single_payload["calibration_gate"].get("rows_by_horizon", {}),
                }
            ],
            "calibration_gate": single_payload["calibration_gate"],
            "labeled_meta": single_payload["labeled_meta"],
        }

    effective_lookback_hours = int(lookback_resolution["effective_lookback_hours"])

    probability_report = output_dir / "probability_branch_alignment_latest.json"
    probability_rows = output_dir / "probability_branch_alignment_rows.csv"
    _run_module(
        "src.scripts.analyze_probability_branch_alignment",
        "--history-path",
        str(args.history_path),
        "--horizons",
        *[str(value) for value in args.audit_horizons],
        "--recent-window",
        str(int(args.recent_window)),
        "--output",
        str(probability_report),
        "--rows-output",
        str(probability_rows),
    )

    marginal_outputs = []
    for horizon in args.audit_horizons:
        normalized_horizon = str(horizon).replace(".", "p")
        report_path = output_dir / f"direction_marginal_{normalized_horizon}_latest.json"
        rows_path = output_dir / f"direction_marginal_{normalized_horizon}_rows.csv"
        command = [
            "--history-path",
            str(args.history_path),
            "--spot-ohlcv-path",
            str(args.spot_ohlcv_path),
            "--horizon",
            str(horizon),
            "--lookback-rows",
            "0",
            "--lookback-hours",
            str(effective_lookback_hours),
            "--output",
            str(report_path),
            "--rows-output",
            str(rows_path),
        ]
        if args.include_reliability_snapshots:
            command.append("--include-reliability-snapshots")
        _run_module("src.scripts.analyze_direction_marginal_calibration", *command)
        marginal_outputs.append(
            {
                "horizon": str(horizon),
                "report_path": str(report_path),
                "rows_path": str(rows_path),
            }
        )
    labeled_meta_payload = lookback_resolution["labeled_meta"]
    calibration_gate = lookback_resolution["calibration_gate"]

    calibration_candidate = output_dir / "recent_downtrend_calibration_candidate.json"
    calibration_candidate_raw = output_dir / "recent_downtrend_calibration_candidate_raw.json"
    calibration_coverage = output_dir / "recent_downtrend_calibration_coverage.json"
    candidate_filter_summary = {
        "allowed_horizons": [str(value) for value in args.candidate_allowed_horizons],
        "removed_keys": [],
        "retained_keys": [],
    }
    recalibration_horizons = [str(value).rstrip("h") for value in args.recalibration_horizons]
    if calibration_gate["eligible"]:
        _run_module(
            "src.scripts.train_platt_calibration",
            "--skip-model-fit",
            "--fit-base-horizons-from-labeled-input",
            "--method",
            str(args.calibration_method),
            "--horizons",
            *recalibration_horizons,
            "--labeled-input",
            str(labeled_csv),
            "--output-path",
            str(calibration_candidate),
            "--coverage-output-path",
            str(calibration_coverage),
            "--min-regime-rows",
            str(int(args.min_regime_rows)),
            "--min-regime-rows-floor",
            str(max(int(args.min_regime_rows), 1)),
        )
        trained_candidate_payload = _read_json(calibration_candidate)
        _write_json(calibration_candidate_raw, trained_candidate_payload)
        filtered_candidate_payload, candidate_filter_summary = _filter_calibration_candidate_by_horizon(
            trained_candidate_payload,
            allowed_horizons=[str(value) for value in args.candidate_allowed_horizons],
        )
        base_calibration_payload = {}
        if args.base_calibration_path.exists():
            base_calibration_payload = _read_json(args.base_calibration_path)
        merged_candidate_payload = _merge_calibration_candidates(base_calibration_payload, filtered_candidate_payload)
        _write_json(calibration_candidate, merged_candidate_payload)

        coverage_payload = _read_json(calibration_coverage)
        if isinstance(coverage_payload, dict):
            coverage_payload["candidate_filter"] = candidate_filter_summary
            coverage_payload["base_calibration_path"] = str(args.base_calibration_path)
            _write_json(calibration_coverage, coverage_payload)
    else:
        _write_json(calibration_candidate, {})
        _write_json(calibration_candidate_raw, {})
        _write_json(
            calibration_coverage,
            {
                "enabled": True,
                "reason": "workflow_sparse_horizon_gate",
                "calibration_gate": calibration_gate,
                "candidate_filter": candidate_filter_summary,
                "base_calibration_path": str(args.base_calibration_path),
                "labeled_meta": labeled_meta_payload,
            },
        )

    manifest = {
        "workflow": "downtrend_bias_remediation",
        "inputs": {
            "history_path": str(args.history_path),
            "spot_ohlcv_path": str(args.spot_ohlcv_path),
            "lookback_hours": int(args.lookback_hours),
            "effective_lookback_hours": effective_lookback_hours,
            "recent_window": int(args.recent_window),
            "audit_horizons": [str(value) for value in args.audit_horizons],
            "recalibration_horizons": [str(value) for value in args.recalibration_horizons],
            "calibration_method": str(args.calibration_method),
            "min_regime_rows": int(args.min_regime_rows),
            "min_labeled_rows": int(args.min_labeled_rows),
            "min_calibration_rows_per_horizon": int(args.min_calibration_rows_per_horizon),
            "auto_expand_lookback": bool(args.auto_expand_lookback),
            "lookback_step_hours": int(args.lookback_step_hours),
            "max_lookback_hours": int(args.max_lookback_hours),
            "include_reliability_snapshots": bool(args.include_reliability_snapshots),
            "include_runtime_runs": bool(args.include_runtime_runs),
            "candidate_allowed_horizons": [str(value) for value in args.candidate_allowed_horizons],
            "base_calibration_path": str(args.base_calibration_path),
        },
        "lookback_resolution": lookback_resolution,
        "calibration_gate": calibration_gate,
        "candidate_filter": candidate_filter_summary,
        "outputs": {
            "probability_branch_alignment": str(probability_report),
            "probability_branch_alignment_rows": str(probability_rows),
            "marginal_audits": marginal_outputs,
            "labeled_backtest_csv": str(labeled_csv),
            "labeled_backtest_meta": str(labeled_meta),
            "calibration_candidate": str(calibration_candidate),
            "calibration_candidate_raw": str(calibration_candidate_raw),
            "calibration_coverage": str(calibration_coverage),
        },
    }
    manifest_path = output_dir / "workflow_manifest.json"
    markdown_path = output_dir / "workflow_manifest.md"
    _write_json(manifest_path, manifest)
    _write_markdown(markdown_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())