from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_CANDIDATE_ROOT = Path("artifacts/models_4h_candidate_ultra_conservative")
DEFAULT_DATASET_PATH = Path("artifacts/datasets/btc_features_multi_horizon_splits.raw_price_levels_ablated.npz")
DEFAULT_HISTORY_PATH = Path("artifacts/predictions/history.json")
DEFAULT_SPOT_OHLCV_PATH = Path("data/spot_klines")
DEFAULT_OUTPUT_JSON = Path("artifacts/analysis/4h_regime_calibration_workflow_latest.json")
DEFAULT_OUTPUT_MD = Path("artifacts/analysis/4h_regime_calibration_workflow_latest.md")


def _parse_horizon_hours(label: str) -> float:
    raw = str(label).strip().lower()
    if raw.endswith("h"):
        return float(raw[:-1])
    if raw.endswith("m"):
        return float(raw[:-1]) / 60.0
    return float(raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build labeled 4h history and fit regime-aware calibration entries for a staged 4h candidate."
        )
    )
    parser.add_argument("--candidate-model-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--spot-ohlcv-path", type=Path, default=DEFAULT_SPOT_OHLCV_PATH)
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=Path("artifacts/monitoring/labeled_backtest_4h_regime.csv"),
    )
    parser.add_argument(
        "--labeled-meta-output",
        type=Path,
        default=Path("artifacts/monitoring/labeled_backtest_4h_regime_meta.json"),
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=Path("artifacts/analysis/4h_regime_calibration_coverage_latest.json"),
    )
    parser.add_argument(
        "--output-calibration-path",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT / "platt_calibration_4h_candidate_regime.json",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--lookback-rows", type=int, default=2000)
    parser.add_argument("--lookback-hours", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument(
        "--build-min-rows",
        type=int,
        default=1,
        help="Minimum row gate passed to build_labeled_backtest_from_history before coverage-based calibration checks run.",
    )
    parser.add_argument("--min-regime-rows", type=int, default=80)
    parser.add_argument("--horizon", default="4h")
    parser.add_argument("--target-regime", default="neutral")
    parser.add_argument(
        "--skip-model-fit",
        action="store_true",
        help="Skip model-validation calibration and fit only labeled-input-derived entries.",
    )
    parser.add_argument(
        "--seed-calibration-path",
        type=Path,
        default=None,
        help="Optional existing calibration JSON whose entries will be merged into the output before adding new entries.",
    )
    parser.add_argument(
        "--fit-base-horizons-from-labeled-input",
        action="store_true",
        help="Also fit a base horizon calibration entry like `12h` from labeled input when skipping model-fit or supplementing seeded calibration.",
    )
    parser.add_argument(
        "--include-reliability-snapshots",
        action="store_true",
        help="Also include archived artifacts/reliability/*/summary/live_predictions_snapshot.json rows when building labeled history.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_markdown(payload: Mapping[str, Any]) -> str:
    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), Mapping) else {}
    target = calibration.get("target_regime_entry") if isinstance(calibration.get("target_regime_entry"), Mapping) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), Mapping) else {}
    lines = ["# 4h Regime Calibration Workflow", ""]
    lines.append("## Inputs")
    lines.append(f"- candidate_model_root: {payload['inputs'].get('candidate_model_root')}")
    lines.append(f"- dataset_path: {payload['inputs'].get('dataset_path')}")
    lines.append(f"- history_path: {payload['inputs'].get('history_path')}")
    lines.append(f"- target_regime: {payload['inputs'].get('target_regime')}")
    lines.append("")
    lines.append("## Labeled Data")
    lines.append(f"- rows: {payload.get('labeled_rows')}")
    lines.append(f"- labeled_output: {payload.get('labeled_output')}")
    lines.append(f"- labeled_meta_output: {payload.get('labeled_meta_output')}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- coverage_reason: {coverage.get('reason')}")
    lines.append(f"- eligible_entry_count: {coverage.get('eligible_entry_count')}")
    lines.append(f"- ineligible_entry_count: {coverage.get('ineligible_entry_count')}")
    lines.append(f"- target_regime_rows: {payload.get('target_regime_rows')}")
    lines.append("")
    lines.append("## Calibration")
    lines.append(f"- output_calibration_path: {calibration.get('output_path')}")
    lines.append(f"- has_target_regime_entry: {calibration.get('has_target_regime_entry')}")
    lines.append(f"- target_regime_key: {calibration.get('target_regime_key')}")
    if target:
        lines.append(f"- method: {target.get('method')}")
        lines.append(f"- a: {target.get('a')}")
        lines.append(f"- b: {target.get('b')}")
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    target_regime = str(args.target_regime).strip().lower()
    target_key = f"{args.horizon}@{target_regime}"
    horizon_hours = _parse_horizon_hours(str(args.horizon))
    calibrator_horizon = str(int(round(horizon_hours))) if float(horizon_hours).is_integer() else str(horizon_hours)

    build_cmd = [
        sys.executable,
        "-m",
        "src.scripts.build_labeled_backtest_from_history",
        "--history-path",
        str(args.history_path),
        "--no-prefer-backtest",
        "--spot-ohlcv-path",
        str(args.spot_ohlcv_path),
        "--horizon",
        str(args.horizon),
        "--lookback-rows",
        str(int(args.lookback_rows)),
        "--lookback-hours",
        str(int(args.lookback_hours)),
        "--min-rows",
        str(int(args.build_min_rows)),
        "--output",
        str(args.labeled_output),
        "--meta-output",
        str(args.labeled_meta_output),
    ]
    if args.include_reliability_snapshots:
        build_cmd.append("--include-reliability-snapshots")
    _run(build_cmd, cwd=repo_root)

    merged_seed: Dict[str, Any] = {}
    calibration_input_path = args.output_calibration_path
    raw_output_calibration_path = args.output_calibration_path
    if args.seed_calibration_path is not None:
        merged_seed = _read_json(args.seed_calibration_path)
        raw_output_calibration_path = args.output_calibration_path.with_name(
            f"{args.output_calibration_path.stem}.new_entries{args.output_calibration_path.suffix}"
        )
        calibration_input_path = raw_output_calibration_path

    calibrate_cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_platt_calibration",
        "--model-root",
        str(args.candidate_model_root),
        "--dataset-multi",
        str(args.dataset_path),
        "--horizons",
        calibrator_horizon,
        "--labeled-input",
        str(args.labeled_output),
        "--output-path",
        str(calibration_input_path),
        "--coverage-output-path",
        str(args.coverage_output),
        "--min-regime-rows",
        str(int(args.min_regime_rows)),
    ]
    if args.skip_model_fit:
        calibrate_cmd.append("--skip-model-fit")
    if args.fit_base_horizons_from_labeled_input:
        calibrate_cmd.append("--fit-base-horizons-from-labeled-input")
    _run(calibrate_cmd, cwd=repo_root)

    if args.seed_calibration_path is not None:
        new_entries = _read_json(raw_output_calibration_path)
        merged_payload = dict(merged_seed)
        merged_payload.update(new_entries)
        _write_json(args.output_calibration_path, merged_payload)

    labeled_meta = _read_json(args.labeled_meta_output)
    coverage = _read_json(args.coverage_output)
    calibration_payload = _read_json(args.output_calibration_path)
    target_entry = calibration_payload.get(target_key) if isinstance(calibration_payload.get(target_key), Mapping) else None
    labeled_rows = int(labeled_meta.get("labeled_rows") or 0)

    eligible_entries = coverage.get("eligible_entries") if isinstance(coverage.get("eligible_entries"), list) else []
    target_eligible = next(
        (
            entry
            for entry in eligible_entries
            if str(entry.get("horizon")) == str(args.horizon)
            and str(entry.get("regime_state", "")).strip().lower() == target_regime
        ),
        None,
    )
    target_rows = None
    if isinstance(target_eligible, Mapping):
        target_rows = int(target_eligible.get("rows") or 0)

    recommendations: list[str] = []
    if labeled_rows < int(args.min_rows):
        recommendations.append(
            f"The labeled {args.horizon} sample is still only {labeled_rows} rows, below the preferred floor of {int(args.min_rows)}; treat any emitted regime calibration as exploratory until replay coverage is broader."
        )
    if target_entry is None:
        recommendations.append(
            f"The workflow did not emit a regime-specific {args.horizon} {target_regime} calibration entry; expand the labeled history window or lower the regime-row floor only if coverage diagnostics justify it."
        )
    else:
        slope = _safe_float(target_entry.get("a"))
        if slope is not None and slope > 0.0:
            recommendations.append(
                f"A usable `{target_key}` calibration entry was emitted. The next step is to point the shadow candidate at this calibration artifact and rerun the same live replay."
            )
        else:
            recommendations.append(
                f"A `{target_key}` entry exists, but its slope is non-positive or invalid, so runtime may still fall back to base `{args.horizon}` calibration."
            )
    if not recommendations:
        recommendations.append("No additional action identified.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "candidate_model_root": str(args.candidate_model_root),
            "dataset_path": str(args.dataset_path),
            "history_path": str(args.history_path),
            "spot_ohlcv_path": str(args.spot_ohlcv_path),
            "horizon": str(args.horizon),
            "target_regime": target_regime,
            "min_rows": int(args.min_rows),
            "build_min_rows": int(args.build_min_rows),
            "min_regime_rows": int(args.min_regime_rows),
            "lookback_rows": int(args.lookback_rows),
            "lookback_hours": int(args.lookback_hours),
            "include_reliability_snapshots": bool(args.include_reliability_snapshots),
            "skip_model_fit": bool(args.skip_model_fit),
            "seed_calibration_path": str(args.seed_calibration_path) if args.seed_calibration_path else None,
            "fit_base_horizons_from_labeled_input": bool(args.fit_base_horizons_from_labeled_input),
        },
        "labeled_output": str(args.labeled_output),
        "labeled_meta_output": str(args.labeled_meta_output),
        "labeled_rows": labeled_rows,
        "rows_by_horizon": labeled_meta.get("rows_by_horizon"),
        "coverage": coverage,
        "target_regime_rows": target_rows,
        "calibration": {
            "output_path": str(args.output_calibration_path),
            "keys": sorted(str(key) for key in calibration_payload.keys()),
            "target_regime_key": target_key,
            "has_target_regime_entry": target_entry is not None,
            "target_regime_entry": dict(target_entry) if isinstance(target_entry, Mapping) else None,
        },
        "recommendations": recommendations,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote workflow JSON: {args.output_json}")
    print(f"Wrote workflow memo: {args.output_md}")


if __name__ == "__main__":
    main()