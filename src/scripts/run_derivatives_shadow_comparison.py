from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import json
from copy import deepcopy

import yaml
from src.scripts.run_shadow_profile_comparison import (
    _copy_file,
    _infer_source_reliability_run_id,
    _load_json,
    _profile_slug,
    _utc_stamp,
)


DEFAULT_BASELINE_CONFIG = Path("configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
DEFAULT_DERIVATIVES_CONFIG = Path("configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml")


def _build_comparison_runtime_config(config_path: Path, output_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    runtime_payload = deepcopy(payload)
    coverage = runtime_payload.get("feature_coverage_policy")
    if not isinstance(coverage, dict):
        coverage = {}
    else:
        coverage = dict(coverage)
    coverage["block_on_violation"] = False
    runtime_payload["feature_coverage_policy"] = coverage
    output_path.write_text(yaml.safe_dump(runtime_payload, sort_keys=False), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the derivatives shadow candidate if needed and run it through the regular shadow-profile "
            "comparison surface against the simplified baseline."
        )
    )
    parser.add_argument("--lhs-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--rhs-config", type=Path, default=DEFAULT_DERIVATIVES_CONFIG)
    parser.add_argument("--lhs-label", default="live_conservative_binance_only")
    parser.add_argument("--rhs-label", default="shadow_derivatives_candidate")
    parser.add_argument("--targets", default="0.25,1,4,8,12")
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--platt-calibration", type=Path, default=None)
    parser.add_argument("--restore-latest-to", default="rhs", choices=["lhs", "rhs", "none"])
    parser.add_argument(
        "--predictions-latest",
        type=Path,
        default=Path("artifacts/predictions/latest.json"),
    )
    parser.add_argument(
        "--monitoring-latest",
        type=Path,
        default=Path("artifacts/monitoring/latest.json"),
    )
    parser.add_argument(
        "--predictions-output-dir",
        type=Path,
        default=Path("artifacts/predictions/comparisons"),
    )
    parser.add_argument(
        "--monitoring-output-dir",
        type=Path,
        default=Path("artifacts/monitoring/comparisons"),
    )
    parser.add_argument(
        "--longitudinal-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.json"),
    )
    parser.add_argument(
        "--summary-markdown-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.md"),
    )
    parser.add_argument(
        "--summary-csv-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv"),
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip regenerating the derivatives shadow package before running the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    python_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    run_id = _utc_stamp()

    if not args.skip_package:
        subprocess.run(
            [str(python_bin), "-m", "src.scripts.package_derivatives_shadow_rollout"],
            check=True,
        )

    profiles = [
        {"label": str(args.lhs_label), "config": Path(args.lhs_config)},
        {"label": str(args.rhs_label), "config": Path(args.rhs_config)},
    ]
    artifacts: dict[str, dict[str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="derivatives-shadow-comparison-") as tmpdir:
        temp_root = Path(tmpdir)
        for profile in profiles:
            label = _profile_slug(profile["label"])
            runtime_config = _build_comparison_runtime_config(
                Path(profile["config"]),
                temp_root / f"{label}_{run_id}.yaml",
            )
            command = [
                str(python_bin),
                "-m",
                "src.scripts.run_live_inference",
                "--config",
                str(runtime_config),
                "--targets",
                str(args.targets),
            ]
            subprocess.run(command, check=True)

            prediction_copy = Path(args.predictions_output_dir) / f"latest_{label}_{run_id}.json"
            monitoring_copy = Path(args.monitoring_output_dir) / f"latest_{label}_{run_id}.json"
            _copy_file(Path(args.predictions_latest), prediction_copy)
            _copy_file(Path(args.monitoring_latest), monitoring_copy)
            artifacts[label] = {
                "config": str(profile["config"]),
                "runtime_config": str(runtime_config),
                "predictions": str(prediction_copy),
                "monitoring": str(monitoring_copy),
            }

    lhs_label = _profile_slug(str(args.lhs_label))
    rhs_label = _profile_slug(str(args.rhs_label))
    comparison_output = Path(args.predictions_output_dir) / f"{lhs_label}_vs_{rhs_label}_{run_id}.json"
    subprocess.run(
        [
            str(python_bin),
            "-m",
            "src.scripts.compare_live_profile_snapshots",
            "--lhs-snapshot",
            artifacts[lhs_label]["predictions"],
            "--rhs-snapshot",
            artifacts[rhs_label]["predictions"],
            "--lhs-label",
            lhs_label,
            "--rhs-label",
            rhs_label,
            "--output",
            str(comparison_output),
        ],
        check=True,
    )

    if str(args.restore_latest_to) != "none":
        restore_label = lhs_label if str(args.restore_latest_to) == "lhs" else rhs_label
        _copy_file(Path(artifacts[restore_label]["predictions"]), Path(args.predictions_latest))
        _copy_file(Path(artifacts[restore_label]["monitoring"]), Path(args.monitoring_latest))

    manifest = {
        "run_id": run_id,
        "generated_at": json.loads(json.dumps({"ts": _utc_stamp()}))["ts"],
        "source_reliability_run_id": _infer_source_reliability_run_id(args.thresholds_json),
        "targets": str(args.targets),
        "thresholds_json": str(args.thresholds_json) if args.thresholds_json is not None else None,
        "platt_calibration": str(args.platt_calibration) if args.platt_calibration is not None else None,
        "profiles": artifacts,
        "comparison_output": str(comparison_output),
        "restore_latest_to": str(args.restore_latest_to),
        "comparison_summary": _load_json(comparison_output).get("overall_summary", {}),
    }
    manifest_path = Path(args.predictions_output_dir) / f"shadow_profile_comparison_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    subprocess.run(
        [
            str(python_bin),
            "-m",
            "src.scripts.update_shadow_profile_comparison_longitudinal",
            "--manifest",
            str(manifest_path),
            "--comparison",
            str(comparison_output),
            "--output",
            str(args.longitudinal_output),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python_bin),
            "-m",
            "src.scripts.summarize_shadow_profile_comparison_longitudinal",
            "--input",
            str(args.longitudinal_output),
            "--output",
            str(args.summary_output),
            "--markdown-output",
            str(args.summary_markdown_output),
            "--csv-output",
            str(args.summary_csv_output),
        ],
        check=True,
    )

    print(json.dumps({"manifest_path": str(manifest_path), "comparison_output": str(comparison_output)}, indent=2))


if __name__ == "__main__":
    main()