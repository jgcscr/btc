from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_command(command: List[str]) -> None:
    subprocess.run(command, check=True)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _profile_slug(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _infer_source_reliability_run_id(thresholds_json: Path | None) -> str | None:
    if thresholds_json is None:
        return None
    threshold_path = Path(thresholds_json)
    summary_dir = threshold_path.parent
    run_dir = summary_dir.parent if summary_dir.name == "summary" else summary_dir
    run_id = str(run_dir.name).strip()
    return run_id or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two runtime profiles, archive their live artifacts, and emit a comparison artifact.",
    )
    parser.add_argument(
        "--lhs-config",
        type=Path,
        default=Path("configs/run_refresh_and_predict.shadow_simplified.yaml"),
    )
    parser.add_argument(
        "--rhs-config",
        type=Path,
        default=Path("configs/run_refresh_and_predict.shadow_chop_suppression.yaml"),
    )
    parser.add_argument("--lhs-label", type=str, default="shadow_simplified")
    parser.add_argument("--rhs-label", type=str, default="shadow_chop_suppression")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--targets", type=str, default="0.25,1,4,8,12")
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--platt-calibration", type=Path, default=None)
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
        "--comparison-output",
        type=Path,
        default=None,
        help="Optional explicit comparison output path. Defaults under artifacts/predictions/comparisons.",
    )
    parser.add_argument(
        "--longitudinal-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json"),
        help="Path to the consolidated longitudinal comparison artifact.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.json"),
        help="Path to the compact summary derived from the longitudinal comparison artifact.",
    )
    parser.add_argument(
        "--summary-markdown-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_summary.md"),
        help="Path to the Markdown operator summary derived from the longitudinal comparison artifact.",
    )
    parser.add_argument(
        "--summary-csv-output",
        type=Path,
        default=Path("artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv"),
        help="Path to the per-run CSV export derived from the longitudinal comparison artifact.",
    )
    parser.add_argument(
        "--restore-latest-to",
        type=str,
        default="rhs",
        choices=["lhs", "rhs", "none"],
    )
    return parser.parse_args()


def _build_refresh_command(
    *,
    python_bin: Path,
    config_path: Path,
    targets: str,
    thresholds_json: Path | None,
    platt_calibration: Path | None,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "src.scripts.run_refresh_and_predict",
        "--config",
        str(config_path),
        "--targets",
        str(targets),
        "--write-artifacts",
    ]
    if thresholds_json is not None:
        command.extend(["--thresholds-json", str(thresholds_json)])
    if platt_calibration is not None:
        command.extend(["--platt-calibration", str(platt_calibration)])
    return command


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id or _utc_stamp())
    python_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    source_reliability_run_id = _infer_source_reliability_run_id(
        Path(args.thresholds_json) if args.thresholds_json is not None else None
    )

    profiles = [
        {"label": str(args.lhs_label), "config": args.lhs_config},
        {"label": str(args.rhs_label), "config": args.rhs_config},
    ]
    for profile in profiles:
        config_path = Path(profile["config"])
        if not config_path.exists():
            raise FileNotFoundError(config_path)

    if args.thresholds_json is not None and not Path(args.thresholds_json).exists():
        raise FileNotFoundError(args.thresholds_json)
    if args.platt_calibration is not None and not Path(args.platt_calibration).exists():
        raise FileNotFoundError(args.platt_calibration)

    artifacts: Dict[str, Dict[str, str]] = {}
    for profile in profiles:
        label = _profile_slug(str(profile["label"]))
        refresh_command = _build_refresh_command(
            python_bin=python_bin,
            config_path=Path(profile["config"]),
            targets=str(args.targets),
            thresholds_json=Path(args.thresholds_json) if args.thresholds_json is not None else None,
            platt_calibration=Path(args.platt_calibration) if args.platt_calibration is not None else None,
        )
        _run_command(refresh_command)

        prediction_copy = Path(args.predictions_output_dir) / f"latest_{label}_{run_id}.json"
        monitoring_copy = Path(args.monitoring_output_dir) / f"latest_{label}_{run_id}.json"
        _copy_file(Path(args.predictions_latest), prediction_copy)
        _copy_file(Path(args.monitoring_latest), monitoring_copy)
        artifacts[label] = {
            "config": str(profile["config"]),
            "predictions": str(prediction_copy),
            "monitoring": str(monitoring_copy),
        }

    lhs_label = _profile_slug(str(args.lhs_label))
    rhs_label = _profile_slug(str(args.rhs_label))
    comparison_output = (
        Path(args.comparison_output)
        if args.comparison_output is not None
        else Path(args.predictions_output_dir) / f"{lhs_label}_vs_{rhs_label}_{run_id}.json"
    )
    compare_command = [
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
    ]
    _run_command(compare_command)

    if str(args.restore_latest_to) != "none":
        restore_label = lhs_label if str(args.restore_latest_to) == "lhs" else rhs_label
        _copy_file(Path(artifacts[restore_label]["predictions"]), Path(args.predictions_latest))
        _copy_file(Path(artifacts[restore_label]["monitoring"]), Path(args.monitoring_latest))

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_reliability_run_id": source_reliability_run_id,
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

    longitudinal_command = [
        str(python_bin),
        "-m",
        "src.scripts.update_shadow_profile_comparison_longitudinal",
        "--manifest",
        str(manifest_path),
        "--comparison",
        str(comparison_output),
        "--output",
        str(args.longitudinal_output),
    ]
    _run_command(longitudinal_command)

    summary_command = [
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
    ]
    _run_command(summary_command)

    manifest["longitudinal_output"] = str(args.longitudinal_output)
    manifest["summary_output"] = str(args.summary_output)
    manifest["summary_markdown_output"] = str(args.summary_markdown_output)
    manifest["summary_csv_output"] = str(args.summary_csv_output)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()