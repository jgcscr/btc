from __future__ import annotations

from pathlib import Path
from typing import List


def build_prediction_coherence_command(*, python: str, summary_dir: Path) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.analyze_prediction_coherence",
        "--history-path",
        "artifacts/predictions/history.json",
        "--output",
        str(summary_dir / "prediction_coherence.json"),
    ]


def build_live_prediction_snapshot_command(
    *,
    python: str,
    run_id: str,
    run_profile_id: str,
    run_profile_name: str,
    live_snapshot_path: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.snapshot_live_predictions",
        "--run-id",
        str(run_id),
        "--profile-id",
        str(run_profile_id),
        "--profile-name",
        str(run_profile_name),
        "--predictions-latest",
        "artifacts/predictions/latest.json",
        "--monitoring-latest",
        "artifacts/monitoring/latest.json",
        "--output",
        str(live_snapshot_path),
    ]


def build_default_vs_midband_live_comparison_command(
    *,
    python: str,
    default_snapshot_path: Path,
    live_snapshot_path: Path,
    summary_dir: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.compare_default_vs_midband_paper_live_snapshots",
        "--default-snapshot",
        str(default_snapshot_path),
        "--midband-snapshot",
        str(live_snapshot_path),
        "--output",
        str(summary_dir / "default_vs_midband_paper_live_comparison.json"),
    ]


def build_default_vs_midband_longitudinal_command(
    *,
    python: str,
    run_root: Path,
    run_id: str,
    run_profile_id: str,
    run_profile_name: str,
    live_snapshot_path: Path,
    summary_dir: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.build_default_vs_midband_paper_live_longitudinal",
        "--run-root",
        str(run_root),
        "--include-run-id",
        str(run_id),
        "--include-profile-id",
        str(run_profile_id),
        "--include-profile-name",
        str(run_profile_name),
        "--include-snapshot",
        str(live_snapshot_path),
        "--output",
        str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
    ]


def build_default_vs_midband_watchlist_command(
    *,
    python: str,
    longitudinal_input: Path,
    output_path: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.build_default_vs_midband_paper_live_watchlist",
        "--longitudinal-input",
        str(longitudinal_input),
        "--output",
        str(output_path),
        "--target-matched-pairs",
        "8",
        "--early-operational-streak",
        "2",
        "--early-actionable-asymmetry-streak",
        "2",
    ]