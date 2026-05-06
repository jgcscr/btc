from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_post_live_command_builders import (
    build_default_vs_midband_live_comparison_command,
    build_default_vs_midband_longitudinal_command,
    build_default_vs_midband_watchlist_command,
    build_live_prediction_snapshot_command,
    build_prediction_coherence_command,
)


def test_build_prediction_coherence_command_targets_history_json(tmp_path: Path) -> None:
    cmd = build_prediction_coherence_command(python="python", summary_dir=tmp_path)

    assert cmd == [
        "python",
        "-m",
        "src.scripts.analyze_prediction_coherence",
        "--history-path",
        "artifacts/predictions/history.json",
        "--output",
        str(tmp_path / "prediction_coherence.json"),
    ]


def test_build_live_prediction_snapshot_command_includes_profile_metadata(tmp_path: Path) -> None:
    cmd = build_live_prediction_snapshot_command(
        python="python",
        run_id="run-1",
        run_profile_id="runtime",
        run_profile_name="Runtime",
        live_snapshot_path=tmp_path / "snapshot.json",
    )

    assert "src.scripts.snapshot_live_predictions" in cmd
    assert "run-1" in cmd
    assert "runtime" in cmd
    assert str(tmp_path / "snapshot.json") in cmd


def test_build_default_vs_midband_watchlist_command_uses_standard_thresholds(tmp_path: Path) -> None:
    cmd = build_default_vs_midband_watchlist_command(
        python="python",
        longitudinal_input=tmp_path / "longitudinal.json",
        output_path=tmp_path / "watchlist.json",
    )

    assert cmd[-6:] == [
        "--target-matched-pairs",
        "8",
        "--early-operational-streak",
        "2",
        "--early-actionable-asymmetry-streak",
        "2",
    ]


def test_build_default_vs_midband_longitudinal_and_comparison_commands_render_paths(tmp_path: Path) -> None:
    comparison = build_default_vs_midband_live_comparison_command(
        python="python",
        default_snapshot_path=tmp_path / "default.json",
        live_snapshot_path=tmp_path / "midband.json",
        summary_dir=tmp_path,
    )
    longitudinal = build_default_vs_midband_longitudinal_command(
        python="python",
        run_root=tmp_path / "runs",
        run_id="run-2",
        run_profile_id="midband_paper_evaluation",
        run_profile_name="Midband",
        live_snapshot_path=tmp_path / "midband.json",
        summary_dir=tmp_path,
    )

    assert str(tmp_path / "default_vs_midband_paper_live_comparison.json") == comparison[-1]
    assert str(tmp_path / "default_vs_midband_paper_live_longitudinal.json") == longitudinal[-1]
    assert "run-2" in longitudinal