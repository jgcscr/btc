from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_champion_command_builders import (
    build_champion_challenger_command,
    build_paired_trigger_overlap_command,
)


def test_build_champion_challenger_command_renders_bootstrap_arguments(tmp_path: Path) -> None:
    cmd = build_champion_challenger_command(
        python="python",
        baseline_path=tmp_path / "baseline.csv",
        candidate_path=tmp_path / "candidate.csv",
        baseline_col="ret_base",
        candidate_col="ret_candidate",
        n_boot=1500,
        alpha=0.1,
        seed=7,
        output_path=tmp_path / "champion.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.evaluate_champion_challenger" in rendered
    assert "--baseline-col ret_base" in rendered
    assert "--candidate-col ret_candidate" in rendered
    assert "--n-boot 1500" in rendered
    assert "--alpha 0.1" in rendered
    assert "--seed 7" in rendered


def test_build_paired_trigger_overlap_command_renders_signal_and_output(tmp_path: Path) -> None:
    cmd = build_paired_trigger_overlap_command(
        python="python",
        candidate_path=tmp_path / "candidate.csv",
        incumbent_path=tmp_path / "incumbent.csv",
        candidate_col="ret_candidate",
        incumbent_col="ret_incumbent",
        signal_col="signal_filtered",
        output_path=tmp_path / "overlap.json",
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.analyze_paired_trigger_overlap",
        "--candidate",
        str(tmp_path / "candidate.csv"),
        "--incumbent",
        str(tmp_path / "incumbent.csv"),
        "--candidate-col",
        "ret_candidate",
        "--incumbent-col",
        "ret_incumbent",
        "--signal-col",
        "signal_filtered",
        "--output",
        str(tmp_path / "overlap.json"),
    ]