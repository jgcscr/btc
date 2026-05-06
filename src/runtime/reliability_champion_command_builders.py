from __future__ import annotations

from pathlib import Path
from typing import List


def build_champion_challenger_command(
    *,
    python: str,
    baseline_path: Path | str,
    candidate_path: Path | str,
    baseline_col: str,
    candidate_col: str,
    n_boot: int,
    alpha: float,
    seed: int,
    output_path: Path | str,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.evaluate_champion_challenger",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--baseline-col",
        str(baseline_col),
        "--candidate-col",
        str(candidate_col),
        "--n-boot",
        str(int(n_boot)),
        "--alpha",
        str(float(alpha)),
        "--seed",
        str(int(seed)),
        "--output",
        str(output_path),
    ]


def build_paired_trigger_overlap_command(
    *,
    python: str,
    candidate_path: Path | str,
    incumbent_path: Path | str,
    candidate_col: str,
    incumbent_col: str,
    signal_col: str,
    output_path: Path | str,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.analyze_paired_trigger_overlap",
        "--candidate",
        str(candidate_path),
        "--incumbent",
        str(incumbent_path),
        "--candidate-col",
        str(candidate_col),
        "--incumbent-col",
        str(incumbent_col),
        "--signal-col",
        str(signal_col),
        "--output",
        str(output_path),
    ]