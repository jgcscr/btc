from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from src.runtime.reliability_champion_command_builders import (
    build_champion_challenger_command,
    build_paired_trigger_overlap_command,
)
from src.runtime.reliability_trade_decision_command_builders import (
    build_trade_decision_policy_backtest_command,
)


@dataclass(frozen=True)
class ShadowPolicyCommandBundle:
    candidate_path: Path
    meta_output_path: Path
    companion_output_path: Path
    overlap_output_path: Path
    build_command: List[str]
    companion_command: List[str]
    overlap_command: List[str]


def build_shadow_policy_command_bundle(
    *,
    python: str,
    variant_name: str,
    candidate_input_path: Path | str,
    model_path: Path | str,
    companion_baseline_path: Path | str,
    summary_dir: Path,
    common_align_args: Sequence[str],
    baseline_col: str,
    candidate_col: str,
    n_boot: int,
    alpha: float,
    seed: int,
    signal_col: str,
    feature_sources: Iterable[Path | str] | None = None,
    extra_policy_args: Sequence[str] | None = None,
) -> ShadowPolicyCommandBundle:
    candidate_path = summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}.csv"
    meta_output_path = summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}_meta.json"
    companion_output_path = summary_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.json"
    overlap_output_path = summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}.json"
    build_command = build_trade_decision_policy_backtest_command(
        python=python,
        input_path=candidate_input_path,
        model_path=model_path,
        output_path=candidate_path,
        meta_output_path=meta_output_path,
        extra_args=[*common_align_args, *(extra_policy_args or [])],
        feature_sources=feature_sources,
    )
    companion_command = build_champion_challenger_command(
        python=python,
        baseline_path=companion_baseline_path,
        candidate_path=candidate_path,
        baseline_col=baseline_col,
        candidate_col=candidate_col,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        output_path=companion_output_path,
    )
    overlap_command = build_paired_trigger_overlap_command(
        python=python,
        candidate_path=candidate_path,
        incumbent_path=companion_baseline_path,
        candidate_col=candidate_col,
        incumbent_col=baseline_col,
        signal_col=signal_col,
        output_path=overlap_output_path,
    )
    return ShadowPolicyCommandBundle(
        candidate_path=candidate_path,
        meta_output_path=meta_output_path,
        companion_output_path=companion_output_path,
        overlap_output_path=overlap_output_path,
        build_command=build_command,
        companion_command=companion_command,
        overlap_command=overlap_command,
    )