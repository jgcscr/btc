from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_shadow_policy_command_builders import (
    build_shadow_policy_command_bundle,
)


def test_build_shadow_policy_command_bundle_renders_expected_paths_and_flags(tmp_path: Path) -> None:
    bundle = build_shadow_policy_command_bundle(
        python="python",
        variant_name="regime_state_veto",
        candidate_input_path=tmp_path / "candidate.csv",
        model_path=tmp_path / "model.pkl",
        companion_baseline_path=tmp_path / "baseline.csv",
        summary_dir=tmp_path,
        common_align_args=["--threshold", "0.55", "--retain-cols", "close"],
        baseline_col="ret_base",
        candidate_col="ret_candidate",
        n_boot=1200,
        alpha=0.1,
        seed=9,
        signal_col="signal_filtered",
        feature_sources=[tmp_path / "quality.csv", tmp_path / "incumbent.csv"],
        extra_policy_args=["--regime-state-candidate-only-veto", "1"],
    )

    assert bundle.candidate_path == tmp_path / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto.csv"
    assert bundle.meta_output_path == tmp_path / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto_meta.json"
    assert bundle.companion_output_path == tmp_path / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.json"
    assert bundle.overlap_output_path == tmp_path / "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto.json"
    rendered_build = " ".join(bundle.build_command)
    assert "--threshold 0.55" in rendered_build
    assert "--retain-cols close" in rendered_build
    assert "--feature-source" in rendered_build
    assert "--regime-state-candidate-only-veto 1" in rendered_build
    rendered_companion = " ".join(bundle.companion_command)
    assert "src.scripts.evaluate_champion_challenger" in rendered_companion
    assert "--baseline-col ret_base" in rendered_companion
    rendered_overlap = " ".join(bundle.overlap_command)
    assert "src.scripts.analyze_paired_trigger_overlap" in rendered_overlap
    assert "--signal-col signal_filtered" in rendered_overlap