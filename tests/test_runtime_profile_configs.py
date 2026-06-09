from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def test_default_research_profile_ignores_stale_auxiliary_sources() -> None:
    config = _load_yaml("configs/run_refresh_and_predict.default.yaml")
    coverage = config["feature_coverage_policy"]

    assert coverage["block_on_violation"] is True
    assert coverage["ignored_sources"] == ["funding", "macro", "onchain"]
    assert "interaction_breakout_volume_8h" in coverage["ignored_columns"]
    assert "trend_path_efficiency_8h" in coverage["ignored_columns"]


def test_research_safe_profile_warns_instead_of_failing() -> None:
    config = _load_yaml("configs/run_refresh_and_predict.research_safe.yaml")
    coverage = config["feature_coverage_policy"]

    assert coverage["block_on_violation"] is False
    assert coverage["ignored_sources"] == ["funding", "macro", "onchain"]


def test_default_profile_enables_long_confirmation_and_directional_degradation() -> None:
    config = _load_yaml("configs/run_refresh_and_predict.default.yaml")

    assert config["execution_policy"]["long_confirmation"] == {"enabled": True, "required_horizons": [1, 4]}
    assert config["degradation_monitoring"]["min_directional_samples"] == 3
    assert config["degradation_monitoring"]["max_long_wrong_ratio"] == 0.65
    assert config["degradation_monitoring"]["max_long_wrong_streak"] == 3


def test_downtrend_remediation_shadow_profile_overrides_calibration_only() -> None:
    config = _load_yaml("configs/run_refresh_and_predict.shadow_downtrend_remediation_candidate.yaml")

    assert config["extends"] == "run_refresh_and_predict.default.yaml"
    assert (
        config["platt_calibration"]
        == "artifacts/analysis/downtrend_bias_remediation/recent_downtrend_calibration_candidate.json"
    )