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