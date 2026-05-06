from __future__ import annotations

from pathlib import Path

from src.runtime.config_composition import load_composed_yaml


def test_load_composed_yaml_merges_parent_configs(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("feature_coverage_policy:\n  block_on_violation: true\n  ignored_sources: [funding]\n", encoding="utf-8")
    child.write_text(
        "extends: base.yaml\nfeature_coverage_policy:\n  ignored_sources: [funding, macro]\nwrite_artifacts: true\n",
        encoding="utf-8",
    )

    payload = load_composed_yaml(child)

    assert payload["feature_coverage_policy"] == {
        "block_on_violation": True,
        "ignored_sources": ["funding", "macro"],
    }
    assert payload["write_artifacts"] is True