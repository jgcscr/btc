from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from src.scripts import package_signal_expansion_rollout as module


def test_render_markdown_includes_meta_threshold_selection_section() -> None:
    markdown = module._render_markdown(
        {
            "generated_at": "2026-05-13T00:00:00Z",
            "next_priority_family": "meta_ensemble",
            "implementation_notes": ["Meta lane remains first."],
            "program_direction": {
                "meta_ensemble": {
                    "recommended_action": "evaluate_before_expanding_base_models",
                    "source_csv": "artifacts/backtests/backtest_signals_meta_ensemble.csv",
                    "config_path": "configs/reliability_workflow.runtime.yaml",
                    "weight_threshold": 0.52,
                    "selected_weight_threshold": 0.54,
                    "auto_threshold_on_oof": True,
                    "threshold_selection": {
                        "trades": 313.0,
                        "net": -0.114309,
                        "hit_rate": 0.507987,
                        "quantile_cap": 0.54,
                    },
                },
                "derivatives": {"status": "shadow_scaffold_ready", "candidate_config": "configs/derivatives.yaml"},
                "featurelift_4h": {"candidate_config": "configs/featurelift.yaml"},
                "state_engineering": {"guarded_shadow_json": "artifacts/analysis/state.json"},
                "macro": {"recommended_action": "keep_deprioritized"},
            },
        }
    )

    assert "## Meta Threshold Selection" in markdown
    assert "- Meta-ensemble configured threshold: 0.52" in markdown
    assert "- Meta-ensemble selected threshold: 0.54" in markdown
    assert "- OOF trades at selected threshold: 313.0" in markdown
    assert "- Quantile cap: 0.54" in markdown


def test_main_loads_trained_meta_threshold_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    base_config = tmp_path / "base.yaml"
    meta_config = tmp_path / "meta.yaml"
    meta_trained_config = tmp_path / "meta_trained.json"
    analysis_dir = tmp_path / "analysis"
    models_root = tmp_path / "models"
    derivatives_config = tmp_path / "shadow_derivatives.yaml"
    featurelift_config = tmp_path / "shadow_featurelift.yaml"
    featurelift_package = tmp_path / "featurelift.md"
    state_guarded_json = tmp_path / "state.json"
    state_guarded_md = tmp_path / "state.md"
    state_guarded_package = tmp_path / "state_package.md"
    output_json = tmp_path / "rollout.json"
    output_markdown = tmp_path / "rollout.md"

    analysis_dir.mkdir(parents=True)
    models_root.mkdir(parents=True)
    base_config.write_text(yaml.safe_dump({"feature_coverage_policy": {}}), encoding="utf-8")
    meta_config.write_text(
        yaml.safe_dump({"search": {"meta_signal_mode": "meta_veto", "meta_weight_threshold": 0.52}}),
        encoding="utf-8",
    )
    meta_trained_config.write_text(
        json.dumps(
            {
                "threshold": 0.54,
                "threshold_selection": {
                    "auto_threshold_on_oof": True,
                    "selected_threshold": 0.54,
                    "trades": 313.0,
                    "net": -0.114309,
                    "hit_rate": 0.507987,
                    "quantile_cap": 0.54,
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_signal_program_dispositions(path: Path) -> dict[str, object]:
        assert path == analysis_dir
        return {"families": {"macro": {"status": "closed", "disposition": "deprioritize_for_now"}, "state_engineering": {"status": "active", "disposition": "guarded_shadow_validation_active"}}}

    def fake_build_derivatives_family_audit(*, config, models_root: Path) -> dict[str, object]:
        assert models_root == models_root
        return {"readiness": {"decision": "shadow_scaffold_ready", "next_action": "run_first_shadow_derivatives_validation"}}

    def fake_build_derivatives_shadow_scaffold(audit: dict[str, object]) -> dict[str, object]:
        return {"runner_status": "ready"}

    def fake_build_derivatives_shadow_candidate_config(base, *, audit):
        return {"feature_coverage_policy": {"ignored_sources": []}}

    def fake_build_signal_expansion_rollout_summary(**kwargs):
        captured.update(kwargs)
        return {
            "next_priority_family": "meta_ensemble",
            "implementation_notes": ["Meta lane remains first."],
            "program_direction": {
                "meta_ensemble": {
                    "recommended_action": "evaluate_before_expanding_base_models",
                    "source_csv": kwargs["meta_baseline_source_csv"],
                    "config_path": kwargs["meta_config_path"],
                    "weight_threshold": kwargs["meta_weight_threshold"],
                    "selected_weight_threshold": kwargs["meta_selected_weight_threshold"],
                    "auto_threshold_on_oof": kwargs["meta_auto_threshold_on_oof"],
                    "threshold_selection": dict(kwargs["meta_threshold_selection"]),
                },
                "derivatives": {
                    "status": "shadow_scaffold_ready",
                    "candidate_config": kwargs["derivatives_config_path"],
                },
                "featurelift_4h": {"candidate_config": kwargs["featurelift_config_path"]},
                "state_engineering": {"guarded_shadow_json": kwargs["state_guarded_json_path"]},
                "macro": {"recommended_action": "keep_deprioritized"},
            },
        }

    monkeypatch.setattr(module, "build_signal_program_dispositions", fake_build_signal_program_dispositions)
    monkeypatch.setattr(module, "build_derivatives_family_audit", fake_build_derivatives_family_audit)
    monkeypatch.setattr(module, "build_derivatives_shadow_scaffold", fake_build_derivatives_shadow_scaffold)
    monkeypatch.setattr(module, "build_derivatives_shadow_candidate_config", fake_build_derivatives_shadow_candidate_config)
    monkeypatch.setattr(module, "build_signal_expansion_rollout_summary", fake_build_signal_expansion_rollout_summary)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_signal_expansion_rollout",
            "--base-config",
            str(base_config),
            "--analysis-dir",
            str(analysis_dir),
            "--models-root",
            str(models_root),
            "--meta-config",
            str(meta_config),
            "--meta-trained-config",
            str(meta_trained_config),
            "--derivatives-config",
            str(derivatives_config),
            "--featurelift-config",
            str(featurelift_config),
            "--featurelift-package-markdown",
            str(featurelift_package),
            "--state-guarded-json",
            str(state_guarded_json),
            "--state-guarded-markdown",
            str(state_guarded_md),
            "--state-guarded-package-markdown",
            str(state_guarded_package),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
    )

    module.main()

    assert captured["meta_selected_weight_threshold"] == 0.54
    assert captured["meta_auto_threshold_on_oof"] is True
    assert captured["meta_threshold_selection"] == {
        "auto_threshold_on_oof": True,
        "selected_threshold": 0.54,
        "trades": 313.0,
        "net": -0.114309,
        "hit_rate": 0.507987,
        "quantile_cap": 0.54,
    }

    rollout_payload = json.loads(output_json.read_text(encoding="utf-8"))
    rollout_markdown = output_markdown.read_text(encoding="utf-8")

    assert rollout_payload["program_direction"]["meta_ensemble"]["selected_weight_threshold"] == 0.54
    assert rollout_payload["program_direction"]["meta_ensemble"]["threshold_selection"]["trades"] == 313.0
    assert rollout_payload["program_direction"]["state_engineering"]["guarded_rollout_package_markdown"] == str(state_guarded_package)
    assert "## Meta Threshold Selection" in rollout_markdown
    assert "- Meta-ensemble selected threshold: 0.54" in rollout_markdown