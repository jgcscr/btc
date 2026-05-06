from __future__ import annotations

import json
from pathlib import Path

from src.runtime.reliability_meta_command_builders import (
    build_meta_ensemble_command,
    resolve_meta_component_weight_spec,
)


def test_build_meta_ensemble_command_includes_component_frame_columns_and_weights(tmp_path: Path) -> None:
    cmd = build_meta_ensemble_command(
        python="python",
        output_csv=tmp_path / "meta.csv",
        config_path=tmp_path / "meta.json",
        search_cfg={"meta_weight_threshold": 0.7, "meta_signal_mode": "blend"},
        component_frame_path=tmp_path / "frame.csv",
        component_columns=["xgb", "lstm"],
        extra_feature_columns=["volatility", "drawdown"],
        component_weight_spec="xgb:0.6,lstm:0.4",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.train_meta_ensemble" in rendered
    assert f"--output-csv {tmp_path / 'meta.csv'}" in rendered
    assert f"--config-path {tmp_path / 'meta.json'}" in rendered
    assert "--weight-threshold 0.7" in rendered
    assert "--signal-mode blend" in rendered
    assert f"--component-frame-csv {tmp_path / 'frame.csv'}" in rendered
    assert rendered.count("--component-column") == 2
    assert rendered.count("--extra-feature-column") == 2
    assert "--component-weight-spec xgb:0.6,lstm:0.4" in rendered


def test_build_meta_ensemble_command_ignores_non_sequence_columns(tmp_path: Path) -> None:
    cmd = build_meta_ensemble_command(
        python="python",
        output_csv=tmp_path / "meta.csv",
        config_path=tmp_path / "meta.json",
        search_cfg={},
        component_frame_path=tmp_path / "frame.csv",
        component_columns="xgb",
        extra_feature_columns=None,
    )

    assert "--component-column" not in cmd
    assert "--extra-feature-column" not in cmd


def test_resolve_meta_component_weight_spec_reads_audit_file(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"weights": "payload"}), encoding="utf-8")

    observed_allowed_components: list[tuple[str, ...]] = []

    def fake_load_json(path: Path) -> dict[str, str]:
        assert path == audit_path
        return {"weights": "payload"}

    def fake_extract_audit_weight_spec(payload: dict[str, str], *, allowed_components: tuple[str, ...]) -> str:
        observed_allowed_components.append(allowed_components)
        assert payload == {"weights": "payload"}
        return "xgb:0.55,lstm:0.45"

    spec, resolved_path, error = resolve_meta_component_weight_spec(
        search_cfg={"meta_component_weights_from_audit_path": str(audit_path)},
        load_json=fake_load_json,
        extract_audit_weight_spec=fake_extract_audit_weight_spec,
    )

    assert spec == "xgb:0.55,lstm:0.45"
    assert resolved_path == audit_path
    assert error is None
    assert observed_allowed_components


def test_resolve_meta_component_weight_spec_returns_error_string_on_failure(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    def fake_load_json(path: Path) -> dict[str, object]:
        assert path == audit_path
        return {}

    def fake_extract_audit_weight_spec(payload: dict[str, object], *, allowed_components: tuple[str, ...]) -> str:
        raise ValueError("broken audit")

    spec, resolved_path, error = resolve_meta_component_weight_spec(
        search_cfg={"meta_component_weights_from_audit_path": str(audit_path)},
        load_json=fake_load_json,
        extract_audit_weight_spec=fake_extract_audit_weight_spec,
    )

    assert spec is None
    assert resolved_path == audit_path
    assert error == "broken audit"