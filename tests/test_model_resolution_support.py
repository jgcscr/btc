from __future__ import annotations

from pathlib import Path

from src.runtime.model_resolution_support import (
    direction_configs_for_horizon,
    model_paths_for_horizon,
    model_suffix_candidates,
    prepare_base_direction_configs,
)


def test_model_suffix_candidates_include_subhour_and_hourly_fallback() -> None:
    suffixes = model_suffix_candidates(0.25, normalize_horizon_value=float)

    assert suffixes == ["15m", "0.25h", "1h"]


def test_model_paths_for_horizon_uses_existing_hourly_fallback(tmp_path: Path) -> None:
    model_root = tmp_path / "models"

    def fake_resolve(base_dir: Path, *, expected_filename: str, version_priority):
        suffix = expected_filename.replace("xgb_", "").replace("_model.json", "")
        return model_root / f"{suffix}.json"

    (model_root / "ret1h.json").parent.mkdir(parents=True, exist_ok=True)
    (model_root / "ret1h.json").write_text("{}", encoding="utf-8")
    (model_root / "dir1h.json").write_text("{}", encoding="utf-8")
    messages: list[str] = []

    reg_path, dir_path = model_paths_for_horizon(
        0.25,
        format_horizon_label=lambda value: f"{int(value * 60)}m" if value < 1 else f"{int(value)}h",
        normalize_horizon_value=float,
        model_root=model_root,
        model_version_priority=("v2", "v1"),
        dir_version_overrides={},
        resolve_best_versioned_model_file_fn=fake_resolve,
        stderr_write=messages.append,
    )

    assert reg_path.name == "ret1h.json"
    assert dir_path.name == "dir1h.json"
    assert any("using 1h model artifacts" in message for message in messages)


def test_prepare_base_direction_configs_passes_path_overrides() -> None:
    captured = {}

    def fake_resolve(defaults, *, config_json_path, weight_spec, path_overrides):
        captured["defaults"] = defaults
        captured["config_json_path"] = config_json_path
        captured["weight_spec"] = weight_spec
        captured["path_overrides"] = path_overrides
        return [{"type": "xgb", "path": "x"}]

    configs = prepare_base_direction_configs(
        config_json_path="cfg.json",
        weight_spec="xgb:1",
        dir_lstm_path="lstm-dir",
        dir_bilstm_path=None,
        dir_gru_path=None,
        dir_cnn_lstm_path=None,
        dir_cnn_bilstm_path=None,
        dir_garch_lstm_path=None,
        dir_transformer_path="transformer-dir",
        default_dir_models_1h=[{"type": "xgb"}],
        resolve_direction_model_configs_fn=fake_resolve,
    )

    assert configs == [{"type": "xgb", "path": "x"}]
    assert captured["config_json_path"] == "cfg.json"
    assert captured["weight_spec"] == "xgb:1"
    assert captured["path_overrides"]["lstm"] == "lstm-dir"
    assert captured["path_overrides"]["transformer"] == "transformer-dir"


def test_direction_configs_for_horizon_applies_overrides_and_adds_lgbm(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    (model_root / "lstm_dir4h_v2").mkdir(parents=True, exist_ok=True)
    lgbm_dir = model_root / "lgbm_dir4h_v2"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    (lgbm_dir / "lgbm_dir4h_model.joblib").write_text("ok", encoding="utf-8")
    regime_logit_dir = model_root / "regime_logit_dir4h_v2"
    regime_logit_dir.mkdir(parents=True, exist_ok=True)
    (regime_logit_dir / "regime_logit_dir4h_model.joblib").write_text("ok", encoding="utf-8")
    transformer_dir = tmp_path / "transformer_dir"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    logs: list[tuple[str, object]] = []

    def fake_clone(configs):
        return [dict(item) for item in configs]

    def fake_apply(configs, overrides):
        logs.append(("overrides", dict(overrides)))
        for item in configs:
            item_type = item.get("type")
            if item_type in overrides:
                item["path"] = overrides[item_type]

    def fake_log(configs, *, label):
        logs.append((label, [dict(item) for item in configs]))

    configs, weight_map = direction_configs_for_horizon(
        [{"name": "xgb", "type": "xgb", "path": "old-xgb", "weight": 1.0}, {"name": "lstm", "type": "lstm", "path": "old-lstm", "weight": 0.5}],
        dir_model_path="new-xgb",
        horizon=4.0,
        horizon_label="4h",
        normalize_horizon_value=float,
        default_transformer_model_dir_by_suffix={"4h": str(transformer_dir)},
        model_root=model_root,
        model_version_priority=("v2", "v1"),
        clone_direction_model_configs_fn=fake_clone,
        apply_path_overrides_fn=fake_apply,
        log_direction_model_configs_fn=fake_log,
        direction_configs_to_weight_map_fn=lambda configs: {str(item["type"]): float(item["weight"]) for item in configs},
        registry_model_exists_fn=lambda name: False,
        env_get=lambda name: None,
    )

    assert any(item["type"] == "lgbm" for item in configs)
    assert any(item["type"] == "regime_logit" for item in configs)
    assert weight_map["xgb"] == 1.0
    assert weight_map["lgbm"] == 1.0
    assert weight_map["regime_logit"] == 0.5
    override_payload = dict(logs[0][1])
    assert override_payload["xgb"] == "new-xgb"
    assert override_payload["lstm"].endswith("lstm_dir4h_v2")
    assert override_payload["transformer"].endswith("transformer_dir")