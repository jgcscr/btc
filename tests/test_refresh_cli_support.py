from __future__ import annotations

from src.runtime.refresh_cli_support import parse_refresh_args


def test_parse_refresh_args_applies_config_defaults_but_preserves_cli_overrides() -> None:
    args = parse_refresh_args(
        ["--config", "cfg.yaml", "--hours", "120", "--disable-monitoring-latest"],
        load_cli_config=lambda path: {"config": path, "hours": 360, "targets": [1.0, 4.0], "write_artifacts": True},
        parse_targets=lambda value: [float(part) for part in value.split(",")],
        default_hours=24,
        default_targets=[0.25, 1.0],
        default_p_up_min=0.45,
        default_ret_min=0.0,
        confidence_min_default=0.0,
        position_size_floor_default=0.0,
        position_size_cap_default=1.0,
        default_dir_model_weights_1h="xgb:1",
    )

    assert args.config == "cfg.yaml"
    assert args.hours == 120
    assert args.targets == [1.0, 4.0]
    assert args.write_artifacts is True
    assert args.disable_monitoring_latest is True