from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_dataset_command_builders import (
    build_canonical_direction_dataset_command,
    build_canonical_hourly_dataset_command,
    build_labeled_dataset_command,
)


def test_build_canonical_hourly_dataset_command_uses_output_dir_override() -> None:
    cmd = build_canonical_hourly_dataset_command(
        python="python",
        canonical_cfg={"output_dir": "artifacts/custom_datasets"},
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.build_training_dataset",
        "--output-dir",
        "artifacts/custom_datasets",
    ]


def test_build_canonical_direction_dataset_command_renders_binary_label_settings() -> None:
    cmd = build_canonical_direction_dataset_command(
        python="python",
        canonical_cfg={
            "output_dir": "artifacts/datasets",
            "threshold": 0.012,
            "no_trade_abs_ret": 0.001,
            "no_trade_vol_mult": 0.8,
            "meta_path": "artifacts/datasets/meta.json",
            "tb_horizon_steps": 6,
            "tb_vol_window": 48,
            "tb_upper_mult": 1.8,
            "tb_lower_mult": 1.2,
        },
        label_policy="binary",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.build_training_dataset_direction" in rendered
    assert "--threshold 0.012" in rendered
    assert "--labeling-scheme binary" in rendered
    assert "--tb-horizon-steps 6" in rendered
    assert "--tb-upper-mult 1.8" in rendered
    assert "--tb-lower-mult 1.2" in rendered


def test_build_labeled_dataset_command_includes_optional_flags(tmp_path: Path) -> None:
    cmd = build_labeled_dataset_command(
        python="python",
        quality_input=tmp_path / "quality.csv",
        labeled_meta_output=tmp_path / "quality_meta.json",
        quality_cfg={
            "fold_size": 24,
            "lookback_rows": 4000,
            "lookback_hours": 72,
            "min_labeled_rows": 250,
            "prefer_backtest": False,
            "include_reliability_snapshots": True,
        },
        resolved_quality_backtest_csv=tmp_path / "backtest.csv",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.build_labeled_backtest_from_history" in rendered
    assert f"--output {tmp_path / 'quality.csv'}" in rendered
    assert f"--meta-output {tmp_path / 'quality_meta.json'}" in rendered
    assert f"--backtest-csv {tmp_path / 'backtest.csv'}" in rendered
    assert "--fold-size 24" in rendered
    assert "--lookback-rows 4000" in rendered
    assert "--lookback-hours 72" in rendered
    assert "--min-rows 250" in rendered
    assert "--no-prefer-backtest" in rendered
    assert "--include-reliability-snapshots" in rendered