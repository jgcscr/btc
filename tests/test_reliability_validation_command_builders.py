from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_validation_command_builders import (
    build_cv_stress_sweep_command,
    build_label_ablation_command,
    build_point_in_time_audit_command,
    build_walkforward_validation_command,
)


def test_build_walkforward_validation_command_renders_all_core_flags(tmp_path: Path) -> None:
    cmd = build_walkforward_validation_command(
        python="python",
        dataset_path=tmp_path / "dataset.npz",
        walkforward_target="y",
        folds=6,
        train_size=168,
        val_size=24,
        test_size=24,
        gap=2,
        purge_size=1,
        embargo_size=3,
        mode="rolling",
        model_kind="xgb",
        signal_threshold=0.57,
        fee_bps=2.0,
        slippage_bps=1.0,
        output_path=tmp_path / "walkforward.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.run_walkforward_validation" in rendered
    assert "--folds 6" in rendered
    assert "--train-size 168" in rendered
    assert "--signal-threshold 0.57" in rendered
    assert f"--output {tmp_path / 'walkforward.json'}" in rendered


def test_build_label_ablation_command_includes_optional_feature_reliability(tmp_path: Path) -> None:
    cmd = build_label_ablation_command(
        python="python",
        summary_dir=tmp_path,
        label_ablation_cfg={"threshold": 0.01},
        quality_cfg={"walkforward_signal_threshold": 0.55, "walkforward_fee_bps": 2.0, "walkforward_slippage_bps": 1.0, "min_trade_count": 10, "min_net_return": 0.0},
        selected_model_kind="meta_stack",
        cv_folds=6,
        cv_train_size=168,
        cv_val_size=24,
        cv_test_size=24,
        cv_gap=0,
        cv_purge_size=0,
        cv_embargo_size=0,
        cv_mode="rolling",
        feature_reliability_json=tmp_path / "feature_reliability.json",
        feature_reliability_min_score=0.6,
    )

    rendered = " ".join(cmd)
    assert "src.scripts.run_label_ablation" in rendered
    assert f"--feature-reliability-json {tmp_path / 'feature_reliability.json'}" in rendered
    assert "--feature-reliability-min-score 0.6" in rendered


def test_build_point_in_time_audit_and_cv_stress_commands_use_defaults(tmp_path: Path) -> None:
    leakage_cmd = build_point_in_time_audit_command(
        python="python",
        walkforward_dataset=tmp_path / "dataset.npz",
        walkforward_target="y",
        leakage_cfg={},
        output_path=tmp_path / "audit.json",
    )
    cv_cmd = build_cv_stress_sweep_command(
        python="python",
        walkforward_dataset=tmp_path / "dataset.npz",
        walkforward_target="y",
        cv_stress_cfg={},
        folds=4,
        train_size=100,
        val_size=20,
        test_size=20,
        cv_gap=0,
        cv_mode="rolling",
        output_path=tmp_path / "cv.json",
    )

    assert leakage_cmd == [
        "python",
        "-m",
        "src.scripts.audit_point_in_time_integrity",
        "--dataset-path",
        str(tmp_path / "dataset.npz"),
        "--y-key",
        "y",
        "--leakage-corr-alert",
        "0.98",
        "--output",
        str(tmp_path / "audit.json"),
    ]
    assert "src.scripts.run_cv_stress_sweep" in " ".join(cv_cmd)
    assert "--purge-list" in cv_cmd
    assert "--embargo-list" in cv_cmd