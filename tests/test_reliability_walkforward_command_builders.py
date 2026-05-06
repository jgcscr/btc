from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_walkforward_command_builders import (
    build_walkforward_model_compare_command,
)


def test_build_walkforward_model_compare_command_renders_core_walkforward_flags(tmp_path: Path) -> None:
    cmd = build_walkforward_model_compare_command(
        python="python",
        dataset_path=tmp_path / "dataset.npz",
        walkforward_target="y",
        compare_cfg={
            "folds": 8,
            "train_size": 192,
            "val_size": 48,
            "test_size": 48,
            "gap": 2,
            "purge_size": 1,
            "embargo_size": 3,
            "mode": "expanding",
            "min_train_size": 50,
            "min_val_size": 20,
            "min_test_size": 20,
            "signal_threshold": 0.56,
            "fee_bps": 3.0,
            "slippage_bps": 1.5,
            "rolling_guard": True,
            "meta_margin": 0.02,
            "meta_min_rolling_trades": 15,
            "selection_policy": "incumbent_guarded",
            "min_auc": 0.57,
        },
        quality_cfg={},
        cv_folds=6,
        cv_train_size=168,
        cv_val_size=24,
        cv_test_size=24,
        cv_gap=0,
        cv_purge_size=0,
        cv_embargo_size=0,
        cv_mode="rolling",
        default_min_train_size=30,
        default_min_val_size=20,
        default_min_test_size=20,
        default_selection_policy="best_cum_ret",
        output_path=tmp_path / "compare.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.compare_walkforward_models" in rendered
    assert f"--dataset-path {tmp_path / 'dataset.npz'}" in rendered
    assert "--folds 8" in rendered
    assert "--train-size 192" in rendered
    assert "--val-size 48" in rendered
    assert "--test-size 48" in rendered
    assert "--rolling-guard" in rendered
    assert "--selection-policy incumbent_guarded" in rendered
    assert "--min-auc 0.57" in rendered


def test_build_walkforward_model_compare_command_omits_rolling_guard_when_disabled(tmp_path: Path) -> None:
    cmd = build_walkforward_model_compare_command(
        python="python",
        dataset_path=tmp_path / "dataset.npz",
        walkforward_target="y",
        compare_cfg={"rolling_guard": False},
        quality_cfg={"walkforward_signal_threshold": 0.51, "walkforward_fee_bps": 2.5, "walkforward_slippage_bps": 0.5},
        cv_folds=6,
        cv_train_size=168,
        cv_val_size=24,
        cv_test_size=24,
        cv_gap=0,
        cv_purge_size=0,
        cv_embargo_size=0,
        cv_mode="rolling",
        default_min_train_size=30,
        default_min_val_size=20,
        default_min_test_size=20,
        default_selection_policy="best_cum_ret",
        output_path=tmp_path / "compare.json",
    )

    assert "--rolling-guard" not in cmd
    assert "0.51" in cmd
    assert "2.5" in cmd
    assert "0.5" in cmd