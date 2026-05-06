from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_calibration_command_builders import (
    build_platt_calibration_command,
    build_regime_weakness_command,
)


def test_build_platt_calibration_command_includes_optional_labeled_input_flags(tmp_path: Path) -> None:
    cmd = build_platt_calibration_command(
        python="python",
        horizons=[1, 4, 12],
        output_path=tmp_path / "calibration.json",
        coverage_output_path=tmp_path / "coverage.json",
        method="platt",
        labeled_input=tmp_path / "labeled.csv",
        fit_base_horizons_from_labeled_input=True,
        skip_model_fit=True,
        regime_col="regime_state",
        min_regime_rows=120,
    )

    rendered = " ".join(cmd)
    assert "src.scripts.train_platt_calibration" in rendered
    assert "--horizons 1 4 12" in rendered
    assert f"--labeled-input {tmp_path / 'labeled.csv'}" in rendered
    assert "--fit-base-horizons-from-labeled-input" in rendered
    assert "--skip-model-fit" in rendered
    assert "--regime-col regime_state" in rendered
    assert "--min-regime-rows 120" in rendered


def test_build_regime_weakness_command_uses_config_defaults(tmp_path: Path) -> None:
    cmd = build_regime_weakness_command(
        python="python",
        calibration_path=tmp_path / "calibration.json",
        walkforward_path=tmp_path / "walkforward.json",
        horizon_key="1h",
        regime_weakness_cfg={"min_net_return": 0.01},
        calibration_cfg={"max_ece_drift": 0.03},
        quality_cfg={"min_recent_auc": 0.58},
        output_path=tmp_path / "regime.json",
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.evaluate_regime_weakness",
        "--calibration",
        str(tmp_path / "calibration.json"),
        "--walkforward",
        str(tmp_path / "walkforward.json"),
        "--horizon",
        "1h",
        "--max-ece-drift",
        "0.03",
        "--min-recent-auc",
        "0.58",
        "--min-net-return",
        "0.01",
        "--output",
        str(tmp_path / "regime.json"),
    ]