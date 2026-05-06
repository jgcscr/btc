from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_trade_decision_command_builders import (
    build_trade_decision_policy_backtest_command,
)


def test_build_trade_decision_policy_backtest_command_renders_base_arguments(tmp_path: Path) -> None:
    cmd = build_trade_decision_policy_backtest_command(
        python="python",
        input_path=tmp_path / "input.csv",
        model_path=tmp_path / "model.json",
        output_path=tmp_path / "output.csv",
        meta_output_path=tmp_path / "meta.json",
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.apply_trade_decision_policy_to_backtest",
        "--input",
        str(tmp_path / "input.csv"),
        "--model",
        str(tmp_path / "model.json"),
        "--output",
        str(tmp_path / "output.csv"),
        "--meta-output",
        str(tmp_path / "meta.json"),
    ]


def test_build_trade_decision_policy_backtest_command_includes_diagnostics_extra_args_and_feature_sources(tmp_path: Path) -> None:
    cmd = build_trade_decision_policy_backtest_command(
        python="python",
        input_path=tmp_path / "input.csv",
        model_path=tmp_path / "model.json",
        output_path=tmp_path / "output.csv",
        meta_output_path=tmp_path / "meta.json",
        diagnostics_output_path=tmp_path / "diag.json",
        diagnostics_only=True,
        extra_args=["--threshold", "0.55", "--min-edge-over-fee", "0.0"],
        feature_sources=[tmp_path / "features_a.csv", tmp_path / "features_b.csv"],
    )

    rendered = " ".join(cmd)
    assert "--diagnostics-output" in rendered
    assert "--diagnostics-only" in rendered
    assert "--threshold 0.55" in rendered
    assert rendered.count("--feature-source") == 2