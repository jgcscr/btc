from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_diagnostic_command_builders import (
    build_feature_reliability_command,
    build_overlap_trust_stability_command,
)


def test_build_feature_reliability_command_renders_windows_and_thresholds(tmp_path: Path) -> None:
    cmd = build_feature_reliability_command(
        python="python",
        input_path=tmp_path / "features.csv",
        feature_cfg={
            "baseline_window": 80,
            "recent_window": 40,
            "min_score": 0.61,
            "max_features": 12,
        },
        output_path=tmp_path / "feature_reliability.json",
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.evaluate_feature_reliability",
        "--input",
        str(tmp_path / "features.csv"),
        "--baseline-window",
        "80",
        "--recent-window",
        "40",
        "--min-score",
        "0.61",
        "--max-features",
        "12",
        "--output",
        str(tmp_path / "feature_reliability.json"),
    ]


def test_build_overlap_trust_stability_command_adds_all_feature_sources(tmp_path: Path) -> None:
    cmd = build_overlap_trust_stability_command(
        python="python",
        full_selected_path=tmp_path / "full.csv",
        overlap_selected_path=tmp_path / "overlap.csv",
        labeled_overlap_dataset=tmp_path / "overlap.npz",
        quality_input=tmp_path / "quality.csv",
        feature_sources=[tmp_path / "a.csv", tmp_path / "b.csv"],
        reconcile_cfg={"ts_col": "timestamp"},
        champ_cfg={"candidate_col": "ret_net"},
        trade_decision_cfg={"signal_col": "signal_filtered"},
        output_path=tmp_path / "stability.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.analyze_overlap_trust_stability" in rendered
    assert rendered.count("--feature-source") == 2
    assert "--ts-col timestamp" in rendered
    assert "--return-col ret_net" in rendered
    assert "--signal-col signal_filtered" in rendered
    assert f"--output {tmp_path / 'stability.json'}" in rendered