from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_overlap_command_builders import (
    build_direction_feature_snapshot_command,
    build_labeled_overlap_dataset_command,
    build_overlap_feature_drift_guard_command,
)


def test_build_direction_feature_snapshot_command_renders_dataset_and_outputs(tmp_path: Path) -> None:
    cmd = build_direction_feature_snapshot_command(
        python="python",
        dataset_path=tmp_path / "input.npz",
        output_path=tmp_path / "snapshot.json",
        meta_output_path=tmp_path / "snapshot_meta.json",
    )

    assert cmd == [
        "python",
        "-m",
        "src.scripts.export_direction_feature_snapshot",
        "--dataset",
        str(tmp_path / "input.npz"),
        "--output",
        str(tmp_path / "snapshot.json"),
        "--meta-output",
        str(tmp_path / "snapshot_meta.json"),
    ]


def test_build_labeled_overlap_dataset_command_includes_fallback_flags(tmp_path: Path) -> None:
    cmd = build_labeled_overlap_dataset_command(
        python="python",
        walkforward_dataset=tmp_path / "walkforward.npz",
        quality_input=tmp_path / "quality.csv",
        reconcile_cfg={
            "ts_col": "timestamp",
            "min_rows": 240,
            "fallback_labeling_scheme": "triple_barrier",
            "fallback_min_coverage_ratio": 0.75,
        },
        labeled_overlap_dataset=tmp_path / "overlap.npz",
        labeled_overlap_meta=tmp_path / "overlap_meta.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.slice_direction_dataset_by_timestamps" in rendered
    assert "--ts-col timestamp" in rendered
    assert "--min-rows 240" in rendered
    assert "--fallback-labeling-scheme triple_barrier" in rendered
    assert "--fallback-min-coverage-ratio 0.75" in rendered


def test_build_overlap_feature_drift_guard_command_includes_feature_filters(tmp_path: Path) -> None:
    cmd = build_overlap_feature_drift_guard_command(
        python="python",
        baseline_pack_path=tmp_path / "baseline.parquet",
        labeled_overlap_dataset=tmp_path / "overlap.npz",
        overlap_drift_guard_cfg={
            "tail_rows": 36,
            "warn_abs_train_std_shift": 1.2,
            "fail_abs_train_std_shift": 2.2,
            "min_failed_features": 3,
            "feature_prefixes": ["momentum_", "vol_"],
            "feature_names": ["feature_a", "feature_b"],
        },
        output_path=tmp_path / "guard.json",
    )

    rendered = " ".join(cmd)
    assert "src.scripts.analyze_overlap_feature_drift_guard" in rendered
    assert "--tail-rows 36" in rendered
    assert "--warn-abs-train-std-shift 1.2" in rendered
    assert "--fail-abs-train-std-shift 2.2" in rendered
    assert "--min-failed-features 3" in rendered
    assert rendered.count("--feature-prefix") == 2
    assert rendered.count("--feature-name") == 2