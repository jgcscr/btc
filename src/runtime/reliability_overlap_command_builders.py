from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping


def build_direction_feature_snapshot_command(
    *,
    python: str,
    dataset_path: Path,
    output_path: Path,
    meta_output_path: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.export_direction_feature_snapshot",
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_path),
        "--meta-output",
        str(meta_output_path),
    ]


def build_labeled_overlap_dataset_command(
    *,
    python: str,
    walkforward_dataset: Path,
    quality_input: Path,
    reconcile_cfg: Mapping[str, Any],
    labeled_overlap_dataset: Path,
    labeled_overlap_meta: Path,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.slice_direction_dataset_by_timestamps",
        "--dataset",
        str(walkforward_dataset),
        "--labeled-csv",
        str(quality_input),
        "--ts-col",
        str(reconcile_cfg.get("ts_col", "ts")),
        "--min-rows",
        str(int(reconcile_cfg.get("min_rows", 120))),
        "--output-dataset",
        str(labeled_overlap_dataset),
        "--output-meta",
        str(labeled_overlap_meta),
    ]
    fallback_labeling_scheme = reconcile_cfg.get("fallback_labeling_scheme")
    if fallback_labeling_scheme:
        cmd.extend(
            [
                "--fallback-labeling-scheme",
                str(fallback_labeling_scheme),
                "--fallback-min-coverage-ratio",
                str(float(reconcile_cfg.get("fallback_min_coverage_ratio", 0.0))),
            ]
        )
    return cmd


def build_overlap_feature_drift_guard_command(
    *,
    python: str,
    baseline_pack_path: Path,
    labeled_overlap_dataset: Path,
    overlap_drift_guard_cfg: Mapping[str, Any],
    output_path: Path,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.analyze_overlap_feature_drift_guard",
        "--baseline-pack",
        str(baseline_pack_path),
        "--current-overlap-dataset",
        str(labeled_overlap_dataset),
        "--tail-rows",
        str(int(overlap_drift_guard_cfg.get("tail_rows", 24))),
        "--warn-abs-train-std-shift",
        str(float(overlap_drift_guard_cfg.get("warn_abs_train_std_shift", 1.5))),
        "--fail-abs-train-std-shift",
        str(float(overlap_drift_guard_cfg.get("fail_abs_train_std_shift", 2.5))),
        "--min-failed-features",
        str(int(overlap_drift_guard_cfg.get("min_failed_features", 2))),
        "--output",
        str(output_path),
    ]
    for prefix in overlap_drift_guard_cfg.get("feature_prefixes", []):
        if str(prefix).strip():
            cmd.extend(["--feature-prefix", str(prefix)])
    for feature_name in overlap_drift_guard_cfg.get("feature_names", []):
        if str(feature_name).strip():
            cmd.extend(["--feature-name", str(feature_name)])
    return cmd