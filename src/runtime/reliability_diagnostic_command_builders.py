from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence


def build_feature_reliability_command(
    *,
    python: str,
    input_path: Path,
    feature_cfg: Mapping[str, Any],
    output_path: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.evaluate_feature_reliability",
        "--input",
        str(input_path),
        "--baseline-window",
        str(int(feature_cfg.get("baseline_window", 240))),
        "--recent-window",
        str(int(feature_cfg.get("recent_window", 120))),
        "--min-score",
        str(float(feature_cfg.get("min_score", 0.55))),
        "--max-features",
        str(int(feature_cfg.get("max_features", 0))),
        "--output",
        str(output_path),
    ]


def build_overlap_trust_stability_command(
    *,
    python: str,
    full_selected_path: Path,
    overlap_selected_path: Path,
    labeled_overlap_dataset: Path,
    quality_input: Path,
    feature_sources: Sequence[Path],
    reconcile_cfg: Mapping[str, Any],
    champ_cfg: Mapping[str, Any],
    trade_decision_cfg: Mapping[str, Any],
    output_path: Path,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.analyze_overlap_trust_stability",
        "--full-walkforward",
        str(full_selected_path),
        "--overlap-walkforward",
        str(overlap_selected_path),
        "--overlap-dataset",
        str(labeled_overlap_dataset),
        "--labeled-csv",
        str(quality_input),
    ]
    for feature_source in feature_sources:
        cmd.extend(["--feature-source", str(feature_source)])
    cmd.extend(
        [
            "--ts-col",
            str(reconcile_cfg.get("ts_col", "ts")),
            "--return-col",
            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
            "--signal-col",
            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
            "--output",
            str(output_path),
        ]
    )
    return cmd