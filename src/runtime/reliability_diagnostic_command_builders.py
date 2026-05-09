from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Mapping, Sequence


def _infer_horizon_from_path(input_path: Path) -> float | None:
    match = re.search(r"(?:^|_)(\d+(?:\.\d+)?)h(?:_|$)", input_path.stem)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def build_feature_reliability_command(
    *,
    python: str,
    input_path: Path,
    feature_cfg: Mapping[str, Any],
    output_path: Path,
) -> List[str]:
    cmd = [
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
    horizon = feature_cfg.get("horizon")
    if horizon is None:
        horizon = _infer_horizon_from_path(input_path)
    if horizon is not None:
        cmd.extend(["--horizon", str(float(horizon))])

    horizon_col = feature_cfg.get("horizon_col")
    if horizon_col:
        cmd.extend(["--horizon-col", str(horizon_col)])

    regime_col = feature_cfg.get("regime_col")
    if regime_col:
        cmd.extend(["--regime-col", str(regime_col)])
    elif bool(feature_cfg.get("derive_regime", True)):
        cmd.append("--derive-regime")

    min_slice_rows = feature_cfg.get("min_slice_rows")
    if min_slice_rows is not None:
        cmd.extend(["--min-slice-rows", str(int(min_slice_rows))])

    return cmd


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