from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping


def build_platt_calibration_command(
    *,
    python: str,
    horizons: Iterable[Any],
    output_path: Path,
    coverage_output_path: Path,
    method: str,
    labeled_input: Path | None = None,
    fit_base_horizons_from_labeled_input: bool = False,
    skip_model_fit: bool = False,
    regime_col: str | None = None,
    min_regime_rows: int | None = None,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.train_platt_calibration",
        "--horizons",
        *[str(horizon) for horizon in horizons],
        "--output-path",
        str(output_path),
        "--coverage-output-path",
        str(coverage_output_path),
        "--method",
        str(method),
    ]
    if labeled_input is not None:
        cmd.extend(["--labeled-input", str(labeled_input)])
    if fit_base_horizons_from_labeled_input:
        cmd.append("--fit-base-horizons-from-labeled-input")
    if skip_model_fit:
        cmd.append("--skip-model-fit")
    if regime_col is not None:
        cmd.extend(["--regime-col", str(regime_col)])
    if min_regime_rows is not None:
        cmd.extend(["--min-regime-rows", str(int(min_regime_rows))])
    return cmd


def build_regime_weakness_command(
    *,
    python: str,
    calibration_path: Path,
    walkforward_path: Path,
    horizon_key: str,
    regime_weakness_cfg: Mapping[str, Any],
    calibration_cfg: Mapping[str, Any],
    quality_cfg: Mapping[str, Any],
    output_path: Path,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.evaluate_regime_weakness",
        "--calibration",
        str(calibration_path),
        "--walkforward",
        str(walkforward_path),
        "--horizon",
        str(horizon_key),
        "--max-ece-drift",
        str(float(regime_weakness_cfg.get("max_ece_drift", calibration_cfg.get("max_ece_drift", 0.02)))),
        "--min-recent-auc",
        str(float(quality_cfg.get("min_recent_auc", 0.0))),
        "--min-net-return",
        str(float(regime_weakness_cfg.get("min_net_return", 0.0))),
        "--output",
        str(output_path),
    ]