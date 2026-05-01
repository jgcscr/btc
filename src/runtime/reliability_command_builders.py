from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def build_rolling_ab_command(
    *,
    python: str,
    baseline_path: Path | str,
    candidate_path: Path | str,
    rolling_cfg: Dict[str, Any],
    output_path: Path,
    output_md_path: Path,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.evaluate_rolling_ab",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--window-size",
        str(int(rolling_cfg.get("window_size", 168))),
        "--step-size",
        str(int(rolling_cfg.get("step_size", 24))),
        "--min-window-trades",
        str(int(rolling_cfg.get("min_window_trades", 5))),
        "--output",
        str(output_path),
        "--output-md",
        str(output_md_path),
    ]
    if bool(rolling_cfg.get("allow_no_trade_baseline", False)):
        cmd.append("--allow-no-trade-baseline")
    return cmd


def build_calibration_robustness_command(
    *,
    python: str,
    input_path: Path,
    output_path: Path,
    calibration_cfg: Dict[str, Any],
    quality_cfg: Dict[str, Any],
    trade_decision_cfg: Dict[str, Any],
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.evaluate_calibration_robustness",
        "--input",
        str(input_path),
        "--baseline-window",
        str(int(calibration_cfg.get("baseline_window", 240))),
        "--recent-window",
        str(int(calibration_cfg.get("recent_window", 120))),
        "--max-ece-drift",
        str(float(calibration_cfg.get("max_ece_drift", 0.02))),
        "--max-recent-ece",
        str(float(quality_cfg.get("max_recent_ece", 1.0))),
        "--min-recent-auc",
        str(float(quality_cfg.get("min_recent_auc", 0.0))),
        "--regime-col",
        str(calibration_cfg.get("regime_col", "regime_state")),
        "--signal-col",
        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
        "--return-col",
        str(calibration_cfg.get("return_col", "ret_ensemble_net")),
        "--selection-scope",
        str(calibration_cfg.get("selection_scope", "all")),
        "--min-selection-rows",
        str(int(calibration_cfg.get("min_selection_rows", 0))),
        "--output",
        str(output_path),
    ]
    adaptive_selection_cfg = (
        calibration_cfg.get("adaptive_selection_rows")
        if isinstance(calibration_cfg.get("adaptive_selection_rows"), dict)
        else {}
    )
    if bool(adaptive_selection_cfg.get("enabled", False)):
        cmd.append("--adaptive-selection-rows")
        cmd.extend(
            [
                "--adaptive-selection-min-floor",
                str(int(adaptive_selection_cfg.get("min_floor", 0))),
                "--adaptive-selection-baseline-ratio",
                str(float(adaptive_selection_cfg.get("baseline_ratio", 0.0))),
                "--adaptive-selection-max-shortfall",
                str(int(adaptive_selection_cfg.get("max_shortfall", 0))),
            ]
        )
    return cmd


def build_directional_objectives_command(
    *,
    python: str,
    input_path: Path,
    output_path: Path,
    directional_cfg: Dict[str, Any],
) -> List[str]:
    def _format_threshold_map(raw: Any) -> str:
        if not isinstance(raw, dict):
            return ""
        parts: List[str] = []
        for key, value in raw.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            label = str(key).strip()
            if not label:
                continue
            parts.append(f"{label}:{numeric}")
        return ",".join(parts)

    cmd = [
        python,
        "-m",
        "src.scripts.evaluate_directional_objectives",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--prob-col",
        str(directional_cfg.get("prob_col", "p_up")),
        "--regime-col",
        str(directional_cfg.get("regime_col", "regime_state")),
        "--threshold",
        str(float(directional_cfg.get("threshold", 0.5))),
        "--min-rows",
        str(int(directional_cfg.get("min_rows", 300))),
        "--group-min-rows",
        str(int(directional_cfg.get("group_min_rows", 80))),
        "--max-brier",
        str(float(directional_cfg.get("max_brier", 0.25))),
        "--max-ece",
        str(float(directional_cfg.get("max_ece", 0.08))),
        "--min-f1",
        str(float(directional_cfg.get("min_f1", 0.45))),
    ]
    label_col = directional_cfg.get("label_col")
    if label_col:
        cmd.extend(["--label-col", str(label_col)])
    for flag, key in (
        ("--min-rows-by-regime", "min_rows_by_regime"),
        ("--max-brier-by-horizon", "max_brier_by_horizon"),
        ("--max-ece-by-horizon", "max_ece_by_horizon"),
        ("--min-f1-by-horizon", "min_f1_by_horizon"),
        ("--max-brier-by-regime", "max_brier_by_regime"),
        ("--max-ece-by-regime", "max_ece_by_regime"),
        ("--min-f1-by-regime", "min_f1_by_regime"),
    ):
        encoded = _format_threshold_map(directional_cfg.get(key))
        if encoded:
            cmd.extend([flag, encoded])
    return cmd