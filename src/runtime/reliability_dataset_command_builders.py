from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping


def build_canonical_hourly_dataset_command(*, python: str, canonical_cfg: Mapping[str, Any]) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.build_training_dataset",
        "--output-dir",
        str(canonical_cfg.get("output_dir", "artifacts/datasets")),
    ]


def build_canonical_direction_dataset_command(
    *,
    python: str,
    canonical_cfg: Mapping[str, Any],
    label_policy: str,
) -> List[str]:
    return [
        python,
        "-m",
        "src.scripts.build_training_dataset_direction",
        "--output-dir",
        str(canonical_cfg.get("output_dir", "artifacts/datasets")),
        "--threshold",
        str(float(canonical_cfg.get("threshold", 0.0))),
        "--labeling-scheme",
        str(label_policy),
        "--no-trade-abs-ret",
        str(float(canonical_cfg.get("no_trade_abs_ret", 0.0))),
        "--no-trade-vol-mult",
        str(float(canonical_cfg.get("no_trade_vol_mult", 0.0))),
        "--meta-path",
        str(canonical_cfg.get("meta_path", "artifacts/datasets/btc_features_1h_direction_meta.json")),
        "--tb-horizon-steps",
        str(int(canonical_cfg.get("tb_horizon_steps", 3))),
        "--tb-vol-window",
        str(int(canonical_cfg.get("tb_vol_window", 24))),
        "--tb-upper-mult",
        str(float(canonical_cfg.get("tb_upper_mult", 1.5))),
        "--tb-lower-mult",
        str(float(canonical_cfg.get("tb_lower_mult", 1.5))),
    ]


def build_labeled_dataset_command(
    *,
    python: str,
    quality_input: Path,
    labeled_meta_output: Path,
    quality_cfg: Mapping[str, Any],
    resolved_quality_backtest_csv: Path | None,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.build_labeled_backtest_from_history",
        "--output",
        str(quality_input),
        "--meta-output",
        str(labeled_meta_output),
        "--fold-size",
        str(int(quality_cfg.get("fold_size", 12))),
        "--lookback-rows",
        str(int(quality_cfg.get("lookback_rows", 2000))),
        "--lookback-hours",
        str(int(quality_cfg.get("lookback_hours", 0))),
        "--min-rows",
        str(int(quality_cfg.get("min_labeled_rows", 200))),
    ]
    if resolved_quality_backtest_csv is not None:
        cmd.extend(["--backtest-csv", str(resolved_quality_backtest_csv)])
    if bool(quality_cfg.get("prefer_backtest", True)):
        cmd.append("--prefer-backtest")
    else:
        cmd.append("--no-prefer-backtest")
    if bool(quality_cfg.get("include_reliability_snapshots", False)):
        cmd.append("--include-reliability-snapshots")
    return cmd