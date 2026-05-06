from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


def build_trade_decision_policy_backtest_command(
    *,
    python: str,
    input_path: Path | str,
    model_path: Path | str,
    output_path: Path | str,
    meta_output_path: Path | str,
    diagnostics_output_path: Path | str | None = None,
    diagnostics_only: bool = False,
    extra_args: Sequence[str] | None = None,
    feature_sources: Iterable[Path | str] | None = None,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.apply_trade_decision_policy_to_backtest",
        "--input",
        str(input_path),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
        "--meta-output",
        str(meta_output_path),
    ]
    if diagnostics_output_path is not None:
        cmd.extend(["--diagnostics-output", str(diagnostics_output_path)])
    if diagnostics_only:
        cmd.append("--diagnostics-only")
    if extra_args:
        cmd.extend([str(value) for value in extra_args])
    if feature_sources:
        for feature_source in feature_sources:
            cmd.extend(["--feature-source", str(feature_source)])
    return cmd