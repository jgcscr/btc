from __future__ import annotations

import argparse
from typing import Sequence

from src.runtime.models import RuntimeMode
from src.runtime.refresh_pipeline import execute_refresh_pipeline
from src.scripts import run_refresh_and_predict as legacy

DEFAULT_LIVE_CONFIG = "configs/run_refresh_and_predict.live_conservative_binance_only.yaml"
DEFAULT_TARGETS = "0.25,1,4,12"


def _targets_include_intrabar(targets: str) -> bool:
    for raw_target in str(targets).split(","):
        value = raw_target.strip()
        if not value:
            continue
        try:
            if abs(float(value) - 0.25) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production-facing live inference path with a constrained option surface.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_LIVE_CONFIG,
        help="Runtime config for live inference.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=360,
        help="Number of hourly candles to fetch before inference.",
    )
    parser.add_argument(
        "--targets",
        default=DEFAULT_TARGETS,
        help="Comma-separated live targets to score.",
    )
    parser.add_argument(
        "--spot-provider",
        choices=("binanceus",),
        default="binanceus",
        help="Spot ingestion provider.",
    )
    parser.add_argument(
        "--disable-monitoring-latest",
        action="store_true",
        help="Skip writing artifacts/monitoring/latest.json.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Do not write trade-ready monitoring artifacts.",
    )
    return parser.parse_args(argv)


def _forwarded_argv(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--config",
        str(args.config),
        "--hours",
        str(args.hours),
        "--targets",
        str(args.targets),
        "--spot-provider",
        str(args.spot_provider),
    ]
    if _targets_include_intrabar(args.targets):
        forwarded.append("--intrabar-enabled")
    if args.disable_monitoring_latest:
        forwarded.append("--disable-monitoring-latest")
    if not args.no_write_artifacts:
        forwarded.append("--write-artifacts")
    return forwarded


def main(argv: Sequence[str] | None = None):
    live_args = parse_args(argv)
    legacy_args = legacy.parse_args(_forwarded_argv(live_args))
    if getattr(legacy_args, "config", None):
        print(f"Loaded CLI defaults from config: {legacy_args.config}")
    return execute_refresh_pipeline(legacy_args, mode=RuntimeMode.LIVE)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
