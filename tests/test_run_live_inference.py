from __future__ import annotations

from src.scripts.run_live_inference import _forwarded_argv, parse_args


def test_live_inference_forwards_constrained_arguments() -> None:
    args = parse_args([
        "--config",
        "configs/run_refresh_and_predict.live_conservative_binance_only.yaml",
        "--hours",
        "240",
        "--targets",
        "1,4,8",
        "--disable-monitoring-latest",
    ])

    forwarded = _forwarded_argv(args)

    assert forwarded == [
        "--config",
        "configs/run_refresh_and_predict.live_conservative_binance_only.yaml",
        "--hours",
        "240",
        "--targets",
        "1,4,8",
        "--spot-provider",
        "binanceus",
        "--disable-monitoring-latest",
        "--write-artifacts",
    ]
