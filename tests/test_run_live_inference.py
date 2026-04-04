from __future__ import annotations

from src.scripts.run_live_inference import _forwarded_argv, _targets_include_intrabar, parse_args


def test_targets_include_intrabar_detects_subhourly_target() -> None:
    assert _targets_include_intrabar("0.25,1,4,12") is True
    assert _targets_include_intrabar("1,4,12") is False


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


def test_live_inference_enables_intrabar_when_15m_target_requested() -> None:
    args = parse_args([])

    forwarded = _forwarded_argv(args)

    assert forwarded == [
        "--config",
        "configs/run_refresh_and_predict.live_conservative_binance_only.yaml",
        "--hours",
        "360",
        "--targets",
        "0.25,1,4,12",
        "--spot-provider",
        "binanceus",
        "--intrabar-enabled",
        "--write-artifacts",
    ]
