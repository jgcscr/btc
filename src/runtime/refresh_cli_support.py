from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence


def parse_refresh_args(
    argv: Sequence[str] | None = None,
    *,
    load_cli_config: Callable[[str | None], dict[str, Any]],
    parse_targets: Callable[[str], Any],
    default_hours: int,
    default_targets: Sequence[float],
    default_p_up_min: float,
    default_ret_min: float,
    confidence_min_default: float,
    position_size_floor_default: float,
    position_size_cap_default: float,
    default_dir_model_weights_1h: str,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML/JSON file that overrides the CLI defaults (hours, targets, thresholds, etc.)."
            " CLI flags still take precedence over config entries."
        ),
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_cli_config(config_args.config)
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Binance US spot data, rebuild local features/datasets, and emit multi-horizon predictions."
        ),
        parents=[config_parser],
    )
    parser.add_argument("--hours", type=int, default=default_hours, help="Number of hourly candles to fetch from Binance US (default: 360).")
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=list(default_targets),
        help="Comma-separated prediction horizons in hours (default: 0.25,1,4,8,12).",
    )
    parser.add_argument("--p-up-min", type=float, default=default_p_up_min, help="Probability threshold for ensemble activation (default: 0.45).")
    parser.add_argument("--ret-min", type=float, default=default_ret_min, help="Return threshold for ensemble activation (default: 0.0).")
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability cutoff used for the direction-only signal. "
            "Values above 0.5 make the model more sensitive to downtrends. "
            "Default 0.5 produces the original behaviour."
        ),
    )
    parser.add_argument(
        "--auto-direction-threshold",
        action="store_true",
        help=(
            "Enable automatic computation of the direction-only threshold based "
            "on calibrated p_up_min values from thresholds_json. "
            "If set, --direction-threshold is ignored."
        ),
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=str(Path("artifacts/models/calibrated_thresholds_merged.json")),
        help="Optional JSON file containing per-horizon thresholds; set to an empty string to disable.",
    )
    parser.add_argument(
        "--platt-calibration",
        type=str,
        default=str(Path("artifacts/models/platt_calibration.json")),
        help="Optional JSON file containing Platt scaling coefficients per horizon.",
    )
    parser.add_argument("--data-quality-enabled", action="store_true", help="Enable hard OHLCV data-quality checks after ingestion/local feature loading.")
    parser.add_argument("--max-staleness-hours", type=float, default=2.0, help="Maximum allowed OHLCV staleness in hours when data quality checks are enabled.")
    parser.add_argument("--max-missing-ratio", type=float, default=0.01, help="Maximum allowed ratio of missing hourly timestamps when quality checks are enabled.")
    parser.add_argument("--max-zero-volume-ratio", type=float, default=0.2, help="Maximum allowed ratio of zero-volume rows when quality checks are enabled.")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum required OHLCV rows when quality checks are enabled.")
    parser.add_argument("--confidence-min", type=float, default=confidence_min_default, help="Minimum confidence score required to keep a non-hold trade action (default: 0.0).")
    parser.add_argument("--position-size-floor", type=float, default=position_size_floor_default, help="Floor for confidence-scaled position size (default: 0.0).")
    parser.add_argument("--position-size-cap", type=float, default=position_size_cap_default, help="Cap for confidence-scaled position size (default: 1.0).")
    parser.add_argument("--dry-run", action="store_true", help="Skip network-dependent steps and reuse cached datasets/models for smoke testing.")
    parser.add_argument(
        "--replay-offset-bars",
        type=int,
        default=0,
        help=(
            "Replay cached hourly predictions from N bars back using the prepared dataset instead of the latest bar. "
            "Intended for hourly horizons such as 1,4,8,12."
        ),
    )
    parser.add_argument("--spot-provider", choices=("binanceus",), default="binanceus", help="Spot ingestion provider for hourly candles (Binance-only; default: binanceus).")
    parser.add_argument(
        "--use-local-features",
        action="store_true",
        help=(
            "Skip ingestion + feature rebuilds and load hourly features directly from the supplied parquet paths."
            " Requires --features-path."
        ),
    )
    parser.add_argument("--features-path", type=str, default=None, help="Path to the merged hourly features parquet/CSV used when --use-local-features is set.")
    parser.add_argument("--macro-path", type=str, default=None, help="Optional macro parquet/CSV merged into local features when --use-local-features is enabled.")
    parser.add_argument("--onchain-path", type=str, default=None, help="Optional on-chain parquet/CSV used only for metadata when --use-local-features is enabled.")
    parser.add_argument("--funding-path", type=str, default=None, help="Optional funding parquet/CSV used only for metadata when --use-local-features is enabled.")
    parser.add_argument("--intrabar-path", type=str, default=None, help="Optional intrabar (15m->1h aggregated) parquet/CSV merged when --use-local-features is enabled.")
    parser.add_argument("--intrabar-enabled", action="store_true", help="Fetch 15m Binance candles and aggregate intrabar features into the live inference bundle.")
    parser.add_argument("--intrabar-interval", type=str, default="15m", help="Binance interval used for intrabar aggregation (default: 15m).")
    parser.add_argument("--intrabar-hours-multiplier", type=int, default=4, help="Multiplier for intrabar fetch size relative to --hours (default: 4 for 15m).")
    parser.add_argument("--intrabar-max-rows", type=int, default=4000, help="Upper bound on intrabar rows fetched from Binance for aggregation.")
    parser.add_argument("--trade-decision-enabled", action="store_true", help="Enable trade decision model override for final trade/no-trade action.")
    parser.add_argument("--trade-decision-disabled", action="store_true", help="Disable trade decision model policy even if enabled in config.")
    parser.add_argument("--trade-decision-model", type=str, default=None, help="Path to JSON trade decision model artifact.")
    parser.add_argument("--trade-decision-threshold", type=float, default=None, help="Optional probability threshold for trade decision model.")
    parser.add_argument("--write-artifacts", action="store_true", help="Update monitoring artifacts (trade_ready_summary + meta baseline) after predictions complete.")
    parser.add_argument("--disable-monitoring-latest", action="store_true", help="Skip writing artifacts/monitoring/latest.json snapshot (default: enabled).")
    parser.add_argument("--dir-lstm-path", type=str, default=None, help="Optional directory containing the LSTM direction model ensemble.")
    parser.add_argument("--dir-bilstm-path", type=str, default=None, help="Optional directory containing the BiLSTM direction model ensemble.")
    parser.add_argument("--dir-gru-path", type=str, default=None, help="Optional directory containing the GRU direction model ensemble.")
    parser.add_argument("--dir-cnn-lstm-path", type=str, default=None, help="Optional directory containing the CNN-LSTM direction model ensemble.")
    parser.add_argument("--dir-cnn-bilstm-path", type=str, default=None, help="Optional directory containing the CNN-BiLSTM direction model ensemble.")
    parser.add_argument("--dir-garch-lstm-path", type=str, default=None, help="Optional directory containing the GARCH-LSTM direction model ensemble.")
    parser.add_argument("--dir-transformer-path", type=str, default=None, help="Optional directory containing the transformer direction model ensemble.")
    parser.add_argument(
        "--dir-model-config-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file describing direction-model entries (list of {type,path,weight}); "
            "overrides the built-in DEFAULT_DIR_MODELS_1H registry."
        ),
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=default_dir_model_weights_1h,
        help=(
            "Legacy comma-separated weights for direction models (e.g. transformer:2,lstm:1,xgb:1). "
            "Applied on top of the resolved structured config."
        ),
    )
    if config_defaults:
        config_defaults = {key: value for key, value in config_defaults.items() if key != "config"}
        if config_defaults:
            parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)