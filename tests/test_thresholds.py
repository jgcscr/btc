from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.trading.thresholds import load_calibrated_thresholds
from src.scripts.backtest_signals_4h import _resolve_thresholds, DEFAULT_P_UP_MIN_4H, DEFAULT_RET_MIN_4H
from src.scripts.backtest_signals_1h4h_confirm import _resolve_confirmation_threshold, DEFAULT_P_UP_MIN_4H as DEFAULT_CONFIRM_P_UP
from src.scripts.run_refresh_and_predict import (
    _build_trade_ready_monitoring_payload,
    _resolve_thresholds_for_horizon,
)


def test_load_calibrated_thresholds_basic(tmp_path: Path) -> None:
    payload = {
        "horizons": {
            "4": {
                "p_up_min": 0.3,
                "ret_min": 0.0005,
                "max_drawdown": -0.08,
                "volatility_ceiling": 0.02,
                "volatility_mult": 1.4,
                "volatility_metric": "volatility_realized_24h",
            },
            "bad": {"p_up_min": 0.9},
        }
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload))

    loaded = load_calibrated_thresholds(path)

    assert loaded == {
        4: {
            "p_up_min": 0.3,
            "ret_min": 0.0005,
            "max_drawdown": -0.08,
            "volatility_ceiling": 0.02,
            "volatility_mult": 1.4,
            "volatility_metric": "volatility_realized_24h",
        }
    }


def test_load_calibrated_thresholds_fractional_keys(tmp_path: Path) -> None:
    payload = {
        "horizons": {
            "0.25": {
                "p_up_min": 0.52,
                "ret_min": 0.0002,
            }
        }
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload))

    loaded = load_calibrated_thresholds(path)

    assert 0.25 in loaded
    assert loaded[0.25]["p_up_min"] == 0.52


def test_load_calibrated_thresholds_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_calibrated_thresholds(missing) == {}


@pytest.mark.parametrize(
    "cli_values,expected",
    [
        ((None, None, {"p_up_min": 0.3, "ret_min": 0.0005}), (0.3, 0.0005)),
        ((0.6, None, {"p_up_min": 0.3, "ret_min": 0.0005}), (0.6, 0.0005)),
        ((None, 0.001, {"p_up_min": 0.3, "ret_min": 0.0005}), (0.3, 0.001)),
        ((0.6, 0.001, {"p_up_min": 0.3, "ret_min": 0.0005}), (0.6, 0.001)),
        ((None, None, None), (DEFAULT_P_UP_MIN_4H, DEFAULT_RET_MIN_4H)),
    ],
)
def test_resolve_thresholds(cli_values, expected, tmp_path: Path) -> None:
    p_arg, ret_arg, thresholds = cli_values
    path = None
    if thresholds is not None:
        payload = {"horizons": {"4": thresholds}}
        file_path = tmp_path / "thresholds.json"
        file_path.write_text(json.dumps(payload))
        path = file_path

    resolved = _resolve_thresholds(p_arg, ret_arg, path)
    assert resolved == expected


@pytest.mark.parametrize(
    "cli_value,threshold_entry,expected",
    [
        (None, {"p_up_min": 0.31, "ret_min": 0.0005}, 0.31),
        (0.6, {"p_up_min": 0.31, "ret_min": 0.0005}, 0.6),
        (None, None, DEFAULT_CONFIRM_P_UP),
    ],
)
def test_resolve_confirmation_threshold(cli_value, threshold_entry, expected, tmp_path: Path) -> None:
    path = None
    if threshold_entry is not None:
        payload = {"horizons": {"4": threshold_entry}}
        file_path = tmp_path / "thresholds.json"
        file_path.write_text(json.dumps(payload))
        path = file_path

    resolved = _resolve_confirmation_threshold(cli_value, path)
    assert resolved == expected


def test_resolve_thresholds_for_horizon_handles_missing() -> None:
    overrides = {4: {"p_up_min": 0.6, "ret_min": 0.0004}}
    resolved = _resolve_thresholds_for_horizon(8, 0.45, 0.0, overrides)
    assert resolved == {"p_up_min": 0.45, "ret_min": 0.0}


def test_resolve_thresholds_for_horizon_includes_max_drawdown() -> None:
    overrides = {4: {"p_up_min": 0.6, "ret_min": 0.0004, "max_drawdown": -0.09}}
    resolved = _resolve_thresholds_for_horizon(4, 0.45, 0.0, overrides)
    assert resolved == {"p_up_min": 0.6, "ret_min": 0.0004, "max_drawdown": -0.09}


def test_resolve_thresholds_for_horizon_includes_volatility_fields() -> None:
    overrides = {
        1: {
            "p_up_min": 0.5,
            "ret_min": 0.0,
            "volatility_ceiling": 0.01,
            "volatility_mult": 2.0,
            "volatility_metric": "volatility_realized_72h",
        }
    }
    resolved = _resolve_thresholds_for_horizon(1, 0.45, 0.0, overrides)
    assert resolved["volatility_ceiling"] == 0.01
    assert resolved["volatility_mult"] == 2.0
    assert resolved["volatility_metric"] == "volatility_realized_72h"


def test_trade_ready_payload_carries_thresholds_snapshot() -> None:
    predictions_payload = {
        "generated_at": "2025-12-30T00:00:00Z",
        "predictions": {
            "1h": {
                "horizon_hours": 1,
                "thresholds": {"p_up_min": 0.6, "ret_min": 0.0005, "max_drawdown": -0.08},
                "volatility": {
                    "snapshot": {"volatility_realized_24h": 0.03},
                    "ceiling": 0.05,
                    "triggered": False,
                },
            },
            "4h": {
                "horizon_hours": 4,
                "thresholds": {"p_up_min": 0.58, "ret_min": 0.0007},
                "volatility": {
                    "snapshot": {"volatility_realized_24h": 0.04},
                    "ceiling": 0.06,
                    "triggered": True,
                },
            },
        },
    }
    args = SimpleNamespace(
        targets=[1, 4],
        spot_provider="binanceus",
        macro_source="vendor",
        onchain_source="cryptocompare",
        funding_provider="binance",
        hours=360,
        dry_run=False,
    )

    payload = _build_trade_ready_monitoring_payload(predictions_payload, args)

    assert payload["generated_at"] == "2025-12-30T00:00:00Z"
    horizons = payload["horizons"]
    assert len(horizons) == 2
    assert horizons[0]["thresholds"]["p_up_min"] == 0.6
    assert horizons[0]["thresholds"]["max_drawdown"] == -0.08
    assert horizons[0]["volatility"]["ceiling"] == 0.05
    assert horizons[1]["thresholds"]["ret_min"] == 0.0007
    assert horizons[1]["volatility"]["triggered"] is True
