from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trading.thresholds import load_calibrated_thresholds
from src.scripts.backtest_signals_4h import _resolve_thresholds, DEFAULT_P_UP_MIN_4H, DEFAULT_RET_MIN_4H
from src.scripts.backtest_signals_1h4h_confirm import _resolve_confirmation_threshold, DEFAULT_P_UP_MIN_4H as DEFAULT_CONFIRM_P_UP


def test_load_calibrated_thresholds_basic(tmp_path: Path) -> None:
    payload = {
        "horizons": {
            "4": {"p_up_min": 0.3, "ret_min": 0.0005},
            "bad": {"p_up_min": 0.9},
        }
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload))

    loaded = load_calibrated_thresholds(path)

    assert loaded == {4: {"p_up_min": 0.3, "ret_min": 0.0005}}


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
