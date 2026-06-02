from __future__ import annotations

from src.scripts.run_downtrend_bias_remediation import (
    _build_calibration_gate_summary,
    _candidate_lookback_hours,
    _filter_calibration_candidate_by_horizon,
    _merge_calibration_candidates,
    _resolve_effective_lookback_hours,
)


def test_build_calibration_gate_summary_marks_sparse_horizons_ineligible() -> None:
    payload = _build_calibration_gate_summary(
        {
            "rows_by_horizon": {
                "4h": 3,
                "8h": 12,
                "12h": 9,
            }
        },
        recalibration_horizons=["4h", "8h", "12h"],
        min_rows_per_horizon=25,
    )

    assert payload["eligible"] is False
    assert payload["reason"] == "insufficient_rows_per_horizon"
    assert payload["insufficient_horizons"] == {"4h": 3, "8h": 12, "12h": 9}


def test_build_calibration_gate_summary_accepts_ready_horizons() -> None:
    payload = _build_calibration_gate_summary(
        {
            "rows_by_horizon": {
                "4h": 32,
                "8h": 28,
                "12h": 30,
            }
        },
        recalibration_horizons=["4h", "8h", "12h"],
        min_rows_per_horizon=25,
    )

    assert payload["eligible"] is True
    assert payload["reason"] == "ready"
    assert payload["insufficient_horizons"] == {}


def test_candidate_lookback_hours_includes_requested_and_cap() -> None:
    assert _candidate_lookback_hours(1080, max_lookback_hours=1800, step_hours=240) == [1080, 1320, 1560, 1800]


def test_resolve_effective_lookback_hours_selects_first_eligible_attempt() -> None:
    rows_by_attempt = {
        1080: {"4h": 3, "8h": 3, "12h": 3},
        1440: {"4h": 18, "8h": 17, "12h": 16},
        1800: {"4h": 27, "8h": 29, "12h": 26},
    }

    payload = _resolve_effective_lookback_hours(
        initial_lookback_hours=1080,
        max_lookback_hours=1800,
        step_hours=360,
        evaluate_lookback=lambda lookback_hours: {
            "lookback_hours": lookback_hours,
            "labeled_meta": {"rows_by_horizon": rows_by_attempt[lookback_hours]},
            "calibration_gate": _build_calibration_gate_summary(
                {"rows_by_horizon": rows_by_attempt[lookback_hours]},
                recalibration_horizons=["4h", "8h", "12h"],
                min_rows_per_horizon=25,
            ),
        },
    )

    assert payload["effective_lookback_hours"] == 1800
    assert payload["auto_expanded"] is True
    assert [attempt["lookback_hours"] for attempt in payload["attempts"]] == [1080, 1440, 1800]
    assert payload["calibration_gate"]["eligible"] is True


def test_resolve_effective_lookback_hours_returns_last_attempt_when_none_eligible() -> None:
    rows_by_attempt = {
        1080: {"4h": 3, "8h": 3, "12h": 3},
        1440: {"4h": 10, "8h": 11, "12h": 12},
    }

    payload = _resolve_effective_lookback_hours(
        initial_lookback_hours=1080,
        max_lookback_hours=1440,
        step_hours=360,
        evaluate_lookback=lambda lookback_hours: {
            "lookback_hours": lookback_hours,
            "labeled_meta": {"rows_by_horizon": rows_by_attempt[lookback_hours]},
            "calibration_gate": _build_calibration_gate_summary(
                {"rows_by_horizon": rows_by_attempt[lookback_hours]},
                recalibration_horizons=["4h", "8h", "12h"],
                min_rows_per_horizon=25,
            ),
        },
    )

    assert payload["effective_lookback_hours"] == 1440
    assert payload["auto_expanded"] is True
    assert payload["calibration_gate"]["eligible"] is False
    assert payload["calibration_gate"]["insufficient_horizons"] == {"4h": 10, "8h": 11, "12h": 12}


def test_filter_calibration_candidate_by_horizon_removes_unallowed_keys() -> None:
    filtered, summary = _filter_calibration_candidate_by_horizon(
        {
            "12h@chop": {"method": "isotonic"},
            "8h@neutral": {"method": "isotonic"},
            "4h": {"method": "isotonic"},
        },
        allowed_horizons=["4h", "8h"],
    )

    assert filtered == {
        "8h@neutral": {"method": "isotonic"},
        "4h": {"method": "isotonic"},
    }
    assert summary["removed_keys"] == ["12h@chop"]
    assert summary["retained_keys"] == ["4h", "8h@neutral"]


def test_merge_calibration_candidates_overlays_filtered_keys_onto_base_payload() -> None:
    merged = _merge_calibration_candidates(
        {
            "4h@neutral": {"method": "isotonic", "x": [0.1, 0.9], "y": [0.2, 0.8]},
            "8h@neutral": {"method": "isotonic", "x": [0.1, 0.9], "y": [0.3, 0.7]},
            "12h@neutral": {"method": "isotonic", "x": [0.1, 0.9], "y": [0.4, 0.6]},
        },
        {
            "8h@neutral": {"method": "isotonic", "x": [0.2, 0.8], "y": [0.1, 0.6]},
        },
    )

    assert merged == {
        "4h@neutral": {"method": "isotonic", "x": [0.1, 0.9], "y": [0.2, 0.8]},
        "8h@neutral": {"method": "isotonic", "x": [0.2, 0.8], "y": [0.1, 0.6]},
        "12h@neutral": {"method": "isotonic", "x": [0.1, 0.9], "y": [0.4, 0.6]},
    }