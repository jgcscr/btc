from __future__ import annotations

import json

from src.runtime.trust_hardening_support import apply_trust_hardening, resolve_trust_hardening_policy


def _coerce_numeric_horizon(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value):
    return float(value)


def _coerce_result_horizon(value):
    return None if value is None else float(value)


def _direction_vote(entry):
    return str(entry.get("direction_next") or "neutral").lower()


def _direction_from_probability(value, *, neutral_band=0.0):
    numeric = float(value)
    if numeric >= 0.5 + neutral_band:
        return "up"
    if numeric <= 0.5 - neutral_band:
        return "down"
    return "neutral"


def _finite_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def test_suspicious_horizon_excluded_from_voting(tmp_path) -> None:
    summary_path = tmp_path / "summary_4h.json"
    summary_path.write_text(
        json.dumps(
            {
                "feature_names": ["open", "ret_4h"],
                "metrics": {
                    "train": {"accuracy": 0.9999},
                    "val": {"accuracy": 0.9999},
                    "test": {"accuracy": 0.9999},
                },
            }
        ),
        encoding="utf-8",
    )
    policy = resolve_trust_hardening_policy(
        {
            "enabled": True,
            "horizons": [4],
            "action_by_horizon": {"4": "exclude"},
            "model_summary_paths_by_horizon": {"4": str(summary_path)},
        },
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    summary = {
        "4h": {
            "horizon_hours": 4.0,
            "direction_next": "up",
            "p_up": 0.7,
            "raw_p_up": 0.7,
            "probability_calibration": {"raw_probability": 0.7, "resolved_probability": 0.7},
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next": "down",
            "p_up": 0.4,
            "raw_p_up": 0.4,
        },
    }

    updated = apply_trust_hardening(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )

    assert updated["4h"]["trust_status"] == "low_trust"
    assert updated["4h"]["excluded_from_voting"] is True
    assert updated["4h"]["voting_weight_after_trust"] == 0.0


def test_calibrated_flip_marks_high_impact_horizon_low_trust() -> None:
    policy = resolve_trust_hardening_policy(
        {
            "enabled": True,
            "horizons": [8],
            "high_impact_horizons": [8],
            "metadata_checks": {"enabled": False},
            "divergence_abs_gap_min": 0.1,
            "divergence_flip_required": True,
            "default_action": "deweight",
            "deweight_factor": 0.25,
        },
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    summary = {
        "8h": {
            "horizon_hours": 8.0,
            "direction_next": "up",
            "p_up": 0.67,
            "raw_p_up": 0.47,
            "probability_calibration": {"raw_probability": 0.47, "resolved_probability": 0.67},
        }
    }

    updated = apply_trust_hardening(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )

    assert updated["8h"]["trust_status"] == "low_trust"
    assert "calibration_flip_divergence" in updated["8h"]["trust_reasons"]
    assert updated["8h"]["excluded_from_voting"] is False
    assert updated["8h"]["voting_weight_after_trust"] == 0.25


def test_fail_closed_vs_fail_open_missing_metadata() -> None:
    base_summary = {
        "4h": {
            "horizon_hours": 4.0,
            "direction_next": "up",
            "p_up": 0.7,
            "raw_p_up": 0.7,
        }
    }
    fail_open_policy = resolve_trust_hardening_policy(
        {
            "enabled": True,
            "horizons": [4],
            "fail_closed": False,
            "model_summary_paths_by_horizon": {"4": "nonexistent.json"},
        },
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    fail_closed_policy = resolve_trust_hardening_policy(
        {
            "enabled": True,
            "horizons": [4],
            "fail_closed": True,
            "model_summary_paths_by_horizon": {"4": "nonexistent.json"},
        },
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )

    open_updated = apply_trust_hardening(
        {"4h": dict(base_summary["4h"])},
        fail_open_policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )
    closed_updated = apply_trust_hardening(
        {"4h": dict(base_summary["4h"])},
        fail_closed_policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )

    assert open_updated["4h"]["trust_status"] == "trusted"
    assert closed_updated["4h"]["trust_status"] == "low_trust"
    assert "missing_required_trust_metadata" in closed_updated["4h"]["trust_reasons"]


def test_trust_hardening_outcome_change_telemetry() -> None:
    policy = resolve_trust_hardening_policy(
        {
            "enabled": True,
            "horizons": [4],
            "metadata_checks": {"enabled": False},
            "default_action": "exclude",
            "high_impact_horizons": [4],
            "divergence_abs_gap_min": 0.1,
            "divergence_flip_required": True,
        },
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    summary = {
        "4h": {
            "horizon_hours": 4.0,
            "direction_next": "up",
            "p_up": 0.7,
            "raw_p_up": 0.4,
            "probability_calibration": {"raw_probability": 0.4, "resolved_probability": 0.7},
        },
        "12h": {
            "horizon_hours": 12.0,
            "direction_next": "down",
            "p_up": 0.4,
            "raw_p_up": 0.4,
        },
    }

    updated = apply_trust_hardening(
        summary,
        policy,
        coerce_result_horizon=_coerce_result_horizon,
        direction_vote=_direction_vote,
        direction_from_probability=_direction_from_probability,
        finite_float_or_none=_finite_float_or_none,
    )

    assert updated["4h"]["trust_hardening_changed_outcome"] is True
    assert updated["12h"]["trust_hardening_changed_outcome"] is True
