from __future__ import annotations

from src.scripts.run_live_wrapper_smoke_check import _validate_trust_behavior


def _base_prediction_entry() -> dict:
    return {
        "trust_status": "trusted",
        "trust_reasons": [],
        "excluded_from_voting": False,
        "voting_weight_after_trust": 1.0,
        "trust_hardening_changed_outcome": False,
        "trust_hardening_action": "none",
    }


def test_validate_trust_behavior_accepts_expected_low_trust_mapping() -> None:
    payload = {
        "predictions": {
            "1h": _base_prediction_entry(),
            "12h": _base_prediction_entry(),
            "4h": {
                **_base_prediction_entry(),
                "trust_status": "low_trust",
                "trust_reasons": ["metadata_implausible_val_accuracy"],
                "excluded_from_voting": False,
                "voting_weight_after_trust": 0.5,
                "trust_hardening_action": "deweight",
            },
        }
    }

    assert _validate_trust_behavior(payload) == []


def test_validate_trust_behavior_accepts_remediated_trusted_4h_mapping() -> None:
    payload = {
        "predictions": {
            "1h": _base_prediction_entry(),
            "12h": _base_prediction_entry(),
            "4h": _base_prediction_entry(),
        }
    }

    assert _validate_trust_behavior(payload) == []


def test_validate_trust_behavior_flags_wrong_4h_and_8h_actions() -> None:
    payload = {
        "predictions": {
            "4h": {
                **_base_prediction_entry(),
                "trust_status": "low_trust",
                "trust_reasons": [],
                "excluded_from_voting": True,
                "voting_weight_after_trust": 0.0,
                "trust_hardening_action": "exclude",
            },
            "8h": {
                **_base_prediction_entry(),
                "trust_status": "trusted",
                "trust_reasons": [],
                "excluded_from_voting": False,
                "voting_weight_after_trust": 1.0,
                "trust_hardening_action": "none",
            },
        }
    }

    errors = _validate_trust_behavior(payload)
    assert any("4h low_trust horizon expected trust_hardening_action=deweight" in err for err in errors)
    assert any("8h horizon should not be emitted" in err for err in errors)


def test_validate_trust_behavior_flags_missing_required_metadata_reason() -> None:
    payload = {
        "predictions": {
            "4h": {
                **_base_prediction_entry(),
                "trust_status": "low_trust",
                "trust_reasons": ["missing_required_trust_metadata"],
                "excluded_from_voting": False,
                "voting_weight_after_trust": 0.5,
                "trust_hardening_action": "deweight",
            },
        }
    }

    errors = _validate_trust_behavior(payload)
    assert any("missing_required_trust_metadata" in err for err in errors)
