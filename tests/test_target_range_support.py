from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.runtime.target_range_support import (
    apply_target_range_overrides,
    evaluate_direction_only_fallback,
    load_target_range_model,
    load_target_range_models,
    predict_target_range_prices,
    resolve_target_range_policy,
    target_range_label,
)


class _FakeModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, vector):
        return [self.value]


def _inactive_direction_fallback(reason: str, **kwargs):
    return {
        "active": False,
        "reason": reason,
        "side": kwargs.get("side"),
        "size_factor": kwargs.get("size_factor", 0.0),
        "cooldown_active": kwargs.get("cooldown_active", False),
    }


def _parse_iso_timestamp(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_predict_target_range_prices_projects_levels_and_confidence() -> None:
    payload = predict_target_range_prices(
        {
            "high": {
                "model": _FakeModel(0.05),
                "feature_names": ["feature_a"],
                "metrics": {"val_rmse": 0.1, "val_residual_std": 0.2},
            },
            "low": {
                "model": _FakeModel(-0.03),
                "feature_names": ["feature_a"],
                "metrics": {"val_rmse": 0.2, "val_residual_std": 0.4},
            },
        },
        pd.Series({"feature_a": 1.0}),
        close=100.0,
        confidence_scale=0.1,
        finite_float_or_none=lambda value: None if value is None else float(value),
    )

    assert payload["projected_high"] == pytest.approx(100.0 * 1.051271096)
    assert payload["projected_low"] == pytest.approx(100.0 * 0.970445534)
    assert payload["projected_high_confidence"] > payload["projected_low_confidence"]
    assert payload["projected_high_residual_std"] == 0.2
    assert payload["projected_low_residual_std"] == 0.4


def test_target_range_label_formats_integer_and_fractional_horizons() -> None:
    assert target_range_label(4.0) == "4h"
    assert target_range_label(0.25) == "0.25h"


def test_resolve_target_range_policy_uses_defaults_and_filters_horizons(tmp_path) -> None:
    policy = resolve_target_range_policy(
        {"enabled": True, "model_dir": str(tmp_path / "models"), "horizons": [4, -1, "8"]},
        target_range_model_dir=tmp_path / "fallback-models",
        default_override_ratio=0.05,
        default_confidence_scale=0.3,
        default_horizons=(1.0, 4.0, 8.0),
    )

    assert policy is not None
    assert policy["enabled"] is True
    assert policy["model_dir"] == (tmp_path / "models")
    assert policy["override_ratio"] == 0.05
    assert policy["confidence_rmse_scale"] == 0.3
    assert policy["horizons"] == [4.0, 8.0]


def test_load_target_range_model_normalizes_feature_names_and_metrics(tmp_path: Path) -> None:
    payload_path = tmp_path / "4h_high.joblib"
    import joblib

    joblib.dump(
        {
            "model": _FakeModel(0.01),
            "feature_names": ["feature_a", 2],
            "metrics": {"val_rmse": 0.2, "ignored": "bad"},
        },
        payload_path,
    )
    messages: list[str] = []

    payload = load_target_range_model(payload_path, stderr_write=messages.append)

    assert messages == []
    assert payload is not None
    assert payload["feature_names"] == ["feature_a", "2"]
    assert payload["metrics"] == {"val_rmse": 0.2}


def test_load_target_range_models_skips_missing_pairs(tmp_path: Path) -> None:
    messages: list[str] = []

    bundles = load_target_range_models(
        {"enabled": True, "model_dir": tmp_path, "horizons": [4.0, 8.0]},
        [4.0, 8.0],
        target_range_model_dir=tmp_path,
        load_target_range_model_fn=lambda path: {"model": object()} if path.name == "4h_high.joblib" else None,
        stderr_write=messages.append,
    )

    assert bundles == {}
    assert messages == [
        "Warning: skipping target-range models for 4h horizon (missing 4h_low.joblib).\n",
        "Warning: skipping target-range models for 8h horizon (missing 8h_high.joblib, 8h_low.joblib).\n",
    ]


def test_apply_target_range_overrides_updates_long_take_profit_and_stop() -> None:
    overrides, updated_stop, updated_take = apply_target_range_overrides(
        98.0,
        103.0,
        {"projected_high": 105.0, "projected_low": 96.0},
        override_ratio=0.01,
        direction=1,
    )

    assert updated_take == 105.0
    assert updated_stop == 96.0
    assert overrides["take_profit"]["reason"] == "target_range_high"
    assert overrides["stop_loss"]["reason"] == "target_range_low"


def test_evaluate_direction_only_fallback_triggers_and_updates_cooldown_state() -> None:
    policy = {
        "enabled": True,
        "prob_threshold": 0.6,
        "max_negative_ev": 0.0005,
        "size_factor": 0.5,
        "stop_take_ratio": 0.01,
        "cooldown_hours": 4.0,
    }

    payload, triggered = evaluate_direction_only_fallback(
        policy,
        p_up=0.68,
        signal_dir_only=1,
        expected_value=-0.0004,
        projected_price=100.0,
        signal_ts="2026-04-02T00:00:00Z",
        trend_prob=0.3,
        trend_threshold=0.6,
        inactive_direction_fallback=_inactive_direction_fallback,
        parse_iso_timestamp=_parse_iso_timestamp,
    )

    assert triggered is True
    assert payload["active"] is True
    assert payload["side"] == "long"
    assert payload["stop_loss_fallback"] == 99.0
    assert payload["take_profit_fallback"] == 101.0
    assert policy["last_trigger_ts"] == "2026-04-02T00:00:00Z"
