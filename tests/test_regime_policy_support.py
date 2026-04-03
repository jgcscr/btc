from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.runtime.regime_policy_support import (
    apply_adaptive_thresholds,
    classify_regime_from_score,
    compute_breakout_scores,
    compute_profile_breakout_score,
    derive_regime_labels_from_frame,
    inactive_direction_fallback,
    load_last_trigger_ts,
    resolve_adaptive_thresholds_policy,
    resolve_direction_fallback_policy,
    resolve_trend_ignition_payload,
    write_last_trigger_ts,
)


def test_load_and_write_last_trigger_ts_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    assert load_last_trigger_ts(state_path) is None
    write_last_trigger_ts(state_path, "2026-04-02T00:00:00Z")

    assert load_last_trigger_ts(state_path) == "2026-04-02T00:00:00Z"


def test_resolve_direction_fallback_policy_preserves_disabled_shape() -> None:
    policy = resolve_direction_fallback_policy(
        {
            "enabled": False,
            "prob_threshold": 0.55,
            "size_factor": 0.25,
        },
        load_state=lambda: "2026-04-02T00:00:00Z",
    )

    assert policy is not None
    assert policy["enabled"] is False
    assert policy["prob_threshold"] == 0.55
    assert policy["last_trigger_ts"] == "2026-04-02T00:00:00Z"


def test_resolve_trend_ignition_payload_handles_missing_model() -> None:
    messages: list[str] = []

    payload = resolve_trend_ignition_payload(
        {"enabled": True, "model_path": "missing.joblib"},
        load_trend_ignition_classifier=lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
        load_state=lambda: None,
        stderr_write=messages.append,
    )

    assert payload is None
    assert messages and "trend ignition support disabled" in messages[0]


def test_apply_adaptive_thresholds_clamps_scaled_values() -> None:
    scaled_p, scaled_ret, scale = apply_adaptive_thresholds(
        {
            "enabled": True,
            "breakout_scale": 0.8,
            "p_up_min_floor": 0.5,
            "ret_min_floor": 0.001,
        },
        base_p_up=0.7,
        base_ret=0.0005,
        regime_state="trend_ignition",
        regime_trend="trend_ignition",
        regime_chop="chop",
    )

    assert scale == 0.8
    assert scaled_p == pytest.approx(0.56)
    assert scaled_ret == 0.001


def test_breakout_scoring_and_regime_labeling_are_consistent() -> None:
    prepared = SimpleNamespace(df_all=pd.DataFrame({"close": [100.0, 101.0]}))
    score = compute_profile_breakout_score(
        prepared,
        1,
        {"volatility_realized_24h": 0.05},
        breakout_vol_normalizer=0.05,
        breakout_ret_normalizer=0.002,
    )

    assert score > 1.0
    assert classify_regime_from_score(
        score,
        {"breakout_score_threshold": 0.8, "chop_score_threshold": 0.3},
        regime_trend="trend_ignition",
        regime_neutral="neutral",
        regime_chop="chop",
    ) == "trend_ignition"

    labels = derive_regime_labels_from_frame(
        pd.DataFrame({"close": [100.0, 101.0, 101.01], "volatility_realized_24h": [0.05, 0.05, 0.0]}),
        volatility_col="volatility_realized_24h",
        breakout_score_threshold=0.8,
        chop_score_threshold=0.05,
        breakout_vol_normalizer=0.05,
        breakout_ret_normalizer=0.002,
        regime_trend="trend_ignition",
        regime_neutral="neutral",
        regime_chop="chop",
    )

    assert labels.iloc[0] == "neutral"
    assert labels.iloc[1] == "trend_ignition"
    assert labels.iloc[-1] in {"neutral", "chop"}


def test_compute_breakout_scores_maps_each_bundle() -> None:
    prepared_a = SimpleNamespace(df_all=pd.DataFrame({"close": [100.0, 101.0]}))
    prepared_b = SimpleNamespace(df_all=pd.DataFrame({"close": [100.0, 99.5]}))

    scores = compute_breakout_scores(
        {
            "1h": (prepared_a, 1, 101.0, "ts-a"),
            "4h": (prepared_b, 1, 99.5, "ts-b"),
        },
        {
            "1h": {"volatility_realized_24h": 0.05},
            "4h": {"volatility_realized_24h": 0.01},
        },
        compute_profile_breakout_score=lambda prepared, index, snapshot: round(float(prepared.df_all["close"].iloc[index]) + float(snapshot.get("volatility_realized_24h", 0.0)), 6),
    )

    assert scores == {"1h": 101.05, "4h": 99.51}


def test_inactive_direction_fallback_preserves_reason_and_flags() -> None:
    payload = inactive_direction_fallback(
        "cooldown_active",
        side="long",
        cooldown_active=True,
        size_factor=0.4,
    )

    assert payload == {
        "active": False,
        "side": "long",
        "size_factor": 0.4,
        "stop_loss_fallback": None,
        "take_profit_fallback": None,
        "reason": "cooldown_active",
        "cooldown_active": True,
    }