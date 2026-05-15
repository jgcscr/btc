from __future__ import annotations

from src.runtime.prediction_result_support import build_prediction_result


def test_build_prediction_result_surfaces_direction_ensemble_metadata() -> None:
	result, _ = build_prediction_result(
		signal={
			"p_up_components": {"xgb": 0.6},
			"direction_ensemble": {"selected_models": ["xgb", "transformer"], "rejected_models": ["gru"]},
		},
		label="1h",
		horizon=1.0,
		signal_ts="2026-05-15T00:00:00Z",
		close=100.0,
		p_up=0.6,
		raw_p_up=0.6,
		ret_pred=0.01,
		trend_prob=0.0,
		ignition_state=0,
		cooldown_active=False,
		signal_ensemble=1,
		signal_dir_only=1,
		confidence_score=0.5,
		position_size=0.5,
		confidence_min=0.0,
		confidence_min_source="default",
		position_size_cap=1.0,
		stop_loss_price=99.0,
		take_profit_price=102.0,
		expected_value=0.01,
		horizon_thresholds={"p_up_min": 0.5, "ret_min": 0.0},
		regime_state="neutral",
		calibration_key=None,
		calibration_used_regime_key=False,
		probability_guard=None,
		regime_weight_policy=None,
		target_projection=None,
		volatility_payload={},
		volatility_flag=False,
		forecast_coherence_policy={},
		direction_output_policy={},
		direction_output_scoped=False,
		trade_decision_policy={},
		abstention_policy={},
		direction_fallback_policy=None,
		trend_payload=None,
		target_range_policy=None,
		regime_score=None,
		adaptive_scale=1.0,
		horizon_p_up=0.5,
		horizon_ret=0.0,
		row_features={},
		optional_feature_fields=[],
		project_price=lambda close, ret: close * (1.0 + ret),
		get_active_regime_weight_override=lambda *_args, **_kwargs: None,
		derive_probability_alignment_features=lambda **_kwargs: {
			"raw_p_up_side": "up",
			"resolved_p_up_side": "up",
			"ret_pred_side": "up",
			"projected_price_side": "up",
			"forecast_consensus_side": "up",
			"probability_calibration_guard_applied": 0.0,
			"probability_calibration_used_regime_key": 0.0,
			"raw_calibrated_probability_gap": 0.0,
			"probability_alignment_gap": 0.0,
			"raw_p_up_ret_mismatch": 0.0,
			"p_up_ret_mismatch": 0.0,
			"raw_p_up_direction_mismatch": 0.0,
			"p_up_direction_mismatch": 0.0,
			"ret_projected_price_consensus": 1.0,
		},
		build_direction_output=lambda **_kwargs: {"direction": "up"},
		apply_target_range_overrides=lambda stop, take, projection, ratio, signal_dir_only: ({}, stop, take),
		evaluate_direction_only_fallback=lambda *_args, **_kwargs: ({"active": False}, False),
		finite_float_or_none=lambda value: float(value) if value is not None else None,
		coerce_row_value=lambda value: float(value) if value is not None else None,
	)

	assert result["direction_ensemble"]["selected_models"] == ["xgb", "transformer"]
	assert result["direction_ensemble"]["rejected_models"] == ["gru"]

import pandas as pd
import pytest

from src.runtime.prediction_result_support import build_prediction_result


def test_build_prediction_result_populates_expected_fields_and_pending_states() -> None:
	signal = {
		"p_up_components": {"xgb": 0.7},
			"component_probability_std": 0.12,
			"component_group_tree_p_up": 0.7,
			"direction_ensemble_selected_count": 3.0,
		"volatility": {"snapshot": {"atr": 1.2}, "triggered": False},
		"volatility_flag": False,
	}
	row_features = pd.Series(
		{
			"range_expansion_1h": 1.1,
			"momentum_slope_2h": 0.2,
		}
	)

	result, fallback_triggered = build_prediction_result(
		signal=signal,
		label="4h",
		horizon=4.0,
		signal_ts="2026-04-01T00:00:00Z",
		close=100000.0,
		p_up=0.72,
		raw_p_up=0.68,
		ret_pred=0.015,
		trend_prob=0.4,
		ignition_state=0,
		cooldown_active=False,
		signal_ensemble=1,
		signal_dir_only=1,
		confidence_score=0.8,
		position_size=0.35,
		confidence_min=0.25,
		confidence_min_source="default",
		position_size_cap=0.5,
		stop_loss_price=99000.0,
		take_profit_price=102000.0,
		expected_value=0.01,
		horizon_thresholds={"volatility_ceiling": 2.0},
		regime_state="neutral",
		calibration_key="4h@neutral",
		calibration_used_regime_key=True,
		probability_guard={"applied": False},
		regime_weight_policy={"enabled": True},
		target_projection={
			"projected_high": 103000.0,
			"projected_low": 99500.0,
			"projected_high_confidence": 0.8,
			"projected_low_confidence": 0.75,
		},
		volatility_payload={"snapshot": {"atr": 1.2}, "triggered": False},
		volatility_flag=False,
		forecast_coherence_policy={"p_up_neutral_band": 0.02},
		direction_output_policy={"enabled": True},
		direction_output_scoped=True,
		trade_decision_policy={"enabled": True},
		abstention_policy={"enabled": True},
		direction_fallback_policy={"enabled": True},
		trend_payload={"threshold": 0.6},
		target_range_policy={"enabled": True, "override_ratio": 0.01},
		regime_score=0.3,
		adaptive_scale=1.1,
		horizon_p_up=0.55,
		horizon_ret=0.002,
		row_features=row_features,
		optional_feature_fields=("range_expansion_1h", "momentum_slope_2h"),
		project_price=lambda close, log_return: close * (1.0 + log_return),
		get_active_regime_weight_override=lambda regime_state, horizon, policy: {"regime": regime_state, "horizon": horizon},
		derive_probability_alignment_features=lambda **kwargs: {
			"raw_p_up_side": "up",
			"resolved_p_up_side": "up",
			"ret_pred_side": "up",
			"projected_price_side": "up",
			"forecast_consensus_side": "up",
			"probability_calibration_guard_applied": False,
		},
		build_direction_output=lambda **kwargs: {"direction": "up", "scoped": kwargs["scoped"]},
		apply_target_range_overrides=lambda stop_loss, take_profit, target_projection, override_ratio, direction: (
			{"stop_loss": stop_loss, "take_profit": take_profit},
			stop_loss + 100.0,
			take_profit + 200.0,
		),
		evaluate_direction_only_fallback=lambda policy, **kwargs: ({"triggered": False}, False),
		finite_float_or_none=lambda value: None if value is None else float(value),
		coerce_row_value=lambda value: None if value is None else float(value),
	)

	assert fallback_triggered is False
	assert result["direction_next"] == "up"
	assert result["trade_action"] == "long"
	assert result["direction_output"]["direction"] == "up"
	assert result["projected_price"] == pytest.approx(101500.0)
	assert result["probability_calibration"]["applied_key"] == "4h@neutral"
	assert result["thresholds"]["adaptive_scale"] == 1.1
	assert result["target_range_overrides"]["stop_loss"] == 99000.0
	assert result["stop_loss"] == 99100.0
	assert result["take_profit"] == 102200.0
	assert result["trade_decision"]["enabled"] is True
	assert result["abstention"]["enabled"] is True
	assert result["uncertainty"]["available"] is False
	assert result["range_expansion_1h"] == 1.1
	assert result["momentum_slope_2h"] == 0.2
	assert result["component_probability_std"] == pytest.approx(0.12)
	assert result["component_group_tree_p_up"] == pytest.approx(0.7)
	assert result["direction_ensemble_selected_count"] == pytest.approx(3.0)
