from __future__ import annotations

import unittest

from src.scripts.analyze_prediction_coherence import _iter_live_rows, _summarize_rows
from src.scripts.run_refresh_and_predict import (
    _apply_forecast_coherence_policy,
    _build_prompt_ready_summary,
    _build_direction_output,
    _compute_directional_stop_take_prices,
    _refine_stop_with_target_range,
    _resolve_execution_target_reward,
    _resolve_direction_signal_for_horizon,
    _resolve_probability_calibration,
    _resolve_direction_threshold_for_horizon,
)


class PredictionCoherenceControlTests(unittest.TestCase):
    def test_probability_calibration_prefers_regime_specific_key(self) -> None:
        key, params, used_regime_key = _resolve_probability_calibration(
            {
                "1h": {"method": "platt", "a": 1.0, "b": 0.0},
                "1h@trend_ignition": {"method": "platt", "a": 2.0, "b": -0.5},
            },
            label="1h",
            regime_state="trend_ignition",
        )

        self.assertEqual(key, "1h@trend_ignition")
        self.assertEqual(params, {"method": "platt", "a": 2.0, "b": -0.5})
        self.assertTrue(used_regime_key)

    def test_probability_calibration_falls_back_to_base_key(self) -> None:
        key, params, used_regime_key = _resolve_probability_calibration(
            {
                "1h": {"method": "platt", "a": 1.0, "b": 0.0},
            },
            label="1h",
            regime_state="trend_ignition",
        )

        self.assertEqual(key, "1h")
        self.assertEqual(params, {"method": "platt", "a": 1.0, "b": 0.0})
        self.assertFalse(used_regime_key)

    def test_probability_calibration_rejects_degenerate_regime_platt_fit(self) -> None:
        key, params, used_regime_key = _resolve_probability_calibration(
            {
                "1h": {"method": "platt", "a": 0.25, "b": 0.05},
                "1h@trend_ignition": {"method": "platt", "a": -0.003, "b": 0.32},
            },
            label="1h",
            regime_state="trend_ignition",
        )

        self.assertEqual(key, "1h")
        self.assertEqual(params, {"method": "platt", "a": 0.25, "b": 0.05})
        self.assertFalse(used_regime_key)

    def test_forecast_coherence_policy_blocks_incoherent_hourly_entry(self) -> None:
        summary = {
            "1h": {
                "horizon_hours": 1.0,
                "close": 100.0,
                "projected_price": 101.0,
                "ret_pred": 0.01,
                "p_up": 0.43,
                "direction_next": "down",
                "direction_next_display": "down",
                "direction_output": {"direction": "down"},
                "trade_action": "short",
                "signal_ensemble": 1,
                "trade_decision": {"triggered": True, "trade_probability": 0.61},
            },
            "4h": {
                "horizon_hours": 4.0,
                "close": 100.0,
                "projected_price": 99.0,
                "ret_pred": -0.01,
                "p_up": 0.3,
                "direction_next": "down",
                "trade_action": "short",
                "signal_ensemble": 1,
                "trade_decision": {},
            },
        }
        policy = {
            "enabled": True,
            "horizons": [1.0, 4.0],
            "block_on_direction_ret_mismatch": True,
            "block_on_direction_projected_price_mismatch": True,
            "block_on_p_up_ret_mismatch": True,
            "p_up_neutral_band": 0.02,
            "min_p_up_edge": 0.05,
            "min_abs_ret_pred": 0.0,
            "exclude_blocked_horizons_from_voting": True,
        }

        updated = _apply_forecast_coherence_policy(summary, policy)

        self.assertEqual(updated["1h"]["trade_action"], "hold")
        self.assertEqual(updated["1h"]["signal_ensemble"], 0)
        self.assertEqual(updated["1h"]["direction_next_display"], "neutral")
        self.assertEqual(updated["1h"]["direction_output"]["direction"], "neutral")
        self.assertEqual(
            updated["1h"]["direction_output"]["coherence_override"]["reason"],
            "forecast_coherence_gate",
        )
        self.assertFalse(updated["1h"]["trade_decision"]["triggered"])
        self.assertTrue(updated["1h"]["trade_decision"]["pre_forecast_coherence_triggered"])
        self.assertTrue(updated["1h"]["trade_decision"]["blocked"])
        self.assertEqual(updated["1h"]["trade_decision"]["blocking_reason"], "forecast_coherence_gate")
        self.assertTrue(updated["1h"]["forecast_coherence"]["triggered"])
        self.assertTrue(updated["1h"]["forecast_coherence"]["exclude_from_voting"])
        self.assertIn("direction_ret_mismatch", updated["1h"]["forecast_coherence"]["reasons"])
        self.assertIn("direction_projected_price_mismatch", updated["1h"]["forecast_coherence"]["reasons"])
        self.assertFalse(updated["4h"]["forecast_coherence"]["triggered"])

    def test_projected_price_mismatch_is_ignored_for_non_positive_close(self) -> None:
        summary = {
            "1h": {
                "horizon_hours": 1.0,
                "close": -2.73,
                "projected_price": -2.74,
                "ret_pred": 0.01,
                "p_up": 0.52,
                "direction_next": "down",
                "trade_action": "short",
                "signal_ensemble": 1,
                "trade_decision": {},
            }
        }
        policy = {
            "enabled": True,
            "horizons": [1.0],
            "block_on_direction_ret_mismatch": True,
            "block_on_direction_projected_price_mismatch": True,
            "block_on_p_up_ret_mismatch": True,
            "p_up_neutral_band": 0.02,
            "min_p_up_edge": 0.05,
            "min_abs_ret_pred": 0.0,
            "exclude_blocked_horizons_from_voting": True,
        }

        updated = _apply_forecast_coherence_policy(summary, policy)

        self.assertTrue(updated["1h"]["forecast_coherence"]["triggered"])
        self.assertIn("direction_ret_mismatch", updated["1h"]["forecast_coherence"]["reasons"])
        self.assertNotIn("direction_projected_price_mismatch", updated["1h"]["forecast_coherence"]["reasons"])

    def test_prediction_coherence_summary_captures_hourly_mismatch_rates(self) -> None:
        history = [
            {
                "generated_at": "2026-03-16T10:00:00Z",
                "predictions": {
                    "1h": {
                        "close": 100.0,
                        "projected_price": 101.0,
                        "ret_pred": 0.01,
                        "p_up": 0.45,
                        "direction_next": "down",
                    },
                    "4h": {
                        "close": 100.0,
                        "projected_price": 99.0,
                        "ret_pred": -0.01,
                        "p_up": 0.40,
                        "direction_next": "down",
                    },
                },
            },
            {
                "generated_at": "2026-03-16T11:00:00Z",
                "predictions": {
                    "1h": {
                        "close": 100.0,
                        "projected_price": 99.0,
                        "ret_pred": -0.01,
                        "p_up": 0.40,
                        "direction_next": "down",
                    }
                },
            },
        ]

        rows = _iter_live_rows(history, neutral_band=0.02)
        payload = _summarize_rows(rows)

        self.assertEqual(payload["snapshots_with_live_rows"], 2)
        self.assertEqual(payload["historical_summary"]["1h"]["rows"], 2)
        self.assertEqual(payload["historical_summary"]["1h"]["direction_ret_mismatch_rate"], 0.5)
        self.assertEqual(payload["historical_summary"]["1h"]["direction_projected_price_mismatch_rate"], 0.5)

    def test_direction_output_can_emit_neutral_display_direction(self) -> None:
        payload = _build_direction_output(
            enabled=True,
            scoped=True,
            label="1h",
            regime_state="neutral",
            signal_dir_only=1,
            raw_probability=0.51,
            trade_probability=0.51,
            ret_pred=0.01,
            close=100.0,
            projected_price=101.0,
            p_up_components={},
            policy={
                "neutral_band": 0.02,
                "use_trade_probability_fallback": True,
                "calibration_map": {},
            },
        )

        self.assertEqual(payload["direction"], "neutral")
        self.assertEqual(payload["source"], "trade_probability")
        self.assertTrue(payload["calibration"]["fallback_to_trade_probability"])

    def test_direction_output_prefers_separate_calibration_map(self) -> None:
        payload = _build_direction_output(
            enabled=True,
            scoped=True,
            label="1h",
            regime_state="trend_ignition",
            signal_dir_only=0,
            raw_probability=0.55,
            trade_probability=0.52,
            ret_pred=0.01,
            close=100.0,
            projected_price=101.0,
            p_up_components={},
            policy={
                "neutral_band": 0.0,
                "use_trade_probability_fallback": True,
                "calibration_map": {
                    "1h@trend_ignition": {"method": "platt", "a": 2.0, "b": 0.0},
                },
            },
        )

        self.assertEqual(payload["source"], "direction_output_calibration")
        self.assertEqual(payload["calibration"]["applied_key"], "1h@trend_ignition")
        self.assertTrue(payload["calibration"]["used_regime_key"])
        self.assertEqual(payload["direction"], "up")

    def test_direction_output_falls_back_to_internal_direction_when_calibration_conflicts_with_forecast(self) -> None:
        payload = _build_direction_output(
            enabled=True,
            scoped=True,
            label="1h",
            regime_state="trend_ignition",
            signal_dir_only=0,
            raw_probability=0.46,
            trade_probability=0.46,
            ret_pred=-0.01,
            close=100.0,
            projected_price=99.0,
            p_up_components={},
            policy={
                "neutral_band": 0.02,
                "use_trade_probability_fallback": True,
                "calibration_map": {
                    "1h@trend_ignition": {
                        "method": "isotonic",
                        "x": [0.46, 0.46],
                        "y": [0.52, 0.52],
                    },
                },
            },
        )

        self.assertEqual(payload["source"], "direction_output_calibration")
        self.assertEqual(payload["direction"], "down")
        self.assertTrue(payload["forecast_alignment_override"]["applied"])
        self.assertEqual(
            payload["forecast_alignment_override"]["reason"],
            "fallback_to_internal_forecast_alignment",
        )

    def test_direction_output_marginal_rerank_only_applies_inside_band(self) -> None:
        payload = _build_direction_output(
            enabled=True,
            scoped=True,
            label="1h",
            regime_state="neutral",
            signal_dir_only=0,
            raw_probability=0.56,
            trade_probability=0.56,
            ret_pred=0.02,
            close=100.0,
            projected_price=102.0,
            p_up_components={"gru": 0.72, "lstm": 0.68, "transformer": 0.41},
            policy={
                "neutral_band": 0.0,
                "use_trade_probability_fallback": True,
                "calibration_map": {
                    "1h": {"method": "platt", "a": 1.0, "b": 0.0},
                },
                "marginal_rerank": {
                    "enabled": True,
                    "horizons": [1.0],
                    "lower": 0.5,
                    "upper": 0.6,
                    "min_component_count": 2,
                    "use_raw_probability_gate": True,
                    "weight_specs": {
                        "default": {"gru": 1.5, "lstm": 1.0},
                    },
                },
            },
        )

        self.assertEqual(payload["source"], "direction_output_marginal_rerank")
        self.assertTrue(payload["marginal_rerank"]["applied"])
        self.assertEqual(payload["direction"], "up")

    def test_auto_direction_threshold_keeps_direction_labels_above_coin_flip(self) -> None:
        self.assertEqual(
            _resolve_direction_threshold_for_horizon(
                direction_threshold=0.6,
                auto_direction_threshold=True,
                horizon_p_up=0.36,
            ),
            0.5,
        )
        self.assertEqual(
            _resolve_direction_threshold_for_horizon(
                direction_threshold=0.6,
                auto_direction_threshold=True,
                horizon_p_up=0.58,
            ),
            0.58,
        )
        self.assertEqual(
            _resolve_direction_threshold_for_horizon(
                direction_threshold=0.6,
                auto_direction_threshold=False,
                horizon_p_up=0.36,
            ),
            0.6,
        )

    def test_directional_stop_take_prices_follow_final_direction(self) -> None:
        stop_loss, take_profit = _compute_directional_stop_take_prices(
            close=100.0,
            ret_pred=0.01,
            residual_std=0.02,
            direction_signal=1,
        )
        self.assertLess(stop_loss, 100.0)
        self.assertGreater(take_profit, 100.0)

        stop_loss, take_profit = _compute_directional_stop_take_prices(
            close=100.0,
            ret_pred=0.01,
            residual_std=0.02,
            direction_signal=0,
        )
        self.assertGreater(stop_loss, 100.0)
        self.assertLess(take_profit, 100.0)

    def test_direction_signal_uses_raw_side_when_base_fallback_conflicts_with_coherent_forecast(self) -> None:
        signal = _resolve_direction_signal_for_horizon(
            raw_probability=0.62,
            calibrated_probability=0.20,
            threshold=0.4,
            close=100.0,
            projected_price=101.0,
            ret_pred=0.01,
            calibration_key="4h",
            calibration_used_regime_key=False,
        )

        self.assertEqual(signal, 1)

    def test_direction_signal_keeps_calibrated_side_when_raw_conflict_lacks_forecast_support(self) -> None:
        signal = _resolve_direction_signal_for_horizon(
            raw_probability=0.62,
            calibrated_probability=0.20,
            threshold=0.4,
            close=100.0,
            projected_price=99.0,
            ret_pred=-0.01,
            calibration_key="4h",
            calibration_used_regime_key=False,
        )

        self.assertEqual(signal, 0)

    def test_direction_signal_uses_directional_threshold_floor_for_regime_specific_conflict(self) -> None:
        signal = _resolve_direction_signal_for_horizon(
            raw_probability=0.38,
            calibrated_probability=0.58,
            threshold=0.36,
            close=100.0,
            projected_price=99.0,
            ret_pred=-0.01,
            calibration_key="1h@trend_ignition",
            calibration_used_regime_key=True,
        )

        self.assertEqual(signal, 0)

    def test_direction_signal_uses_forecast_consensus_when_both_classifier_sides_conflict(self) -> None:
        signal = _resolve_direction_signal_for_horizon(
            raw_probability=0.79,
            calibrated_probability=0.82,
            threshold=0.6,
            close=100.0,
            projected_price=98.0,
            ret_pred=-0.02,
            calibration_key="4h@neutral",
            calibration_used_regime_key=True,
        )

        self.assertEqual(signal, 0)

    def test_direction_signal_keeps_classifier_side_without_two_way_forecast_consensus(self) -> None:
        signal = _resolve_direction_signal_for_horizon(
            raw_probability=0.79,
            calibrated_probability=0.82,
            threshold=0.6,
            close=100.0,
            projected_price=100.0,
            ret_pred=-0.02,
            calibration_key="4h@neutral",
            calibration_used_regime_key=True,
        )

        self.assertEqual(signal, 1)

    def test_direction_output_marginal_rerank_does_not_apply_outside_band(self) -> None:
        payload = _build_direction_output(
            enabled=True,
            scoped=True,
            label="1h",
            regime_state="neutral",
            signal_dir_only=0,
            raw_probability=0.63,
            trade_probability=0.63,
            ret_pred=-0.01,
            close=100.0,
            projected_price=99.0,
            p_up_components={"gru": 0.72, "lstm": 0.68},
            policy={
                "neutral_band": 0.0,
                "use_trade_probability_fallback": True,
                "calibration_map": {},
                "marginal_rerank": {
                    "enabled": True,
                    "horizons": [1.0],
                    "lower": 0.5,
                    "upper": 0.6,
                    "min_component_count": 2,
                    "use_raw_probability_gate": True,
                    "weight_specs": {
                        "default": {"gru": 1.5, "lstm": 1.0},
                    },
                },
            },
        )

        self.assertEqual(payload["source"], "trade_probability")
        self.assertFalse(payload["marginal_rerank"]["applied"])

    def test_execution_target_reward_can_adapt_to_mfe_headroom(self) -> None:
        payload = _resolve_execution_target_reward(
            side="long",
            planned_entry=100.0,
            existing_take=120.0,
            projected_high=117.0,
            projected_low=None,
            analytics_payload={"available": True, "mfe_distance": 0.17},
            risk_unit=10.0,
            horizon=4.0,
            policy={
                "minimum_rr_by_horizon": {4.0: 2.0},
                "analytics": {"regime_volatility_buckets": {"max_projection_mfe_ratio": 1.25}},
                "adaptive_take_profit": {"enabled": True, "min_rr_fraction_of_floor": 0.85},
            },
            regime_template={"tp_multiplier": 1.0},
        )

        self.assertEqual(payload["status"], "pass")
        self.assertTrue(payload["target_management"]["adapted_to_mfe_headroom"])
        self.assertAlmostEqual(payload["risk_reward_ratio"], 1.7)
        self.assertAlmostEqual(payload["selected_take"], 117.0)

    def test_execution_target_reward_rejects_when_feasible_rr_remains_too_low(self) -> None:
        payload = _resolve_execution_target_reward(
            side="long",
            planned_entry=100.0,
            existing_take=120.0,
            projected_high=115.0,
            projected_low=None,
            analytics_payload={"available": True, "mfe_distance": 0.15},
            risk_unit=10.0,
            horizon=4.0,
            policy={
                "minimum_rr_by_horizon": {4.0: 2.0},
                "analytics": {"regime_volatility_buckets": {"max_projection_mfe_ratio": 1.25}},
                "adaptive_take_profit": {"enabled": True, "min_rr_fraction_of_floor": 0.85},
            },
            regime_template={"tp_multiplier": 1.0},
        )

        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["reason"], "insufficient_mfe_headroom")

    def test_target_range_stop_refinement_tightens_long_stop_with_high_confidence_projection(self) -> None:
        payload = _refine_stop_with_target_range(
            side="long",
            planned_entry=100.0,
            selected_stop=92.0,
            risk_unit=8.0,
            atr_distance=4.0,
            horizon=8.0,
            projected_high=109.0,
            projected_low=98.5,
            projected_high_confidence=0.8,
            projected_low_confidence=0.78,
            projected_high_residual_std=0.002,
            projected_low_residual_std=0.003,
            policy={
                "target_range_stop_refinement": {
                    "enabled": True,
                    "horizons": [8.0, 12.0],
                    "confidence_min": 0.72,
                    "buffer_std_mult": 1.5,
                    "min_tighten_fraction": 0.1,
                }
            },
            guards_cfg={
                "enabled": True,
                "min_stop_distance_atr_mult": 0.35,
            },
        )

        self.assertTrue(payload["applied"])
        self.assertGreater(payload["stop_loss"], 92.0)
        self.assertLess(payload["risk_unit"], 8.0)
        self.assertEqual(payload["details"]["type"], "target_range_stop_tightened")

    def test_target_range_stop_refinement_respects_confidence_floor(self) -> None:
        payload = _refine_stop_with_target_range(
            side="long",
            planned_entry=100.0,
            selected_stop=92.0,
            risk_unit=8.0,
            atr_distance=4.0,
            horizon=8.0,
            projected_high=109.0,
            projected_low=98.5,
            projected_high_confidence=0.8,
            projected_low_confidence=0.6,
            projected_high_residual_std=0.002,
            projected_low_residual_std=0.003,
            policy={
                "target_range_stop_refinement": {
                    "enabled": True,
                    "horizons": [8.0, 12.0],
                    "confidence_min": 0.72,
                    "buffer_std_mult": 1.5,
                    "min_tighten_fraction": 0.1,
                }
            },
            guards_cfg={
                "enabled": True,
                "min_stop_distance_atr_mult": 0.35,
            },
        )

        self.assertFalse(payload["applied"])
        self.assertEqual(payload["stop_loss"], 92.0)
        self.assertEqual(payload["risk_unit"], 8.0)

    def test_prompt_ready_summary_prefers_bias_ready_midterm_horizon(self) -> None:
        payload = _build_prompt_ready_summary(
            {
                "1h": {
                    "horizon_hours": 1.0,
                    "direction_next_display": "neutral",
                    "confidence_score": 0.12,
                    "forecast_coherence": {"triggered": True, "reasons": ["direction_ret_mismatch"]},
                    "execution_plan": {"status": "rejected", "reason": "forecast_coherence_gate"},
                },
                "4h": {
                    "horizon_hours": 4.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.48,
                    "entry_price": 100.0,
                    "stop_loss": 94.0,
                    "take_profit": 110.2,
                    "risk_reward_ratio": 1.7,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "bias_only_ready",
                        "reason": "upstream_model_hold",
                        "target_management": {"adapted_to_mfe_headroom": True},
                    },
                },
            }
        )

        self.assertEqual(payload["market_outlook_strategy"]["selected_direction"], "Long")
        self.assertEqual(payload["market_outlook_strategy"]["preferred_horizon"], "4h")
        self.assertEqual(payload["market_outlook_strategy"]["execution_state"], "bias_only_ready")
        self.assertEqual(payload["trade_execution_plan_usd"]["entry_point"], 100.0)
        self.assertIn("Take-profit was resized to empirical MFE headroom", payload["analysis_summary"]["rationale"])

    def test_prompt_ready_summary_includes_projected_range_in_trend_forecast(self) -> None:
        payload = _build_prompt_ready_summary(
            {
                "8h": {
                    "horizon_hours": 8.0,
                    "direction_next_display": "up",
                    "projected_high": 108.0,
                    "projected_low": 97.0,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {"status": "rejected", "reason": "insufficient_mfe_headroom"},
                }
            }
        )

        self.assertIn("projected range $97.00 to $108.00", payload["analysis_summary"]["trend_forecast"][0])

    def test_prompt_ready_summary_prefers_broader_high_timeframe_bias_when_midterm_conflicts(self) -> None:
        payload = _build_prompt_ready_summary(
            {
                "4h": {
                    "horizon_hours": 4.0,
                    "direction_next_display": "down",
                    "confidence_score": 0.55,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "bias_direction_conflict",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.25,
                    },
                },
                "8h": {
                    "horizon_hours": 8.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.49,
                    "entry_price": 100.0,
                    "stop_loss": 95.0,
                    "take_profit": 109.0,
                    "risk_reward_ratio": 1.8,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "low_execution_confluence",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.5,
                    },
                },
                "12h": {
                    "horizon_hours": 12.0,
                    "direction_next_display": "up",
                    "confidence_score": 0.46,
                    "entry_price": 99.0,
                    "stop_loss": 94.0,
                    "take_profit": 111.0,
                    "risk_reward_ratio": 2.0,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "low_execution_confluence",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.5,
                    },
                },
            }
        )

        self.assertEqual(payload["market_outlook_strategy"]["selected_direction"], "Long")
        self.assertEqual(payload["market_outlook_strategy"]["preferred_horizon"], "8h")
        self.assertEqual(payload["market_outlook_strategy"]["execution_state"], "rejected")
        self.assertIn("wins side arbitration for long bias across 8h, 12h", payload["analysis_summary"]["rationale"])

    def test_prompt_ready_summary_keeps_15m_as_timing_when_hourly_stack_exists(self) -> None:
        payload = _build_prompt_ready_summary(
            {
                "15m": {
                    "horizon_hours": 0.25,
                    "direction_next_display": "down",
                    "confidence_score": 0.61,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "waiting_pullback",
                        "reason": "await_pullback_entry_zone",
                        "confluence_tier": "medium",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.75,
                    },
                },
                "1h": {
                    "horizon_hours": 1.0,
                    "direction_next_display": "down",
                    "confidence_score": 0.43,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "insufficient_mfe_headroom",
                        "confluence_tier": "medium",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.75,
                    },
                },
                "4h": {
                    "horizon_hours": 4.0,
                    "direction_next_display": "down",
                    "confidence_score": 0.48,
                    "entry_price": 100.0,
                    "stop_loss": 104.0,
                    "take_profit": 94.0,
                    "risk_reward_ratio": 1.5,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "waiting_pullback",
                        "reason": "await_pullback_entry_zone",
                        "confluence_tier": "medium",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.75,
                    },
                },
                "8h": {
                    "horizon_hours": 8.0,
                    "direction_next_display": "down",
                    "confidence_score": 0.44,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "insufficient_mfe_headroom",
                        "confluence_tier": "medium",
                        "bias_alignment_ratio": 2.0 / 3.0,
                        "execution_alignment_ratio": 0.75,
                    },
                },
            }
        )

        self.assertEqual(payload["market_outlook_strategy"]["selected_direction"], "Short")
        self.assertEqual(payload["market_outlook_strategy"]["preferred_horizon"], "4h")
        self.assertEqual(payload["market_outlook_strategy"]["execution_state"], "waiting_pullback")
        self.assertNotIn("Preferred horizon 15m", payload["analysis_summary"]["rationale"])

    def test_prompt_ready_summary_keeps_blocked_high_timeframe_bias_for_outlook(self) -> None:
        payload = _build_prompt_ready_summary(
            {
                "15m": {
                    "horizon_hours": 0.25,
                    "direction_next": "up",
                    "direction_next_display": "up",
                    "confidence_score": 0.84,
                    "projected_price": 74107.2,
                    "forecast_coherence": {"triggered": False, "reasons": []},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "low_execution_confluence",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 1.0,
                        "execution_alignment_ratio": 0.0,
                    },
                },
                "1h": {
                    "horizon_hours": 1.0,
                    "direction_next": "down",
                    "direction_next_display": "neutral",
                    "confidence_score": 0.16,
                    "projected_price": 73933.56,
                    "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "forecast_coherence_gate",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 0.0,
                        "execution_alignment_ratio": 0.0,
                    },
                },
                "4h": {
                    "horizon_hours": 4.0,
                    "direction_next": "down",
                    "direction_next_display": "neutral",
                    "confidence_score": 0.35,
                    "projected_high": 74250.7,
                    "projected_low": 73874.32,
                    "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "forecast_coherence_gate",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 0.0,
                        "execution_alignment_ratio": 0.0,
                    },
                },
                "8h": {
                    "horizon_hours": 8.0,
                    "direction_next": "down",
                    "direction_next_display": "neutral",
                    "confidence_score": 0.44,
                    "projected_high": 74315.76,
                    "projected_low": 73879.47,
                    "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "forecast_coherence_gate",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 0.0,
                        "execution_alignment_ratio": 0.0,
                    },
                },
                "12h": {
                    "horizon_hours": 12.0,
                    "direction_next": "down",
                    "direction_next_display": "neutral",
                    "confidence_score": 0.41,
                    "projected_high": 74349.99,
                    "projected_low": 73693.78,
                    "forecast_coherence": {"triggered": True, "reasons": ["p_up_ret_mismatch"]},
                    "execution_plan": {
                        "status": "rejected",
                        "reason": "forecast_coherence_gate",
                        "confluence_tier": "low",
                        "bias_alignment_ratio": 0.0,
                        "execution_alignment_ratio": 0.0,
                    },
                },
            }
        )

        self.assertEqual(payload["market_outlook_strategy"]["selected_direction"], "Short")
        self.assertEqual(payload["market_outlook_strategy"]["preferred_horizon"], "8h")
        self.assertIn("1h: down, projected price $73,933.56 (coherence blocked)", payload["analysis_summary"]["trend_forecast"])
        self.assertIn("Preferred horizon 8h carries the strongest post-policy bias.", payload["analysis_summary"]["rationale"])
        self.assertIn("forecast coherence blocks it from execution", payload["analysis_summary"]["rationale"])


if __name__ == "__main__":
    unittest.main()