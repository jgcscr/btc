from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from src.scripts.run_refresh_and_predict import (
    _apply_execution_policy,
    _apply_uncertainty_abstention,
    _build_blocked_trade_analytics,
    _build_degradation_monitoring,
    _build_prompt_ready_summary,
    _resolve_execution_policy,
    _resolve_uncertainty_policy,
    _summarize_bias_context,
)


class ExecutionPolicyExtensionTests(unittest.TestCase):
    def test_bias_context_uses_horizon_weights(self) -> None:
        summary = {
            "4h": {
                "horizon_hours": 4.0,
                "direction_next": "down",
                "direction_next_display": "down",
                "confidence_score": 0.4,
            },
            "8h": {
                "horizon_hours": 8.0,
                "direction_next": "up",
                "direction_next_display": "up",
                "confidence_score": 1.0,
            },
            "12h": {
                "horizon_hours": 12.0,
                "direction_next": "down",
                "direction_next_display": "down",
                "confidence_score": 0.8,
            },
        }
        policy = _resolve_execution_policy(
            {
                "enabled": True,
                "bias_horizons": [4.0, 8.0, 12.0],
                "execution_horizons": [4.0, 8.0, 12.0],
                "horizon_bias_weights": {"4": 1.0, "8": 0.4, "12": 1.2},
            }
        )

        context = _summarize_bias_context(summary, policy)

        self.assertEqual(context["bias_direction"], "down")
        self.assertGreater(context["bias_alignment_ratio"], 0.5)
        self.assertEqual(context["direction_support_horizons"]["down"], ["4h", "12h"])

    def test_bias_context_penalizes_low_trust_horizon_votes(self) -> None:
        summary = {
            "4h": {
                "horizon_hours": 4.0,
                "direction_next": "up",
                "direction_next_display": "up",
                "confidence_score": 1.0,
                "forecast_coherence": {
                    "triggered": False,
                    "low_trust": True,
                    "ret_pred_side": "down",
                    "projected_price_side": "down",
                    "p_up_side": "up",
                },
                "ret_pred": -0.01,
                "projected_price": 99.0,
                "close": 100.0,
                "p_up": 0.7,
            },
            "8h": {
                "horizon_hours": 8.0,
                "direction_next": "down",
                "direction_next_display": "down",
                "confidence_score": 0.7,
                "forecast_coherence": {
                    "triggered": False,
                    "low_trust": False,
                    "ret_pred_side": "down",
                    "projected_price_side": "down",
                    "p_up_side": "down",
                },
                "ret_pred": -0.01,
                "projected_price": 99.0,
                "close": 100.0,
                "p_up": 0.3,
            },
            "12h": {
                "horizon_hours": 12.0,
                "direction_next": "down",
                "direction_next_display": "down",
                "confidence_score": 0.7,
                "forecast_coherence": {
                    "triggered": False,
                    "low_trust": False,
                    "ret_pred_side": "down",
                    "projected_price_side": "down",
                    "p_up_side": "down",
                },
                "ret_pred": -0.01,
                "projected_price": 99.0,
                "close": 100.0,
                "p_up": 0.3,
            },
        }
        policy = _resolve_execution_policy(
            {
                "enabled": True,
                "bias_horizons": [4.0, 8.0, 12.0],
                "execution_horizons": [4.0, 8.0, 12.0],
                "horizon_bias_weights": {"4": 1.0, "8": 1.0, "12": 1.0},
                "coherence_weighting": {
                    "enabled": True,
                    "low_trust_penalty": 0.75,
                    "p_up_conflict_penalty": 0.3,
                    "consensus_bonus": 0.05,
                },
            }
        )

        context = _summarize_bias_context(summary, policy)

        self.assertEqual(context["bias_direction"], "down")
        four_hour_detail = next(detail for detail in context["bias_scores"]["details"] if detail["label"] == "4h")
        self.assertLess(four_hour_detail["coherence_multiplier"], 1.0)

    def test_uncertainty_policy_uses_horizon_regime_override(self) -> None:
        policy = _resolve_uncertainty_policy(
            {
                "enabled": True,
                "alpha": 0.2,
                "hold_prob_center": 0.5,
                "max_interval_width": 0.5,
                "require_center_cross": False,
                "thresholds_by_horizon_regime": {
                    "8": {
                        "neutral": {
                            "max_interval_width": 0.95,
                        }
                    }
                },
            }
        )

        abstain, reason, payload = _apply_uncertainty_abstention(
            trade_action="long",
            p_up_components={"a": 0.1, "b": 0.9, "c": 0.85},
            horizon=8.0,
            regime_state="neutral",
            policy=policy,
        )

        self.assertFalse(abstain)
        self.assertEqual(reason, "pass")
        self.assertAlmostEqual(payload["effective_policy"]["max_interval_width"], 0.95)

    def test_pullback_quality_downgrades_extended_entry(self) -> None:
        summary = {
            "4h": {
                "horizon_hours": 4.0,
                "close": 106.0,
                "entry_price": 106.0,
                "stop_loss": 103.0,
                "take_profit": 112.0,
                "risk_reward_ratio": 2.0,
                "direction_next": "up",
                "direction_next_display": "up",
                "trade_action": "long",
                "signal_ensemble": 1,
                "confluence_support_ratio": 1.0,
                "confluence_mid_term_ratio": 1.0,
                "regime_state": "neutral",
                "forecast_coherence": {"triggered": False, "reasons": []},
                "position_size": 1.0,
                "range_expansion_1h": 2.0,
                "momentum_slope_2h": -0.03,
            }
        }
        contexts = {
            "4h": {
                "prepared": SimpleNamespace(
                    df_all=pd.DataFrame(
                        {
                            "high": [100.5, 100.5, 100.5, 106.5],
                            "low": [99.5, 99.5, 99.5, 105.5],
                            "close": [100.0, 100.0, 100.0, 106.0],
                            "volume": [1.0, 1.0, 1.0, 1.0],
                        }
                    )
                ),
                "index": 3,
                "horizon": 4.0,
                "residual_std": 0.01,
            }
        }
        policy = _resolve_execution_policy(
            {
                "enabled": True,
                "bias_horizons": [4.0],
                "execution_horizons": [4.0],
                "require_bias_alignment": True,
                "immediate_entry_min_support_ratio": 0.8,
                "pullback_entry_min_support_ratio": 0.5,
                "immediate_entry_min_mid_ratio": 0.8,
                "pullback_entry_min_mid_ratio": 0.5,
                "high_execution_alignment_ratio": 0.5,
                "medium_execution_alignment_ratio": 0.5,
                "entry_zone_atr_mult": 0.25,
                "max_chase_atr_mult": 0.35,
                "session_lookback_bars": 4,
                "swing_lookback_bars": 4,
                "structure_buffer_atr_mult": 0.2,
                "minimum_rr_by_horizon": {"4": 1.2},
                "time_stop_bars_by_horizon": {"4": 2},
                "analytics": {"enabled": False},
                "no_trade_guards": {"enabled": False},
                "partial_take_profit": {"enabled": False},
                "trailing_stop": {"enabled": False},
                "adaptive_take_profit": {"enabled": False},
                "pullback_quality": {
                    "enabled": True,
                    "min_score_by_horizon": {"4": 0.95},
                    "max_vwap_deviation_atr": 1.0,
                    "max_candle_expansion_ratio": 1.2,
                    "candle_expansion_window": 3,
                    "range_expansion_penalty_threshold": 1.1,
                },
                "disagreement_severity": {"enabled": False},
                "regime_templates": {"neutral": {}},
            }
        )

        updated = _apply_execution_policy(summary, contexts, policy)
        plan = updated["4h"]["execution_plan"]

        self.assertTrue(plan["pullback_quality"]["triggered"])
        self.assertEqual(plan["entry_mode"], "pullback")
        self.assertIn(plan["status"], {"waiting_pullback", "rejected"})

    def test_prompt_ready_summary_includes_compact_operator_view(self) -> None:
        summary = {
            "4h": {
                "horizon_hours": 4.0,
                "direction_next": "up",
                "direction_next_display": "up",
                "confidence_score": 0.8,
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "risk_reward_ratio": 2.0,
                "forecast_coherence": {"triggered": False, "reasons": []},
                "execution_plan": {
                    "status": "bias_only_ready",
                    "reason": "confluence_gate",
                    "pending_trade_action": "long",
                    "execution_alignment_ratio": 1.0,
                    "bias_alignment_ratio": 1.0,
                    "execution_score": 1.0,
                    "bias_score": 1.0,
                    "confluence_tier": "high",
                    "disagreement_severity": {"score": 0.2, "triggered": False},
                    "pullback_quality": {"triggered": False},
                },
            }
        }

        payload = _build_prompt_ready_summary(summary)
        compact = payload["operator_summary_compact"]

        self.assertEqual(compact["recommended_operator_action"], "bias_only")
        self.assertEqual(compact["primary_blocker"], "confluence_gate")
        self.assertEqual(compact["preferred_horizon"], "4h")

    def test_blocked_trade_analytics_counts_reasons(self) -> None:
        analytics = _build_blocked_trade_analytics(
            {
                "4h": {
                    "trade_action": "hold",
                    "execution_plan": {"status": "rejected", "reason": "short_term_disagreement"},
                },
                "12h": {
                    "trade_action": "hold",
                    "execution_plan": {"status": "bias_only_ready", "reason": "confluence_gate"},
                },
            }
        )

        self.assertEqual(analytics["blocked_total"], 2)
        self.assertEqual(analytics["reason_counts"]["short_term_disagreement"], 1)
        self.assertEqual(analytics["reason_counts"]["confluence_gate"], 1)

    def test_degradation_monitoring_flags_repeated_blocking(self) -> None:
        history = [
            {
                "predictions": {
                    "4h": {
                        "confidence_score": 0.2,
                        "trade_decision": {"expected_net": 0.0},
                        "execution_plan": {"status": "rejected", "reason": "bias_direction_conflict"},
                    }
                }
            }
            for _ in range(4)
        ]

        payload = _build_degradation_monitoring(
            history,
            policy={
                "enabled": True,
                "lookback_snapshots": 4,
                "min_snapshots": 3,
                "min_ready_ratio": 0.5,
                "max_blocked_ratio": 0.6,
                "min_expected_net": 0.001,
                "min_confidence": 0.5,
            },
        )

        self.assertTrue(payload["by_horizon"]["4h"]["alarm"])
        self.assertIn("ready_ratio_below_floor", payload["by_horizon"]["4h"]["reasons"])
        self.assertIn("blocked_ratio_above_ceiling", payload["by_horizon"]["4h"]["reasons"])


if __name__ == "__main__":
    unittest.main()