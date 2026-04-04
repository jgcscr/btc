from __future__ import annotations

import unittest

from src.runtime.family_outcome_confirmation import (
    _assess_two_window_stability,
    _resolve_effective_window_size,
    split_two_non_overlapping_recent_windows,
)


class OrderFlowTwoWindowStabilityTests(unittest.TestCase):
    def test_resolve_effective_window_size_caps_to_available(self) -> None:
        self.assertEqual(_resolve_effective_window_size(snapshot_count=1783, requested_window_size=1000), 891)
        self.assertEqual(_resolve_effective_window_size(snapshot_count=1783, requested_window_size=0), 891)

    def test_split_two_non_overlapping_recent_windows(self) -> None:
        snapshots = [{"generated_at": f"t{i}"} for i in range(10)]
        windows = split_two_non_overlapping_recent_windows(snapshots, window_size=4)
        self.assertEqual(len(windows["window_older"]), 4)
        self.assertEqual(len(windows["window_newer"]), 4)
        self.assertEqual(windows["window_older"][0]["generated_at"], "t2")
        self.assertEqual(windows["window_newer"][0]["generated_at"], "t6")

    def test_assess_two_window_stability_ready(self) -> None:
        window_summaries = {
            "window_older": {
                "overall_delta": {
                    "net_return_proxy_mean_delta": 0.0005,
                    "direction_accuracy_proxy_delta": 0.02,
                },
                "veto_precision": 0.75,
            },
            "window_newer": {
                "overall_delta": {
                    "net_return_proxy_mean_delta": 0.00045,
                    "direction_accuracy_proxy_delta": 0.018,
                },
                "veto_precision": 0.78,
            },
        }
        window_decisions = {
            "window_older": {"decision": "go", "harmful_veto_rate": 0.2},
            "window_newer": {"decision": "go", "harmful_veto_rate": 0.18},
        }
        aggregate_summary = {
            "by_horizon": {
                "4h": {"delta": {"net_return_proxy_mean_delta": 0.00025}},
                "8h": {"delta": {"net_return_proxy_mean_delta": 0.0002}},
                "12h": {"delta": {"net_return_proxy_mean_delta": 0.00015}},
            },
            "by_regime": {
                "neutral": {"delta": {"net_return_proxy_mean_delta": 0.0003}},
                "chop": {"delta": {"net_return_proxy_mean_delta": 0.0002}},
            },
        }
        aggregate_decision = {"decision": "go"}
        thresholds = {
            "maximum_window_gain_concentration_share": 0.75,
            "maximum_horizon_gain_concentration_share": 0.80,
            "maximum_regime_gain_concentration_share": 0.80,
            "immediate_disable_on_any_window_net_delta_below": -0.0005,
            "immediate_disable_on_any_window_accuracy_delta_below": -0.02,
            "immediate_disable_on_any_window_harmful_veto_rate_above": 0.50,
            "immediate_disable_on_any_window_veto_precision_below": 0.50,
        }

        out = _assess_two_window_stability(
            window_summaries=window_summaries,
            window_decisions=window_decisions,
            aggregate_summary=aggregate_summary,
            aggregate_decision=aggregate_decision,
            thresholds=thresholds,
        )

        self.assertTrue(out["both_windows_pass_guardrails"])
        self.assertTrue(out["robust_enough_for_guarded_shadow"])
        self.assertEqual(out["readiness_recommendation"], "ready_for_shadow_production")

    def test_assess_two_window_stability_not_ready_when_one_window_fails(self) -> None:
        window_summaries = {
            "window_older": {
                "overall_delta": {
                    "net_return_proxy_mean_delta": 0.00055,
                    "direction_accuracy_proxy_delta": 0.021,
                },
                "veto_precision": 0.74,
            },
            "window_newer": {
                "overall_delta": {
                    "net_return_proxy_mean_delta": -0.0008,
                    "direction_accuracy_proxy_delta": -0.03,
                },
                "veto_precision": 0.45,
            },
        }
        window_decisions = {
            "window_older": {"decision": "go", "harmful_veto_rate": 0.2},
            "window_newer": {"decision": "hold", "harmful_veto_rate": 0.52},
        }
        aggregate_summary = {
            "by_horizon": {
                "4h": {"delta": {"net_return_proxy_mean_delta": 0.0003}},
                "8h": {"delta": {"net_return_proxy_mean_delta": 0.00002}},
                "12h": {"delta": {"net_return_proxy_mean_delta": 0.00001}},
            },
            "by_regime": {
                "neutral": {"delta": {"net_return_proxy_mean_delta": 0.0003}},
                "chop": {"delta": {"net_return_proxy_mean_delta": 0.0}},
            },
        }
        aggregate_decision = {"decision": "hold"}
        thresholds = {
            "maximum_window_gain_concentration_share": 0.75,
            "maximum_horizon_gain_concentration_share": 0.80,
            "maximum_regime_gain_concentration_share": 0.80,
            "immediate_disable_on_any_window_net_delta_below": -0.0005,
            "immediate_disable_on_any_window_accuracy_delta_below": -0.02,
            "immediate_disable_on_any_window_harmful_veto_rate_above": 0.50,
            "immediate_disable_on_any_window_veto_precision_below": 0.50,
        }

        out = _assess_two_window_stability(
            window_summaries=window_summaries,
            window_decisions=window_decisions,
            aggregate_summary=aggregate_summary,
            aggregate_decision=aggregate_decision,
            thresholds=thresholds,
        )

        self.assertFalse(out["both_windows_pass_guardrails"])
        self.assertFalse(out["robust_enough_for_guarded_shadow"])
        self.assertEqual(out["readiness_recommendation"], "not_ready_more_validation_needed")
        self.assertTrue(out["immediate_disable_triggered"])


if __name__ == "__main__":
    unittest.main()
