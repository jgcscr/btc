from __future__ import annotations

import unittest

from src.runtime.family_outcome_confirmation import (
    _classify_rolling_stability,
    split_rolling_non_overlapping_windows,
)


class OrderFlowRollingStabilityTests(unittest.TestCase):
    def test_split_rolling_windows_uses_full_non_overlapping_chunks(self) -> None:
        snapshots = [{"generated_at": f"t{i}"} for i in range(23)]
        windows = split_rolling_non_overlapping_windows(snapshots, window_size=5)
        self.assertEqual(len(windows), 4)
        self.assertEqual(len(windows[0]["snapshots"]), 5)
        self.assertEqual(windows[0]["snapshots"][0]["generated_at"], "t3")
        self.assertEqual(windows[-1]["snapshots"][-1]["generated_at"], "t22")

    def test_split_rolling_windows_respects_max_windows(self) -> None:
        snapshots = [{"generated_at": f"t{i}"} for i in range(40)]
        windows = split_rolling_non_overlapping_windows(snapshots, window_size=5, max_windows=3)
        self.assertEqual(len(windows), 3)
        self.assertEqual([w["window_label"] for w in windows], ["window_1", "window_2", "window_3"])

    def test_classify_majority_failures_deprioritize(self) -> None:
        window_results = [
            {"go_hold": {"decision": "hold"}},
            {"go_hold": {"decision": "hold"}},
            {"go_hold": {"decision": "hold"}},
            {"go_hold": {"decision": "go"}},
        ]
        failure_clusters = {
            "dominant_failure_regime": "chop",
            "dominant_failure_horizon": "8h",
            "regime_failure_cluster": {"chop": {"share": 0.9}},
            "horizon_failure_cluster": {"8h": {"share": 0.8}},
        }
        positive_dependency = {
            "horizon": {"dominant_positive_share": 0.85, "dominant_positive_bucket": "4h"},
            "regime": {"dominant_positive_share": 0.9, "dominant_positive_bucket": "neutral"},
            "confidence": {"dominant_positive_bucket": "mid"},
        }
        out = _classify_rolling_stability(
            window_results=window_results,
            failure_clusters=failure_clusters,
            positive_dependency=positive_dependency,
        )
        self.assertEqual(out["classification"], "unstable")
        self.assertEqual(out["disposition"], "deprioritize_for_now")

    def test_classify_conditional_narrow_scope(self) -> None:
        window_results = [
            {"go_hold": {"decision": "go"}},
            {"go_hold": {"decision": "hold"}},
            {"go_hold": {"decision": "go"}},
            {"go_hold": {"decision": "hold"}},
            {"go_hold": {"decision": "go"}},
        ]
        failure_clusters = {
            "dominant_failure_regime": "chop",
            "dominant_failure_horizon": "12h",
            "regime_failure_cluster": {"chop": {"share": 0.7}},
            "horizon_failure_cluster": {"12h": {"share": 0.7}},
        }
        positive_dependency = {
            "horizon": {"dominant_positive_share": 0.85, "dominant_positive_bucket": "4h"},
            "regime": {"dominant_positive_share": 0.82, "dominant_positive_bucket": "neutral"},
            "confidence": {"dominant_positive_bucket": "mid"},
        }
        out = _classify_rolling_stability(
            window_results=window_results,
            failure_clusters=failure_clusters,
            positive_dependency=positive_dependency,
        )
        self.assertEqual(out["classification"], "conditionally_stable")
        self.assertEqual(out["disposition"], "narrow_scope_followup_validation")


if __name__ == "__main__":
    unittest.main()
