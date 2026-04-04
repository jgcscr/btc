from __future__ import annotations

import unittest

from src.runtime.family_outcome_confirmation import (
    _assess_state_narrow_scope_candidate,
    _state_candidate_rank_tuple,
    _state_narrow_scope_thresholds,
)


class StateEngineeringNarrowScopeTests(unittest.TestCase):
    def test_positive_slice_can_still_fail_on_trade_count(self) -> None:
        assessment = _assess_state_narrow_scope_candidate(
            {
                "shadow": {"snapshot_count": 176, "trade_count": 7},
                "overall_delta": {
                    "net_return_proxy_mean_delta": 0.00164,
                    "direction_accuracy_proxy_delta": 0.0179,
                },
                "veto_count": 6,
                "removed_good_trade_count": 0,
                "veto_precision": 1.0,
                "changed_snapshot_count": 12,
            },
            _state_narrow_scope_thresholds(),
        )

        self.assertEqual(assessment["classification"], "positive_but_too_sparse")
        self.assertFalse(assessment["viable"])
        self.assertFalse(assessment["checks"]["shadow_trade_count"])

    def test_viable_candidate_ranks_above_sparse_positive_candidate(self) -> None:
        viable = {
            "assessment": {
                "classification": "viable",
                "net_return_proxy_mean_delta": 0.0005,
                "shadow_trade_count": 40,
                "snapshot_count": 140,
                "changed_snapshot_count": 30,
            }
        }
        sparse = {
            "assessment": {
                "classification": "positive_but_too_sparse",
                "net_return_proxy_mean_delta": 0.0015,
                "shadow_trade_count": 7,
                "snapshot_count": 176,
                "changed_snapshot_count": 12,
            }
        }

        self.assertGreater(_state_candidate_rank_tuple(viable), _state_candidate_rank_tuple(sparse))


if __name__ == "__main__":
    unittest.main()