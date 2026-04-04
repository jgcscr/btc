from __future__ import annotations

import unittest

from src.runtime.state_guarded_shadow import _build_fail_close_summary, _validate_guarded_scope


class StateGuardedShadowTests(unittest.TestCase):
    def test_validate_guarded_scope_requires_exact_4h_scope(self) -> None:
        payload = {
            "final_recommendation": {"decision": "continue_narrow_scope_validation"},
            "best_candidate": {"scope": "horizon=4h"},
        }
        out = _validate_guarded_scope(payload)
        self.assertTrue(out["ready_for_guarded_shadow"])
        self.assertEqual(out["blockers"], [])

    def test_validate_guarded_scope_blocks_non_4h_scope(self) -> None:
        payload = {
            "final_recommendation": {"decision": "continue_narrow_scope_validation"},
            "best_candidate": {"scope": "horizon=4h | regime=chop"},
        }
        out = _validate_guarded_scope(payload)
        self.assertFalse(out["ready_for_guarded_shadow"])
        self.assertIn("best_narrow_scope_is_not_exactly_horizon_4h", out["blockers"])

    def test_build_fail_close_summary_uses_stale_and_unavailable_counts(self) -> None:
        out = _build_fail_close_summary(
            {
                "state_status_counts": {
                    "available": 10,
                    "stale": 3,
                    "unavailable": 2,
                }
            }
        )
        self.assertEqual(out["disabled_snapshot_count"], 5)
        self.assertEqual(out["disabled_stale_snapshot_count"], 3)
        self.assertEqual(out["disabled_unavailable_snapshot_count"], 2)


if __name__ == "__main__":
    unittest.main()