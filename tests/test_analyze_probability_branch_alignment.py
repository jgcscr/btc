from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.scripts.analyze_probability_branch_alignment import build_probability_branch_alignment_report


class AnalyzeProbabilityBranchAlignmentTests(unittest.TestCase):
    def test_report_separates_calibration_introduced_and_fixed_mismatches(self) -> None:
        history_payload = [
            {
                "generated_at": "2026-03-17T00:00:00Z",
                "predictions": {
                    "1h": {
                        "close": 100.0,
                        "p_up": 0.62,
                        "ret_pred": -0.01,
                        "projected_price": 99.0,
                        "direction_next": "down",
                        "regime_state": "trend_ignition",
                        "probability_calibration": {
                            "applied_key": "1h@trend_ignition",
                            "used_regime_key": True,
                        },
                        "direction_output": {
                            "raw_probability": 0.38,
                        },
                    },
                    "4h": {
                        "close": 100.0,
                        "p_up": 0.41,
                        "ret_pred": -0.02,
                        "projected_price": 98.0,
                        "direction_next": "down",
                        "regime_state": "neutral",
                        "probability_calibration": {
                            "applied_key": "4h",
                            "used_regime_key": False,
                        },
                        "direction_output": {
                            "raw_probability": 0.58,
                        },
                    },
                },
            }
        ]

        with TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            history_path.write_text(__import__("json").dumps(history_payload), encoding="utf-8")
            payload = build_probability_branch_alignment_report(
                history_path,
                horizons=["1h", "4h"],
                neutral_band=0.02,
                recent_window=10,
            )

        self.assertEqual(payload["by_horizon"]["1h"]["ret_alignment_buckets"]["calibration_introduced_mismatch"]["count"], 1)
        self.assertEqual(payload["by_horizon"]["4h"]["ret_alignment_buckets"]["calibration_fixed_mismatch"]["count"], 1)
        self.assertEqual(payload["by_horizon"]["1h"]["raw_vs_ret_match_rate"], 1.0)
        self.assertEqual(payload["by_horizon"]["1h"]["calibrated_vs_ret_match_rate"], 0.0)
        self.assertEqual(payload["by_horizon"]["4h"]["raw_vs_ret_match_rate"], 0.0)
        self.assertEqual(payload["by_horizon"]["4h"]["calibrated_vs_ret_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()