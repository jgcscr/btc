from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.runtime.family_outcome_confirmation import (
    _decision_outcome,
    _evaluate_go_hold,
    _promotion_guardrails,
    _top_variants_from_shadow_artifact,
    load_spot_ohlcv_with_outcomes,
)


class FamilyOutcomeConfirmationTests(unittest.TestCase):
    def test_load_spot_outcomes_builds_realized_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {
                    "ts": pd.date_range("2026-04-01", periods=20, freq="h", tz="UTC"),
                    "open": [100 + i for i in range(20)],
                    "high": [101 + i for i in range(20)],
                    "low": [99 + i for i in range(20)],
                    "close": [100 + i for i in range(20)],
                    "volume": [1.0] * 20,
                }
            )
            frame.to_parquet(root / "spot.parquet", index=False)

            out = load_spot_ohlcv_with_outcomes(root, horizons=["4h", "8h"])
            self.assertIn("ret_4h_realized", out.columns)
            self.assertIn("ret_8h_realized", out.columns)
            self.assertIn("high_fwd_4h", out.columns)
            self.assertIn("low_fwd_8h", out.columns)

    def test_decision_outcome_signed_return(self) -> None:
        ts = pd.Timestamp("2026-04-03T10:00:00Z")
        lookup = {
            (ts, "4h"): {
                "entry_close": 100.0,
                "close_next": 110.0,
                "ret_realized": 0.10,
                "high_fwd": 112.0,
                "low_fwd": 98.0,
            }
        }

        long_outcome = _decision_outcome(
            strategy={"tradeable": True, "selected_direction": "Long", "preferred_horizon": "4h"},
            generated_at=ts,
            outcome_lookup=lookup,
        )
        short_outcome = _decision_outcome(
            strategy={"tradeable": True, "selected_direction": "Short", "preferred_horizon": "4h"},
            generated_at=ts,
            outcome_lookup=lookup,
        )

        self.assertTrue(long_outcome.has_trade)
        self.assertAlmostEqual(float(long_outcome.signed_return), 0.10, places=6)
        self.assertAlmostEqual(float(short_outcome.signed_return), -0.10, places=6)

    def test_top_variant_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.json"
            path.write_text(
                """
                {
                  "sweep": {
                    "families": {
                      "order_flow": {
                        "variant_rankings": [
                          {"policy": "a"},
                          {"policy": "b"},
                          {"policy": "c"}
                        ]
                      },
                      "state_engineering": {
                        "variant_rankings": [
                          {"policy": "x"},
                          {"policy": "y"},
                          {"policy": "z"}
                        ]
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )
            mapping = _top_variants_from_shadow_artifact(path, top_n=2)
            self.assertEqual(mapping["order_flow"], ["a", "b"])
            self.assertEqual(mapping["state_engineering"], ["x", "y"])

    def test_go_hold_guardrail_logic(self) -> None:
        guardrails = _promotion_guardrails()
        summary = {
            "shadow": {"trade_count": 200},
            "overall_delta": {
                "net_return_proxy_mean_delta": 0.001,
                "direction_accuracy_proxy_delta": 0.03,
            },
            "veto_count": 100,
            "removed_good_trade_count": 20,
            "veto_precision": 0.8,
            "positive_target_horizon_count": 2,
            "positive_target_regime_count": 2,
        }
        decision = _evaluate_go_hold(summary, guardrails)
        self.assertEqual(decision["decision"], "go")


if __name__ == "__main__":
    unittest.main()
