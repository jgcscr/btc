from __future__ import annotations

import argparse
import unittest

import pandas as pd

from src.scripts.train_trade_decision_model import (
    FEATURE_COLUMNS,
    _apply_reference_feature_controls,
)


class TradeDecisionModelControlTests(unittest.TestCase):
    def test_disable_on_source_mismatch_zeros_reference_features(self) -> None:
        X = pd.DataFrame({column: [0.0, 0.0] for column in FEATURE_COLUMNS})
        X["incumbent_signal_reference"] = [1.0, 0.0]
        X["candidate_only_reference"] = [0.0, 1.0]
        X["candidate_incumbent_disagreement"] = [1.0, 1.0]
        args = argparse.Namespace(
            feature_meta_path=None,
            reference_feature_mode="disable_on_source_mismatch",
            reference_feature_expected_source="expected.csv",
            reference_feature_max_abs_value=None,
        )

        adjusted, meta = _apply_reference_feature_controls(
            X,
            feature_meta={"incumbent_reference": {"source": "other.csv"}},
            args=args,
        )

        self.assertTrue(meta["disabled"])
        self.assertEqual(meta["disable_reason"], "source_mismatch")
        self.assertTrue((adjusted["incumbent_signal_reference"] == 0.0).all())
        self.assertTrue((adjusted["candidate_only_reference"] == 0.0).all())
        self.assertTrue((adjusted["candidate_incumbent_disagreement"] == 0.0).all())

    def test_max_abs_value_clips_reference_features(self) -> None:
        X = pd.DataFrame({column: [0.0, 0.0] for column in FEATURE_COLUMNS})
        X["incumbent_signal_reference"] = [2.0, -2.0]
        X["candidate_only_reference"] = [1.5, -1.5]
        X["candidate_incumbent_disagreement"] = [0.25, -0.25]
        args = argparse.Namespace(
            feature_meta_path=None,
            reference_feature_mode="allow",
            reference_feature_expected_source=None,
            reference_feature_max_abs_value=0.5,
        )

        adjusted, meta = _apply_reference_feature_controls(
            X,
            feature_meta={"incumbent_reference": {"source": "expected.csv"}},
            args=args,
        )

        self.assertFalse(meta["disabled"])
        self.assertEqual(meta["clipped_columns"], [
            "incumbent_signal_reference",
            "candidate_only_reference",
            "candidate_incumbent_disagreement",
        ])
        self.assertEqual(adjusted["incumbent_signal_reference"].tolist(), [0.5, -0.5])
        self.assertEqual(adjusted["candidate_only_reference"].tolist(), [0.5, -0.5])
        self.assertEqual(adjusted["candidate_incumbent_disagreement"].tolist(), [0.25, -0.25])


if __name__ == "__main__":
    unittest.main()
