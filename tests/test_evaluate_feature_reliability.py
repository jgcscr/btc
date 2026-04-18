from __future__ import annotations

import unittest

import pandas as pd

from src.scripts.evaluate_feature_reliability import _feature_score, _resolve_target_series


class EvaluateFeatureReliabilityTests(unittest.TestCase):
    def test_resolve_target_series_prefers_realized_target_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, 4.0] * 8,
                "ret_4h_realized": [0.1, -0.1, 0.2, -0.2] * 8,
            }
        )

        target, name = _resolve_target_series(frame)

        self.assertEqual(name, "ret_4h_realized")
        self.assertIsNotNone(target)

    def test_feature_score_includes_predictive_strength_when_target_available(self) -> None:
        feature = pd.Series([0.0, 0.0, 1.0, 1.0] * 20)
        target = pd.Series([0.0, 0.0, 1.0, 1.0] * 20)

        scored = _feature_score(feature, baseline_window=24, recent_window=24, target=target)

        self.assertGreater(scored["predictive_strength_recent"], 0.5)
        self.assertGreater(scored["score"], 0.5)


if __name__ == "__main__":
    unittest.main()