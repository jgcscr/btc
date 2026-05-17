from __future__ import annotations

import unittest

import pandas as pd

from src.scripts import build_training_dataset
from src.scripts import build_training_dataset_multi_horizon


class FeatureLeakageGuardTests(unittest.TestCase):
    def test_hourly_excluded_features_cover_forward_returns(self) -> None:
        self.assertTrue({"ret_4h", "ret_8h", "ret_12h"}.issubset(build_training_dataset.EXCLUDED_FEATURES))

    def test_hourly_core_model_features_exclude_forward_targets_and_labels(self) -> None:
        forbidden = {
            "ret_1h",
            "ret_4h",
            "ret_8h",
            "ret_12h",
            build_training_dataset.TREND_IGNITION_LABEL,
        }
        self.assertTrue(forbidden.isdisjoint(build_training_dataset.CORE_MODEL_FEATURES))

    def test_hourly_drop_excluded_features_removes_forward_returns(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [1.0],
                "ret_1h": [0.1],
                "ret_4h": [0.2],
                "ret_8h": [0.3],
                "ret_12h": [0.4],
            }
        )
        filtered = build_training_dataset._drop_excluded_features(frame)
        self.assertEqual(list(filtered.columns), ["open"])

    def test_hourly_drop_excluded_features_preserves_funding_zscore(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [1.0],
                "funding_rate_zscore_24h": [0.25],
                "ret_1h": [0.1],
            }
        )
        filtered = build_training_dataset._drop_excluded_features(frame)
        self.assertEqual(list(filtered.columns), ["open", "funding_rate_zscore_24h"])

    def test_multi_horizon_excluded_features_cover_forward_returns(self) -> None:
        self.assertTrue({"ret_4h", "ret_8h", "ret_12h"}.issubset(build_training_dataset_multi_horizon.EXCLUDED_FEATURES))

    def test_multi_horizon_core_model_features_exclude_forward_returns(self) -> None:
        self.assertTrue(
            {"ret_1h", "ret_4h", "ret_8h", "ret_12h"}.isdisjoint(
                build_training_dataset_multi_horizon.CORE_MODEL_FEATURES
            )
        )

    def test_multi_horizon_drop_excluded_features_removes_forward_returns(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [1.0],
                "ret_1h": [0.1],
                "ret_4h": [0.2],
                "ret_8h": [0.3],
                "ret_12h": [0.4],
            }
        )
        filtered = build_training_dataset_multi_horizon._drop_excluded_features(frame)
        self.assertEqual(list(filtered.columns), ["close"])

    def test_multi_horizon_drop_excluded_features_preserves_funding_zscore(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [1.0],
                "funding_rate_zscore_24h": [0.25],
                "ret_1h": [0.1],
            }
        )
        filtered = build_training_dataset_multi_horizon._drop_excluded_features(frame)
        self.assertEqual(list(filtered.columns), ["close", "funding_rate_zscore_24h"])

    def test_hourly_trend_ignition_label_is_not_allowed_feature(self) -> None:
        self.assertNotIn(build_training_dataset.TREND_IGNITION_LABEL, build_training_dataset.CORE_MODEL_FEATURES)


if __name__ == "__main__":
    unittest.main()
