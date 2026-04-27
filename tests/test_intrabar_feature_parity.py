from __future__ import annotations

import unittest

import pandas as pd

from src.trading.intrabar_features import compute_hourly_intrabar_features


class IntrabarFeatureParityTests(unittest.TestCase):
    def test_compute_hourly_intrabar_features_emits_shared_schema(self) -> None:
        ts = pd.date_range("2026-03-10T00:15:00Z", periods=8, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "ts": ts,
                "open": [100, 101, 102, 103, 104, 105, 106, 107],
                "high": [101, 102, 103, 104, 105, 106, 107, 108],
                "low": [99, 100, 101, 102, 103, 104, 105, 106],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
                "volume": [10, 11, 12, 13, 10, 11, 12, 13],
                "quote_volume": [1000, 1100, 1200, 1300, 1050, 1150, 1250, 1350],
                "num_trades": [1, 2, 3, 4, 1, 2, 3, 4],
                "taker_buy_base_volume": [5, 6, 7, 8, 5, 6, 7, 8],
                "taker_buy_quote_volume": [500, 600, 700, 800, 520, 620, 720, 820],
            }
        )

        output = compute_hourly_intrabar_features(frame)

        self.assertEqual(len(output), 2)
        self.assertIn("intrabar_path_range", output.columns)
        self.assertIn("intrabar_taker_imbalance_persistence", output.columns)
        self.assertIn("intrabar_path_efficiency_1h", output.columns)
        self.assertIn("intrabar_flow_acceleration_3h", output.columns)
        self.assertIn("intrabar_taker_imbalance_early_late_delta", output.columns)
        self.assertIn("intrabar_reversal_score_1h", output.columns)
        self.assertIn("intrabar_wick_asymmetry_shift", output.columns)
        self.assertIn("intrabar_breakout_failure_1h", output.columns)
        self.assertIn("intrabar_return_dispersion_regime_3h", output.columns)
        self.assertIn("intrabar_return_dispersion_regime_6h", output.columns)

    def test_compute_hourly_intrabar_features_emits_transition_signals(self) -> None:
        ts = pd.date_range("2026-03-10T00:15:00Z", periods=4, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "ts": ts,
                "open": [100.0, 102.0, 104.0, 101.0],
                "high": [102.0, 105.0, 105.0, 102.0],
                "low": [99.0, 101.0, 100.0, 97.0],
                "close": [102.0, 104.0, 101.0, 98.0],
                "volume": [10.0, 10.0, 10.0, 10.0],
                "quote_volume": [1000.0, 1020.0, 1010.0, 980.0],
                "num_trades": [10.0, 10.0, 10.0, 10.0],
                "taker_buy_base_volume": [8.0, 8.0, 2.0, 2.0],
                "taker_buy_quote_volume": [800.0, 816.0, 202.0, 196.0],
            }
        )

        output = compute_hourly_intrabar_features(frame)

        self.assertEqual(len(output), 1)
        row = output.iloc[0]
        self.assertLess(row["intrabar_taker_imbalance_early_late_delta"], 0.0)
        self.assertLess(row["intrabar_reversal_score_1h"], 0.0)
        self.assertNotEqual(row["intrabar_wick_asymmetry_shift"], 0.0)
        self.assertGreater(row["intrabar_breakout_failure_1h"], 0.0)