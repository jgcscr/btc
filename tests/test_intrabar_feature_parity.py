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