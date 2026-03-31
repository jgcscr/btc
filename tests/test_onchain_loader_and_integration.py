from __future__ import annotations

import unittest

import pandas as pd

from src.data import onchain_loader
from src.scripts import build_training_dataset


class OnchainLoaderTests(unittest.TestCase):
    def test_build_onchain_feature_frame_adds_expected_columns(self) -> None:
        ts = pd.date_range("2026-03-01T00:00:00Z", periods=36, freq="h", tz="UTC")
        raw = pd.DataFrame(
            {
                "ts": ts,
                "active_addresses": [1000 + idx for idx in range(len(ts))],
                "new_addresses": [300 + idx for idx in range(len(ts))],
                "transaction_count": [2000 + idx * 2 for idx in range(len(ts))],
                "hashrate": [500 + idx * 0.5 for idx in range(len(ts))],
                "difficulty": [700 + idx for idx in range(len(ts))],
            }
        )

        frame = onchain_loader.build_onchain_feature_frame(raw_frame=raw)

        self.assertEqual(frame["ts"].iloc[0], ts[0])
        self.assertIn("onchain_active_addresses", frame.columns)
        self.assertIn("onchain_active_addresses_change_1h", frame.columns)
        self.assertIn("onchain_transaction_count_zscore_24h", frame.columns)
        self.assertIn("onchain_hashrate_trend_6h", frame.columns)


class OnchainIntegrationTests(unittest.TestCase):
    def test_drop_external_source_columns_preserves_approved_onchain_features(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-03-10T00:00:00Z"], utc=True),
                "close": [100.0],
                "onchain_active_addresses": [1234.0],
                "onchain_active_addresses_change_1h": [12.0],
                "onchain_unapproved_raw": [999.0],
            }
        )

        cleaned = build_training_dataset._drop_external_source_columns(frame)

        self.assertIn("onchain_active_addresses", cleaned.columns)
        self.assertIn("onchain_active_addresses_change_1h", cleaned.columns)
        self.assertNotIn("onchain_unapproved_raw", cleaned.columns)