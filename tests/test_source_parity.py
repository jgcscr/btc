from __future__ import annotations

import unittest

import pandas as pd

from src.data.source_parity import (
    drop_unready_source_family_features,
    evaluate_source_family_readiness,
)


class SourceParityTests(unittest.TestCase):
    def test_unready_source_families_are_dropped_from_allowed_features(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
                "macro_us10y": [None, None, None, None],
                "onchain_active_addresses": [None, None, None, None],
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        )

        readiness = evaluate_source_family_readiness(frame, recent_rows=4)
        filtered, dropped = drop_unready_source_family_features(
            ["close", "macro_us10y", "onchain_active_addresses"],
            readiness,
        )

        self.assertEqual(filtered, ["close"])
        self.assertIn("macro", dropped)
        self.assertIn("onchain", dropped)


if __name__ == "__main__":
    unittest.main()