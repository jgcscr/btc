from __future__ import annotations

import unittest

import pandas as pd

from src.data.derivatives_loader import _assemble_derivatives_feature_frame


class DerivativesLoaderTests(unittest.TestCase):
    def test_assemble_derivatives_feature_frame_aligns_and_annualizes(self) -> None:
        futures = pd.DataFrame(
            {
                "ts": pd.to_datetime([
                    "2026-04-01T00:59:59.999Z",
                    "2026-04-01T01:59:59.999Z",
                ], utc=True),
                "fut_open": [100.0, 101.0],
                "fut_high": [101.0, 102.0],
                "fut_low": [99.5, 100.5],
                "fut_close": [100.5, 101.5],
                "fut_volume": [10.0, 11.0],
            }
        )
        open_interest = pd.DataFrame(
            {
                "ts": pd.to_datetime([
                    "2026-04-01T00:59:59.999Z",
                    "2026-04-01T01:59:59.999Z",
                ], utc=True),
                "open_interest": [1000.0, 1010.0],
            }
        )
        funding = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-04-01T00:59:59.999Z"], utc=True),
                "funding_rate": [0.0001],
            }
        )

        out = _assemble_derivatives_feature_frame(futures, open_interest, funding)

        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out.loc[1, "funding_rate"]), 0.0001, places=8)
        self.assertAlmostEqual(float(out.loc[0, "funding_rate_annualized"]), 0.1095, places=6)
        self.assertAlmostEqual(float(out.loc[1, "open_interest"]), 1010.0, places=6)


if __name__ == "__main__":
    unittest.main()