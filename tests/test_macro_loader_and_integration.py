from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.data import macro_loader
from src.scripts import build_training_dataset


class MacroLoaderTests(unittest.TestCase):
    def test_fetch_fred_series_parses_public_csv(self) -> None:
        csv_text = "observation_date,DGS10\n2026-03-25,4.25\n2026-03-26,4.30\n"

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = csv_text

        class _Session:
            def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
                return _Response()

        frame = macro_loader.fetch_fred_series(
            "DGS10",
            column="macro_us10y",
            start_date="2026-03-25",
            end_date="2026-03-26",
            session=_Session(),
        )

        self.assertEqual(list(frame.columns), ["macro_source_date", "macro_us10y"])
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(float(frame["macro_us10y"].iloc[-1]), 4.30)

    @patch("src.data.macro_loader.fetch_frankfurter_eurusd")
    @patch("src.data.macro_loader.fetch_fred_series")
    def test_build_macro_feature_frame_adds_safe_effective_timestamp(
        self,
        mock_fetch_fred_series,
        mock_fetch_frankfurter,
    ) -> None:
        source_dates = pd.date_range("2026-03-01", periods=40, freq="D", tz="UTC")

        def _fred_side_effect(series_id: str, **kwargs) -> pd.DataFrame:  # noqa: ANN003
            column = kwargs["column"]
            base = 100.0 if column == "macro_dollar_proxy" else 4.0
            return pd.DataFrame(
                {
                    "macro_source_date": source_dates,
                    column: [base + idx for idx in range(len(source_dates))],
                }
            )

        mock_fetch_fred_series.side_effect = _fred_side_effect
        mock_fetch_frankfurter.return_value = pd.DataFrame(
            {
                "macro_source_date": source_dates,
                "macro_eurusd": [1.05 + idx * 0.001 for idx in range(len(source_dates))],
            }
        )

        frame = macro_loader.build_macro_feature_frame(
            start_date="2026-03-01",
            end_date="2026-04-09",
        )

        self.assertEqual(frame["ts"].iloc[0], source_dates[0] + pd.Timedelta(days=1))
        self.assertIn("macro_us10y_change_1d", frame.columns)
        self.assertIn("macro_dollar_proxy_zscore_30d", frame.columns)
        self.assertIn("macro_eurusd_trend_5d", frame.columns)
        self.assertTrue(frame["macro_us10y_zscore_30d"].iloc[-1] == frame["macro_us10y_zscore_30d"].iloc[-1])

    def test_resolve_incremental_start_date_uses_overlap(self) -> None:
        existing = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-03-11T00:00:00Z", "2026-03-12T00:00:00Z"], utc=True),
                "macro_source_date": pd.to_datetime(["2026-03-10T00:00:00Z", "2026-03-11T00:00:00Z"], utc=True),
                "macro_us10y": [4.1, 4.2],
                "macro_dollar_proxy": [100.0, 100.1],
                "macro_eurusd": [1.08, 1.09],
            }
        )

        start_date = macro_loader.resolve_incremental_start_date(existing, overlap_days=5)

        self.assertEqual(start_date, "2026-03-06")


class MacroIntegrationTests(unittest.TestCase):
    def test_drop_external_source_columns_preserves_approved_macro_features(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-03-10T00:00:00Z"], utc=True),
                "close": [100.0],
                "macro_source_date": pd.to_datetime(["2026-03-09T00:00:00Z"], utc=True),
                "macro_us10y": [4.2],
                "macro_us10y_change_1d": [0.1],
                "macro_unapproved_raw": [123.0],
            }
        )

        cleaned = build_training_dataset._drop_external_source_columns(frame)

        self.assertIn("macro_us10y", cleaned.columns)
        self.assertIn("macro_us10y_change_1d", cleaned.columns)
        self.assertNotIn("macro_source_date", cleaned.columns)
        self.assertNotIn("macro_unapproved_raw", cleaned.columns)


if __name__ == "__main__":
    unittest.main()
