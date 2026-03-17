from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts.build_labeled_backtest_from_history import _build_multi_horizon_from_history, _load_history_rows


class BuildLabeledBacktestFromHistoryTests(unittest.TestCase):
    def test_build_multi_horizon_history_labels_include_explicit_horizons(self) -> None:
        history_payload = [
            {
                "generated_at": "2026-01-01T00:05:00Z",
                "predictions": {
                    "1h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.61,
                        "ret_pred": 0.01,
                        "signal_dir_only": 1.0,
                        "expected_value": 0.002,
                        "regime_state": "trend_ignition",
                    },
                    "4h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.58,
                        "ret_pred": 0.02,
                        "signal_dir_only": 1.0,
                        "expected_value": 0.003,
                        "regime_state": "trend_ignition",
                    },
                },
            },
            {
                "generated_at": "2026-01-01T01:05:00Z",
                "predictions": {
                    "1h": {
                        "timestamp": "2026-01-01T01:00:00Z",
                        "p_up": 0.43,
                        "ret_pred": -0.01,
                        "signal_dir_only": 0.0,
                        "expected_value": -0.001,
                        "regime_state": "neutral",
                    },
                    "4h": {
                        "timestamp": "2026-01-01T01:00:00Z",
                        "p_up": 0.47,
                        "ret_pred": 0.015,
                        "signal_dir_only": 1.0,
                        "expected_value": 0.002,
                        "regime_state": "neutral",
                    },
                },
            },
        ]
        ohlcv = pd.DataFrame(
            {
                "ts": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                    "2026-01-01T04:00:00Z",
                    "2026-01-01T05:00:00Z",
                ],
                "close": [100.0, 99.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            history_path = tmp_path / "history.json"
            ohlcv_path = tmp_path / "ohlcv.csv"
            history_path.write_text(json.dumps(history_payload), encoding="utf-8")
            ohlcv.to_csv(ohlcv_path, index=False)

            labeled, meta = _build_multi_horizon_from_history(
                history_path=history_path,
                horizons=["1h", "4h"],
                spot_ohlcv_path=ohlcv_path,
                fold_size=2,
                lookback_rows=None,
                lookback_hours=None,
            )

        self.assertEqual(set(labeled["horizon"]), {"1h", "4h"})
        self.assertEqual(set(labeled["horizon_hours"]), {1.0, 4.0})
        self.assertIn("ret_realized", labeled.columns)
        self.assertIn("close_target", labeled.columns)
        self.assertEqual(meta["source"], "history_plus_ohlcv_multi_horizon")
        self.assertEqual(meta["rows_by_horizon"], {"1h": 2, "4h": 2})
        self.assertTrue((labeled.loc[labeled["horizon"] == "4h", "ret_realized"] > 0).all())

    def test_load_history_rows_can_include_archived_reliability_snapshots(self) -> None:
        snapshot_payload = {
            "generated_at": "2026-01-02T00:05:00Z",
            "horizons": {
                "1h": {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "p_up": 0.62,
                    "ret_pred": 0.01,
                    "signal_dir_only": 1.0,
                    "expected_value": 0.002,
                    "regime_state": "neutral",
                },
                "4h": {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "p_up": 0.59,
                    "ret_pred": 0.015,
                    "signal_dir_only": 1.0,
                    "expected_value": 0.003,
                    "regime_state": "trend_ignition",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            history_path = tmp_path / "artifacts" / "predictions" / "history.json"
            snapshot_path = tmp_path / "artifacts" / "reliability" / "20260102T000500Z" / "summary" / "live_predictions_snapshot.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("[]", encoding="utf-8")
            snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                loaded = _load_history_rows(
                    Path("artifacts/predictions/history.json"),
                    "4h",
                    include_reliability_snapshots=True,
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["horizon"], "4h")
        self.assertEqual(loaded.iloc[0]["regime_state"], "trend_ignition")


if __name__ == "__main__":
    unittest.main()