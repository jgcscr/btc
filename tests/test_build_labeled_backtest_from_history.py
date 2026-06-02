from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts.build_labeled_backtest_from_history import (
    _build_from_backtest,
    _build_multi_horizon_from_history,
    _load_history_rows,
)


class BuildLabeledBacktestFromHistoryTests(unittest.TestCase):
    def test_load_history_rows_derives_cross_horizon_consensus_features(self) -> None:
        history_payload = [
            {
                "generated_at": "2026-01-01T00:05:00Z",
                "predictions": {
                    "1h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.64,
                        "ret_pred": 0.01,
                        "direction_next": "up",
                    },
                    "4h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.34,
                        "ret_pred": -0.02,
                        "direction_next": "down",
                    },
                    "12h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.30,
                        "ret_pred": -0.03,
                        "direction_next": "down",
                    },
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "history.json"
            history_path.write_text(json.dumps(history_payload), encoding="utf-8")
            loaded = _load_history_rows(history_path, "1h")

        self.assertEqual(len(loaded), 1)
        row = loaded.iloc[0]
        self.assertIn("horizon_consensus_support_ratio", loaded.columns)
        self.assertIn("horizon_weighted_p_up", loaded.columns)
        self.assertAlmostEqual(float(row["horizon_consensus_support_ratio"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(row["horizon_directional_agreement_ratio"]), 1.0 / 3.0)
        self.assertEqual(float(row["horizon_directional_disagreement_count"]), 2.0)
        self.assertEqual(float(row["horizon_bias_conflict"]), 1.0)

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

    def test_load_history_rows_can_include_archived_runtime_runs(self) -> None:
        runtime_payload = {
            "generated_at": "2026-01-03T00:05:00Z",
            "predictions": {
                "4h": {
                    "timestamp": "2026-01-03T00:00:00Z",
                    "p_up": 0.63,
                    "ret_pred": 0.02,
                    "signal_dir_only": 1.0,
                    "expected_value": 0.004,
                    "regime_state": "neutral",
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            history_path = tmp_path / "artifacts" / "predictions" / "history.json"
            runtime_path = tmp_path / "artifacts" / "runtime_runs" / "research-20260103T000500-test" / "predictions.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("[]", encoding="utf-8")
            runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")

            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                loaded = _load_history_rows(
                    Path("artifacts/predictions/history.json"),
                    "4h",
                    include_runtime_runs=True,
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["horizon"], "4h")
        self.assertEqual(loaded.iloc[0]["regime_state"], "neutral")

    def test_build_multi_horizon_history_labels_support_non_1h_only_requests(self) -> None:
        history_payload = [
            {
                "generated_at": "2026-01-01T00:05:00Z",
                "predictions": {
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
                horizons=["4h"],
                spot_ohlcv_path=ohlcv_path,
                fold_size=2,
                lookback_rows=None,
                lookback_hours=None,
            )

        self.assertEqual(set(labeled["horizon"]), {"4h"})
        self.assertEqual(set(labeled["horizon_hours"]), {4.0})
        self.assertIn("ret_realized", labeled.columns)
        self.assertIn("close_target", labeled.columns)
        self.assertEqual(meta["rows_by_horizon"], {"4h": 2})
        self.assertTrue((labeled["ret_realized"] > 0).all())

    def test_build_from_backtest_enriches_component_columns_from_history(self) -> None:
        backtest = pd.DataFrame(
            {
                "ts": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                "ret_1h": [0.01, -0.02],
                "p_up": [0.61, 0.41],
                "p_up_xgb": [0.60, 0.40],
                "y_true": [1, 0],
            }
        )
        history_payload = [
            {
                "generated_at": "2026-01-01T00:05:00Z",
                "predictions": {
                    "1h": {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "p_up": 0.61,
                        "ret_pred": 0.01,
                        "signal_dir_only": 1,
                        "expected_value": 0.002,
                        "regime_state": "trend_ignition",
                        "p_up_components": {
                            "xgb": 0.60,
                            "lstm": 0.62,
                            "bilstm": 0.63,
                            "gru": 0.59,
                            "cnn_lstm": 0.58,
                            "transformer": 0.64,
                        },
                    }
                },
            },
            {
                "generated_at": "2026-01-01T01:05:00Z",
                "predictions": {
                    "1h": {
                        "timestamp": "2026-01-01T01:00:00Z",
                        "p_up": 0.41,
                        "ret_pred": -0.02,
                        "signal_dir_only": 0,
                        "expected_value": -0.001,
                        "regime_state": "neutral",
                        "p_up_components": {
                            "xgb": 0.40,
                            "lstm": 0.42,
                            "bilstm": 0.43,
                            "gru": 0.39,
                            "cnn_lstm": 0.38,
                            "transformer": 0.44,
                        },
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backtest_path = tmp_path / "backtest.csv"
            history_path = tmp_path / "history.json"
            ohlcv_path = tmp_path / "ohlcv.csv"
            backtest.to_csv(backtest_path, index=False)
            history_path.write_text(json.dumps(history_payload), encoding="utf-8")
            pd.DataFrame(
                {
                    "ts": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
                    "close": [100.0, 101.0, 102.0],
                }
            ).to_csv(ohlcv_path, index=False)

            labeled, meta = _build_from_backtest(
                backtest_csv=backtest_path,
                history_path=history_path,
                horizon="1h",
                spot_ohlcv_path=ohlcv_path,
                fold_size=2,
                lookback_rows=None,
                lookback_hours=None,
            )

        self.assertEqual(meta["source"], "backtest_csv")
        self.assertIn("p_up_lstm", labeled.columns)
        self.assertIn("p_up_bilstm", labeled.columns)
        self.assertIn("p_up_gru", labeled.columns)
        self.assertIn("p_up_cnn_lstm", labeled.columns)
        self.assertIn("p_up_transformer", labeled.columns)
        self.assertAlmostEqual(float(labeled.loc[0, "p_up_lstm"]), 0.62)
        self.assertAlmostEqual(float(labeled.loc[1, "p_up_transformer"]), 0.44)


if __name__ == "__main__":
    unittest.main()