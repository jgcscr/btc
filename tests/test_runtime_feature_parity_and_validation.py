from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.scripts.audit_feature_parity import audit_feature_parity
from src.scripts.run_refresh_and_predict import (
    _apply_trade_decision_stage,
    _resolve_abstention_policy,
    _resolve_abstention_policy_for_horizon,
    _normalize_threshold_overrides,
    _resolve_confidence_min_for_horizon,
    _resolve_trade_decision_policy,
)
from src.trading.direction_config import apply_weight_overrides
from src.trading.signals import PreparedData, prepare_data_for_signals_from_ohlcv
from src.trading.thresholds import load_calibrated_thresholds


class RuntimeFeatureParityAndValidationTests(unittest.TestCase):
    def test_trade_decision_stage_uses_confluence_features(self) -> None:
        summary = {
            "1h": {
                "regime_state": "neutral",
                "trade_action": "long",
                "signal_ensemble": 1,
                "signal_dir_only": 1,
                "p_up": 0.62,
                "ret_pred": 0.01,
                "expected_value": 0.005,
                "confidence_score": 0.8,
                "position_size": 0.25,
                "confluence_support_ratio": 0.9,
                "confluence_short_term_ratio": 1.0,
                "confluence_mid_term_ratio": 1.0,
                "confluence_direction_matches_dominant": 1.0,
                "gate_trace": [],
            }
        }
        policy = {
            "enabled": True,
            "threshold": 0.55,
            "replace_threshold_rule": True,
            "model": {
                "feature_columns": ["confluence_support_ratio"],
                "coefficients": [8.0],
                "intercept": -4.0,
            },
        }

        updated = _apply_trade_decision_stage(summary, {"1h": {"residual_std": 0.01}}, policy)

        self.assertTrue(updated["1h"]["trade_decision"]["triggered"])
        self.assertEqual(updated["1h"]["trade_decision"]["feature_snapshot"]["confluence_support_ratio"], 0.9)
        self.assertEqual(updated["1h"]["trade_action"], "long")

    def test_trade_decision_stage_respects_upstream_gates(self) -> None:
        summary = {
            "1h": {
                "regime_state": "neutral",
                "trade_action": "hold",
                "signal_ensemble": 0,
                "signal_dir_only": 1,
                "p_up": 0.62,
                "ret_pred": 0.01,
                "expected_value": 0.005,
                "confidence_score": 0.8,
                "position_size": 0.25,
                "confluence_support_ratio": 0.9,
                "forecast_coherence": {"triggered": True},
                "gate_trace": [],
            }
        }
        policy = {
            "enabled": True,
            "threshold": 0.55,
            "replace_threshold_rule": True,
            "model": {
                "feature_columns": ["confluence_support_ratio"],
                "coefficients": [8.0],
                "intercept": -4.0,
            },
        }

        updated = _apply_trade_decision_stage(summary, {"1h": {"residual_std": 0.01}}, policy)

        self.assertFalse(updated["1h"]["trade_decision"]["triggered"])
        self.assertTrue(updated["1h"]["trade_decision"]["blocked"])
        self.assertIn("forecast_coherence_gate", updated["1h"]["trade_decision"]["blocking_reason"])

    def test_trade_decision_stage_applies_horizon_regime_threshold_override(self) -> None:
        summary = {
            "8h": {
                "regime_state": "trend_ignition",
                "trade_action": "hold",
                "signal_ensemble": 0,
                "signal_dir_only": 1,
                "p_up": 0.7,
                "ret_pred": 0.01,
                "expected_value": 0.005,
                "confidence_score": 0.8,
                "position_size": 0.25,
                "confluence_support_ratio": 0.9,
                "confluence_short_term_ratio": 1.0,
                "confluence_mid_term_ratio": 1.0,
                "confluence_direction_matches_dominant": 1.0,
                "gate_trace": [],
            }
        }
        policy = {
            "enabled": True,
            "threshold": 0.55,
            "thresholds_by_horizon_regime": {8.0: {"trend_ignition": 0.4}},
            "replace_threshold_rule": True,
            "model": {
                "feature_columns": ["confluence_support_ratio"],
                "coefficients": [4.0],
                "intercept": -2.0,
            },
        }

        updated = _apply_trade_decision_stage(summary, {"8h": {"residual_std": 0.01}}, policy)

        self.assertTrue(updated["8h"]["trade_decision"]["triggered"])
        self.assertEqual(updated["8h"]["trade_decision"]["threshold_source"], "8h@trend_ignition")
        self.assertAlmostEqual(updated["8h"]["trade_decision"]["threshold"], 0.4)

    def test_resolve_trade_decision_policy_normalizes_horizon_regime_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trade_decision_model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "feature_columns": ["confluence_support_ratio"],
                        "coefficients": [1.0],
                        "intercept": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            resolved = _resolve_trade_decision_policy(
                {
                    "enabled": True,
                    "model_path": str(model_path),
                    "thresholds_by_horizon_regime": {
                        "8": {"trend_ignition": 0.42},
                        "12": {"default": {"threshold": 0.44}},
                    },
                }
            )

        self.assertEqual(resolved["thresholds_by_horizon_regime"][8.0]["trend_ignition"], 0.42)
        self.assertEqual(resolved["thresholds_by_horizon_regime"][12.0]["default"], 0.44)

    def test_resolve_confidence_min_for_horizon_prefers_regime_override(self) -> None:
        value, source = _resolve_confidence_min_for_horizon(
            0.33,
            {8.0: {"trend_ignition": 0.23}},
            horizon=8.0,
            regime_state="trend_ignition",
        )

        self.assertEqual(value, 0.23)
        self.assertEqual(source, "8h@trend_ignition")

    def test_resolve_abstention_policy_for_horizon_prefers_regime_override(self) -> None:
        policy = _resolve_abstention_policy(
            {
                "enabled": True,
                "hold_prob_center": 0.5,
                "hold_prob_band": 0.03,
                "thresholds_by_horizon_regime": {
                    "8": {
                        "trend_ignition": {
                            "hold_prob_band": 0.0,
                        }
                    }
                },
            }
        )

        resolved = _resolve_abstention_policy_for_horizon(
            policy,
            horizon=8.0,
            regime_state="trend_ignition",
        )

        self.assertEqual(resolved["hold_prob_band"], 0.0)

    def test_prepare_data_for_signals_from_ohlcv_supports_subhourly_volatility(self) -> None:
        periods = 120
        ts = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
        close = np.linspace(100.0, 112.0, periods) + np.sin(np.arange(periods) / 5.0)
        frame = pd.DataFrame(
            {
                "ts": ts,
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": np.linspace(10.0, 25.0, periods),
            }
        )

        prepared = prepare_data_for_signals_from_ohlcv(
            frame,
            feature_names=["open", "high", "low", "close", "volume"],
            expected_freq=pd.Timedelta(minutes=15),
            periods_per_hour=4,
        )

        self.assertIn("volatility_realized_24h", prepared.df_all.columns)
        self.assertGreater(float(prepared.df_all["volatility_realized_24h"].iloc[-1]), 0.0)

    def test_apply_weight_overrides_rejects_unknown_keys(self) -> None:
        configs = [{"name": "xgb", "type": "xgb", "weight": 1.0, "path": "foo"}]
        with self.assertRaises(ValueError):
            apply_weight_overrides(configs, "unknown:1.0")

    def test_normalize_threshold_overrides_rejects_duplicate_horizons(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_threshold_overrides(
                {
                    "1": {"p_up_min": 0.5, "ret_min": 0.0},
                    "1.0": {"p_up_min": 0.6, "ret_min": 0.001},
                }
            )

    def test_load_calibrated_thresholds_rejects_duplicate_normalized_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "horizons": {
                            "1": {"p_up_min": 0.5, "ret_min": 0.0},
                            "1.0": {"p_up_min": 0.6, "ret_min": 0.001},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_calibrated_thresholds(path)

    def test_audit_feature_parity_reports_match_for_synthetic_bundle(self) -> None:
        feature_names = ["open", "high", "low", "close", "volume"]
        raw_rows = np.asarray(
            [
                [100.0, 101.0, 99.0, 100.5, 10.0],
                [101.0, 102.0, 100.0, 101.5, 11.0],
                [102.0, 103.0, 101.0, 102.5, 12.0],
            ],
            dtype=float,
        )
        raw_frame = pd.DataFrame(raw_rows, columns=feature_names)
        scaler = StandardScaler().fit(raw_frame.iloc[:2])
        scaled = scaler.transform(raw_frame)
        ts_all = pd.to_datetime([
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                X_train=scaled[:1],
                X_val=scaled[1:2],
                X_test=scaled[2:],
                ts_all=ts_all.to_numpy(dtype="datetime64[ns]"),
                feature_names=np.array(feature_names),
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
            )

            prepared = PreparedData(
                df_all=pd.DataFrame({"ts": ts_all}),
                X_all_ordered=pd.DataFrame(raw_rows, columns=feature_names),
                scaler=scaler,
                feature_names=feature_names,
                volatility_columns=[],
            )

            report = audit_feature_parity(
                dataset_path=str(dataset_path),
                features_path="unused.parquet",
                target_column="ret_1h",
                split_index=2,
                prepared=prepared,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["mismatched_feature_count"], 0)


if __name__ == "__main__":
    unittest.main()