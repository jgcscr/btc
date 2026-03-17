from __future__ import annotations

import unittest

import pandas as pd

from src.scripts.analyze_direction_marginal_calibration import _build_regime_specific_override_analysis


class AnalyzeDirectionMarginalCalibrationTests(unittest.TestCase):
    def test_regime_override_analysis_selects_chop_specific_override_when_ece_improves(self) -> None:
        rows = []
        for idx in range(16):
            label = 1 if idx < 11 else 0
            rows.append(
                {
                    "regime_state": "chop",
                    "y_true": label,
                    "ret_realized": 0.001 if label == 1 else -0.001,
                    "p_up_lstm": 0.92 if label == 1 else 0.08,
                    "p_up_xgb": 0.85,
                }
            )
        for idx in range(20):
            label = idx % 2
            rows.append(
                {
                    "regime_state": "neutral",
                    "y_true": label,
                    "ret_realized": 0.001 if label == 1 else -0.001,
                    "p_up_lstm": 0.55 if label == 1 else 0.45,
                    "p_up_xgb": 0.56 if label == 1 else 0.44,
                }
            )
        frame = pd.DataFrame(rows)

        payload = _build_regime_specific_override_analysis(
            frame,
            fallback_spec="transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:1.5,lgbm:0.0",
            benchmark_specs=None,
            regime_min_rows=15,
            min_ece_improvement=0.01,
        )

        self.assertEqual(
            payload["selected_regime_overrides"],
            {
                "chop": "transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0"
            },
        )
        self.assertFalse(payload["apply_fallback_for_missing_regimes"])
        self.assertTrue(payload["per_regime"]["chop"]["selected"])
        self.assertEqual(
            payload["per_regime"]["chop"]["selection_reason"],
            "ece_improved_without_accuracy_regression",
        )

    def test_regime_override_analysis_respects_current_benchmark_spec(self) -> None:
        rows = []
        for idx in range(16):
            label = 1 if idx < 11 else 0
            rows.append(
                {
                    "regime_state": "neutral",
                    "y_true": label,
                    "ret_realized": 0.001 if label == 1 else -0.001,
                    "p_up_lstm": 0.80 if label == 1 else 0.20,
                    "p_up_xgb": 0.78 if label == 1 else 0.22,
                    "p_up_transformer": 0.66 if label == 1 else 0.34,
                    "p_up_gru": 0.68 if label == 1 else 0.32,
                }
            )
        frame = pd.DataFrame(rows)

        payload = _build_regime_specific_override_analysis(
            frame,
            fallback_spec="transformer:1.0,transformer_large:0.0,lstm:0.0,bilstm:0.0,gru:1.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0",
            benchmark_specs={
                "neutral": "transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:1.5,lgbm:0.0"
            },
            regime_min_rows=15,
            min_ece_improvement=0.01,
        )

        self.assertEqual(payload["selected_regime_overrides"], {})
        self.assertEqual(
            payload["per_regime"]["neutral"]["benchmark_spec"],
            "transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:1.5,lgbm:0.0",
        )
        self.assertFalse(payload["per_regime"]["neutral"]["selected"])


if __name__ == "__main__":
    unittest.main()