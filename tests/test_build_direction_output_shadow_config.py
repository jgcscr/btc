from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.scripts.build_direction_output_shadow_config import build_shadow_config


class BuildDirectionOutputShadowConfigTests(unittest.TestCase):
    def test_build_shadow_config_applies_direction_output_policy_and_marginal_rerank(self) -> None:
        base_config = {
            "direction_output_policy": {"enabled": False},
            "regime_model_weights": {
                "enabled": True,
                "neutral": {"1": "xgb:1.0,lstm:1.0"},
                "trend_ignition": {"1": "xgb:1.0,lstm:1.0"},
                "chop": {"1": "xgb:1.0,lstm:1.0"},
            },
            "write_artifacts": True,
        }
        audit_payload = {
            "weight_recommendations": {
                "recommended_weight_spec_1h": "transformer:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:1.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0,transformer_large:0.0"
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_path = tmp_path / "base.yaml"
            calib_path = tmp_path / "direction_output_isotonic_1h.json"
            audit_path = tmp_path / "marginal_audit.json"
            output_path = tmp_path / "shadow.yaml"
            base_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
            calib_path.write_text(json.dumps({"1h": {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}}), encoding="utf-8")
            audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

            meta = build_shadow_config(
                base_config_path=base_path,
                direction_output_calibration_path=calib_path,
                output_path=output_path,
                marginal_audit_path=audit_path,
                neutral_band=0.03,
                horizons=[1.0],
            )
            built = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        self.assertTrue(meta["audit_weights_applied"])
        self.assertEqual(built["direction_output_policy"]["calibration_path"], str(calib_path))
        self.assertEqual(built["direction_output_policy"]["neutral_band"], 0.03)
        self.assertFalse(built["write_artifacts"])
        self.assertTrue(built["direction_output_policy"]["marginal_rerank"]["enabled"])
        self.assertEqual(
            built["direction_output_policy"]["marginal_rerank"]["weight_specs"]["default"],
            audit_payload["weight_recommendations"]["recommended_weight_spec_1h"],
        )
        self.assertEqual(built["regime_model_weights"]["neutral"]["1"], "xgb:1.0,lstm:1.0")

    def test_build_shadow_config_accepts_generic_multi_horizon_weight_keys(self) -> None:
        base_config = {
            "direction_output_policy": {"enabled": False},
            "write_artifacts": True,
        }
        audit_payload = {
            "horizon": "4h",
            "weight_recommendations": {
                "recommended_weight_spec": "transformer:0.0,lstm:1.0,bilstm:0.0,gru:1.5,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0,transformer_large:0.0",
                "recommended_regime_weights": {
                    "neutral": "transformer:0.0,lstm:1.0,bilstm:0.0,gru:1.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0,transformer_large:0.0"
                },
            },
            "marginal_band": {"lower": 0.48, "upper": 0.62},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_path = tmp_path / "base.yaml"
            calib_path = tmp_path / "direction_output_isotonic_4h.json"
            audit_path = tmp_path / "marginal_audit.json"
            output_path = tmp_path / "shadow.yaml"
            base_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
            calib_path.write_text(json.dumps({"4h": {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}}), encoding="utf-8")
            audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

            meta = build_shadow_config(
                base_config_path=base_path,
                direction_output_calibration_path=calib_path,
                output_path=output_path,
                marginal_audit_path=audit_path,
                neutral_band=0.02,
                horizons=[4.0, 8.0],
            )
            built = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        self.assertTrue(meta["audit_weights_applied"])
        self.assertEqual(built["direction_output_policy"]["marginal_rerank"]["horizons"], [4.0, 8.0])
        self.assertEqual(built["direction_output_policy"]["marginal_rerank"]["lower"], 0.48)
        self.assertEqual(
            built["direction_output_policy"]["marginal_rerank"]["weight_specs"]["default"],
            audit_payload["weight_recommendations"]["recommended_weight_spec"],
        )
        self.assertEqual(
            built["direction_output_policy"]["marginal_rerank"]["weight_specs"]["neutral"],
            audit_payload["weight_recommendations"]["recommended_regime_weights"]["neutral"],
        )


if __name__ == "__main__":
    unittest.main()