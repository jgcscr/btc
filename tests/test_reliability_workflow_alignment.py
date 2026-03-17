from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scripts.run_reliability_workflow import (
    _build_regime_max_p_up_shadow,
    _build_regime_abs_ret_pred_floor_shadow,
    _apply_trade_decision_model_shift_guard,
    _augment_selection_guard_candidate_floors,
    _build_champion_gate_alignment_check,
    _build_trade_decision_ablation_comparison,
    _build_trade_decision_model_shift,
    _build_trade_decision_model_shift_guard,
    _derive_selection_calibration_guard_rules,
    _evaluate_selection_calibration_guard_rule_viability,
    _format_reference_feature_ablation_abs_ret_pred_variant_name,
    _format_reference_feature_ablation_neutral_p_up_cap_variant_name,
    _format_reference_feature_ablation_selection_guard_variant_name,
    _format_reference_feature_ablation_threshold_variant_name,
    _extract_trade_decision_reference_source,
    _is_supported_official_shadow_variant,
    _official_shadow_overlap_triggered_trade_diag_path,
    _load_reusable_selection_calibration_guard_rules,
    _resolve_direction_output_shadow_horizons,
    _resolve_trade_decision_model_path_for_variant,
    _resolve_effective_champion_gate,
    _shadow_variant_uses_reference_feature_ablation_model,
    _summarize_trade_decision_stage_distribution,
    _derive_trade_decision_regime_midband_candidate,
    _write_trade_decision_midband_candidate_config,
    _write_upstream_direction_candidate_config,
    _write_direction_output_shadow_config,
)


class ChampionGateAlignmentCheckTests(unittest.TestCase):
    def test_resolve_effective_champion_gate_uses_selected_shadow_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            selected_companion = summary_dir / "champion_challenger_policy_aligned_shadow_reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499_companion.json"
            selected_companion.write_text(
                json.dumps(
                    {
                        "promote": True,
                        "stats": {
                            "mean_diff": 0.0002988664105767271,
                            "pvalue_one_sided": 0.039,
                        },
                    }
                ),
                encoding="utf-8",
            )

            effective_path, effective_payload, resolution = _resolve_effective_champion_gate(
                summary_dir=summary_dir,
                champion_gate_payload={"promote": False, "stats": {}},
                official_shadow_variant="reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
                champion_gate_source="auto",
                policy_aligned_gate_path=selected_companion,
            )

        self.assertEqual(effective_path, selected_companion)
        self.assertTrue(bool(effective_payload and effective_payload.get("promote")))
        self.assertEqual(resolution["selected_source"], "policy_aligned")

    def test_none_variant_ignores_companion_metric_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            result = _build_champion_gate_alignment_check(
                summary_dir=summary_dir,
                official_shadow_variant="none",
                champion_gate_source="auto",
                selection_payload={
                    "candidates": [
                        {
                            "variant": "none",
                            "companion": {
                                "promote": False,
                                "mean_diff": 0.000017471393588651585,
                                "pvalue_one_sided": 0.3795,
                            },
                        }
                    ]
                },
                effective_champion_gate_path=summary_dir / "champion_challenger_gate.json",
                effective_champion_gate_payload={
                    "promote": False,
                    "stats": {
                        "mean_diff": -0.00016703358045914006,
                        "pvalue_one_sided": 0.835,
                    },
                },
                champion_gate_resolution={"selected_source": "labeled"},
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_source"], "labeled")
        self.assertEqual(result["selected_source"], "labeled")
        self.assertEqual(result["errors"], [])

    def test_none_variant_still_enforces_source_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            result = _build_champion_gate_alignment_check(
                summary_dir=summary_dir,
                official_shadow_variant="none",
                champion_gate_source="auto",
                selection_payload={"candidates": []},
                effective_champion_gate_path=summary_dir / "champion_challenger_gate.json",
                effective_champion_gate_payload={"promote": False, "stats": {}},
                champion_gate_resolution={"selected_source": "policy_aligned"},
            )

        self.assertFalse(result["passed"])
        self.assertIn("selected_source_mismatch", result["errors"][0])

    def test_policy_aligned_variant_requires_metric_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            result = _build_champion_gate_alignment_check(
                summary_dir=summary_dir,
                official_shadow_variant="threshold_0p56",
                champion_gate_source="auto",
                selection_payload={
                    "candidates": [
                        {
                            "variant": "threshold_0p56",
                            "companion": {
                                "promote": True,
                                "mean_diff": 0.0004,
                                "pvalue_one_sided": 0.03,
                            },
                        }
                    ]
                },
                effective_champion_gate_path=summary_dir / "champion_challenger_policy_aligned_companion.json",
                effective_champion_gate_payload={
                    "promote": False,
                    "stats": {
                        "mean_diff": 0.0001,
                        "pvalue_one_sided": 0.2,
                    },
                },
                champion_gate_resolution={"selected_source": "policy_aligned"},
            )

        self.assertFalse(result["passed"])
        self.assertTrue(any(error.startswith("effective_promote_mismatch") for error in result["errors"]))
        self.assertTrue(any(error.startswith("effective_mean_diff_mismatch") for error in result["errors"]))
        self.assertTrue(any(error.startswith("effective_pvalue_mismatch") for error in result["errors"]))

    def test_policy_aligned_variant_accepts_selected_shadow_companion_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            selected_companion = summary_dir / "champion_challenger_policy_aligned_shadow_reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499_companion.json"
            result = _build_champion_gate_alignment_check(
                summary_dir=summary_dir,
                official_shadow_variant="reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
                champion_gate_source="auto",
                selection_payload={
                    "candidates": [
                        {
                            "variant": "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
                            "companion": {
                                "promote": True,
                                "mean_diff": 0.0002988664105767271,
                                "pvalue_one_sided": 0.039,
                            },
                        }
                    ]
                },
                effective_champion_gate_path=selected_companion,
                effective_champion_gate_payload={
                    "promote": True,
                    "stats": {
                        "mean_diff": 0.0002988664105767271,
                        "pvalue_one_sided": 0.039,
                    },
                },
                champion_gate_resolution={"selected_source": "policy_aligned"},
                policy_aligned_gate_path=selected_companion,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_gate_path"], str(selected_companion))


class SelectionCalibrationGuardReuseTests(unittest.TestCase):
    def test_resolve_direction_output_shadow_horizons_filters_invalid_values(self) -> None:
        self.assertEqual(
            _resolve_direction_output_shadow_horizons({"horizons": [1, "4", 0, "bad", -2, 1]}),
            [1.0, 4.0],
        )
        self.assertEqual(_resolve_direction_output_shadow_horizons({}), [1.0])

    def test_write_direction_output_shadow_config_writes_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.yaml"
            calib_path = tmp_path / "direction_output_isotonic_1h.json"
            audit_path = tmp_path / "direction_marginal_1h.json"
            output_path = tmp_path / "shadow.yaml"
            meta_path = tmp_path / "shadow_meta.json"
            base_path.write_text(
                "regime_model_weights:\n  enabled: true\n  neutral:\n    '1': 'xgb:1.0,lstm:1.0'\nwrite_artifacts: true\n",
                encoding="utf-8",
            )
            calib_path.write_text(
                json.dumps({"1h": {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}}),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps({"weight_recommendations": {"recommended_weight_spec_1h": "xgb:1.5,lstm:1.0"}}),
                encoding="utf-8",
            )

            payload = _write_direction_output_shadow_config(
                base_config_path=base_path,
                direction_output_calibration_path=calib_path,
                output_path=output_path,
                meta_output_path=meta_path,
                marginal_audit_path=audit_path,
                neutral_band=0.03,
                horizons=[1.0],
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(meta_path.exists())
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_meta["output_path"], str(output_path))
            self.assertTrue(payload["audit_weights_applied"])

    def test_write_upstream_direction_candidate_config_updates_only_1h_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.yaml"
            audit_path = tmp_path / "direction_marginal_1h.json"
            output_path = tmp_path / "candidate.yaml"
            meta_path = tmp_path / "candidate_meta.json"
            base_path.write_text(
                "regime_model_weights:\n"
                "  enabled: true\n"
                "  neutral:\n"
                "    '1': 'xgb:1.0,lstm:1.0'\n"
                "    '4': 'xgb:0.5,lstm:1.5'\n"
                "  trend_ignition:\n"
                "    '1': 'xgb:1.0,lstm:1.0'\n"
                "  chop:\n"
                "    '1': 'xgb:1.0,lstm:1.0'\n",
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "weight_recommendations": {
                            "recommended_weight_spec_1h": "gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0",
                            "recommended_regime_weights_1h": {
                                "neutral": "gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0",
                                "trend_ignition": "gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0",
                                "chop": "gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            payload = _write_upstream_direction_candidate_config(
                base_config_path=base_path,
                marginal_audit_path=audit_path,
                output_path=output_path,
                meta_output_path=meta_path,
                apply_to_paper_live=False,
            )

            rendered = output_path.read_text(encoding="utf-8")
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["internal_direction_weight_update_applied"])
            self.assertEqual(saved_meta["output_path"], str(output_path))
            self.assertIn("'1': gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0", rendered)
            self.assertIn("'4': xgb:0.5,lstm:1.5", rendered)

    def test_write_upstream_direction_candidate_config_can_override_chop_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.yaml"
            audit_path = tmp_path / "direction_marginal_1h.json"
            output_path = tmp_path / "candidate.yaml"
            meta_path = tmp_path / "candidate_meta.json"
            base_path.write_text(
                "regime_model_weights:\n"
                "  enabled: true\n"
                "  neutral:\n"
                "    '1': 'xgb:1.5,lstm:1.5'\n"
                "    '4': 'xgb:0.5,lstm:1.5'\n"
                "  trend_ignition:\n"
                "    '1': 'xgb:1.5,lstm:1.5'\n"
                "  chop:\n"
                "    '1': 'xgb:1.5,lstm:1.5'\n",
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "weight_recommendations": {
                            "recommended_weight_spec_1h": "gru:1.5,lstm:1.0,xgb:0.0,lgbm:0.0",
                            "recommended_regime_weights_1h": {
                                "chop": "transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0"
                            },
                            "apply_fallback_for_missing_regimes": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            payload = _write_upstream_direction_candidate_config(
                base_config_path=base_path,
                marginal_audit_path=audit_path,
                output_path=output_path,
                meta_output_path=meta_path,
                apply_to_paper_live=False,
            )

            rendered = output_path.read_text(encoding="utf-8")
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["internal_direction_weight_update_applied"])
            self.assertFalse(bool(saved_meta["apply_fallback_for_missing_regimes"]))
            self.assertIn("neutral:\n    '1': xgb:1.5,lstm:1.5", rendered)
            self.assertIn("trend_ignition:\n    '1': xgb:1.5,lstm:1.5", rendered)
            self.assertIn("chop:\n    '1': transformer:0.0,transformer_large:0.0,lstm:1.5,bilstm:0.0,gru:0.0,cnn_lstm:0.0,cnn_bilstm:0.0,garch_lstm:0.0,xgb:0.0,lgbm:0.0", rendered)
            self.assertIn("'4': xgb:0.5,lstm:1.5", rendered)

    def test_derive_trade_decision_regime_midband_candidate_builds_chop_band_from_recent_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            candidate_path = tmp_path / "candidate.csv"
            candidate_path.write_text(
                "ts,signal_ensemble,p_up,ret_pred,ret_ensemble_net,regime_state,volatility_realized_24h\n"
                "2026-03-01T00:00:00Z,1,0.412,-0.0002,-0.0020,chop,0.010\n"
                "2026-03-01T01:00:00Z,1,0.497,-0.0030,-0.0010,chop,0.011\n"
                "2026-03-01T02:00:00Z,1,0.556,0.0080,-0.0030,chop,0.012\n"
                "2026-03-01T03:00:00Z,1,0.520,0.0020,0.0040,neutral,0.004\n",
                encoding="utf-8",
            )

            payload = _derive_trade_decision_regime_midband_candidate(
                candidate_path=candidate_path,
                recent_window_rows=10,
                signal_col="signal_ensemble",
                p_col="p_up",
                ret_pred_col="ret_pred",
                return_col="ret_ensemble_net",
                regime_col="regime_state",
                min_regime_rows=2,
                require_overall_regime_negative=True,
            )

            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["selected_regimes"], ["chop"])
            self.assertEqual(payload["p_up_low"], 0.41)
            self.assertEqual(payload["p_up_high"], 0.56)
            self.assertEqual(payload["min_abs_ret_pred"], 0.0002)
            self.assertIsNone(payload["max_abs_ret_pred"])

    def test_write_trade_decision_midband_candidate_config_updates_only_midband_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_path = tmp_path / "base.yaml"
            candidate_path = tmp_path / "candidate.csv"
            output_path = tmp_path / "suppression_candidate.yaml"
            meta_path = tmp_path / "suppression_candidate_meta.json"
            base_path.write_text(
                "trade_decision_policy:\n"
                "  enabled: true\n"
                "  threshold: 0.55\n"
                "  midband_veto:\n"
                "    enabled: true\n"
                "    p_up_low: 0.56\n"
                "    p_up_high: 0.59\n"
                "    high_inclusive: false\n"
                "    min_abs_ret_pred: 0.001\n"
                "    max_abs_ret_pred: null\n"
                "    regime_states:\n"
                "    - chop\n"
                "regime_model_weights:\n"
                "  enabled: true\n"
                "  chop:\n"
                "    '1': 'xgb:1.5,lstm:1.5'\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                "ts,signal_ensemble,p_up,ret_pred,ret_ensemble_net,regime_state,volatility_realized_24h\n"
                "2026-03-01T00:00:00Z,1,0.412,-0.0002,-0.0020,chop,0.010\n"
                "2026-03-01T01:00:00Z,1,0.497,-0.0030,-0.0010,chop,0.011\n"
                "2026-03-01T02:00:00Z,1,0.556,0.0080,-0.0030,chop,0.012\n"
                "2026-03-01T03:00:00Z,1,0.520,0.0020,0.0040,neutral,0.004\n",
                encoding="utf-8",
            )

            payload = _write_trade_decision_midband_candidate_config(
                base_config_path=base_path,
                candidate_path=candidate_path,
                output_path=output_path,
                meta_output_path=meta_path,
            )

            rendered = output_path.read_text(encoding="utf-8")
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["trade_decision_midband_update_applied"])
            self.assertEqual(saved_meta["output_path"], str(output_path))
            self.assertIn("p_up_low: 0.41", rendered)
            self.assertIn("p_up_high: 0.56", rendered)
            self.assertIn("high_inclusive: true", rendered)
            self.assertIn("min_abs_ret_pred: 0.0002", rendered)
            self.assertIn("regime_states:\n    - chop", rendered)
            self.assertIn("'1': xgb:1.5,lstm:1.5", rendered)


    def test_extract_trade_decision_reference_source_reads_incumbent_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            meta_path.write_text(
                json.dumps({"incumbent_reference": {"source": "artifacts/monitoring/labeled_backtest_1h_incumbent.csv"}}),
                encoding="utf-8",
            )

            result = _extract_trade_decision_reference_source(meta_path)

        self.assertEqual(result, "artifacts/monitoring/labeled_backtest_1h_incumbent.csv")

    def test_resolve_trade_decision_model_path_for_variant_prefers_ablation_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            ablation_model = summary_dir / "trade_decision_model_reference_feature_ablation.json"
            base_model = summary_dir / "trade_decision_model.json"
            ablation_model.write_text("{}", encoding="utf-8")
            base_model.write_text("{}", encoding="utf-8")

            result = _resolve_trade_decision_model_path_for_variant(
                summary_dir,
                "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
            )

        self.assertEqual(result, ablation_model)

    def test_official_shadow_overlap_triggered_trade_diag_path_uses_variant_specific_name(self) -> None:
        summary_dir = Path("/tmp/reliability")

        self.assertEqual(
            _official_shadow_overlap_triggered_trade_diag_path(
                summary_dir,
                "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
            ),
            summary_dir
            / "overlap_triggered_trade_diagnostics_shadow_reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499.json",
        )
        self.assertEqual(
            _official_shadow_overlap_triggered_trade_diag_path(summary_dir, "none"),
            summary_dir / "overlap_triggered_trade_diagnostics.json",
        )

    def test_reference_feature_ablation_threshold_variants_are_valid_official_shadows(self) -> None:
        self.assertTrue(_is_supported_official_shadow_variant("reference_feature_ablation_threshold_0p555"))
        self.assertTrue(
            _is_supported_official_shadow_variant(
                "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499"
            )
        )

    def test_reference_feature_ablation_variant_uses_ablation_model(self) -> None:
        self.assertTrue(_shadow_variant_uses_reference_feature_ablation_model("reference_feature_ablation"))
        self.assertTrue(
            _shadow_variant_uses_reference_feature_ablation_model("reference_feature_ablation_threshold_0p6")
        )
        self.assertFalse(_shadow_variant_uses_reference_feature_ablation_model("none"))
        self.assertFalse(_shadow_variant_uses_reference_feature_ablation_model("threshold_0p56"))

    def test_reference_feature_ablation_threshold_variant_name(self) -> None:
        self.assertEqual(
            _format_reference_feature_ablation_threshold_variant_name(0.6),
            "reference_feature_ablation_threshold_0p6",
        )

    def test_reference_feature_ablation_selection_guard_variant_name(self) -> None:
        self.assertEqual(
            _format_reference_feature_ablation_selection_guard_variant_name(0.555),
            "reference_feature_ablation_threshold_0p555_selection_calibration_guard",
        )

    def test_reference_feature_ablation_abs_ret_pred_variant_name(self) -> None:
        self.assertEqual(
            _format_reference_feature_ablation_abs_ret_pred_variant_name(0.555, 0.00212),
            "reference_feature_ablation_threshold_0p555_neutral_abs_ret_pred_floor_0p00212",
        )

    def test_reference_feature_ablation_neutral_p_up_cap_variant_name(self) -> None:
        self.assertEqual(
            _format_reference_feature_ablation_neutral_p_up_cap_variant_name(0.555, 0.499),
            "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499",
        )

    def test_reference_feature_ablation_selection_guard_variant_uses_ablation_model(self) -> None:
        self.assertTrue(
            _shadow_variant_uses_reference_feature_ablation_model(
                "reference_feature_ablation_threshold_0p555_selection_calibration_guard"
            )
        )

    def test_reference_feature_ablation_abs_ret_pred_variant_uses_ablation_model(self) -> None:
        self.assertTrue(
            _shadow_variant_uses_reference_feature_ablation_model(
                "reference_feature_ablation_threshold_0p555_neutral_abs_ret_pred_floor_0p00212"
            )
        )

    def test_reference_feature_ablation_neutral_p_up_cap_variant_uses_ablation_model(self) -> None:
        self.assertTrue(
            _shadow_variant_uses_reference_feature_ablation_model(
                "reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499"
            )
        )

    def test_build_regime_abs_ret_pred_floor_shadow_blocks_only_low_magnitude_neutral_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "candidate.csv"
            output_path = root / "shadow.csv"
            meta_path = root / "shadow_meta.json"
            input_path.write_text(
                """ts,signal_ensemble,ret_ensemble_net,regime_state,ret_pred\n2026-03-01T00:00:00Z,1,0.01,neutral,0.001\n2026-03-01T01:00:00Z,1,0.02,neutral,0.003\n2026-03-01T02:00:00Z,1,0.03,chop,0.0005\n""".strip(),
                encoding="utf-8",
            )

            payload = _build_regime_abs_ret_pred_floor_shadow(
                input_path=input_path,
                output_path=output_path,
                meta_path=meta_path,
                signal_col="signal_ensemble",
                return_col="ret_ensemble_net",
                regime_col="regime_state",
                ret_pred_col="ret_pred",
                regime_state="neutral",
                min_abs_ret_pred=0.002,
            )

            shadow = output_path.read_text(encoding="utf-8")
            meta_text = meta_path.read_text(encoding="utf-8")

        self.assertEqual(payload["trade_count"], 2)
        self.assertIn('"blocked_rows": 1', meta_text)
        self.assertIn(",0,0.0,neutral,0.001", shadow)

    def test_build_regime_max_p_up_shadow_blocks_only_high_probability_neutral_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "candidate.csv"
            output_path = root / "shadow.csv"
            meta_path = root / "shadow_meta.json"
            input_path.write_text(
                """ts,signal_ensemble,ret_ensemble_net,regime_state,p_up\n2026-03-01T00:00:00Z,1,0.01,neutral,0.500\n2026-03-01T01:00:00Z,1,0.02,neutral,0.480\n2026-03-01T02:00:00Z,1,0.03,chop,0.600\n""".strip(),
                encoding="utf-8",
            )

            payload = _build_regime_max_p_up_shadow(
                input_path=input_path,
                output_path=output_path,
                meta_path=meta_path,
                signal_col="signal_ensemble",
                return_col="ret_ensemble_net",
                regime_col="regime_state",
                p_col="p_up",
                regime_state="neutral",
                max_p_up_exclusive=0.499,
            )

            shadow = output_path.read_text(encoding="utf-8")
            meta_text = meta_path.read_text(encoding="utf-8")

        self.assertEqual(payload["trade_count"], 2)
        self.assertIn('"blocked_rows": 1', meta_text)
        self.assertIn(",0,0.0,neutral,0.5", shadow)

    def test_trade_decision_model_shift_guard_fails_on_large_jumps(self) -> None:
        guard = _build_trade_decision_model_shift_guard(
            model_shift_payload={
                "available": True,
                "top_coefficient_deltas": [
                    {"feature": "__intercept__", "coef_delta": -0.35},
                    {"feature": "incumbent_signal_reference", "coef_delta": -0.25},
                    {"feature": "candidate_only_reference", "coef_delta": 0.24},
                    {"feature": "candidate_incumbent_disagreement", "coef_delta": 0.24},
                ],
                "reference_sources": {
                    "current": {"source": "current.csv"},
                    "source": {"source": "source.csv"},
                },
                "counterfactual_threshold_pass": {"source_not_current_count": 85},
            },
            guard_cfg={
                "enabled": True,
                "require_reference_source_stable": True,
                "max_abs_intercept_delta": 0.25,
                "max_abs_reference_coef_delta": 0.15,
                "max_source_not_current_count": 50,
            },
        )

        self.assertTrue(guard["enabled"])
        self.assertFalse(guard["passed"])
        self.assertIn("intercept_delta_ok", guard["failed_checks"])
        self.assertIn("reference_coef_delta_ok", guard["failed_checks"])
        self.assertIn("source_not_current_count_ok", guard["failed_checks"])
        self.assertIn("reference_source_stable", guard["failed_checks"])

    def test_trade_decision_ablation_comparison_reports_stage_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.csv"
            base_model_path = root / "base_model.json"
            ablation_model_path = root / "ablation_model.json"
            candidate_path.write_text(
                """ts,p_up,ret_pred,signal_dir_only,signal_ensemble,regime_state,incumbent_signal_reference,candidate_only_reference,candidate_incumbent_disagreement,volatility_realized_24h,volatility_ewm_24h,volatility_garch_like
2026-03-01T00:00:00Z,0.49,0.001,1,0,neutral,1,0,1,0.01,0.01,0.01
2026-03-01T01:00:00Z,0.49,0.001,1,0,neutral,0,0,0,0.01,0.01,0.01
""".strip(),
                encoding="utf-8",
            )
            base_model_path.write_text(
                json.dumps(
                    {
                        "feature_columns": ["p_up", "incumbent_signal_reference", "candidate_incumbent_disagreement", "regime_is_neutral"],
                        "coefficients": [1.0, -1.0, 0.5, 0.0],
                        "intercept": -0.1,
                        "threshold": 0.55,
                        "reference_feature_controls": {"mode": "allow"},
                        "oof_expected_value": {"bins": [{"p_min": 0.0, "p_max": 1.0, "samples": 10, "mean_ret_net": 0.01}]},
                    }
                ),
                encoding="utf-8",
            )
            ablation_model_path.write_text(
                json.dumps(
                    {
                        "feature_columns": ["p_up", "incumbent_signal_reference", "candidate_incumbent_disagreement", "regime_is_neutral"],
                        "coefficients": [1.0, 0.0, 0.0, 0.0],
                        "intercept": 0.3,
                        "threshold": 0.55,
                        "reference_feature_controls": {"mode": "disable"},
                        "oof_expected_value": {"bins": [{"p_min": 0.0, "p_max": 1.0, "samples": 10, "mean_ret_net": 0.01}]},
                    }
                ),
                encoding="utf-8",
            )

            result = _build_trade_decision_ablation_comparison(
                candidate_path=candidate_path,
                base_model_path=base_model_path,
                ablation_model_path=ablation_model_path,
                trade_policy_cfg={
                    "threshold": 0.55,
                    "replace_threshold_rule": True,
                    "require_direction_ret_alignment": True,
                    "use_oof_expected_value": True,
                    "oof_expected_value_mode": "max_with_raw_calibrated",
                    "enforce_positive_oof_envelope": True,
                    "positive_oof_envelope_mode": "populated_bin_sign",
                    "block_when_no_positive_oof_bin": True,
                    "allow_raw_ev_fallback_when_no_positive_oof_bin": False,
                    "min_expected_net": 0.0,
                    "min_edge_over_fee": 0.0,
                    "midband_veto": {"enabled": False},
                },
                fee_bps=2.0,
                slippage_bps=1.0,
            )

        self.assertTrue(result["available"])
        stage_map = {item["stage"]: item for item in result["stage_deltas"]}
        self.assertEqual(stage_map["threshold_pass"]["base_count"], 1)
        self.assertEqual(stage_map["threshold_pass"]["ablation_count"], 2)
        self.assertEqual(stage_map["threshold_pass"]["count_delta"], 1)

    def test_apply_trade_decision_model_shift_guard_updates_synthetic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_dir = Path(tmpdir)
            result = _apply_trade_decision_model_shift_guard(
                summary_dir=summary_dir,
                promotion_gate_payload={
                    "promote": False,
                    "reason": "champion_challenger_blocked",
                },
                trade_decision_cfg={
                    "model_shift_guard": {
                        "enabled": True,
                        "require_reference_source_stable": True,
                        "max_abs_intercept_delta": 0.25,
                        "max_abs_reference_coef_delta": 0.15,
                        "max_source_not_current_count": 50,
                    }
                },
                model_shift_payload={
                    "available": True,
                    "top_coefficient_deltas": [
                        {"feature": "__intercept__", "coef_delta": -0.35},
                        {"feature": "incumbent_signal_reference", "coef_delta": -0.25},
                    ],
                    "reference_sources": {
                        "current": {"source": "current.csv"},
                        "source": {"source": "source.csv"},
                    },
                    "counterfactual_threshold_pass": {"source_not_current_count": 85},
                },
            )

            guard_path = summary_dir / "trade_decision_model_shift_guard.json"
            self.assertTrue(guard_path.exists())

        self.assertEqual(result["reason"], "trade_decision_model_shift_guard_failed")
        self.assertIn("trade_decision_model_shift_guard", result)
        self.assertIn(
            "trade_decision_model_shift_guard:intercept_delta_ok",
            result["failed_checks"],
        )

    def test_trade_decision_model_shift_captures_reference_source_and_counterfactual_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            current_dir = root / "current"
            source_dir.mkdir()
            current_dir.mkdir()

            csv_text = """ts,p_up,ret_pred,regime_state,incumbent_signal_reference,candidate_only_reference,candidate_incumbent_disagreement
2026-03-01T00:00:00Z,0.48,0.001,neutral,0.0,0.0,0.0
2026-03-01T01:00:00Z,0.49,0.001,neutral,1.0,0.0,1.0
2026-03-01T02:00:00Z,0.50,-0.001,chop,0.0,1.0,1.0
""".strip()
            (source_dir / "features.csv").write_text(csv_text, encoding="utf-8")
            (current_dir / "features.csv").write_text(csv_text, encoding="utf-8")

            source_model = {
                "feature_columns": ["p_up", "incumbent_signal_reference", "candidate_incumbent_disagreement", "regime_is_neutral"],
                "coefficients": [1.0, 0.0, 0.0, 0.0],
                "intercept": 0.3,
                "threshold": 0.55,
            }
            current_model = {
                "feature_columns": ["p_up", "incumbent_signal_reference", "candidate_incumbent_disagreement", "regime_is_neutral"],
                "coefficients": [1.0, -1.0, 0.5, 0.0],
                "intercept": -0.1,
                "threshold": 0.55,
            }
            (source_dir / "trade_decision_model.json").write_text(json.dumps(source_model), encoding="utf-8")
            (current_dir / "trade_decision_model.json").write_text(json.dumps(current_model), encoding="utf-8")
            (source_dir / "feature_meta.json").write_text(
                json.dumps(
                    {
                        "incumbent_reference": {
                            "source": "historical.csv",
                            "rows_with_reference": 0,
                            "candidate_only_rows": 2,
                            "disagreement_rows": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (current_dir / "feature_meta.json").write_text(
                json.dumps(
                    {
                        "incumbent_reference": {
                            "source": "incumbent.csv",
                            "rows_with_reference": 3,
                            "candidate_only_rows": 1,
                            "disagreement_rows": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = _build_trade_decision_model_shift(
                current_candidate_path=current_dir / "features.csv",
                current_model_path=current_dir / "trade_decision_model.json",
                current_feature_meta_path=current_dir / "feature_meta.json",
                source_candidate_path=source_dir / "features.csv",
                source_model_path=source_dir / "trade_decision_model.json",
                source_feature_meta_path=source_dir / "feature_meta.json",
                source_run_id="run-source",
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["reference_sources"]["source"]["source"], "historical.csv")
        self.assertEqual(result["reference_sources"]["current"]["source"], "incumbent.csv")
        self.assertEqual(result["counterfactual_threshold_pass"]["current_rows_under_source_model"], 3)
        self.assertEqual(result["counterfactual_threshold_pass"]["current_rows_under_current_model"], 2)
        self.assertEqual(result["counterfactual_threshold_pass"]["source_not_current_count"], 1)
        self.assertEqual(result["counterfactual_threshold_pass"]["source_not_current_regime_counts"], {"neutral": 1})

    def test_trade_decision_stage_distribution_captures_threshold_alignment_and_midband(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.csv"
            model_path = root / "trade_decision_model.json"
            candidate_path.write_text(
                """ts,p_up,ret_pred,signal_dir_only,signal_ensemble,regime_state,volatility_realized_24h,volatility_ewm_24h,volatility_garch_like
2026-03-01T00:00:00Z,0.40,0.001,1,0,trend_ignition,0.01,0.01,0.01
2026-03-01T01:00:00Z,0.56,0.001,1,0,chop,0.01,0.01,0.01
2026-03-01T02:00:00Z,0.65,-0.001,1,0,neutral,0.01,0.01,0.01
2026-03-01T03:00:00Z,0.70,0.001,1,0,neutral,0.01,0.01,0.01
""".strip(),
                encoding="utf-8",
            )
            model_path.write_text(
                json.dumps(
                    {
                        "feature_columns": ["p_up"],
                        "coefficients": [10.0],
                        "intercept": -5.0,
                        "threshold": 0.55,
                        "oof_expected_value": {
                            "bins": [
                                {"p_min": 0.0, "p_max": 0.6, "mean_ret_net": 0.01, "samples": 10},
                                {"p_min": 0.6, "p_max": 1.0, "mean_ret_net": 0.02, "samples": 10},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = _summarize_trade_decision_stage_distribution(
                candidate_path=candidate_path,
                model_path=model_path,
                trade_policy_cfg={
                    "threshold": 0.55,
                    "replace_threshold_rule": True,
                    "require_direction_ret_alignment": True,
                    "use_oof_expected_value": True,
                    "oof_expected_value_mode": "max_with_raw_calibrated",
                    "enforce_positive_oof_envelope": True,
                    "positive_oof_envelope_mode": "populated_bin_sign",
                    "block_when_no_positive_oof_bin": True,
                    "positive_oof_min_samples": 4,
                    "allow_raw_ev_fallback_when_no_positive_oof_bin": True,
                    "raw_ev_fallback_quantile": 0.9,
                    "raw_ev_fallback_min_edge_over_fee": 0.0,
                    "min_expected_net": 0.0,
                    "min_edge_over_fee": 0.0,
                    "midband_veto": {
                        "enabled": True,
                        "p_up_low": 0.55,
                        "p_up_high": 0.60,
                        "high_inclusive": False,
                        "min_abs_ret_pred": None,
                        "max_abs_ret_pred": None,
                        "regime_states": [],
                    },
                },
                fee_bps=2.0,
                slippage_bps=1.0,
            )

        self.assertTrue(result["available"])
        stages = {item["stage"]: item for item in result["stages"]}
        self.assertEqual(stages["all_rows"]["count"], 4)
        self.assertEqual(stages["threshold_pass"]["count"], 3)
        self.assertEqual(stages["direction_alignment_pass"]["count"], 2)
        self.assertEqual(stages["positive_envelope_pass"]["count"], 2)
        self.assertEqual(stages["policy_midband_pass"]["count"], 1)
        self.assertEqual(stages["triggered"]["count"], 1)
        self.assertEqual(stages["triggered"]["regime_counts"], {"neutral": 1})

    def test_augment_floors_adds_deployed_rule_neighborhood(self) -> None:
        result = _augment_selection_guard_candidate_floors(
            base_floors=[0.46, 0.47, 0.48, 0.49, 0.53],
            reference_rules=[
                {"regime_state": "neutral", "min_p_up": 0.46},
                {"regime_state": "chop", "min_p_up": 0.53},
            ],
            step=0.01,
            lower_steps=2,
            upper_steps=0,
        )

        self.assertIn(0.44, result)
        self.assertIn(0.45, result)
        self.assertIn(0.51, result)
        self.assertIn(0.52, result)

    def test_reuses_last_deployed_guard_when_manifest_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rule_path = root / "selection_calibration_guard_rule_1h.json"
            manifest_path = root / "reliability_promotion_deploy_manifest.json"
            rule_path.write_text(
                """
{
  "enabled": true,
  "regime_col": "regime_state",
  "p_col": "p_up",
  "rules": [
    {"regime_state": "neutral", "min_p_up": 0.46},
    {"regime_state": "chop", "min_p_up": 0.53}
  ]
}
""".strip(),
                encoding="utf-8",
            )
            manifest_path.write_text(
                '{"run_id": "20260315T062250Z", "official_shadow_variant": "selection_calibration_guard"}',
                encoding="utf-8",
            )

            result = _load_reusable_selection_calibration_guard_rules(
                deployed_rule_path=rule_path,
                deploy_manifest_path=manifest_path,
                expected_regime_col="regime_state",
                expected_p_col="p_up",
            )

        self.assertTrue(result["enabled"])
        self.assertEqual(result["reason"], "reused_last_deployed")
        self.assertEqual(len(result["rules"]), 2)
        self.assertIsNone(result.get("source_candidate_path"))

    def test_does_not_reuse_guard_when_last_deployed_variant_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rule_path = root / "selection_calibration_guard_rule_1h.json"
            manifest_path = root / "reliability_promotion_deploy_manifest.json"
            rule_path.write_text(
                '{"enabled": true, "regime_col": "regime_state", "p_col": "p_up", "rules": [{"regime_state": "neutral", "min_p_up": 0.46}]}',
                encoding="utf-8",
            )
            manifest_path.write_text(
                '{"run_id": "20260315T070000Z", "official_shadow_variant": "none"}',
                encoding="utf-8",
            )

            result = _load_reusable_selection_calibration_guard_rules(
                deployed_rule_path=rule_path,
                deploy_manifest_path=manifest_path,
                expected_regime_col="regime_state",
                expected_p_col="p_up",
            )

        self.assertFalse(result["enabled"])
        self.assertEqual(result["reason"], "last_deployed_variant_mismatch")

    def test_reused_guard_viability_rejects_over_pruned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.csv"
            candidate_path.write_text(
                """ts,signal_ensemble,ret_ensemble_net,regime_state,p_up,y_true
2026-03-01T00:00:00Z,1,0.01,chop,0.52,1
2026-03-01T01:00:00Z,1,-0.02,chop,0.51,0
2026-03-01T02:00:00Z,1,0.01,chop,0.50,1
2026-03-01T03:00:00Z,1,0.01,neutral,0.45,1
2026-03-01T04:00:00Z,1,-0.01,neutral,0.44,0
2026-03-01T05:00:00Z,1,0.01,neutral,0.43,1
""".strip(),
                encoding="utf-8",
            )

            result = _evaluate_selection_calibration_guard_rule_viability(
                candidate_path=candidate_path,
                rules=[
                    {"regime_state": "chop", "min_p_up": 0.53},
                    {"regime_state": "neutral", "min_p_up": 0.46},
                ],
                recent_window_rows=3,
                baseline_window_rows=3,
                signal_col="signal_ensemble",
                return_col="ret_ensemble_net",
                regime_col="regime_state",
                p_col="p_up",
                y_col="y_true",
                min_selection_rows=2,
                adaptive_selection_cfg={"enabled": True, "min_floor": 1, "baseline_ratio": 0.6, "max_shortfall": 1},
                min_candidate_trades=2,
            )

        self.assertFalse(result["enabled"])
        self.assertEqual(result["reason"], "guard_reuse_not_viable")
        self.assertIn("recent_selection_rows_below_effective_min", result["errors"])
        self.assertIn("guarded_trade_count_below_min_candidate_trades", result["errors"])

    def test_sparse_auto_derive_can_emit_partial_chop_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.csv"
            candidate_path.write_text(
                """ts,signal_ensemble,ret_ensemble_net,regime_state,p_up,y_true
2026-03-01T00:00:00Z,0,0.0,neutral,0.44,1
2026-03-01T01:00:00Z,0,0.0,neutral,0.45,0
2026-03-01T02:00:00Z,0,0.0,neutral,0.46,1
2026-03-01T03:00:00Z,1,-0.02,chop,0.520187,1
2026-03-01T04:00:00Z,1,0.0035,chop,0.524213,0
2026-03-01T05:00:00Z,1,0.0006,chop,0.528736,1
""".strip(),
                encoding="utf-8",
            )

            result = _derive_selection_calibration_guard_rules(
                candidate_path=candidate_path,
                recent_window_rows=3,
                baseline_window_rows=3,
                signal_col="signal_ensemble",
                return_col="ret_ensemble_net",
                regime_col="regime_state",
                p_col="p_up",
                y_col="y_true",
                min_selection_rows=15,
                adaptive_selection_cfg={"enabled": True, "min_floor": 8, "baseline_ratio": 0.6, "max_shortfall": 2},
                floors=[0.53],
                min_blocked_recent_rows=3,
                max_rules=1,
                min_recent_ece_improvement=0.0,
                min_ece_drift_improvement=0.0,
                max_recent_ece=0.08,
                max_ece_drift=0.02,
                min_recent_auc=0.5,
                require_blocked_recent_net_nonpositive=False,
                max_blocked_recent_net_return_total=0.02,
                require_recent_net_nonnegative=True,
                sparse_active_trade_cap=5,
                sparse_min_blocked_recent_rows=1,
                sparse_min_retained_recent_rows=1,
                sparse_allow_row_policy_override=True,
                sparse_allow_missing_baseline=True,
                sparse_use_observed_p_up_values=True,
            )

        self.assertTrue(result["enabled"])
        self.assertEqual(len(result["rules"]), 1)
        self.assertEqual(result["rules"][0]["regime_state"], "chop")
        self.assertGreater(result["rules"][0]["min_p_up"], 0.52)
        self.assertLess(result["rules"][0]["min_p_up"], 0.53)


if __name__ == "__main__":
    unittest.main()