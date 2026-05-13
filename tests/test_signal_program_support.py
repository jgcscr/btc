from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.runtime.signal_program_support import (
    build_derivatives_shadow_candidate_config,
    build_derivatives_family_audit,
    build_signal_expansion_rollout_summary,
    build_derivatives_shadow_scaffold,
    build_signal_program_dispositions,
)


class SignalProgramSupportTests(unittest.TestCase):
    def test_build_signal_program_dispositions_parses_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "macro_shadow_enforcement_latest.json").write_text(
                json.dumps(
                    {
                        "sweep": {
                            "recommendation": {
                                "macro_disposition": "deprioritize_for_now",
                                "best_assessment": "neutral",
                                "advance_to_next_validation_stage": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "state_orderflow_outcome_confirmation_latest.json").write_text(
                json.dumps(
                    {
                        "family_best_rankings": [
                            {
                                "family": "state_engineering",
                                "best_variant": "weak_signal_veto_only",
                                "net_return_proxy_mean_delta": -0.001,
                                "direction_accuracy_proxy_delta": 0.02,
                                "robustness": "narrow",
                                "go_hold": {"decision": "hold"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "orderflow_rolling_stability_latest.json").write_text(
                json.dumps(
                    {
                        "rolling_stability_classification": {
                            "classification": "unstable",
                            "disposition": "deprioritize_for_now",
                            "pass_count": 0,
                            "fail_count": 5,
                        }
                    }
                ),
                encoding="utf-8",
            )

            payload = build_signal_program_dispositions(root)

        self.assertEqual(payload["families"]["macro"]["disposition"], "deprioritize_for_now")
        self.assertEqual(payload["families"]["order_flow"]["disposition"], "deprioritize_for_now")
        self.assertEqual(payload["families"]["state_engineering"]["disposition"], "hold")
        self.assertEqual(payload["next_priority_family"], "derivatives")

    def test_derivatives_audit_blocks_when_funding_data_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_root = root / "models"
            (models_root / "xgb_dir4h_v1").mkdir(parents=True, exist_ok=True)
            (models_root / "xgb_ret4h_v1").mkdir(parents=True, exist_ok=True)
            (models_root / "xgb_dir4h_v1" / "model_metadata_direction.json").write_text(
                json.dumps({"feature_names": ["open", "funding_rate_zscore_24h", "fut_close", "open_interest"]}),
                encoding="utf-8",
            )
            (models_root / "xgb_ret4h_v1" / "model_metadata.json").write_text(
                json.dumps({"feature_names": ["funding_rate_zscore_24h", "fut_close"]}),
                encoding="utf-8",
            )
            spot_dir = root / "spot_klines"
            spot_dir.mkdir(parents=True, exist_ok=True)
            (spot_dir / "btc.parquet").write_text("placeholder", encoding="utf-8")

            payload = build_derivatives_family_audit(
                config={
                    "targets": [4],
                    "feature_coverage_policy": {"ignored_columns": ["fut_close", "open_interest"]},
                },
                models_root=models_root,
                funding_dir=root / "funding",
                spot_dir=spot_dir,
            )

        self.assertEqual(payload["readiness"]["decision"], "needs_data_wiring_first")
        self.assertIn("missing_local_funding_dataset", payload["readiness"]["blockers"])
        self.assertTrue(payload["training_derivatives_union"])

    def test_derivatives_scaffold_reflects_blocked_status(self) -> None:
        scaffold = build_derivatives_shadow_scaffold(
            {
                "readiness": {
                    "decision": "needs_data_wiring_first",
                    "blockers": ["missing_local_funding_dataset"],
                },
                "shadow_policy_scaffold": [{"name": "funding_conflict_veto_weak"}],
            }
        )

        self.assertEqual(scaffold["runner_status"], "blocked")
        self.assertEqual(scaffold["readiness_decision"], "needs_data_wiring_first")

    def test_derivatives_audit_marks_dataset_ready_when_npz_has_derivatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_root = root / "models"
            models_root.mkdir(parents=True, exist_ok=True)
            funding_dir = root / "funding"
            funding_dir.mkdir(parents=True, exist_ok=True)
            (funding_dir / "hourly_features.parquet").write_text("placeholder", encoding="utf-8")
            spot_dir = root / "spot_klines"
            spot_dir.mkdir(parents=True, exist_ok=True)
            (spot_dir / "btc.parquet").write_text("placeholder", encoding="utf-8")

            import numpy as np

            np.savez_compressed(
                root / "btc_features_1h_splits.npz",
                feature_names=np.array(["open", "fut_close", "funding_rate_zscore_24h"], dtype=object),
            )

            from src.runtime import signal_program_support as module

            original_multi = module.DEFAULT_DATASET_MULTI_PATH
            original_1h = module.DEFAULT_DATASET_1H_PATH
            module.DEFAULT_DATASET_MULTI_PATH = root / "missing_multi.npz"
            module.DEFAULT_DATASET_1H_PATH = root / "btc_features_1h_splits.npz"
            try:
                payload = build_derivatives_family_audit(
                    config={"targets": [1, 4]},
                    models_root=models_root,
                    funding_dir=funding_dir,
                    spot_dir=spot_dir,
                )
            finally:
                module.DEFAULT_DATASET_MULTI_PATH = original_multi
                module.DEFAULT_DATASET_1H_PATH = original_1h

        self.assertEqual(payload["readiness"]["decision"], "dataset_ready_retrain_required")
        self.assertIn("checked_model_metadata_not_refreshed_after_derivatives_wiring", payload["readiness"]["blockers"])
        self.assertGreater(payload["dataset_derivatives_family_count"], 0)

    def test_derivatives_shadow_candidate_config_requires_derivatives_in_coverage(self) -> None:
        config = build_derivatives_shadow_candidate_config(
            {
                "feature_coverage_policy": {
                    "ignored_sources": ["funding", "macro", "onchain"],
                    "ignored_columns": ["fut_close", "open_interest", "macro_us10y"],
                }
            },
            audit={"readiness": {"decision": "shadow_scaffold_ready", "next_action": "run_first_shadow_derivatives_validation"}},
        )

        coverage = config["feature_coverage_policy"]
        self.assertEqual(coverage["ignored_sources"], ["macro", "onchain"])
        self.assertEqual(coverage["ignored_columns"], ["macro_us10y"])
        self.assertNotIn("derivatives_shadow_validation", config)

    def test_signal_expansion_rollout_summary_keeps_macro_deprioritized(self) -> None:
        payload = build_signal_expansion_rollout_summary(
            signal_payload={
                "families": {
                    "macro": {"status": "closed", "disposition": "deprioritize_for_now"},
                    "state_engineering": {"status": "active", "disposition": "guarded_shadow_validation_active"},
                }
            },
            derivatives_audit={"readiness": {"decision": "shadow_scaffold_ready", "next_action": "run_first_shadow_derivatives_validation"}},
            derivatives_scaffold={"runner_status": "ready"},
            meta_baseline_source_csv="artifacts/backtests/backtest_signals_meta_ensemble.csv",
            meta_config_path="configs/reliability_workflow.runtime.yaml",
            meta_signal_mode="meta_veto",
            meta_weight_threshold=0.52,
            meta_selected_weight_threshold=0.54,
            meta_auto_threshold_on_oof=True,
            meta_threshold_selection={"trades": 313.0, "net": -0.114309, "quantile_cap": 0.54},
            derivatives_config_path="configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml",
            featurelift_config_path="configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml",
            featurelift_package_path="artifacts/analysis/featurelift_20260331_rerun/shadow_rollout_4h_package.md",
            state_guarded_json_path="artifacts/analysis/state_engineering_guarded_shadow_4h_latest.json",
            state_guarded_md_path="artifacts/analysis/state_engineering_guarded_shadow_4h_latest.md",
        )

        self.assertEqual(payload["next_priority_family"], "meta_ensemble")
        self.assertEqual(payload["program_direction"]["macro"]["recommended_action"], "keep_deprioritized")
        self.assertEqual(payload["program_direction"]["meta_ensemble"]["selected_weight_threshold"], 0.54)
        self.assertTrue(payload["program_direction"]["meta_ensemble"]["auto_threshold_on_oof"])
        self.assertEqual(payload["program_direction"]["meta_ensemble"]["threshold_selection"]["trades"], 313.0)
        self.assertEqual(
            payload["program_direction"]["derivatives"]["candidate_config"],
            "configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml",
        )


if __name__ == "__main__":
    unittest.main()