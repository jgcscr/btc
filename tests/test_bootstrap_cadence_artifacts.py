from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scripts.bootstrap_cadence_artifacts import (
    bootstrap_cadence_artifacts,
    validate_cadence_artifacts,
)


class BootstrapCadenceArtifactsTests(unittest.TestCase):
    def test_validate_rejects_remote_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            repo_root.mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(ValueError, "local filesystem path"):
                validate_cadence_artifacts(
                    artifacts_root_uri="gs://btc-artifacts",
                    repo_root=repo_root,
                )

    def test_validate_reports_ok_when_required_local_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_artifacts = root / "remote_artifacts"
            repo_root = root / "repo"
            summary_dir = remote_artifacts / "reliability" / "20260317T014743Z" / "summary"
            monitoring_dir = remote_artifacts / "monitoring"
            models_dir = remote_artifacts / "models"

            summary_dir.mkdir(parents=True, exist_ok=True)
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            repo_root.mkdir(parents=True, exist_ok=True)

            (summary_dir / "edge_trustworthiness.json").write_text(
                json.dumps({"edge_trustworthy": True}),
                encoding="utf-8",
            )
            (summary_dir / "calibrated_thresholds.json").write_text(
                json.dumps({"1": {"threshold": 0.55}}),
                encoding="utf-8",
            )
            (summary_dir / "platt_calibration.json").write_text(
                json.dumps({"1": {"a": 1.0, "b": 0.0}}),
                encoding="utf-8",
            )
            (monitoring_dir / "calibrated_thresholds_last_deployable.json").write_text(
                json.dumps({"1": {"threshold": 0.55}}),
                encoding="utf-8",
            )
            (models_dir / "direction_output_isotonic_1h.json").write_text(
                json.dumps({"calibration": "ok"}),
                encoding="utf-8",
            )
            (monitoring_dir / "reliability_promotion_deploy_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260317T014743Z",
                        "deployed_files": {
                            "thresholds_json": {
                                "source": "artifacts/monitoring/calibrated_thresholds_last_deployable.json",
                                "target": "artifacts/models/calibrated_thresholds_merged.json",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = validate_cadence_artifacts(
                artifacts_root_uri=str(remote_artifacts),
                repo_root=repo_root,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["missing"], [])

    def test_restores_manifest_summary_and_deployed_files_from_local_artifacts_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_artifacts = root / "remote_artifacts"
            repo_root = root / "repo"
            summary_dir = remote_artifacts / "reliability" / "20260317T014743Z" / "summary"
            monitoring_dir = remote_artifacts / "monitoring"
            models_dir = remote_artifacts / "models"

            summary_dir.mkdir(parents=True, exist_ok=True)
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            repo_root.mkdir(parents=True, exist_ok=True)

            (summary_dir / "edge_trustworthiness.json").write_text(
                json.dumps({"edge_trustworthy": True}),
                encoding="utf-8",
            )
            (summary_dir / "calibrated_thresholds.json").write_text(
                json.dumps({"1": {"threshold": 0.55}}),
                encoding="utf-8",
            )
            (summary_dir / "platt_calibration.json").write_text(
                json.dumps({"1": {"a": 1.0, "b": 0.0}}),
                encoding="utf-8",
            )
            (monitoring_dir / "calibrated_thresholds_last_deployable.json").write_text(
                json.dumps({"1": {"threshold": 0.55}}),
                encoding="utf-8",
            )
            (summary_dir / "trade_decision_model_reference_feature_ablation.json").write_text(
                json.dumps({"model": "ok"}),
                encoding="utf-8",
            )
            model_family_dir = models_dir / "lstm_dir1h_v1"
            model_family_dir.mkdir(parents=True, exist_ok=True)
            (model_family_dir / "model.keras").write_text("stub", encoding="utf-8")

            manifest = {
                "run_id": "20260317T014743Z",
                "deployed_files": {
                    "thresholds_json": {
                        "source": "artifacts/monitoring/calibrated_thresholds_last_deployable.json",
                        "target": "artifacts/models/calibrated_thresholds_merged.json",
                    },
                    "platt_calibration": {
                        "source": "artifacts/reliability/20260317T014743Z/summary/platt_calibration.json",
                        "target": "artifacts/models/platt_calibration.json",
                    },
                    "trade_decision_model": {
                        "source": "artifacts/reliability/20260317T014743Z/summary/trade_decision_model_reference_feature_ablation.json",
                        "target": "artifacts/models/trade_decision_model.json",
                    },
                },
            }
            (monitoring_dir / "reliability_promotion_deploy_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            result = bootstrap_cadence_artifacts(
                artifacts_root_uri=str(remote_artifacts),
                repo_root=repo_root,
            )

            self.assertEqual(result["run_id"], "20260317T014743Z")
            self.assertTrue(
                (repo_root / "artifacts" / "monitoring" / "reliability_promotion_deploy_manifest.json").exists()
            )
            self.assertTrue(
                (repo_root / "artifacts" / "reliability" / "20260317T014743Z" / "summary" / "edge_trustworthiness.json").exists()
            )
            self.assertTrue((repo_root / "artifacts" / "models" / "calibrated_thresholds_merged.json").exists())
            self.assertTrue((repo_root / "artifacts" / "models" / "platt_calibration.json").exists())
            self.assertTrue((repo_root / "artifacts" / "models" / "trade_decision_model.json").exists())
            self.assertTrue((repo_root / "artifacts" / "models" / "lstm_dir1h_v1" / "model.keras").exists())

    def test_bootstrap_replaces_existing_models_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_artifacts = root / "remote_artifacts"
            repo_root = root / "repo"
            summary_dir = remote_artifacts / "reliability" / "20260317T014743Z" / "summary"
            monitoring_dir = remote_artifacts / "monitoring"
            models_dir = remote_artifacts / "models"

            summary_dir.mkdir(parents=True, exist_ok=True)
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "artifacts" / "models" / "stale_dir1h_v9").mkdir(parents=True, exist_ok=True)
            (repo_root / "artifacts" / "models" / "stale_dir1h_v9" / "model.keras").write_text(
                "stale",
                encoding="utf-8",
            )

            (summary_dir / "edge_trustworthiness.json").write_text(
                json.dumps({"edge_trustworthy": True}),
                encoding="utf-8",
            )
            (summary_dir / "calibrated_thresholds.json").write_text(
                json.dumps({"1": {"threshold": 0.55}}),
                encoding="utf-8",
            )
            (summary_dir / "platt_calibration.json").write_text(
                json.dumps({"1": {"a": 1.0, "b": 0.0}}),
                encoding="utf-8",
            )
            (monitoring_dir / "reliability_promotion_deploy_manifest.json").write_text(
                json.dumps({"run_id": "20260317T014743Z", "deployed_files": {}}),
                encoding="utf-8",
            )
            (models_dir / "lstm_dir1h_v1").mkdir(parents=True, exist_ok=True)
            (models_dir / "lstm_dir1h_v1" / "model.keras").write_text("fresh", encoding="utf-8")

            bootstrap_cadence_artifacts(
                artifacts_root_uri=str(remote_artifacts),
                repo_root=repo_root,
            )

            self.assertFalse((repo_root / "artifacts" / "models" / "stale_dir1h_v9").exists())
            self.assertTrue((repo_root / "artifacts" / "models" / "lstm_dir1h_v1" / "model.keras").exists())
