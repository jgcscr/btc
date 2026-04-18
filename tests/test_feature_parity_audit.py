from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.runtime.feature_parity_audit import (
    build_parity_audit,
    classify_feature_family,
    rank_candidates,
)


class FeatureParityAuditTests(unittest.TestCase):
    def test_classify_feature_family_maps_known_families(self) -> None:
        self.assertEqual(classify_feature_family("macro_us10y"), "macro")
        self.assertEqual(classify_feature_family("onchain_active_addresses"), "onchain")
        self.assertEqual(classify_feature_family("intrabar_volume_sum"), "intrabar")
        self.assertEqual(classify_feature_family("funding_rate_zscore_24h"), "derivatives")
        self.assertEqual(classify_feature_family("ret_4h"), "forward_return_proxy")

    def test_rank_candidates_penalizes_forward_return_proxy(self) -> None:
        rows = rank_candidates(
            available_families={"macro", "forward_return_proxy", "state_engineering"},
            live_enforced_families={"state_engineering"},
            ignored_families={"macro"},
            reliability_payload={
                "accepted_features": ["macro_us10y", "macro_eurusd", "ret_4h"],
                "feature_scores": {
                    "macro_us10y": {"score": 0.9},
                    "macro_eurusd": {"score": 0.91},
                    "ret_4h": {"score": 0.88},
                },
            },
            leakage_payload={"multi_horizon_leakage_warning": "present"},
        )

        by_family = {row.family: row for row in rows}
        self.assertIn("forward_return_proxy", by_family)
        self.assertGreaterEqual(by_family["forward_return_proxy"].implementation_risk, 0.9)

    def test_build_parity_audit_captures_ignored_and_stale_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            out_dir = model_dir / "xgb_dir1h_v1"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "model_metadata_direction.json").write_text(
                json.dumps(
                    {
                        "feature_names": [
                            "open",
                            "volume",
                            "macro_us10y",
                            "onchain_active_addresses",
                            "funding_rate",
                            "trend_path_efficiency_4h",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            reg_dir = model_dir / "xgb_ret1h_v1"
            reg_dir.mkdir(parents=True, exist_ok=True)
            (reg_dir / "model_metadata.json").write_text(
                json.dumps({"feature_names": ["open", "ret_4h", "macro_eurusd"]}),
                encoding="utf-8",
            )

            reliability_path = root / "reliability.json"
            reliability_path.write_text(
                json.dumps(
                    {
                        "accepted_features": ["macro_us10y", "trend_path_efficiency_4h"],
                        "feature_scores": {
                            "macro_us10y": {"score": 0.9},
                            "trend_path_efficiency_4h": {"score": 0.8},
                        },
                    }
                ),
                encoding="utf-8",
            )
            featurelift_path = root / "comparison_report.md"
            featurelift_path.write_text(
                "This report supersedes previous runs after leakage fixes. Multi-horizon degradations reflect removed leaked edge.",
                encoding="utf-8",
            )

            payload = build_parity_audit(
                horizons=[1.0],
                models_root=model_dir,
                ignored_columns=["macro_us10y", "funding_rate"],
                ignored_sources=["onchain"],
                max_source_lag_hours=6.0,
                reliability_path=reliability_path,
                featurelift_report_path=featurelift_path,
            )

        self.assertIn("macro", payload["ignored_families"])
        self.assertIn("derivatives", payload["ignored_families"])
        self.assertIn("onchain", payload["stale_tolerated_families"])
        self.assertIn("likely_untapped_candidates", payload)
        self.assertIn("source_family_artifacts", payload)
        self.assertIn("macro", payload["source_family_artifacts"])
        self.assertIn("onchain", payload["source_family_artifacts"])
        self.assertTrue(payload["likely_untapped_candidates"])


if __name__ == "__main__":
    unittest.main()
