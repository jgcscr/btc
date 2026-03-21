from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.scripts.summarize_shadow_profile_comparison_longitudinal import (
    build_markdown_summary,
    build_run_rows,
    build_summary,
)
from src.scripts.update_shadow_profile_comparison_longitudinal import _extract_run_record, main


class ShadowProfileComparisonLongitudinalTests(unittest.TestCase):
    def test_extract_run_record_captures_actionable_and_difference_horizons(self) -> None:
        manifest_payload = {
            "run_id": "20260321T010000Z",
            "generated_at": "2026-03-21T01:00:00Z",
            "source_reliability_run_id": "20260320T224912Z",
            "restore_latest_to": "rhs",
            "targets": "0.25,1,4,8,12",
            "thresholds_json": "artifacts/reliability/latest/summary/calibrated_thresholds.json",
            "platt_calibration": "artifacts/reliability/latest/summary/platt_calibration.json",
            "profiles": {
                "shadow_simplified": {"config": "lhs.yaml"},
                "shadow_chop_suppression": {"config": "rhs.yaml"},
            },
        }
        comparison_payload = {
            "overall_summary": {
                "profiles_differ": True,
                "difference_only_probabilities_or_scores": False,
                "either_profile_actionable": True,
                "both_resolve_to_hold": False,
                "operationally_meaningful_difference": True,
                "operational_diff_horizons": ["1h"],
                "decision_state_only_diff_horizons": [],
                "score_only_diff_horizons": ["4h"],
            },
            "per_horizon": {
                "1h": {
                    "flags": {
                        "differs_operationally": True,
                        "differs_decision_state": True,
                        "differs_score_level": True,
                        "shadow_simplified_actionable": False,
                        "shadow_chop_suppression_actionable": True,
                    }
                },
                "4h": {
                    "flags": {
                        "differs_operationally": False,
                        "differs_decision_state": False,
                        "differs_score_level": True,
                        "shadow_simplified_actionable": False,
                        "shadow_chop_suppression_actionable": False,
                    }
                },
            },
        }

        record = _extract_run_record(
            manifest_payload,
            comparison_payload,
            manifest_path=Path("manifest.json"),
            comparison_path=Path("comparison.json"),
        )

        self.assertEqual(record["profile_pair"], "shadow_simplified_vs_shadow_chop_suppression")
        self.assertEqual(record["source_reliability_run_id"], "20260320T224912Z")
        self.assertEqual(record["operational_diff_horizons"], ["1h"])
        self.assertEqual(record["score_only_diff_horizons"], ["4h"])
        self.assertEqual(record["rhs_actionable_horizons"], ["1h"])
        self.assertEqual(record["differing_horizons"], ["1h", "4h"])

    def test_main_updates_longitudinal_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            comparison_path = root / "comparison.json"
            output_path = root / "longitudinal.json"

            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "20260321T020000Z",
                        "generated_at": "2026-03-21T02:00:00Z",
                        "source_reliability_run_id": "20260320T224912Z",
                        "restore_latest_to": "rhs",
                        "profiles": {
                            "shadow_simplified": {"config": "lhs.yaml"},
                            "shadow_chop_suppression": {"config": "rhs.yaml"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            comparison_path.write_text(
                json.dumps(
                    {
                        "overall_summary": {
                            "profiles_differ": True,
                            "difference_only_probabilities_or_scores": True,
                            "either_profile_actionable": False,
                            "both_resolve_to_hold": True,
                            "operationally_meaningful_difference": False,
                            "operational_diff_horizons": [],
                            "decision_state_only_diff_horizons": [],
                            "score_only_diff_horizons": ["15m"],
                        },
                        "per_horizon": {
                            "15m": {
                                "flags": {
                                    "differs_operationally": False,
                                    "differs_decision_state": False,
                                    "differs_score_level": True,
                                    "shadow_simplified_actionable": False,
                                    "shadow_chop_suppression_actionable": False,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "update_shadow_profile_comparison_longitudinal.py",
                    "--manifest",
                    str(manifest_path),
                    "--comparison",
                    str(comparison_path),
                    "--output",
                    str(output_path),
                ],
            ):
                main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            track = payload["tracks"]["shadow_simplified_vs_shadow_chop_suppression"]
            self.assertEqual(track["latest_run_id"], "20260321T020000Z")
            self.assertEqual(track["latest"]["score_only_diff_horizons"], ["15m"])
            self.assertEqual(len(track["runs"]), 1)

    def test_main_preserves_latest_when_backfilling_older_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "longitudinal.json"

            newer_manifest = root / "manifest_newer.json"
            newer_comparison = root / "comparison_newer.json"
            newer_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "20260321T020000Z",
                        "generated_at": "2026-03-21T02:00:00Z",
                        "source_reliability_run_id": "20260320T224912Z",
                        "restore_latest_to": "rhs",
                        "profiles": {
                            "shadow_simplified": {"config": "lhs.yaml"},
                            "shadow_chop_suppression": {"config": "rhs.yaml"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            newer_comparison.write_text(
                json.dumps(
                    {
                        "overall_summary": {"score_only_diff_horizons": ["15m"]},
                        "per_horizon": {"15m": {"flags": {}}},
                    }
                ),
                encoding="utf-8",
            )

            older_manifest = root / "manifest_older.json"
            older_comparison = root / "comparison_older.json"
            older_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "20260320T230000Z",
                        "generated_at": "2026-03-20T23:00:00Z",
                        "source_reliability_run_id": "20260319T190000Z",
                        "restore_latest_to": "rhs",
                        "profiles": {
                            "shadow_simplified": {"config": "lhs.yaml"},
                            "shadow_chop_suppression": {"config": "rhs.yaml"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            older_comparison.write_text(
                json.dumps(
                    {
                        "overall_summary": {"score_only_diff_horizons": ["1h"]},
                        "per_horizon": {"1h": {"flags": {}}},
                    }
                ),
                encoding="utf-8",
            )

            for manifest_path, comparison_path in (
                (newer_manifest, newer_comparison),
                (older_manifest, older_comparison),
            ):
                with patch(
                    "sys.argv",
                    [
                        "update_shadow_profile_comparison_longitudinal.py",
                        "--manifest",
                        str(manifest_path),
                        "--comparison",
                        str(comparison_path),
                        "--output",
                        str(output_path),
                    ],
                ):
                    main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            track = payload["tracks"]["shadow_simplified_vs_shadow_chop_suppression"]
            self.assertEqual(track["latest_run_id"], "20260321T020000Z")
            self.assertEqual(track["latest"]["run_id"], "20260321T020000Z")
            self.assertEqual([row["run_id"] for row in track["runs"]], ["20260320T230000Z", "20260321T020000Z"])

    def test_main_uses_generated_at_for_latest_with_mixed_run_id_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "longitudinal.json"

            newer_manifest = root / "manifest_newer.json"
            newer_comparison = root / "comparison_newer.json"
            newer_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "20260321T005206Z",
                        "generated_at": "2026-03-21T00:53:46Z",
                        "profiles": {
                            "shadow_simplified": {"config": "lhs.yaml"},
                            "shadow_chop_suppression": {"config": "rhs.yaml"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            newer_comparison.write_text(
                json.dumps(
                    {
                        "overall_summary": {"score_only_diff_horizons": ["15m"]},
                        "per_horizon": {"15m": {"flags": {}}},
                    }
                ),
                encoding="utf-8",
            )

            older_manifest = root / "manifest_older.json"
            older_comparison = root / "comparison_older.json"
            older_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "20260321Tcontinuetest",
                        "generated_at": "2026-03-21T00:40:57Z",
                        "profiles": {
                            "shadow_simplified": {"config": "lhs.yaml"},
                            "shadow_chop_suppression": {"config": "rhs.yaml"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            older_comparison.write_text(
                json.dumps(
                    {
                        "overall_summary": {"score_only_diff_horizons": ["15m"]},
                        "per_horizon": {"15m": {"flags": {}}},
                    }
                ),
                encoding="utf-8",
            )

            for manifest_path, comparison_path in (
                (older_manifest, older_comparison),
                (newer_manifest, newer_comparison),
            ):
                with patch(
                    "sys.argv",
                    [
                        "update_shadow_profile_comparison_longitudinal.py",
                        "--manifest",
                        str(manifest_path),
                        "--comparison",
                        str(comparison_path),
                        "--output",
                        str(output_path),
                    ],
                ):
                    main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            track = payload["tracks"]["shadow_simplified_vs_shadow_chop_suppression"]
            self.assertEqual(track["latest_run_id"], "20260321T005206Z")
            self.assertEqual(track["latest"]["generated_at"], "2026-03-21T00:53:46Z")

    def test_build_summary_aggregates_run_outcomes_and_source_counts(self) -> None:
        payload = {
            "updated_at": "2026-03-21T02:10:00Z",
            "tracked_comparison": "shadow_profile_comparison",
            "tracks": {
                "shadow_simplified_vs_shadow_chop_suppression": {
                    "lhs_label": "shadow_simplified",
                    "rhs_label": "shadow_chop_suppression",
                    "runs": [
                        {
                            "run_id": "20260321T010000Z",
                            "generated_at": "2026-03-21T01:00:00Z",
                            "thresholds_json": "artifacts/reliability/20260320T224912Z/summary/calibrated_thresholds.json",
                            "profiles_differ": True,
                            "difference_only_probabilities_or_scores": True,
                            "either_profile_actionable": False,
                            "both_resolve_to_hold": True,
                            "operationally_meaningful_difference": False,
                            "operational_diff_horizons": [],
                            "decision_state_only_diff_horizons": [],
                            "score_only_diff_horizons": ["15m"],
                            "differing_horizons": ["15m"],
                            "lhs_actionable_horizons": [],
                            "rhs_actionable_horizons": [],
                        },
                        {
                            "run_id": "20260321T020000Z",
                            "generated_at": "2026-03-21T02:00:00Z",
                            "source_reliability_run_id": "20260320T224912Z",
                            "profiles_differ": True,
                            "difference_only_probabilities_or_scores": False,
                            "either_profile_actionable": True,
                            "both_resolve_to_hold": False,
                            "operationally_meaningful_difference": True,
                            "operational_diff_horizons": ["1h"],
                            "decision_state_only_diff_horizons": [],
                            "score_only_diff_horizons": [],
                            "differing_horizons": ["1h"],
                            "lhs_actionable_horizons": [],
                            "rhs_actionable_horizons": ["1h"],
                        },
                    ],
                    "latest_run_id": "20260321T020000Z",
                    "latest": {
                        "run_id": "20260321T020000Z",
                        "generated_at": "2026-03-21T02:00:00Z",
                        "source_reliability_run_id": "20260320T224912Z",
                        "profiles_differ": True,
                        "difference_only_probabilities_or_scores": False,
                        "either_profile_actionable": True,
                        "both_resolve_to_hold": False,
                        "operationally_meaningful_difference": True,
                        "operational_diff_horizons": ["1h"],
                        "decision_state_only_diff_horizons": [],
                        "score_only_diff_horizons": [],
                        "lhs_actionable_horizons": [],
                        "rhs_actionable_horizons": ["1h"],
                    },
                }
            },
            "legacy": {
                "active_track": "shadow_simplified_vs_shadow_chop_suppression",
            },
        }

        summary = build_summary(payload)

        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["latest_run_id"], "20260321T020000Z")
        self.assertEqual(summary["aggregate_counts"]["profiles_differ_runs"], 2)
        self.assertEqual(summary["aggregate_counts"]["operationally_meaningful_difference_runs"], 1)
        self.assertEqual(summary["aggregate_counts"]["shadow_chop_suppression_actionable_runs"], 1)
        self.assertEqual(summary["horizon_counts"]["score_only_diff"], {"15m": 1})
        self.assertEqual(summary["horizon_counts"]["operational_diff"], {"1h": 1})
        self.assertEqual(summary["horizon_counts"]["shadow_chop_suppression"], {"1h": 1})
        self.assertEqual(summary["source_reliability_run_counts"], {"20260320T224912Z": 2})

        markdown = build_markdown_summary(summary)
        self.assertIn("# Shadow Profile Comparison Summary", markdown)
        self.assertIn("Latest run: 20260321T020000Z", markdown)
        self.assertIn("score_only_diff: 15m=1", markdown)

        rows = build_run_rows(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_id"], "20260321T010000Z")
        self.assertEqual(rows[0]["source_reliability_run_id"], "20260320T224912Z")
        self.assertEqual(rows[1]["operational_diff_horizons"], "1h")


if __name__ == "__main__":
    unittest.main()