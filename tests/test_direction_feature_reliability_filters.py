import json
import tempfile
import unittest
from pathlib import Path

from src.scripts.build_training_dataset_direction import _filter_features_by_reliability


class DirectionFeatureReliabilityFilterTests(unittest.TestCase):
    def test_accepted_features_are_authoritative_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "feature_reliability.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "accepted_features": ["feature_b", "feature_c"],
                        "feature_scores": {
                            "feature_a": {"score": 0.99},
                            "feature_b": {"score": 0.20},
                            "feature_c": {"score": 0.10},
                        },
                    }
                ),
                encoding="utf-8",
            )

            filtered = _filter_features_by_reliability(
                ["feature_a", "feature_b", "feature_c"],
                str(payload_path),
                min_score=0.95,
            )

        self.assertEqual(filtered, ["feature_b", "feature_c"])

    def test_score_filter_is_used_when_no_accepted_features_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "feature_reliability.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "feature_scores": {
                            "feature_a": {"score": 0.80},
                            "feature_b": {"score": 0.40},
                            "feature_c": {"score": 0.70},
                        },
                    }
                ),
                encoding="utf-8",
            )

            filtered = _filter_features_by_reliability(
                ["feature_a", "feature_b", "feature_c"],
                str(payload_path),
                min_score=0.65,
            )

        self.assertEqual(filtered, ["feature_a", "feature_c"])

    def test_empty_accepted_features_falls_back_to_score_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "feature_reliability.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "accepted_features": [],
                        "feature_scores": {
                            "feature_a": {"score": 0.80},
                            "feature_b": {"score": 0.40},
                            "feature_c": {"score": 0.70},
                        },
                    }
                ),
                encoding="utf-8",
            )

            filtered = _filter_features_by_reliability(
                ["feature_a", "feature_b", "feature_c"],
                str(payload_path),
                min_score=0.65,
            )

        self.assertEqual(filtered, ["feature_a", "feature_c"])

    def test_horizon_regime_union_is_preferred_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "feature_reliability.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "accepted_features": ["feature_a"],
                        "accepted_features_by_horizon_regime": {
                            "1h": {
                                "trend": ["feature_b"],
                                "chop": ["feature_c"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            filtered = _filter_features_by_reliability(
                ["feature_a", "feature_b", "feature_c"],
                str(payload_path),
                min_score=0.95,
                target_horizon=1.0,
            )

        self.assertEqual(filtered, ["feature_b", "feature_c"])

    def test_horizon_regime_score_filter_overrides_union_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "feature_reliability.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "accepted_features_by_horizon_regime": {
                            "1h": {
                                "trend": ["feature_a", "feature_b"],
                                "chop": ["feature_c"],
                            }
                        },
                        "horizon_regime_feature_scores": {
                            "1h": {
                                "trend": {
                                    "feature_a": {"score": 0.91},
                                    "feature_b": {"score": 0.52},
                                },
                                "chop": {
                                    "feature_c": {"score": 0.49},
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            filtered = _filter_features_by_reliability(
                ["feature_a", "feature_b", "feature_c"],
                str(payload_path),
                min_score=0.75,
                target_horizon=1.0,
            )

        self.assertEqual(filtered, ["feature_a"])


if __name__ == "__main__":
    unittest.main()
