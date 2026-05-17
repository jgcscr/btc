import unittest

import numpy as np

from src.trading.ensembles import select_diverse_models


class DirectionEnsemblePolicyTests(unittest.TestCase):
    def test_select_diverse_models_prunes_highly_correlated_peer(self) -> None:
        probabilities = {
            "xgb": 0.62,
            "lstm": 0.61,
            "bilstm": 0.60,
            "transformer": 0.58,
        }
        history = {
            "xgb": np.array([0.52, 0.55, 0.57, 0.60, 0.62]),
            "lstm": np.array([0.53, 0.56, 0.58, 0.61, 0.63]),
            "bilstm": np.array([0.5305, 0.5605, 0.5805, 0.6105, 0.6305]),
            "transformer": np.array([0.40, 0.46, 0.50, 0.54, 0.58]),
        }

        payload = select_diverse_models(
            probabilities,
            {"xgb": 1.5, "lstm": 1.25, "bilstm": 1.0, "transformer": 1.0},
            history=history,
            priority_order=["xgb", "lstm", "bilstm", "transformer"],
            max_active_models=4,
            model_groups={"xgb": "tree", "lstm": "recurrent", "bilstm": "recurrent", "transformer": "attention"},
            max_models_per_group={"tree": 1, "recurrent": 2, "attention": 1},
            max_correlation=0.985,
            min_mean_abs_probability_gap=0.01,
            min_history_points=4,
        )

        self.assertIn("xgb", payload["selected_models"])
        self.assertIn("lstm", payload["selected_models"])
        self.assertIn("transformer", payload["selected_models"])
        self.assertNotIn("bilstm", payload["selected_models"])
        rejected = {entry["name"]: entry["reason"] for entry in payload["rejected_models"]}
        self.assertEqual(rejected.get("bilstm"), "orthogonality")

    def test_select_diverse_models_respects_group_caps(self) -> None:
        payload = select_diverse_models(
            {"xgb": 0.6, "lgbm": 0.59, "lstm": 0.58, "transformer": 0.57},
            {"xgb": 1.0, "lgbm": 0.9, "lstm": 1.0, "transformer": 1.0},
            priority_order=["xgb", "lgbm", "lstm", "transformer"],
            model_groups={"xgb": "tree", "lgbm": "tree", "lstm": "recurrent", "transformer": "attention"},
            max_models_per_group={"tree": 1, "recurrent": 1, "attention": 1},
        )

        self.assertEqual(payload["selected_models"], ["xgb", "lstm", "transformer"])
        rejected = {entry["name"]: entry["reason"] for entry in payload["rejected_models"]}
        self.assertEqual(rejected.get("lgbm"), "group_cap:tree")

    def test_select_diverse_models_falls_back_to_top_ranked_model(self) -> None:
        payload = select_diverse_models(
            {"xgb": 0.51, "lstm": 0.52},
            {"xgb": 1.0, "lstm": 0.8},
            priority_order=["xgb", "lstm"],
            max_active_models=1,
        )

        self.assertEqual(payload["selected_models"], ["xgb"])
        self.assertEqual(payload["effective_weights"]["xgb"], 1.0)

    def test_select_diverse_models_prefers_group_coverage_before_extra_recurrent_models(self) -> None:
        payload = select_diverse_models(
            {"xgb": 0.62, "lstm": 0.61, "gru": 0.60, "transformer": 0.59},
            {"xgb": 1.2, "lstm": 1.0, "gru": 0.8, "transformer": 1.1},
            priority_order=["xgb", "lstm", "gru", "transformer"],
            preferred_groups=["tree", "recurrent", "attention"],
            model_groups={
                "xgb": "tree",
                "lstm": "recurrent",
                "gru": "recurrent",
                "transformer": "attention",
            },
            max_models_per_group={"tree": 1, "recurrent": 2, "attention": 1},
            max_active_models=3,
        )

        self.assertEqual(payload["selected_models"], ["xgb", "lstm", "transformer"])
        self.assertEqual(payload["selected_groups"], ["tree", "recurrent", "attention"])
        self.assertEqual(payload["missing_preferred_groups"], [])

    def test_select_diverse_models_ignores_zero_weight_models(self) -> None:
        payload = select_diverse_models(
            {"xgb": 0.62, "lstm": 0.61, "cnn_lstm": 0.60, "transformer": 0.59},
            {"xgb": 1.2, "lstm": 0.8, "cnn_lstm": 0.0, "transformer": 1.1},
            priority_order=["xgb", "lstm", "cnn_lstm", "transformer"],
            model_groups={
                "xgb": "tree",
                "lstm": "recurrent",
                "cnn_lstm": "recurrent",
                "transformer": "attention",
            },
            max_models_per_group={"tree": 1, "recurrent": 2, "attention": 1},
            max_active_models=4,
        )

        self.assertEqual(payload["selected_models"], ["xgb", "lstm", "transformer"])
        self.assertEqual(payload["base_weights"].get("cnn_lstm"), 0.0)
        self.assertNotIn("cnn_lstm", payload["effective_weights"])


if __name__ == "__main__":
    unittest.main()