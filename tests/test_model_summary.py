from src.utils.model_summary import build_model_summary


def test_build_model_summary_exposes_normalized_split_fields() -> None:
    summary = build_model_summary(
        model_type="xgboost_classifier",
        target="direction_1h",
        dataset_path="artifacts/datasets/example.npz",
        model_path="artifacts/models/example/model.json",
        metrics={
            "train": {"accuracy": 0.6},
            "val": {"accuracy": 0.55, "roc_auc": 0.58},
            "test": {"accuracy": 0.53},
        },
    )

    assert summary["metrics"]["val"]["roc_auc"] == 0.58
    assert summary["train_metrics"]["accuracy"] == 0.6
    assert summary["val_metrics"]["accuracy"] == 0.55
    assert summary["test_metrics"]["accuracy"] == 0.53