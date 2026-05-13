import json
from pathlib import Path

import pandas as pd

from src.scripts.train_meta_ensemble import (
    label_backtest_splits,
    load_component_frame,
    save_meta_config,
    select_threshold_from_oof,
)


def test_load_component_frame_uses_all_component_columns_by_default(tmp_path: Path) -> None:
    path = tmp_path / "components.csv"
    pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "ret_1h": [0.01, -0.02],
            "p_up_xgb": [0.6, 0.4],
            "p_up_lstm": [0.58, 0.42],
            "p_up_meta": [0.59, 0.41],
        }
    ).to_csv(path, index=False)

    frame, columns, extra_columns = load_component_frame(path)

    assert columns == ["p_up_lstm", "p_up_xgb"]
    assert extra_columns == []
    assert list(frame.columns) == ["ts", "ret_1h", "p_up_lstm", "p_up_xgb"]


def test_load_component_frame_accepts_alias_names(tmp_path: Path) -> None:
    path = tmp_path / "components.csv"
    pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00Z"],
            "ret_1h": [0.01],
            "p_up_transformer": [0.61],
            "p_up_xgb": [0.55],
        }
    ).to_csv(path, index=False)

    frame, columns, extra_columns = load_component_frame(path, ["transformer"])

    assert columns == ["p_up_transformer"]
    assert extra_columns == []
    assert list(frame.columns) == ["ts", "ret_1h", "p_up_transformer"]


def test_load_component_frame_drops_all_nan_columns_and_imputes_sparse_rows(tmp_path: Path) -> None:
    path = tmp_path / "components.csv"
    pd.DataFrame(
        {
            "ts": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
            ],
            "ret_1h": [0.01, -0.02, 0.03],
            "p_up_xgb": [0.60, 0.40, 0.70],
            "p_up_lstm": [0.58, 0.42, 0.68],
            "p_up_gru": [None, 0.43, None],
            "p_up_lgbm": [None, None, None],
        }
    ).to_csv(path, index=False)

    frame, columns, extra_columns = load_component_frame(path)

    assert columns == ["p_up_gru", "p_up_lstm", "p_up_xgb"]
    assert extra_columns == []
    assert "p_up_lgbm" not in frame.columns
    assert frame[columns].isna().sum().sum() == 0
    assert frame.loc[0, "p_up_gru"] == (frame.loc[0, "p_up_lstm"] + frame.loc[0, "p_up_xgb"]) / 2


def test_select_threshold_from_oof_caps_sparse_tail_threshold() -> None:
    oof_frame = pd.DataFrame(
        {
            "ret_1h": [-0.01, -0.01, -0.01, 0.02, 0.02],
            "p_up_gate": [0.60, 0.60, 0.60, 0.60, 0.60],
            "p_up_meta": [0.50, 0.54, 0.54, 0.57, 0.57],
        }
    )

    threshold, summary = select_threshold_from_oof(
        oof_frame,
        fallback_threshold=0.52,
        signal_mode="meta_veto",
        min_trades=1,
        max_threshold_quantile=0.5,
    )

    assert threshold == 0.54
    assert summary["quantile_cap"] == 0.54
    assert summary["trades"] == 4.0
    assert summary["net"] < 0.0394


def test_label_backtest_splits_marks_warmup_rows_separately() -> None:
    splits = label_backtest_splits(
        pd.Series([None, None, 0.51, 0.53, 0.55]),
        n_test_start=4,
    )

    assert splits.tolist() == [
        "train_warmup",
        "train_warmup",
        "trainval_oof",
        "trainval_oof",
        "test_holdout",
    ]


def test_save_meta_config_persists_threshold_selection_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "meta_config.json"

    save_meta_config(
        config_path,
        feature_columns=["p_up_xgb", "meta_prob_mean"],
        intercept=0.1,
        coefficients=[0.2, -0.05],
        threshold=0.54,
        signal_mode="meta_veto",
        schedules=[{"fee_bps": 2.0, "slippage_bps": 1.0, "label": "fee_20_10"}],
        oof_metrics={"accuracy": 0.5},
        trainval_metrics={"accuracy": 0.51},
        oof_splits=5,
        component_columns=["p_up_xgb", "p_up_lstm"],
        extra_feature_columns=["meta_prob_mean"],
        component_weights={"p_up_xgb": 1.5, "p_up_lstm": 1.0},
        threshold_selection={
            "auto_threshold_on_oof": True,
            "fallback_threshold": 0.52,
            "selected_threshold": 0.54,
            "signal_mode": "meta_veto",
            "min_threshold_trades": 25,
            "max_auto_threshold_quantile": 0.95,
            "quantile_cap": 0.54,
            "trades": 313,
            "net": -0.114309,
            "hit_rate": 0.507987,
        },
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["threshold"] == 0.54
    assert payload["threshold_selection"] == {
        "auto_threshold_on_oof": True,
        "fallback_threshold": 0.52,
        "selected_threshold": 0.54,
        "signal_mode": "meta_veto",
        "min_threshold_trades": 25.0,
        "max_auto_threshold_quantile": 0.95,
        "quantile_cap": 0.54,
        "trades": 313.0,
        "net": -0.114309,
        "hit_rate": 0.507987,
    }