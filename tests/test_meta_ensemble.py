from pathlib import Path

import pandas as pd

from src.scripts.train_meta_ensemble import load_component_frame


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

    frame, columns = load_component_frame(path)

    assert columns == ["p_up_lstm", "p_up_xgb"]
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

    frame, columns = load_component_frame(path, ["transformer"])

    assert columns == ["p_up_transformer"]
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

    frame, columns = load_component_frame(path)

    assert columns == ["p_up_gru", "p_up_lstm", "p_up_xgb"]
    assert "p_up_lgbm" not in frame.columns
    assert frame[columns].isna().sum().sum() == 0
    assert frame.loc[0, "p_up_gru"] == (frame.loc[0, "p_up_lstm"] + frame.loc[0, "p_up_xgb"]) / 2