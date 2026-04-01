from src.scripts.backtest_signals import _component_probability_frame


def test_component_probability_frame_flattens_component_maps() -> None:
    frame = _component_probability_frame(
        [
            {"p_up_components": {"xgb": 0.6, "lstm": 0.58}},
            {"p_up_components": {"xgb": 0.4, "gru": 0.43}},
        ]
    )

    assert list(frame.columns) == ["p_up_gru", "p_up_lstm", "p_up_xgb"]
    assert frame.loc[0, "p_up_xgb"] == 0.6
    assert frame.loc[0, "p_up_lstm"] == 0.58
    assert frame.loc[1, "p_up_gru"] == 0.43


def test_component_probability_frame_handles_missing_component_maps() -> None:
    frame = _component_probability_frame(
        [
            {"p_up_components": {"transformer": 0.61}},
            {},
        ]
    )

    assert list(frame.columns) == ["p_up_transformer"]
    assert frame.loc[0, "p_up_transformer"] == 0.61
    assert frame["p_up_transformer"].isna().iloc[1]