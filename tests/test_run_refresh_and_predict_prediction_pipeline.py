from __future__ import annotations

from src.scripts import run_refresh_and_predict as legacy
import src.runtime.prediction_execution as prediction_execution


def test_run_predictions_wires_direction_fallback_dependency(monkeypatch) -> None:
    captured = {}

    def fake_run_predictions(**kwargs):
        captured.update(kwargs)
        return {"1h": {"trade_action": "hold"}}

    monkeypatch.setattr(prediction_execution, "run_predictions", fake_run_predictions)

    result = legacy.run_predictions(
        targets=[1.0],
        p_up_min=0.45,
        ret_min=0.0,
        direction_only_fallback={"enabled": True},
    )

    assert result == {"1h": {"trade_action": "hold"}}
    assert captured["direction_only_fallback"] == {"enabled": True}