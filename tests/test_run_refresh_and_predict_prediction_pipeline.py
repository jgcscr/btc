from __future__ import annotations

from src.scripts import run_refresh_and_predict as legacy


def test_run_predictions_wires_direction_fallback_dependency(monkeypatch) -> None:
    captured = {}

    def fake_execute(config, deps):
        captured["config"] = config
        captured["deps"] = deps
        return {"1h": {"trade_action": "hold"}}

    monkeypatch.setattr(legacy, "runtime_execute_prediction_pipeline", fake_execute)

    result = legacy.run_predictions(
        targets=[1.0],
        p_up_min=0.45,
        ret_min=0.0,
        direction_only_fallback={"enabled": True},
    )

    assert result == {"1h": {"trade_action": "hold"}}
    assert captured["config"].direction_only_fallback == {"enabled": True}
    assert captured["deps"].evaluate_direction_only_fallback is legacy._evaluate_direction_only_fallback