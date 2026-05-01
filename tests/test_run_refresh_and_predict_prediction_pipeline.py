from __future__ import annotations

from src.scripts import run_refresh_and_predict as legacy
from src.runtime.horizon_support import normalize_horizon_value
import src.runtime.prediction_dependency_support as dependency_support
from src.runtime.prediction_dependency_support import compute_position_size, parse_iso_timestamp
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


def test_runtime_prediction_execution_uses_runtime_dependency_support(monkeypatch) -> None:
    def fail(*args, **kwargs):  # pragma: no cover - guard against legacy regression
        raise AssertionError("legacy helper should not be called")

    monkeypatch.setattr(legacy, "_normalize_horizon_value", fail)
    monkeypatch.setattr(legacy, "_resolve_abstention_policy", fail)
    monkeypatch.setattr(legacy, "_resolve_execution_policy", fail)
    monkeypatch.setattr(legacy, "_compute_position_size", fail)
    monkeypatch.setattr(legacy, "_parse_iso_timestamp", fail)
    monkeypatch.setattr(legacy, "resolve_direction_model_configs", fail)
    monkeypatch.setattr(legacy, "_summarize_bias_context", fail)
    monkeypatch.setattr(legacy, "_resolve_execution_upstream_hold_reason", fail)
    monkeypatch.setattr(legacy, "_append_gate_trace", fail)

    def fake_resolve_direction_model_configs(*args, **kwargs):
        return ["runtime-owned-config"]

    monkeypatch.setattr(dependency_support, "resolve_direction_model_configs", fake_resolve_direction_model_configs)

    def fake_apply_execution_policy(summary, contexts, policy, **kwargs):
        kwargs["summarize_bias_context"]({}, {})
        kwargs["resolve_execution_upstream_hold_reason"]({})
        return summary

    monkeypatch.setattr(dependency_support, "apply_execution_policy", fake_apply_execution_policy)
    monkeypatch.setattr(dependency_support, "summarize_bias_context", lambda summary, policy: {"bias_direction": "neutral", "bias_alignment_ratio": 0.0, "execution_entries": [], "bias_scores": {}, "execution_scores": {}, "direction_support_horizons": {}})
    monkeypatch.setattr(dependency_support, "resolve_execution_upstream_hold_reason", lambda entry: "upstream_model_hold")

    def fake_execute_prediction_pipeline(config, deps):
        assert deps.normalize_horizon_value(0.25) == normalize_horizon_value(0.25)
        assert deps.resolve_abstention_policy({"enabled": True})["enabled"] is True
        assert deps.resolve_execution_policy({"enabled": True})["enabled"] is True
        assert deps.prepare_base_direction_configs(
            config_json_path=None,
            weight_spec=None,
            dir_lstm_path=None,
            dir_bilstm_path=None,
            dir_gru_path=None,
            dir_cnn_lstm_path=None,
            dir_cnn_bilstm_path=None,
            dir_garch_lstm_path=None,
            dir_transformer_path=None,
        ) == ["runtime-owned-config"]
        assert deps.compute_position_size(0.8, confidence_min=0.3, size_floor=0.0, size_cap=0.5) == compute_position_size(
            0.8,
            confidence_min=0.3,
            size_floor=0.0,
            size_cap=0.5,
        )
        assert deps.parse_iso_timestamp("2026-04-30T00:00:00Z") == parse_iso_timestamp("2026-04-30T00:00:00Z")
        assert deps.apply_execution_policy({}, {}, {"enabled": True}) == {}
        post_trade_summary = deps.apply_post_trade_gates(
            {
                "1h": {
                    "trade_action": "long",
                    "confidence_score": 0.1,
                }
            },
            confidence_min=0.5,
            abstention_policy={"enabled": False},
            uncertainty_policy={"enabled": False},
        )
        assert post_trade_summary["1h"]["gate_trace"][0]["stage"] == "confidence_filter"
        return {"1h": {"trade_action": "hold"}}

    monkeypatch.setattr(prediction_execution, "execute_prediction_pipeline", fake_execute_prediction_pipeline)

    result = prediction_execution.run_predictions(
        targets=[1.0],
        p_up_min=0.45,
        ret_min=0.0,
        abstention_policy={"enabled": True},
        execution_policy={"enabled": True},
    )

    assert result == {"1h": {"trade_action": "hold"}}