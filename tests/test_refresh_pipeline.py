from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.runtime.models import RuntimeMode
from src.runtime.persistence import RuntimeStateStore
import src.runtime.refresh_pipeline as refresh_pipeline


def test_persist_outputs_uses_runtime_output_functions_not_legacy_helpers(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fail(*args, **kwargs):  # pragma: no cover - guard against legacy regression
        raise AssertionError("legacy output helper should not be called")

    def fake_write_prediction_summary(*args, **kwargs):
        calls.append("write_prediction_summary")
        return {
            "generated_at": "2026-04-02T00:00:00Z",
            "predictions": {"1h": {"trade_action": "hold"}},
            "prompt_ready_summary": {
                "market_outlook_strategy": {
                    "selected_direction": "Neutral",
                    "preferred_horizon": "1h",
                    "tradeable": False,
                }
            },
        }

    def fake_write_monitoring_artifact(*args, **kwargs):
        calls.append(f"write_monitoring:{kwargs['output_path'].name}")
        return {"source": "test", "generated_at": "2026-04-02T00:00:00Z"}

    def fake_refresh_meta_baseline(**kwargs):
        calls.append("refresh_meta_baseline")

    monkeypatch.setattr(refresh_pipeline.legacy, "write_summary", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_write_monitoring_latest", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_write_trade_ready_monitoring", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_refresh_meta_baseline", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_build_prompt_ready_summary", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_build_blocked_trade_analytics", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_build_degradation_monitoring", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_horizon_sort_key", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_format_horizon_label", fail)
    monkeypatch.setattr(refresh_pipeline.legacy, "_append_detected_meta_columns", fail)
    monkeypatch.setattr(refresh_pipeline, "runtime_write_prediction_summary", fake_write_prediction_summary)
    monkeypatch.setattr(refresh_pipeline, "runtime_write_monitoring_artifact", fake_write_monitoring_artifact)
    monkeypatch.setattr(refresh_pipeline, "runtime_refresh_meta_baseline", fake_refresh_meta_baseline)

    args = argparse.Namespace(
        disable_monitoring_latest=False,
        write_artifacts=True,
        degradation_monitoring=None,
        targets=[1.0],
        spot_provider="binanceus",
        hours=360,
        dry_run=True,
        confidence_min=0.0,
        position_size_floor=0.0,
        position_size_cap=1.0,
        position_size_cap_by_horizon=None,
        confidence_min_by_horizon_regime=None,
        data_quality={},
        local_feature_metadata=None,
    )
    store = RuntimeStateStore(tmp_path)
    run_paths = store.start_run(mode=RuntimeMode.RESEARCH, request={"targets": [1.0]})

    result = refresh_pipeline._persist_outputs(
        args,
        RuntimeMode.RESEARCH,
        {"1h": {"trade_action": "hold"}},
        run_paths,
        store,
    )

    assert result.predictions_payload["prompt_ready_summary"]["market_outlook_strategy"]["preferred_horizon"] == "1h"
    assert calls == [
        "write_prediction_summary",
        "write_monitoring:latest.json",
        "write_monitoring:trade_ready_summary.json",
        "refresh_meta_baseline",
    ]


def test_execute_refresh_pipeline_coordinates_runtime_stages(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    store = RuntimeStateStore(tmp_path)

    monkeypatch.setattr(refresh_pipeline, "RuntimeStateStore", lambda: store)
    monkeypatch.setattr(refresh_pipeline, "normalize_refresh_args", lambda args: calls.append(("normalize", args.hours)))

    def fake_prepare_market_data(args, run_paths, runtime_store):
        calls.append(("prepare_market_data", run_paths.run_id))
        assert runtime_store is store
        return SimpleNamespace(prepared_override="prepared", latest_close=123.45)

    def fake_apply_replay_override(args, run_paths, runtime_store, *, prepared_override, latest_close):
        calls.append(("apply_replay_override", prepared_override, latest_close))
        assert runtime_store is store
        return SimpleNamespace(prepared_override="replayed", latest_close=234.56)

    def fake_load_prediction_inputs(args, run_paths, runtime_store):
        calls.append(("load_prediction_inputs", run_paths.run_id))
        assert runtime_store is store
        return "inputs"

    def fake_run_prediction_stage(args, run_paths, runtime_store, *, prepared_override, latest_close, prediction_inputs):
        calls.append(("run_prediction_stage", prepared_override, latest_close, prediction_inputs))
        assert runtime_store is store
        return {"1h": {"trade_action": "hold"}}

    def fake_persist_outputs(args, mode, summary, run_paths, runtime_store):
        calls.append(("persist_outputs", mode.value, tuple(summary.keys())))
        assert runtime_store is store
        return SimpleNamespace(
            run_id=run_paths.run_id,
            mode=mode,
            run_root=run_paths.root,
            predictions_payload={"prompt_ready_summary": {"market_outlook_strategy": {"selected_direction": "Neutral"}}},
            monitoring_payload=None,
            trade_ready_payload=None,
            metadata={},
        )

    monkeypatch.setattr(refresh_pipeline, "prepare_market_data", fake_prepare_market_data)
    monkeypatch.setattr(refresh_pipeline, "apply_replay_override", fake_apply_replay_override)
    monkeypatch.setattr(refresh_pipeline, "_load_prediction_inputs", fake_load_prediction_inputs)
    monkeypatch.setattr(refresh_pipeline, "_run_prediction_stage", fake_run_prediction_stage)
    monkeypatch.setattr(refresh_pipeline, "_persist_outputs", fake_persist_outputs)

    args = argparse.Namespace(hours=360)
    result = refresh_pipeline.execute_refresh_pipeline(args, mode=RuntimeMode.RESEARCH)

    assert result.mode is RuntimeMode.RESEARCH
    assert [call[0] for call in calls] == [
        "normalize",
        "prepare_market_data",
        "apply_replay_override",
        "load_prediction_inputs",
        "run_prediction_stage",
        "persist_outputs",
    ]