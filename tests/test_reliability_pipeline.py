from __future__ import annotations

import argparse
import json

from src.runtime.models import RuntimeMode
from src.runtime.persistence import RuntimeStateStore
import src.runtime.reliability_pipeline as reliability_pipeline
from src.scripts import run_reliability_workflow as workflow


def test_run_with_step_event_sink_scopes_nested_events() -> None:
    outer_events: list[tuple[str, str, dict | None]] = []
    inner_events: list[tuple[str, str, dict | None]] = []

    def outer_sink(name: str, status: str, details):
        outer_events.append((name, status, details))

    def inner_sink(name: str, status: str, details):
        inner_events.append((name, status, details))

    def callback() -> None:
        workflow._emit_step_event("outer-before", "started", {"scope": "outer"})
        workflow._run_with_step_event_sink(
            inner_sink,
            lambda: workflow._emit_step_event("inner", "completed", {"scope": "inner"}),
        )
        workflow._emit_step_event("outer-after", "completed", {"scope": "outer"})

    workflow._run_with_step_event_sink(outer_sink, callback)
    workflow._emit_step_event("outside", "started", {"scope": "none"})

    assert outer_events == [
        ("outer-before", "started", {"scope": "outer"}),
        ("outer-after", "completed", {"scope": "outer"}),
    ]
    assert inner_events == [("inner", "completed", {"scope": "inner"})]


def test_execute_reliability_pipeline_passes_step_event_sink(tmp_path, monkeypatch) -> None:
    store = RuntimeStateStore(tmp_path)

    monkeypatch.setattr(reliability_pipeline, "RuntimeStateStore", lambda: store)

    class FakeRegistry:
        def record_workflow_manifest(self, manifest):
            return {"run_id": manifest.get("run_id")}

    monkeypatch.setattr(reliability_pipeline, "ReliabilityRunRegistry", FakeRegistry)

    captured: dict[str, object] = {}

    def fake_execute_reliability_workflow(args, *, step_event_sink=None):
        captured["step_event_sink"] = step_event_sink
        assert step_event_sink is not None
        step_event_sink("train_models", "started", {"dry_run": True})
        step_event_sink("train_models", "completed", {"returncode": 0})
        return {
            "run_id": "reliability-test-run",
            "run_dir": "artifacts/reliability/reliability-test-run",
            "profile": {"id": "test"},
            "steps": [{"name": "train_models"}],
        }

    monkeypatch.setattr(reliability_pipeline, "execute_reliability_workflow", fake_execute_reliability_workflow)

    args = argparse.Namespace(config="configs/reliability_workflow.runtime.yaml")
    result = reliability_pipeline.execute_reliability_pipeline(args)

    assert captured["step_event_sink"] is not None
    assert result.mode is RuntimeMode.RELIABILITY
    events = [
        json.loads(line)
        for line in result.run_root.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    workflow_events = [event for event in events if event["stage"] == "workflow_step"]
    assert [event["status"] for event in workflow_events] == ["started", "completed"]
    assert workflow_events[0]["details"]["step_name"] == "train_models"
