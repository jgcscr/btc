from __future__ import annotations

import argparse
from typing import Any

from src.runtime.models import PipelineExecutionResult, RuntimeMode
from src.runtime.persistence import RuntimeStateStore
from src.runtime.reliability_registry import ReliabilityRunRegistry
from src.scripts import run_reliability_workflow as legacy


def execute_reliability_pipeline(args: argparse.Namespace) -> PipelineExecutionResult:
    store = RuntimeStateStore()
    run_paths = store.start_run(mode=RuntimeMode.RELIABILITY, request=vars(args))
    store.append_event(run_paths, stage="pipeline", status="started", details={"mode": RuntimeMode.RELIABILITY.value})

    def sink(step_name: str, status: str, details: dict[str, Any] | None) -> None:
        payload = {"step_name": step_name}
        if details:
            payload.update(details)
        store.append_event(run_paths, stage="workflow_step", status=status, details=payload)

    legacy._set_step_event_sink(sink)
    try:
        manifest = legacy.execute_reliability_workflow(args)
    except Exception as exc:
        store.append_event(run_paths, stage="pipeline", status="failed", details={"error": str(exc)})
        store.finalize(
            run_paths,
            mode=RuntimeMode.RELIABILITY,
            status="failed",
            summary={"error": str(exc)},
        )
        raise
    finally:
        legacy._set_step_event_sink(None)

    store.write_predictions(run_paths, {"workflow_manifest": manifest})
    registry_payload = ReliabilityRunRegistry().record_workflow_manifest(manifest if isinstance(manifest, dict) else {})
    profile = manifest.get("profile") if isinstance(manifest, dict) else {}
    run_dir = manifest.get("run_dir") if isinstance(manifest, dict) else None
    store.finalize(
        run_paths,
        mode=RuntimeMode.RELIABILITY,
        status="succeeded",
        summary={
            "profile": profile,
            "run_dir": run_dir,
            "reliability_registry": registry_payload,
            "step_count": len(manifest.get("steps", [])) if isinstance(manifest, dict) else None,
        },
    )
    store.append_event(run_paths, stage="pipeline", status="completed", details={"run_id": run_paths.run_id})
    return PipelineExecutionResult(
        run_id=run_paths.run_id,
        mode=RuntimeMode.RELIABILITY,
        run_root=run_paths.root,
        predictions_payload={"workflow_manifest": manifest},
        metadata={"summary_path": run_paths.summary_path.as_posix()},
    )
