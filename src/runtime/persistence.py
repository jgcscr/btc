from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.runtime.models import PipelineStage, PipelineStatus, RuntimeEvent, RuntimeMode, RuntimeRunPaths
from src.runtime.run_registry import RuntimeRunRegistry
from src.runtime.storage import ArtifactStorage, FileSystemArtifactStorage


class RuntimeStateStore:
    def __init__(
        self,
        root: Path | str = Path("artifacts/runtime_runs"),
        *,
        artifact_storage: ArtifactStorage | None = None,
    ) -> None:
        self.root = Path(root)
        self.registry = RuntimeRunRegistry(self.root)
        self.artifact_storage = artifact_storage or FileSystemArtifactStorage()

    def start_run(self, *, mode: RuntimeMode, request: Mapping[str, Any]) -> RuntimeRunPaths:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_id = f"{mode.value}-{timestamp}-{uuid4().hex[:8]}"
        run_root = self.root / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        paths = RuntimeRunPaths(
            run_id=run_id,
            root=run_root,
            request_path=run_root / "request.json",
            events_path=run_root / "events.jsonl",
            summary_path=run_root / "summary.json",
            predictions_path=run_root / "predictions.json",
            monitoring_path=run_root / "monitoring.json",
            trade_ready_path=run_root / "trade_ready.json",
        )
        self._write_json(paths.request_path, request)
        return paths

    def append_event(
        self,
        paths: RuntimeRunPaths,
        *,
        stage: PipelineStage | str,
        status: PipelineStatus | str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        event = RuntimeEvent(
            ts=datetime.now().astimezone().isoformat(),
            stage=stage.value if isinstance(stage, PipelineStage) else str(stage),
            status=status.value if isinstance(status, PipelineStatus) else str(status),
            details=self._to_jsonable(details or {}),
        )
        self.artifact_storage.append_event(paths.events_path, event)

    def write_predictions(self, paths: RuntimeRunPaths, payload: Mapping[str, Any]) -> None:
        self._write_json(paths.predictions_path, payload)

    def write_monitoring(self, paths: RuntimeRunPaths, payload: Mapping[str, Any]) -> None:
        self._write_json(paths.monitoring_path, payload)

    def write_trade_ready(self, paths: RuntimeRunPaths, payload: Mapping[str, Any]) -> None:
        self._write_json(paths.trade_ready_path, payload)

    def finalize(
        self,
        paths: RuntimeRunPaths,
        *,
        mode: RuntimeMode,
        status: str,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "run_id": paths.run_id,
            "mode": mode.value,
            "status": status,
            "completed_at": datetime.now().astimezone().isoformat(),
            "artifacts": {
                "request": paths.request_path.as_posix(),
                "events": paths.events_path.as_posix(),
                "summary": paths.summary_path.as_posix(),
                "predictions": paths.predictions_path.as_posix(),
                "monitoring": paths.monitoring_path.as_posix(),
                "trade_ready": paths.trade_ready_path.as_posix(),
            },
            "summary": self._to_jsonable(summary or {}),
        }
        self._write_json(paths.summary_path, payload)
        self.registry.record_finalized_run(
            paths,
            mode=mode,
            status=status,
            summary=summary,
        )

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.artifact_storage.write_json(path, payload)

    def _to_jsonable(self, value: Any) -> Any:
        return self.artifact_storage.to_jsonable(value)
