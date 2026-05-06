from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class RuntimeMode(str, Enum):
    LIVE = "live"
    RESEARCH = "research"
    RELIABILITY = "reliability"


class PipelineStage(str, Enum):
    PIPELINE = "pipeline"
    DATA_PREPARATION = "data_preparation"
    MODEL_INPUT_RESOLUTION = "model_input_resolution"
    PREDICTION = "prediction"
    ARTIFACT_WRITING = "artifact_writing"
    WORKFLOW_STEP = "workflow_step"


class PipelineStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeRunPaths:
    run_id: str
    root: Path
    request_path: Path
    events_path: Path
    summary_path: Path
    predictions_path: Path
    monitoring_path: Path
    trade_ready_path: Path


@dataclass(frozen=True)
class RuntimeEvent:
    ts: str
    stage: str
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecutionResult:
    run_id: str
    mode: RuntimeMode
    run_root: Path
    predictions_payload: Mapping[str, Any]
    monitoring_payload: Mapping[str, Any] | None = None
    trade_ready_payload: Mapping[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
