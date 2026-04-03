from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class RuntimeMode(str, Enum):
    LIVE = "live"
    RESEARCH = "research"
    RELIABILITY = "reliability"


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


@dataclass
class PipelineExecutionResult:
    run_id: str
    mode: RuntimeMode
    run_root: Path
    predictions_payload: Mapping[str, Any]
    monitoring_payload: Mapping[str, Any] | None = None
    trade_ready_payload: Mapping[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
