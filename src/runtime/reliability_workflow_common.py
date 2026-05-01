from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import yaml


@dataclass(frozen=True)
class StepResult:
    name: str
    command: List[str]
    returncode: int
    log_path: Path


_STEP_EVENT_SINK: ContextVar[Callable[[str, str, Mapping[str, Any] | None], None] | None] = ContextVar(
    "_STEP_EVENT_SINK",
    default=None,
)


def run_with_step_event_sink(
    sink: Callable[[str, str, Mapping[str, Any] | None], None],
    callback: Callable[[], Any],
) -> Any:
    token = _STEP_EVENT_SINK.set(sink)
    try:
        return callback()
    finally:
        _STEP_EVENT_SINK.reset(token)


def emit_step_event(name: str, status: str, details: Mapping[str, Any] | None = None) -> None:
    sink = _STEP_EVENT_SINK.get()
    if sink is not None:
        sink(name, status, details)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow config must be a mapping: {path}")
    return payload