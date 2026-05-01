from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Mapping

from src.scripts import run_reliability_workflow as legacy


def execute_reliability_workflow(
    args: argparse.Namespace,
    *,
    step_event_sink: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
) -> Dict[str, Any]:
    return legacy.execute_reliability_workflow(args, step_event_sink=step_event_sink)