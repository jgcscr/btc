from __future__ import annotations

from typing import Any, Dict


def append_gate_trace(
    entry: Dict[str, Any],
    *,
    stage: str,
    reason: str,
    triggered: bool,
    blocking: bool,
) -> None:
    trace = entry.get("gate_trace")
    if not isinstance(trace, list):
        trace = []
        entry["gate_trace"] = trace
    trace.append(
        {
            "stage": stage,
            "reason": reason,
            "triggered": bool(triggered),
            "blocking": bool(blocking),
        }
    )