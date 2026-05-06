from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from src.runtime.reliability_workflow_common import StepResult, emit_step_event


def run_step(
    name: str,
    cmd: List[str],
    log_path: Path,
    dry_run: bool,
    *,
    allowed_returncodes: Sequence[int] | None = None,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(part) for part in cmd)
    allowed = set(int(code) for code in (allowed_returncodes or [0]))
    emit_step_event(
        name,
        "started",
        {"command": list(cmd), "log_path": str(log_path), "dry_run": bool(dry_run)},
    )
    if dry_run:
        log_path.write_text(f"[dry-run] {rendered}\n", encoding="utf-8")
        print(f"[dry-run] {name}: {rendered}")
        emit_step_event(name, "completed", {"returncode": 0, "dry_run": True, "log_path": str(log_path)})
        return StepResult(name=name, command=cmd, returncode=0, log_path=log_path)

    print(f"\n>>> {name}")
    print(rendered)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode not in allowed:
        emit_step_event(name, "failed", {"returncode": process.returncode, "log_path": str(log_path)})
        raise RuntimeError(f"Step '{name}' failed (exit={process.returncode}). See {log_path}")
    if process.returncode != 0:
        print(
            f"Warning: step '{name}' returned non-zero exit {process.returncode} but is configured as allowed.",
            file=sys.stderr,
        )
    emit_step_event(name, "completed", {"returncode": process.returncode, "log_path": str(log_path)})
    return StepResult(name=name, command=cmd, returncode=process.returncode, log_path=log_path)