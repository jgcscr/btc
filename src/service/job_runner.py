from __future__ import annotations

import importlib
import io
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass(frozen=True)
class JobSpec:
    name: str
    module_name: str
    accepts_argv: bool
    description: str


@dataclass
class JobRunResult:
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    job_name: str
    run_id: str | None = None


JOB_SPECS: dict[str, JobSpec] = {
    "live-inference": JobSpec(
        name="live-inference",
        module_name="src.scripts.run_live_inference",
        accepts_argv=True,
        description="Constrained live inference refresh path.",
    ),
    "research-refresh": JobSpec(
        name="research-refresh",
        module_name="src.scripts.run_research_refresh",
        accepts_argv=True,
        description="Full research refresh and prediction workflow.",
    ),
    "reliability-workflow": JobSpec(
        name="reliability-workflow",
        module_name="src.scripts.run_reliability_pipeline",
        accepts_argv=True,
        description="Reliability recalibration and deployment workflow.",
    ),
    "walkforward-validation": JobSpec(
        name="walkforward-validation",
        module_name="src.scripts.run_walkforward_validation",
        accepts_argv=False,
        description="Walkforward validation workflow.",
    ),
}


def list_jobs() -> list[JobSpec]:
    return [JOB_SPECS[key] for key in sorted(JOB_SPECS.keys())]


def run_job(name: str, args: Iterable[str] | None = None) -> JobRunResult:
    spec = JOB_SPECS.get(name)
    if spec is None:
        raise KeyError(name)
    argv = list(args or [])
    module = importlib.import_module(spec.module_name)
    main_callable = getattr(module, "main", None)
    if not callable(main_callable):
        raise RuntimeError(f"Module {spec.module_name} does not expose a callable main().")

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter()
    run_id = None
    returncode = 0

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            if spec.accepts_argv:
                result = main_callable(argv)
            else:
                with _temporary_argv(spec.module_name, argv):
                    result = main_callable()
        run_id = getattr(result, "run_id", None)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        returncode = int(code)
    except Exception:
        returncode = 1
        traceback.print_exc(file=stderr_buffer)

    duration = time.perf_counter() - start
    return JobRunResult(
        returncode=returncode,
        duration_seconds=duration,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
        job_name=name,
        run_id=run_id,
    )


@contextmanager
def _temporary_argv(module_name: str, argv: List[str]):
    original = sys.argv[:]
    sys.argv = [module_name, *argv]
    try:
        yield
    finally:
        sys.argv = original
