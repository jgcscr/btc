import os
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.service.job_runner import JOB_SPECS, list_jobs, run_job
from src.service.orchestration import (
    ENDPOINT_SPECS,
    build_endpoint_request,
    run_service_job,
)

app = FastAPI(title="BTC Trading Service", version="1.0.0")

DEFAULT_PYTHON = sys.executable


class RunRequest(BaseModel):
    args: Optional[List[str]] = None
    dry_run: Optional[bool] = None


class RunResponse(BaseModel):
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    job_name: str
    run_id: Optional[str] = None


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        env["PYTHONPATH"] = os.getcwd()
    return env


def _run_registered_job(job_name: str, extra_args: Optional[List[str]] = None) -> RunResponse:
    try:
        result = run_service_job(
            build_endpoint_request(
                next(
                    (endpoint_name for endpoint_name, spec in ENDPOINT_SPECS.items() if spec.job_name == job_name),
                    job_name,
                ),
                extra_args,
            ),
            available_jobs=JOB_SPECS,
            run_job_fn=lambda name, args: run_job(name, args=args),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_name}") from None
    return RunResponse(
        returncode=result.returncode,
        duration_seconds=result.duration_seconds,
        stdout=result.stdout,
        stderr=result.stderr,
        job_name=result.job_name,
        run_id=result.run_id,
    )


@app.get("/jobs")
def jobs() -> Dict[str, List[Dict[str, str]]]:
    return {
        "jobs": [
            {
                "name": job.name,
                "module_name": job.module_name,
                "description": job.description,
            }
            for job in list_jobs()
        ]
    }


@app.post("/jobs/{job_name}", response_model=RunResponse)
def run_registered_job(job_name: str, req: RunRequest) -> RunResponse:
    return _run_registered_job(job_name, req.args)


def _run_endpoint_job(endpoint_name: str, req: RunRequest) -> RunResponse:
    try:
        service_request = build_endpoint_request(endpoint_name, req.args, dry_run=bool(req.dry_run))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {endpoint_name}") from None
    return _run_registered_job(service_request.job_name, service_request.args)


def _register_endpoint_handlers() -> None:
    for endpoint_name, spec in ENDPOINT_SPECS.items():
        def handler(req: RunRequest, endpoint_name: str = endpoint_name) -> RunResponse:
            return _run_endpoint_job(endpoint_name, req)

        handler.__name__ = spec.handler_name
        globals()[spec.handler_name] = handler
        app.post(spec.path, response_model=RunResponse)(handler)


_register_endpoint_handlers()


@app.post("/run-papertrade", response_model=RunResponse)
def run_papertrade(req: RunRequest) -> RunResponse:
    raise HTTPException(status_code=501, detail="Paper trading service endpoint is not implemented in this workspace.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "jobs": str(len(JOB_SPECS)), "python": DEFAULT_PYTHON}
