import os
import time
import subprocess
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        env["PYTHONPATH"] = os.getcwd()
    return env


def _invoke(module: str, extra_args: Optional[List[str]] = None) -> RunResponse:
    args = [DEFAULT_PYTHON, "-m", module]
    if extra_args:
        args.extend(extra_args)
    start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True, env=_build_env(), cwd=os.getcwd())
    duration = time.perf_counter() - start
    return RunResponse(
        returncode=proc.returncode,
        duration_seconds=duration,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@app.post("/run-signal", response_model=RunResponse)
def run_signal(req: RunRequest) -> RunResponse:
    args: List[str] = []
    if req.dry_run or req.dry_run is None:
        args.append("--dry-run")
    if req.args:
        args.extend(req.args)
    return _invoke("src.scripts.run_signal_realtime", args)


@app.post("/run-walkforward", response_model=RunResponse)
def run_walkforward(req: RunRequest) -> RunResponse:
    args = req.args or []
    return _invoke("src.scripts.run_walkforward_weekly", args)


@app.post("/run-papertrade", response_model=RunResponse)
def run_papertrade(req: RunRequest) -> RunResponse:
    args: List[str] = req.args or []
    if req.dry_run:
        args.append("--dry-run")
    return _invoke("src.scripts.paper_trade_loop", args)


@app.post("/run-dataset-refresh", response_model=RunResponse)
def run_dataset_refresh(req: RunRequest) -> RunResponse:
    args = req.args or []
    return _invoke("src.scripts.run_daily_refresh", args)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
