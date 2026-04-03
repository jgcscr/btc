from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from src.service.job_runner import JobRunResult


@dataclass(frozen=True)
class ServiceJobRequest:
    job_name: str
    args: list[str]


@dataclass(frozen=True)
class ServiceEndpointSpec:
    name: str
    path: str
    job_name: str
    handler_name: str
    apply_dry_run: bool = False


def build_live_inference_request(args: Sequence[str] | None = None) -> ServiceJobRequest:
    return ServiceJobRequest(job_name="live-inference", args=list(args or []))


def build_walkforward_request(args: Sequence[str] | None = None) -> ServiceJobRequest:
    return ServiceJobRequest(job_name="walkforward-validation", args=list(args or []))


def build_reliability_workflow_request(args: Sequence[str] | None = None) -> ServiceJobRequest:
    return ServiceJobRequest(job_name="reliability-workflow", args=list(args or []))


def build_dataset_refresh_request(
    args: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> ServiceJobRequest:
    request_args = list(args or [])
    if dry_run:
        request_args.append("--dry-run")
    return ServiceJobRequest(job_name="research-refresh", args=request_args)


ENDPOINT_SPECS: dict[str, ServiceEndpointSpec] = {
    "run-signal": ServiceEndpointSpec(
        name="run-signal",
        path="/run-signal",
        job_name="live-inference",
        handler_name="run_signal",
    ),
    "run-walkforward": ServiceEndpointSpec(
        name="run-walkforward",
        path="/run-walkforward",
        job_name="walkforward-validation",
        handler_name="run_walkforward",
    ),
    "run-dataset-refresh": ServiceEndpointSpec(
        name="run-dataset-refresh",
        path="/run-dataset-refresh",
        job_name="research-refresh",
        handler_name="run_dataset_refresh",
        apply_dry_run=True,
    ),
    "run-reliability-workflow": ServiceEndpointSpec(
        name="run-reliability-workflow",
        path="/run-reliability-workflow",
        job_name="reliability-workflow",
        handler_name="run_reliability_workflow",
    ),
}


def build_endpoint_request(endpoint_name: str, args: Sequence[str] | None = None, *, dry_run: bool = False) -> ServiceJobRequest:
    spec = ENDPOINT_SPECS.get(endpoint_name)
    if spec is None:
        raise KeyError(endpoint_name)
    if spec.job_name == "live-inference":
        return build_live_inference_request(args)
    if spec.job_name == "walkforward-validation":
        return build_walkforward_request(args)
    if spec.job_name == "reliability-workflow":
        return build_reliability_workflow_request(args)
    if spec.job_name == "research-refresh":
        return build_dataset_refresh_request(args, dry_run=dry_run and spec.apply_dry_run)
    raise KeyError(spec.job_name)


def run_service_job(
    request: ServiceJobRequest,
    *,
    available_jobs: Mapping[str, object],
    run_job_fn: Callable[[str, Sequence[str] | None], JobRunResult],
) -> JobRunResult:
    if request.job_name not in available_jobs:
        raise KeyError(request.job_name)
    return run_job_fn(request.job_name, request.args)