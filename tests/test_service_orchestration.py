from __future__ import annotations

from src.service.job_runner import JobRunResult
from src.service.orchestration import (
    ENDPOINT_SPECS,
    ServiceJobRequest,
    build_dataset_refresh_request,
    build_endpoint_request,
    build_live_inference_request,
    build_reliability_workflow_request,
    build_walkforward_request,
    run_service_job,
)


def test_build_dataset_refresh_request_appends_dry_run_flag() -> None:
    request = build_dataset_refresh_request(["--config", "configs/run_refresh_and_predict.default.yaml"], dry_run=True)

    assert request == ServiceJobRequest(
        job_name="research-refresh",
        args=["--config", "configs/run_refresh_and_predict.default.yaml", "--dry-run"],
    )


def test_service_request_builders_map_to_expected_jobs() -> None:
    assert build_live_inference_request(["--dry-run"]).job_name == "live-inference"
    assert build_walkforward_request().job_name == "walkforward-validation"
    assert build_reliability_workflow_request(["--config", "x"]).job_name == "reliability-workflow"


def test_endpoint_registry_builds_dataset_refresh_request_with_dry_run() -> None:
    request = build_endpoint_request(
        "run-dataset-refresh",
        ["--config", "configs/run_refresh_and_predict.default.yaml"],
        dry_run=True,
    )

    assert ENDPOINT_SPECS["run-dataset-refresh"].job_name == "research-refresh"
    assert request == ServiceJobRequest(
        job_name="research-refresh",
        args=["--config", "configs/run_refresh_and_predict.default.yaml", "--dry-run"],
    )


def test_run_service_job_dispatches_registered_job() -> None:
    captured = {}

    def fake_run_job(name, args=None):
        captured["name"] = name
        captured["args"] = list(args or [])
        return JobRunResult(
            returncode=0,
            duration_seconds=0.1,
            stdout="ok",
            stderr="",
            job_name=name,
            run_id="run-123",
        )

    result = run_service_job(
        ServiceJobRequest(job_name="research-refresh", args=["--dry-run"]),
        available_jobs={"research-refresh": object()},
        run_job_fn=fake_run_job,
    )

    assert captured == {"name": "research-refresh", "args": ["--dry-run"]}
    assert result.run_id == "run-123"