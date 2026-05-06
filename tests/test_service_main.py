from __future__ import annotations

from src.service.job_state import ServiceJobRecord
from src.service.main import RunRequest, RunResponse
import src.service.main as service_main
from src.service.orchestration import ServiceJobRequest


def test_jobs_endpoint_lists_registered_jobs() -> None:
    payload = service_main.jobs()
    names = {item["name"] for item in payload["jobs"]}
    assert {"live-inference", "research-refresh", "reliability-workflow", "walkforward-validation"}.issubset(names)


def test_run_dataset_refresh_uses_research_job(monkeypatch) -> None:
    recorded = {}

    def fake_build(endpoint_name, args=None, *, dry_run=False):
        recorded["endpoint_name"] = endpoint_name
        recorded["builder_args"] = list(args or [])
        recorded["builder_dry_run"] = dry_run
        return ServiceJobRequest("research-refresh", [*(args or []), *(["--dry-run"] if dry_run else [])])

    def fake_run(job_name: str, extra_args=None):
        recorded["job_name"] = job_name
        recorded["args"] = list(extra_args or [])
        return RunResponse(
            returncode=0,
            duration_seconds=0.01,
            stdout="ok",
            stderr="",
            job_name=job_name,
            job_id="job-123",
            run_id="test-run",
        )

    monkeypatch.setattr(service_main, "build_endpoint_request", fake_build)
    monkeypatch.setattr(service_main, "_run_registered_job", fake_run)

    response = service_main.run_dataset_refresh(
        RunRequest(args=["--config", "configs/run_refresh_and_predict.default.yaml"], dry_run=True)
    )

    assert recorded == {
        "endpoint_name": "run-dataset-refresh",
        "builder_args": ["--config", "configs/run_refresh_and_predict.default.yaml"],
        "builder_dry_run": True,
        "job_name": "research-refresh",
        "args": ["--config", "configs/run_refresh_and_predict.default.yaml", "--dry-run"],
    }
    assert response.job_name == "research-refresh"
    assert response.job_id == "job-123"


def test_run_reliability_workflow_uses_registered_job(monkeypatch) -> None:
    recorded = {}

    def fake_build(endpoint_name, args=None, *, dry_run=False):
        recorded["endpoint_name"] = endpoint_name
        recorded["builder_args"] = list(args or [])
        recorded["builder_dry_run"] = dry_run
        return ServiceJobRequest("reliability-workflow", list(args or []))

    def fake_run(job_name: str, extra_args=None):
        recorded["job_name"] = job_name
        recorded["args"] = list(extra_args or [])
        return RunResponse(
            returncode=0,
            duration_seconds=0.01,
            stdout="ok",
            stderr="",
            job_name=job_name,
            job_id="job-234",
            run_id="reliability-run",
        )

    monkeypatch.setattr(service_main, "build_endpoint_request", fake_build)
    monkeypatch.setattr(service_main, "_run_registered_job", fake_run)

    response = service_main.run_reliability_workflow(RunRequest(args=["--dry-run"]))

    assert recorded == {
        "endpoint_name": "run-reliability-workflow",
        "builder_args": ["--dry-run"],
        "builder_dry_run": False,
        "job_name": "reliability-workflow",
        "args": ["--dry-run"],
    }
    assert response.job_name == "reliability-workflow"


def test_get_job_run_reads_persisted_state(monkeypatch) -> None:
    class FakeStore:
        def get_job(self, job_id: str):
            return ServiceJobRecord(
                job_id=job_id,
                job_name="research-refresh",
                status="succeeded",
                args=["--dry-run"],
                created_at="2026-05-06T00:00:00Z",
                started_at="2026-05-06T00:00:00Z",
                finished_at="2026-05-06T00:01:00Z",
                run_id="run-123",
                returncode=0,
                error=None,
            )

    monkeypatch.setattr(service_main, "ServiceJobStateStore", FakeStore)

    response = service_main.get_job_run("job-123")

    assert response.job_id == "job-123"
    assert response.status == "succeeded"
