from __future__ import annotations

import pytest

from src.service.job_state import ServiceJobStateStore


def test_service_job_state_store_records_and_reads_job(tmp_path) -> None:
    store = ServiceJobStateStore(tmp_path)
    record = store.start_job("research-refresh", ["--dry-run"])
    store.complete_job(record, returncode=0, run_id="run-123")

    loaded = store.get_job(record.job_id)

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.run_id == "run-123"


def test_service_job_state_store_blocks_concurrent_same_job(tmp_path) -> None:
    store = ServiceJobStateStore(tmp_path)
    record = store.start_job("research-refresh", [])

    with pytest.raises(RuntimeError, match="already active"):
        store.start_job("research-refresh", [])

    store.complete_job(record, returncode=0)