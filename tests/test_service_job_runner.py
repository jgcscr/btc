from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from src.service.job_state import ServiceJobStateStore
from src.service.job_runner import execute_job_in_process, run_job


def test_execute_job_in_process_captures_run_id(monkeypatch) -> None:
    class FakeResult:
        run_id = "run-123"

    class FakeModule:
        @staticmethod
        def main(argv):
            print(f"argv={argv}")
            return FakeResult()

    monkeypatch.setattr("importlib.import_module", lambda _: FakeModule)

    result = execute_job_in_process("research-refresh", ["--dry-run"])

    assert result.returncode == 0
    assert result.run_id == "run-123"
    assert "argv=['--dry-run']" in result.stdout


def test_run_job_uses_worker_metadata(monkeypatch, tmp_path) -> None:
    def fake_run(command, capture_output, text, env, check):
        metadata_path = Path(command[command.index("--metadata-path") + 1])
        metadata_path.write_text(
            json.dumps(
                {
                    "job_name": "research-refresh",
                    "run_id": "worker-run-1",
                    "returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        return CompletedProcess(command, 0, stdout="worker stdout", stderr="")

    class FakeTemporaryDirectory:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("tempfile.TemporaryDirectory", lambda prefix=None: FakeTemporaryDirectory())
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("src.service.job_runner.ServiceJobStateStore", lambda: ServiceJobStateStore(tmp_path / "jobs"))

    result = run_job("research-refresh", ["--dry-run"])

    assert result.returncode == 0
    assert result.job_id is not None
    assert result.run_id == "worker-run-1"
    assert result.stdout == "worker stdout"