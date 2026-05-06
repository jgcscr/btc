from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ServiceJobRecord:
    job_id: str
    job_name: str
    status: str
    args: list[str]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    returncode: int | None = None
    error: str | None = None


class ServiceJobStateStore:
    def __init__(self, root: Path | str = Path("artifacts/service_jobs")) -> None:
        self.root = Path(root)
        self.records_root = self.root / "records"
        self.active_root = self.root / "active"

    def start_job(self, job_name: str, args: list[str]) -> ServiceJobRecord:
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        record = ServiceJobRecord(
            job_id=job_id,
            job_name=job_name,
            status="running",
            args=list(args),
            created_at=_utc_now(),
            started_at=_utc_now(),
        )
        self._acquire(job_name, job_id)
        self._write_record(record)
        return record

    def complete_job(
        self,
        record: ServiceJobRecord,
        *,
        returncode: int,
        run_id: str | None = None,
        error: str | None = None,
    ) -> ServiceJobRecord:
        record.status = "succeeded" if returncode == 0 else "failed"
        record.finished_at = _utc_now()
        record.returncode = int(returncode)
        record.run_id = run_id
        record.error = error
        self._write_record(record)
        self._release(record.job_name, record.job_id)
        return record

    def get_job(self, job_id: str) -> ServiceJobRecord | None:
        path = self.records_root / f"{job_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ServiceJobRecord(**payload)

    def _write_record(self, record: ServiceJobRecord) -> None:
        self.records_root.mkdir(parents=True, exist_ok=True)
        (self.records_root / f"{record.job_id}.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    def _acquire(self, job_name: str, job_id: str) -> None:
        self.active_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.active_root / f"{job_name}.lock"
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(job_id)
        except FileExistsError:
            active_job_id = lock_path.read_text(encoding="utf-8").strip() if lock_path.exists() else "unknown"
            raise RuntimeError(f"Job '{job_name}' is already active under {active_job_id}")

    def _release(self, job_name: str, job_id: str) -> None:
        lock_path = self.active_root / f"{job_name}.lock"
        if not lock_path.exists():
            return
        current = lock_path.read_text(encoding="utf-8").strip()
        if current == job_id:
            lock_path.unlink()