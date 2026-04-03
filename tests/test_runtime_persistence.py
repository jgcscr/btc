from __future__ import annotations

from pathlib import Path

from src.runtime.models import RuntimeMode
from src.runtime.persistence import RuntimeStateStore


def test_runtime_state_store_writes_structured_run(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path)
    paths = store.start_run(mode=RuntimeMode.LIVE, request={"targets": [1, 4, 8]})

    store.append_event(paths, stage="prediction", status="started", details={"targets": [1, 4, 8]})
    store.write_predictions(paths, {"generated_at": "2026-04-01T00:00:00Z"})
    store.write_monitoring(paths, {"source": "test"})
    store.write_trade_ready(paths, {"status": "ready"})
    store.finalize(paths, mode=RuntimeMode.LIVE, status="succeeded", summary={"preferred_horizon": "4h"})

    assert paths.request_path.exists()
    assert paths.events_path.exists()
    assert paths.summary_path.exists()
    assert paths.predictions_path.exists()
    assert paths.monitoring_path.exists()
    assert paths.trade_ready_path.exists()
    assert 'preferred_horizon' in paths.summary_path.read_text(encoding='utf-8')
