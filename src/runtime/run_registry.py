from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.runtime.models import RuntimeMode, RuntimeRunPaths


class RuntimeRunRegistry:
    def __init__(self, root: Path | str = Path("artifacts/runtime_runs")) -> None:
        self.root = Path(root)

    def record_finalized_run(
        self,
        paths: RuntimeRunPaths,
        *,
        mode: RuntimeMode,
        status: str,
        summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "run_id": paths.run_id,
            "mode": mode.value,
            "status": status,
            "recorded_at": datetime.now().astimezone().isoformat(),
            "run_root": paths.root.as_posix(),
            "summary_path": paths.summary_path.as_posix(),
            "artifacts": {
                "request": paths.request_path.as_posix(),
                "events": paths.events_path.as_posix(),
                "summary": paths.summary_path.as_posix(),
                "predictions": paths.predictions_path.as_posix(),
                "monitoring": paths.monitoring_path.as_posix(),
                "trade_ready": paths.trade_ready_path.as_posix(),
            },
            "summary": _to_jsonable(summary or {}),
        }
        self._write_json(self.root / "latest.json", payload)
        self._write_json(self.root / "latest_by_mode" / f"{mode.value}.json", payload)
        return payload

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    return value