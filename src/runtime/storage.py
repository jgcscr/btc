from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.runtime.models import RuntimeEvent


class ArtifactStorage(Protocol):
    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        ...

    def append_event(self, path: Path, event: RuntimeEvent) -> None:
        ...

    def to_jsonable(self, value: Any) -> Any:
        ...


class FileSystemArtifactStorage:
    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(payload), indent=2), encoding="utf-8")

    def append_event(self, path: Path, event: RuntimeEvent) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": event.ts,
                        "stage": event.stage,
                        "status": event.status,
                        "details": self.to_jsonable(dict(event.details)),
                    }
                )
            )
            handle.write("\n")

    def to_jsonable(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self.to_jsonable(inner) for key, inner in value.items()}
        if isinstance(value, Namespace):
            return {key: self.to_jsonable(inner) for key, inner in vars(value).items()}
        if isinstance(value, (list, tuple, set)):
            return [self.to_jsonable(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value