from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCandidate:
    path: Path
    target_column: str
    base_horizon: float
    offline_only: bool = False


@dataclass(frozen=True)
class DatasetProfile:
    key: str
    candidates: tuple[DatasetCandidate, ...]