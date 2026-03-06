from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class TimeSeriesFold:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def build_time_series_folds(
    n_samples: int,
    *,
    n_splits: int,
    train_size: int,
    val_size: int,
    test_size: int,
    gap: int = 0,
    step_size: int | None = None,
    mode: str = "expanding",
) -> List[TimeSeriesFold]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("train_size, val_size, and test_size must be positive")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'")

    step = step_size if step_size is not None else test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    folds: List[TimeSeriesFold] = []
    for idx in range(n_splits):
        offset = idx * step
        if mode == "expanding":
            train_start = 0
            train_end = train_size + offset
        else:
            train_start = offset
            train_end = train_start + train_size

        val_start = train_end + gap
        val_end = val_start + val_size
        test_start = val_end + gap
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        folds.append(
            TimeSeriesFold(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    if not folds:
        raise ValueError("Insufficient samples for requested time-series folds")

    return folds


def iter_fold_indices(folds: Sequence[TimeSeriesFold]) -> Iterable[tuple[slice, slice, slice]]:
    for fold in folds:
        yield fold.train_slice, fold.val_slice, fold.test_slice
