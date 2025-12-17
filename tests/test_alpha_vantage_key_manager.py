import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from data.ingestors.alpha_vantage_macro import (
    AlphaVantageInvalidKeyError,
    AlphaVantageKeyManager,
    AlphaVantageRateLimitError,
)


class Clock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


@pytest.fixture()
def clock() -> Clock:
    return Clock(datetime(2025, 1, 1, tzinfo=timezone.utc))


@pytest.fixture()
def usage_path(tmp_path: Path) -> Path:
    return tmp_path / "usage.json"


def test_key_rotation_on_rate_limit(clock: Clock, usage_path: Path) -> None:
    manager = AlphaVantageKeyManager(
        ["key1", "key2"],
        usage_path=usage_path,
        base_backoff=1.0,
        now_fn=clock.now,
    )

    assert manager.acquire() == "key1"
    manager.mark_rate_limit("key1", retry_after=2.0)

    # Next acquire should rotate to key2 because key1 is backing off.
    assert manager.acquire() == "key2"
    manager.mark_success("key2")

    # Advance clock beyond key1 backoff so it becomes eligible again.
    clock.advance(2.0)
    assert manager.acquire() == "key1"


def test_rate_limit_when_all_keys_wait(clock: Clock, usage_path: Path) -> None:
    manager = AlphaVantageKeyManager(
        ["key1", "key2"],
        usage_path=usage_path,
        base_backoff=1.0,
        now_fn=clock.now,
    )
    manager.mark_rate_limit("key1", retry_after=5.0)
    manager.mark_rate_limit("key2", retry_after=10.0)

    with pytest.raises(AlphaVantageRateLimitError) as exc:
        manager.acquire()
    assert exc.value.wait_seconds == pytest.approx(5.0)


def test_invalid_key_removal(clock: Clock, usage_path: Path) -> None:
    manager = AlphaVantageKeyManager(
        ["key1", "key2"],
        usage_path=usage_path,
        base_backoff=1.0,
        now_fn=clock.now,
    )

    manager.mark_invalid("key1")
    assert manager.acquire() == "key2"
    manager.mark_invalid("key2")
    with pytest.raises(AlphaVantageInvalidKeyError):
        manager.acquire()


def test_usage_persistence(clock: Clock, usage_path: Path) -> None:
    manager = AlphaVantageKeyManager(
        ["key1"],
        usage_path=usage_path,
        base_backoff=1.0,
        now_fn=clock.now,
    )

    key = manager.acquire()
    manager.mark_success(key)
    manager.mark_rate_limit(key, retry_after=1.0)

    data = json.loads(usage_path.read_text())
    today = clock.now().date().isoformat()
    assert data[today]["key1"]["calls"] == 1
    assert data[today]["key1"]["rate_limit_hits"] == 1
    assert "last_updated" in data[today]["key1"]