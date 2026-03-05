"""Pytest configuration helpers and fixtures for the BTC project."""

import os
import sys

import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


@pytest.fixture(autouse=True)
def _disable_live_vendor_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default tests to sample payloads unless explicitly opted-in."""

    if _env_truthy(os.getenv("LIVE_DATA_OK")):
        return
    monkeypatch.setenv("LIVE_DATA_OK", "0")


@pytest.fixture(name="requests_mock")
def _requests_mock_fixture():
    """Provide a lightweight requests-mock harness without relying on pytest plugins."""

    import requests_mock as requests_mock_lib

    with requests_mock_lib.Mocker() as mocker:
        yield mocker
