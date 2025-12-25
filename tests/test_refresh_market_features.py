from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

import src.scripts.refresh_market_features as refresh


class StubError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def reset_module_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(refresh, "ONCHAIN_RAW_ROOT", tmp_path / "onchain_raw")
    monkeypatch.setattr(refresh, "FUNDING_RAW_ROOT", tmp_path / "funding_raw")
    monkeypatch.setattr(refresh, "FUNDING_OUTPUT_PATH", tmp_path / "funding.parquet")
    monkeypatch.setattr(refresh, "TECHNICAL_SUMMARY_PATH", tmp_path / "technical_summary.json")
    monkeypatch.setattr(refresh, "ONCHAIN_SUMMARY_PATH", tmp_path / "onchain_summary.json")
    monkeypatch.setattr(refresh, "FUNDING_SUMMARY_PATH", tmp_path / "funding_summary.json")
    monkeypatch.setattr(refresh, "ONCHAIN_METRICS", ["metric_a", "metric_b"])


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    onchain_calls = []
    funding_calls = []
    technical_calls = []

    def fake_ingest(*, metrics, limit, output_root, api_key):
        onchain_calls.append({
            "metrics": metrics,
            "limit": limit,
            "output_root": output_root,
            "api_key": api_key,
        })
        paths = []
        for metric in metrics:
            dest = tmp_path / f"raw_{metric}.parquet"
            dest.touch()
            paths.append(dest)
        return paths

    def fake_onchain_features() -> Path:
        path = tmp_path / "onchain.parquet"
        path.touch()
        _write_summary(refresh.ONCHAIN_SUMMARY_PATH, {"latest_timestamp": "2024-01-01T00:00:00Z"})
        return path

    def fake_funding_features(**kwargs: Any) -> Path:
        funding_calls.append(kwargs)
        path = tmp_path / "funding.parquet"
        path.touch()
        _write_summary(refresh.FUNDING_SUMMARY_PATH, {"latest_timestamp": "2024-01-02T00:00:00Z"})
        return path

    def fake_technical_features(**kwargs: Any) -> Path:
        technical_calls.append(kwargs)
        path = tmp_path / "technical.parquet"
        path.touch()
        _write_summary(refresh.TECHNICAL_SUMMARY_PATH, {"latest_timestamp": "2024-01-03T00:00:00Z"})
        return path

    monkeypatch.setattr(refresh, "ingest_cryptocompare_metrics", fake_ingest)
    monkeypatch.setattr(refresh, "process_onchain_features", fake_onchain_features)
    monkeypatch.setattr(refresh, "process_funding_features", fake_funding_features)
    monkeypatch.setattr(refresh, "process_technical_features", fake_technical_features)

    exit_code = refresh.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert onchain_calls[0]["metrics"] == refresh.ONCHAIN_METRICS
    assert onchain_calls[0]["limit"] == refresh.DEFAULT_ONCHAIN_LIMIT
    assert onchain_calls[0]["output_root"] == refresh.ONCHAIN_RAW_ROOT
    assert onchain_calls[0]["api_key"] is None
    assert funding_calls[0]["pair"] == refresh.FUNDING_PAIR
    assert funding_calls[0]["live_fetch"] is True
    assert funding_calls[0]["live_limit"] == refresh.DEFAULT_FUNDING_LIMIT
    assert technical_calls[0]["include_history"] is True
    assert technical_calls[0]["history_limit"] == refresh.TECHNICAL_HISTORY_LIMIT
    output = json.loads(captured.out)
    assert "onchain" in output
    assert "funding" in output
    assert "technical" in output


def test_skip_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_onchain(*_: Any, **__: Any) -> Path:
        raise AssertionError("onchain should be skipped")

    def fake_funding(**_: Any) -> Path:
        raise AssertionError("funding should be skipped")

    def fake_technical_features(**kwargs: Any) -> Path:
        _write_summary(refresh.TECHNICAL_SUMMARY_PATH, {"latest_timestamp": "2024-01-03T00:00:00Z"})
        return tmp_path / "technical.parquet"

    monkeypatch.setattr(refresh, "ingest_cryptocompare_metrics", fake_onchain)
    monkeypatch.setattr(refresh, "process_onchain_features", fake_onchain)
    monkeypatch.setattr(refresh, "process_funding_features", fake_funding)
    monkeypatch.setattr(refresh, "process_technical_features", fake_technical_features)

    exit_code = refresh.main(["--skip-onchain", "--skip-funding"])
    captured = capsys.readouterr()

    assert exit_code == 0
    output = json.loads(captured.out)
    assert "onchain" not in output
    assert "funding" not in output
    assert "technical" in output


def test_failure_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_technical(**kwargs: Any) -> Path:
        raise StubError("technical boom")

    monkeypatch.setattr(refresh, "ingest_cryptocompare_metrics", lambda **kwargs: [Path("ignored")])
    monkeypatch.setattr(refresh, "process_onchain_features", lambda: Path("ignored"))
    monkeypatch.setattr(refresh, "process_funding_features", lambda **_: Path("ignored"))
    monkeypatch.setattr(refresh, "process_technical_features", fake_technical)

    exit_code = refresh.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "technical boom" in captured.err
