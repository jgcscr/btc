import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from data.ingestors import alpha_vantage_macro as macro


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MACRO_PROVIDER", raising=False)
    monkeypatch.delenv(macro.TWELVE_DATA_API_KEY_ENV, raising=False)


def _build_payload(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "values": values,
    }


def test_twelve_ingest_maps_alias_and_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_symbols: List[str] = []

    def fake_get(url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        assert url == macro.TWELVE_DATA_URL
        captured_symbols.append(params["symbol"])
        assert params["interval"] == "1day"
        return _FakeResponse(200, _build_payload([
            {
                "datetime": "2023-02-08 15:30:00",
                "open": "27.93",
                "high": "27.94",
                "low": "27.88",
                "close": "27.915",
                "volume": "193260",
            },
        ]))

    monkeypatch.setenv(macro.TWELVE_DATA_API_KEY_ENV, "test-key")
    monkeypatch.setattr("data.ingestors.alpha_vantage_macro.requests.get", fake_get)

    output_root = tmp_path / "macro"
    path, summary = macro.ingest_series(
        function="TIME_SERIES_DAILY",
        params={},
        output_root=output_root,
        canonical_symbol="DXY",
        provider="twelve",
        return_summary=True,
    )

    assert captured_symbols == ["UUP"]
    assert path.exists()
    persisted = pd.read_parquet(path)
    assert set(persisted["metric"].unique()) == {
        "DXY_open",
        "DXY_high",
        "DXY_low",
        "DXY_close",
        "DXY_volume",
    }
    assert summary["provider"] == "twelve"
    assert summary["params"].get("provider_symbol") == "UUP"


def test_twelve_ingest_falls_back_to_secondary_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: List[Dict[str, Any]] = []

    responses = [
        _FakeResponse(429, {"status": "error", "message": "Rate limit"}),
        _FakeResponse(200, _build_payload([
            {
                "datetime": "2023-02-08 19:30:00",
                "open": "189.82",
                "high": "191.5",
                "low": "189.48",
                "close": "189.8",
                "volume": "495532",
            },
        ])),
    ]

    def fake_get(url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        attempts.append(params.copy())
        return responses.pop(0)

    monkeypatch.setenv(macro.TWELVE_DATA_API_KEY_ENV, "test-key")
    monkeypatch.setattr("data.ingestors.alpha_vantage_macro.requests.get", fake_get)

    path, summary = macro.ingest_series(
        function="TIME_SERIES_DAILY",
        params={},
        output_root=tmp_path,
        canonical_symbol="VIX",
        provider="twelve",
        return_summary=True,
        aliases=["VIXY"],
    )

    attempted_symbols = [item["symbol"] for item in attempts]
    assert attempted_symbols == ["VIXY", "VIX"]
    assert summary["params"].get("provider_symbol") == "VIX"
    frame = pd.read_parquet(path)
    assert not frame.empty
    assert (frame["metric"].str.startswith("VIX_")).all()


def test_provider_resolves_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MACRO_PROVIDER", "twelve_data")
    monkeypatch.setenv(macro.TWELVE_DATA_API_KEY_ENV, "test-key")

    def fake_get(url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        return _FakeResponse(200, _build_payload([
            {
                "datetime": "2023-01-01 00:00:00",
                "open": "27",
                "high": "27.1",
                "low": "26.9",
                "close": "27.05",
                "volume": "1000",
            },
        ]))

    monkeypatch.setattr("data.ingestors.alpha_vantage_macro.requests.get", fake_get)

    output_path = macro.ingest_series(
        function="TIME_SERIES_DAILY",
        params={},
        output_root=tmp_path,
        canonical_symbol="DXY",
        return_summary=False,
    )

    assert Path(output_path).exists()


def test_twelve_non_200_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(macro.TWELVE_DATA_API_KEY_ENV, "test-key")

    def fake_get(url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        return _FakeResponse(500, {"status": "error", "message": "Internal error"})

    monkeypatch.setattr("data.ingestors.alpha_vantage_macro.requests.get", fake_get)

    with pytest.raises(macro.TwelveDataIngestionError, match="Internal error"):
        macro.ingest_series(
            function="TIME_SERIES_DAILY",
            params={},
            output_root=tmp_path,
            canonical_symbol="DXY",
            provider="twelve",
        )
