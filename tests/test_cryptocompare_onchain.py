from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from data.ingestors import cryptocompare_onchain as cc


class DummyResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.text = "payload"

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_fetch_hourly_metrics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "Response": "Success",
        "Data": {
            "Data": [
                {"time": 1700000000, "value": 5},
                {"time": 1700003600, "value": 6},
            ],
        },
    }

    def fake_get(url: str, params: Dict[str, Any], timeout: int):
        assert url == cc.BASE_URL
        assert params["fsym"] == "BTC"
        assert params["metric"] == "activeaddresses"
        assert params["limit"] == 10
        assert "api_key" not in params
        assert timeout == 30
        return DummyResponse(200, payload)

    monkeypatch.setattr(cc.requests, "get", fake_get)

    frame = cc.fetch_hourly_metrics(metric="active_addresses", limit=10, api_key=None)

    assert list(frame.columns) == ["ts", "metric", "value", "source"]
    assert frame.iloc[0]["metric"] == "active_addresses"
    assert frame.iloc[0]["value"] == 5.0
    assert frame.iloc[0]["source"] == cc.SOURCE_NAME
    assert pd.api.types.is_datetime64tz_dtype(frame["ts"])


def test_fetch_hourly_metrics_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, params: Dict[str, Any], timeout: int):
        response = DummyResponse(500, {"error": "boom"})
        response.text = "boom"
        return response

    monkeypatch.setattr(cc.requests, "get", fake_get)

    with pytest.raises(cc.CryptoCompareIngestionError) as exc:
        cc.fetch_hourly_metrics(metric="active_addresses", limit=1, api_key=None)

    assert "status 500" in str(exc.value)


def test_fetch_hourly_metrics_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"Response": "Success", "Data": {"Data": []}}

    def fake_get(url: str, params: Dict[str, Any], timeout: int):
        return DummyResponse(200, payload)

    monkeypatch.setattr(cc.requests, "get", fake_get)

    with pytest.raises(cc.CryptoCompareIngestionError) as exc:
        cc.fetch_hourly_metrics(metric="active_addresses", limit=5, api_key=None)

    assert "did not include any records" in str(exc.value)


def test_ingest_metrics_writes_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frames = [
        pd.DataFrame({
            "ts": pd.to_datetime([1700000000, 1700003600], unit="s", utc=True),
            "metric": ["active_addresses", "active_addresses"],
            "value": [5.0, 6.0],
            "source": [cc.SOURCE_NAME, cc.SOURCE_NAME],
        })
    ]

    def fake_fetch(metric: str, limit: int, api_key: str | None) -> pd.DataFrame:
        assert metric == "active_addresses"
        assert limit == 3
        assert api_key == "secret"
        return frames[0]

    monkeypatch.setenv("CRYPTOCOMPARE_API_KEY", "secret")
    monkeypatch.setattr(cc, "fetch_hourly_metrics", fake_fetch)

    paths = cc.ingest_metrics(metrics=["active_addresses"], limit=3, output_root=tmp_path, api_key=None)

    assert len(paths) == 1
    assert paths[0].exists()
    df = pd.read_parquet(paths[0])
    assert list(df.columns) == ["ts", "metric", "value", "source"]
    assert len(df) == 2


def test_fetch_hourly_metrics_unsupported_metric() -> None:
    with pytest.raises(cc.CryptoCompareIngestionError) as exc:
        cc.fetch_hourly_metrics(metric="unsupported", limit=10, api_key=None)

    assert "Supported" in str(exc.value)