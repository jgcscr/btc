import pandas as pd
import pytest

from src.data import onchain_loader


class DummyResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload
        self.text = "payload"

    def json(self):
        return self._payload


def test_fetch_onchain_metrics_success(monkeypatch):
    payload = [
        {
            "ts": "2025-01-01T00:00:00Z",
            "active_addresses": 1,
            "transaction_count": 2,
            "hash_rate": 3.5,
            "market_cap": 4.1,
        },
        {
            "ts": "2025-01-01T01:00:00Z",
            "metrics": {
                "active_addresses": 5,
                "transaction_count": 6,
                "hash_rate": 7.5,
                "market_cap": 8.1,
            },
        },
    ]

    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return DummyResponse(200, payload)

    monkeypatch.setattr(onchain_loader, "ONCHAIN_API_BASE_URL", "https://example.com/api")
    monkeypatch.setattr(onchain_loader, "ONCHAIN_API_KEY", "secret")
    monkeypatch.setattr(onchain_loader, "ONCHAIN_METRICS", [
        "active_addresses",
        "transaction_count",
        "hash_rate",
        "market_cap",
    ])
    monkeypatch.setattr(onchain_loader.requests, "get", fake_get)

    df = onchain_loader.fetch_onchain_metrics(
        start_ts="2025-01-01T00:00:00Z",
        end_ts="2025-01-01T02:00:00Z",
        interval="1h",
    )

    assert captured["url"] == "https://example.com/api"
    assert captured["params"]["interval"] == "1h"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert list(df.columns) == [
        "ts",
        "active_addresses",
        "transaction_count",
        "hash_rate",
        "market_cap",
    ]
    assert len(df) == 2
    assert df.loc[0, "active_addresses"] == 1
    assert df.loc[1, "hash_rate"] == 7.5


def test_load_onchain_cached(tmp_path):
    csv_path = tmp_path / "cached.csv"
    csv_path.write_text(
        "ts,active_addresses,transaction_count,hash_rate,market_cap\n"
        "2025-01-01T00:00:00Z,1,2,3.5,4.1\n"
        "2025-01-01T01:00:00Z,5,6,7.5,8.1\n"
    )

    df = onchain_loader.load_onchain_cached(str(csv_path))

    assert list(df.columns) == [
        "ts",
        "active_addresses",
        "transaction_count",
        "hash_rate",
        "market_cap",
    ]
    assert df["ts"].dtype == "datetime64[ns, UTC]"
    assert df["hash_rate"].iloc[0] == pytest.approx(3.5)