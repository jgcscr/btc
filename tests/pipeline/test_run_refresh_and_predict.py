import json
from pathlib import Path

from src.scripts import run_refresh_and_predict

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_KLINES_PATH = FIXTURES_DIR / "binance_us_1h.json"


class DummyResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def json(self):
        return json.loads(self._payload)

    def raise_for_status(self):
        return None


def test_dry_run_produces_stub_predictions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def forbid(*_args, **_kwargs):
        raise AssertionError("Dry run should not trigger network-dependent steps")

    monkeypatch.setattr(run_refresh_and_predict, "run_ingestion", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "run_feature_builders", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "rebuild_datasets", forbid)

    run_refresh_and_predict.main(["--dry-run", "--targets", "1,4"])

    latest_path = Path("artifacts/predictions/latest.json")
    history_path = Path("artifacts/predictions/history.json")

    assert latest_path.exists(), "Latest prediction JSON should be written in dry-run mode"
    assert history_path.exists(), "History file should be appended in dry-run mode"

    payload = json.loads(latest_path.read_text())
    assert "generated_at" in payload
    assert "predictions" in payload
    assert set(payload["predictions"].keys()) == {"1h", "4h"}
    for entry in payload["predictions"].values():
        assert set(entry.keys()) >= {
            "timestamp",
            "horizon_hours",
            "close",
            "p_up",
            "ret_pred",
            "projected_price",
            "signal_ensemble",
            "signal_dir_only",
            "thresholds",
        }


def test_full_flow_with_mocked_binance(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from data.ingestors import binance_us_spot as binance_module

    fixture_payload = FIXTURE_KLINES_PATH.read_text()

    class MockRequests:
        @staticmethod
        def get(url, params=None, timeout=None):
            return DummyResponse(fixture_payload)

    monkeypatch.setattr(binance_module, "requests", MockRequests)

    feature_calls = []
    dataset_calls = []

    monkeypatch.setattr(run_refresh_and_predict, "run_feature_builders", lambda **_: feature_calls.append("features"))
    monkeypatch.setattr(run_refresh_and_predict, "rebuild_datasets", lambda *_: dataset_calls.append("datasets"))

    def fake_predictions(targets, p_up_min, ret_min, offline):
        assert not offline
        result = {}
        for horizon in sorted(set(targets)):
            result[f"{horizon}h"] = {
                "timestamp": "2023-01-01T00:00:00Z",
                "horizon_hours": horizon,
                "close": 60250.0,
                "p_up": 0.6,
                "ret_pred": 0.001,
                "projected_price": 60310.0,
                "signal_ensemble": 1,
                "signal_dir_only": 1,
                "thresholds": {"p_up_min": p_up_min, "ret_min": ret_min},
            }
        return result

    monkeypatch.setattr(run_refresh_and_predict, "run_predictions", fake_predictions)

    run_refresh_and_predict.main(["--hours", "2", "--targets", "1,4", "--p-up-min", "0.45", "--ret-min", "0.0"])

    latest_path = Path("artifacts/predictions/latest.json")
    history_path = Path("artifacts/predictions/history.json")

    assert latest_path.exists()
    assert history_path.exists()

    payload = json.loads(latest_path.read_text())
    predictions = payload["predictions"]
    assert set(predictions.keys()) == {"1h", "4h"}
    for entry in predictions.values():
        assert entry["horizon_hours"] in {1, 4}
        assert isinstance(entry["p_up"], float)
        assert isinstance(entry["signal_ensemble"], int)

    history = json.loads(history_path.read_text())
    assert isinstance(history, list)
    assert history, "History should accumulate entries"

    kline_root = Path("data/raw/market/binanceus")
    assert kline_root.exists(), "Mocked Binance ingestion should write parquet output"
    assert feature_calls == ["features"]
    assert dataset_calls == ["datasets"]