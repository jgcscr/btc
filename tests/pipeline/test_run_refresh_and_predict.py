import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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


class DummyRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, array):  # pylint: disable=unused-argument
        return np.full((array.shape[0],), self.value, dtype=float)


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
    monitoring_latest_path = Path("artifacts/monitoring/latest.json")

    assert latest_path.exists(), "Latest prediction JSON should be written in dry-run mode"
    assert history_path.exists(), "History file should be appended in dry-run mode"
    assert monitoring_latest_path.exists(), "Monitoring latest snapshot should be written in dry-run mode"
    payload = json.loads(latest_path.read_text())
    assert "generated_at" in payload
    assert set(payload["predictions"].keys()) == {"1h", "4h"}
    for entry in payload["predictions"].values():
        assert entry["horizon_hours"] in {1, 4}
        assert "projected_high" in entry and entry["projected_high"] >= entry["projected_low"]
        assert "projected_high_confidence" in entry
        assert "projected_low_confidence" in entry
        assert "target_range_overrides" in entry
        fallback = entry["direction_only_fallback"]
        assert fallback["active"] is False
        assert {"stop_loss_fallback", "take_profit_fallback"}.issubset(fallback.keys())
        assert "p_up_min_effective" in entry["thresholds"]
        assert "adaptive_scale" in entry["thresholds"]

    history = json.loads(history_path.read_text())
    assert isinstance(history, list)
    assert history, "History should accumulate entries"

    monitoring_payload = json.loads(monitoring_latest_path.read_text())
    assert monitoring_payload["request"]["dry_run"] is True
    assert monitoring_payload["horizons"], "Monitoring payload should include per-horizon entries"
    for entry in monitoring_payload["horizons"]:
        assert "thresholds" in entry
        assert "volatility" in entry
        assert "p_up_components" in entry


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

    def fake_predictions(targets, p_up_min, ret_min, offline, **kwargs):
        assert not offline
        assert "thresholds_by_horizon" in kwargs
        result = {}
        for horizon in sorted(set(targets)):
            label = run_refresh_and_predict._format_horizon_label(horizon)
            result[label] = {
                "timestamp": "2023-01-01T00:00:00Z",
                "horizon_hours": horizon,
                "close": 60250.0,
                "p_up": 0.6,
                "p_trend_ignition": 0.55,
                "ignition_state": 1,
                "ignition_cooldown_active": False,
                "ret_pred": 0.001,
                "projected_price": 60310.0,
                "signal_ensemble": 1,
                "signal_dir_only": 1,
                "p_up_components": {"xgb": 0.6},
                "thresholds": {
                    "p_up_min": p_up_min,
                    "ret_min": ret_min,
                    "p_up_min_effective": p_up_min,
                    "ret_min_effective": ret_min,
                    "adaptive_scale": 1.0,
                },
                "volatility": {
                    "snapshot": {"volatility_realized_24h": 0.02},
                    "ceiling": 0.05,
                    "triggered": False,
                },
                "volatility_flag": False,
                "regime_state": "neutral",
                "regime_score": 0.5,
                "direction_only_fallback": {
                    "active": False,
                    "side": None,
                    "size_factor": 0.0,
                    "stop_loss_fallback": None,
                    "take_profit_fallback": None,
                    "reason": "mock",
                    "cooldown_active": False,
                },
            }
        return result

    monkeypatch.setattr(run_refresh_and_predict, "run_predictions", fake_predictions)

    run_refresh_and_predict.main(["--hours", "2", "--targets", "1,4", "--p-up-min", "0.45", "--ret-min", "0.0"])

    latest_path = Path("artifacts/predictions/latest.json")
    history_path = Path("artifacts/predictions/history.json")
    monitoring_latest_path = Path("artifacts/monitoring/latest.json")

    assert latest_path.exists()
    assert history_path.exists()
    assert monitoring_latest_path.exists()

    payload = json.loads(latest_path.read_text())
    predictions = payload["predictions"]
    assert set(predictions.keys()) == {"1h", "4h"}
    for entry in predictions.values():
        assert entry["horizon_hours"] in {1, 4}
        assert isinstance(entry["p_up"], float)
        assert isinstance(entry["signal_ensemble"], int)
        assert "p_trend_ignition" in entry
        assert "ignition_state" in entry
        assert "volatility" in entry
        assert "volatility_flag" in entry
        assert "direction_only_fallback" in entry

    history = json.loads(history_path.read_text())
    assert isinstance(history, list)
    assert history, "History should accumulate entries"

    monitoring_payload = json.loads(monitoring_latest_path.read_text())
    assert monitoring_payload["horizons"], "Monitoring payload should contain per-horizon entries"
    assert all("p_up_components" in entry for entry in monitoring_payload["horizons"])

    kline_root = Path("data/raw/market/binanceus")
    assert kline_root.exists(), "Mocked Binance ingestion should write parquet output"
    assert feature_calls == ["features"]
    assert dataset_calls == ["datasets"]


def test_run_predictions_uses_structured_config(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    dataset_path.write_text("stub")
    reg_model_path = tmp_path / "reg.json"
    dir_model_path = tmp_path / "dir.json"
    reg_model_path.write_text("{}"); dir_model_path.write_text("{}")

    trend_state_path = tmp_path / "ti_state.json"
    monkeypatch.setattr(run_refresh_and_predict, "TREND_IGNITION_STATE_PATH", trend_state_path)
    direction_state_path = tmp_path / "direction_state.json"
    monkeypatch.setattr(run_refresh_and_predict, "DIRECTION_FALLBACK_STATE_PATH", direction_state_path)

    monkeypatch.setattr(run_refresh_and_predict, "DATASET_MULTI_PATH", dataset_path)
    monkeypatch.setattr(run_refresh_and_predict, "DATASET_1H_PATH", dataset_path)
    monkeypatch.setattr(
        run_refresh_and_predict,
        "_model_paths_for_horizon",
        lambda horizon: (reg_model_path, dir_model_path),
    )

    df_stub = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=1, freq="h"),
            "dummy_feature": [1.0],
        }
    )
    prepared_stub = SimpleNamespace(df_all=df_stub, volatility_columns=[])
    monkeypatch.setattr(
        run_refresh_and_predict,
        "_load_prepared",
        lambda *_args, **_kwargs: (prepared_stub, 0, 100.0, "2024-01-01T00:00:00Z"),
    )
    monkeypatch.setattr(run_refresh_and_predict, "populate_sequence_cache_from_prepared", lambda *a, **k: None)
    monkeypatch.setattr(run_refresh_and_predict, "load_residual_std_from_dataset", lambda *a, **k: {1: 0.0004})

    logged_configs: list[tuple[str | None, list[dict[str, object]]]] = []

    def fake_logger(configs, label=None):
        logged_configs.append((label, configs))
        return ""

    monkeypatch.setattr(run_refresh_and_predict, "log_direction_model_configs", fake_logger)

    loaded_configs: list[list[dict[str, object]] | None] = []

    def fake_load_models(*_args, **kwargs):
        loaded_configs.append(kwargs.get("direction_model_configs"))
        return {}

    monkeypatch.setattr(run_refresh_and_predict, "load_models", fake_load_models)

    monkeypatch.setattr(
        run_refresh_and_predict,
        "load_trend_ignition_classifier",
        lambda *_args, **_kwargs: {"model": object(), "feature_names": []},
    )

    score_responses = [
        {"hourly": 0.95},
        {"hourly": 0.1},
    ]

    def fake_breakout_scores(*_args, **_kwargs):
        assert score_responses, "Unexpected breakout score request"
        return score_responses.pop(0)

    monkeypatch.setattr(run_refresh_and_predict, "_compute_breakout_scores", fake_breakout_scores)

    def fake_target_models(*_args, **_kwargs):
        return {
            1.0: {
                "high": {
                    "model": DummyRegressor(0.02),
                    "feature_names": ["dummy_feature"],
                    "metrics": {"val_rmse": 0.002},
                },
                "low": {
                    "model": DummyRegressor(-0.015),
                    "feature_names": ["dummy_feature"],
                    "metrics": {"val_rmse": 0.003},
                },
            }
        }

    monkeypatch.setattr(run_refresh_and_predict, "_load_target_range_models", fake_target_models)

    captured_weights: list[dict[str, float] | None] = []

    def fake_compute_signal(*_args, **kwargs):
        captured_weights.append(kwargs.get("dir_model_weights"))
        return {
            "ts": "2024-01-01T00:00:00Z",
            "p_up": 0.68,
            "p_trend_ignition": 0.72,
            "ret_pred": 0.0001,
            "signal_ensemble": 1,
            "signal_dir_only": 1,
        }

    monkeypatch.setattr(run_refresh_and_predict, "compute_signal_for_index", fake_compute_signal)

    json_config_path = tmp_path / "direction_config.json"
    json_config_path.write_text(
        json.dumps(
            [
                {
                    "name": "json_xgb",
                    "type": "xgb",
                    "path": "placeholder",
                    "weight": 3.0,
                },
            ],
        ),
    )

    summary = run_refresh_and_predict.run_predictions(
        targets=[1],
        p_up_min=0.5,
        ret_min=0.0002,
        dir_model_config_json=str(json_config_path),
        dir_model_weights="json_xgb:5",
        trend_ignition={"model_path": "fake_model.joblib", "probability_threshold": 0.7, "cooldown_hours": 12},
        direction_only_fallback={
            "enabled": True,
            "prob_threshold": 0.6,
            "max_negative_ev": 0.0002,
            "size_factor": 0.4,
            "stop_take_ratio": 0.01,
            "cooldown_hours": 6,
            "ignition_ev_extension": 0.0001,
        },
        adaptive_thresholds={
            "enabled": True,
            "breakout_score_threshold": 0.8,
            "chop_score_threshold": 0.2,
            "breakout_scale": 0.8,
            "chop_scale": 1.2,
            "p_up_min_floor": 0.3,
            "p_up_min_ceiling": 0.9,
            "ret_min_floor": 0.0,
            "ret_min_ceiling": 0.01,
        },
        target_range_models={
            "enabled": True,
            "model_dir": "unused",
            "override_ratio": 0.01,
            "confidence_rmse_scale": 0.01,
            "horizons": [1],
        },
    )

    assert "1h" in summary
    assert loaded_configs and loaded_configs[0][0]["name"] == "json_xgb"
    assert logged_configs and "(1h)" in (logged_configs[0][0] or "")

    logged_entry = logged_configs[0][1][0]
    assert logged_entry["name"] == "json_xgb"
    assert logged_entry["path"] == str(dir_model_path)
    assert logged_entry["weight"] == 5.0

    assert captured_weights == [{"json_xgb": 5.0}]

    entry = summary["1h"]
    assert entry["p_trend_ignition"] == 0.72
    assert entry["ignition_state"] == 1
    assert entry["ignition_cooldown_active"] is False
    assert entry["regime_state"] == "trend_ignition"
    assert entry["thresholds"]["p_up_min_effective"] == pytest.approx(0.4, rel=0, abs=1e-9)
    assert entry["thresholds"]["ret_min_effective"] == pytest.approx(0.00016, rel=0, abs=1e-9)
    assert entry["thresholds"]["adaptive_scale"] == pytest.approx(0.8, rel=0, abs=1e-9)
    assert entry["projected_high"] > entry["projected_low"]
    assert entry["projected_high_confidence"] > 0.0
    overrides = entry["target_range_overrides"]
    assert overrides["take_profit"]["reason"] == "target_range_high"
    assert overrides["take_profit"]["updated"] == pytest.approx(entry["take_profit"], rel=0, abs=1e-9)
    fallback = entry["direction_only_fallback"]
    assert fallback["active"] is True
    assert fallback["side"] == "long"
    assert fallback["reason"].startswith("ev_within_band")
    assert fallback["size_factor"] == 0.4
    assert fallback["stop_loss_fallback"] < fallback["take_profit_fallback"]

    # direction_threshold should override the default 0.5 behaviour
    # refresh breakout scores so fake_breakout_scores can still service later
    score_responses.extend([
        {"hourly": 0.95},
        {"hourly": 0.1},
    ])
    summary2 = run_refresh_and_predict.run_predictions(
        targets=[1],
        p_up_min=0.5,
        ret_min=0.0002,
        direction_threshold=0.9,
        dir_model_config_json=str(json_config_path),
        dir_model_weights="json_xgb:5",
        trend_ignition={"model_path": "fake_model.joblib", "probability_threshold": 0.7, "cooldown_hours": 12},
        direction_only_fallback={
            "enabled": True,
            "prob_threshold": 0.6,
            "max_negative_ev": 0.0002,
            "size_factor": 0.4,
            "stop_take_ratio": 0.01,
            "cooldown_hours": 6,
            "ignition_ev_extension": 0.0001,
        },
        adaptive_thresholds={
            "enabled": True,
            "breakout_score_threshold": 0.8,
            "chop_score_threshold": 0.2,
            "breakout_scale": 0.8,
            "chop_scale": 1.2,
            "p_up_min_floor": 0.3,
            "p_up_min_ceiling": 0.9,
            "ret_min_floor": 0.0,
            "ret_min_ceiling": 0.01,
        },
        target_range_models={
            "enabled": True,
            "model_dir": "unused",
            "override_ratio": 0.01,
            "confidence_rmse_scale": 0.01,
            "horizons": [1],
        },
    )
    assert summary2["1h"]["signal_dir_only"] == 0

    # exercise auto_threshold: threshold should be taken from thresholds_by_horizon
    thresholds_map = {1: {"p_up_min": 0.8, "ret_min": 0.0}}
    auto_summary = run_refresh_and_predict.run_predictions(
        targets=[1],
        p_up_min=0.5,
        ret_min=0.0002,
        direction_threshold=0.1,  # ignored when auto enabled
        auto_direction_threshold=True,
        thresholds_by_horizon=thresholds_map,
        dir_model_config_json=str(json_config_path),
        dir_model_weights="json_xgb:5",
        trend_ignition={"model_path": "fake_model.joblib", "probability_threshold": 0.7, "cooldown_hours": 12},
        direction_only_fallback={
            "enabled": True,
            "prob_threshold": 0.6,
            "max_negative_ev": 0.0002,
            "size_factor": 0.4,
            "stop_take_ratio": 0.01,
            "cooldown_hours": 6,
            "ignition_ev_extension": 0.0001,
        },
        adaptive_thresholds={
            "enabled": True,
            "breakout_score_threshold": 0.8,
            "chop_score_threshold": 0.2,
            "breakout_scale": 0.8,
            "chop_scale": 1.2,
            "p_up_min_floor": 0.3,
            "p_up_min_ceiling": 0.9,
            "ret_min_floor": 0.0,
            "ret_min_ceiling": 0.01,
        },
        target_range_models={
            "enabled": True,
            "model_dir": "unused",
            "override_ratio": 0.01,
            "confidence_rmse_scale": 0.01,
            "horizons": [1],
        },
    )
    # fake signal p_up=0.68 < auto threshold 0.8
    assert auto_summary["1h"]["signal_dir_only"] == 0

    # reset breakout scores so the next run produces a chop regime
    score_responses[:] = [{"hourly": 0.1}]
    assert trend_state_path.exists()
    state_payload = json.loads(trend_state_path.read_text())
    assert state_payload["last_trigger_ts"] == "2024-01-01T00:00:00Z"
    assert direction_state_path.exists()
    fallback_state = json.loads(direction_state_path.read_text())
    assert fallback_state["last_trigger_ts"] == "2024-01-01T00:00:00Z"

    cooldown_summary = run_refresh_and_predict.run_predictions(
        targets=[1],
        p_up_min=0.5,
        ret_min=0.0002,
        dir_model_config_json=str(json_config_path),
        dir_model_weights="json_xgb:5",
        trend_ignition={"model_path": "fake_model.joblib", "probability_threshold": 0.7, "cooldown_hours": 12},
        direction_only_fallback={
            "enabled": True,
            "prob_threshold": 0.6,
            "max_negative_ev": 0.0002,
            "size_factor": 0.4,
            "stop_take_ratio": 0.01,
            "cooldown_hours": 6,
            "ignition_ev_extension": 0.0001,
        },
        adaptive_thresholds={
            "enabled": True,
            "breakout_score_threshold": 0.8,
            "chop_score_threshold": 0.2,
            "breakout_scale": 0.8,
            "chop_scale": 1.2,
            "p_up_min_floor": 0.3,
            "p_up_min_ceiling": 0.9,
            "ret_min_floor": 0.0,
            "ret_min_ceiling": 0.01,
        },
        target_range_models={
            "enabled": True,
            "model_dir": "unused",
            "override_ratio": 0.01,
            "confidence_rmse_scale": 0.01,
            "horizons": [1],
        },
    )
    cooldown_entry = cooldown_summary["1h"]
    assert cooldown_entry["regime_state"] == "chop"
    assert cooldown_entry["thresholds"]["p_up_min_effective"] == pytest.approx(0.6, rel=0, abs=1e-9)
    cooldown_overrides = cooldown_entry["target_range_overrides"]
    assert cooldown_overrides["take_profit"]["updated"] == pytest.approx(
        cooldown_entry["take_profit"], rel=0, abs=1e-9
    )
    cooldown_fallback = cooldown_entry["direction_only_fallback"]
    assert cooldown_fallback["active"] is False
    assert cooldown_fallback["cooldown_active"] is True
    assert cooldown_fallback["reason"] == "cooldown_active"


def test_local_feature_overrides_skip_ingestion(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    hours = 12
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    base_df = pd.DataFrame(
        {
            "ts": timestamps,
            "close": 50000 + pd.Series(range(hours), dtype=float),
            "volume": 1000 + pd.Series(range(hours), dtype=float),
            "ma_close_7h": 50000.0,
            "ma_close_24h": 50000.0,
            "fut_close": 50010.0,
        }
    )
    features_path = tmp_path / "live_features.parquet"
    base_df.to_parquet(features_path)

    override_template = pd.DataFrame({"ts": timestamps, "value": 1.0})
    macro_path = tmp_path / "macro.parquet"
    override_template.rename(columns={"value": "macro_signal"}).to_parquet(macro_path)
    onchain_path = tmp_path / "onchain.parquet"
    override_template.rename(columns={"value": "onchain_metric"}).to_parquet(onchain_path)
    cryptoquant_path = tmp_path / "cryptoquant.parquet"
    override_template.rename(columns={"value": "cq_metric"}).to_parquet(cryptoquant_path)
    funding_path = tmp_path / "funding.parquet"
    override_template.rename(columns={"value": "funding_rate"}).to_parquet(funding_path)

    def forbid(*_args, **_kwargs):
        raise AssertionError("Local feature mode should not run network-dependent stages")

    monkeypatch.setattr(run_refresh_and_predict, "run_ingestion", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "run_feature_builders", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "rebuild_datasets", forbid)

    captured: dict[str, object] = {}

    def fake_run_predictions(targets, p_up_min, ret_min, offline, **kwargs):
        prepared_override = kwargs.get("prepared_override")
        assert prepared_override is not None, "Should pass PreparedData override when using local features"
        captured["close"] = prepared_override[2]
        result = {}
        for horizon in sorted(set(targets)):
            label = run_refresh_and_predict._format_horizon_label(horizon)
            result[label] = {
                "timestamp": "2025-01-01T00:00:00Z",
                "horizon_hours": horizon,
                "close": prepared_override[2],
                "p_up": 0.6,
                "p_trend_ignition": 0.3,
                "ignition_state": 0,
                "ignition_cooldown_active": False,
                "ret_pred": 0.001,
                "projected_price": prepared_override[2],
                "signal_ensemble": 1,
                "signal_dir_only": 1,
                "p_up_components": {"xgb": 0.6},
                "thresholds": {
                    "p_up_min": p_up_min,
                    "ret_min": ret_min,
                    "p_up_min_effective": p_up_min,
                    "ret_min_effective": ret_min,
                    "adaptive_scale": 1.0,
                },
                "volatility": {
                    "snapshot": {"volatility_realized_24h": 0.02},
                    "ceiling": 0.05,
                    "triggered": False,
                },
                "volatility_flag": False,
                "projected_high": prepared_override[2],
                "projected_low": prepared_override[2],
                "projected_high_confidence": 0.0,
                "projected_low_confidence": 0.0,
                "target_range_overrides": {"stop_loss": None, "take_profit": None},
                "regime_state": "neutral",
                "regime_score": 0.0,
                "direction_only_fallback": {
                    "active": False,
                    "side": None,
                    "size_factor": 0.0,
                    "stop_loss_fallback": None,
                    "take_profit_fallback": None,
                    "reason": "mock",
                    "cooldown_active": False,
                },
            }
        return result

    monkeypatch.setattr(run_refresh_and_predict, "run_predictions", fake_run_predictions)

    run_refresh_and_predict.main(
        [
            "--targets",
            "1,4",
            "--use-local-features",
            "--features-path",
            str(features_path),
            "--macro-path",
            str(macro_path),
            "--onchain-path",
            str(onchain_path),
            "--cryptoquant-path",
            str(cryptoquant_path),
            "--funding-path",
            str(funding_path),
        ]
    )

    assert captured["close"] == base_df["close"].iloc[-1]

    latest_path = Path("artifacts/predictions/latest.json")
    monitoring_latest_path = Path("artifacts/monitoring/latest.json")
    assert latest_path.exists()
    assert monitoring_latest_path.exists()

    monitoring_payload = json.loads(monitoring_latest_path.read_text())
    overrides = monitoring_payload["request"].get("local_feature_overrides")
    assert overrides is not None, "Metadata for local overrides should be recorded"
    assert set(overrides.keys()) == {"features", "macro", "onchain", "cryptoquant", "funding"}
    assert overrides["features"]["path"].endswith("live_features.parquet")


def test_config_file_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def forbid(*_args, **_kwargs):
        raise AssertionError("Dry-run config should skip network-dependent stages")

    monkeypatch.setattr(run_refresh_and_predict, "run_ingestion", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "run_feature_builders", forbid)
    monkeypatch.setattr(run_refresh_and_predict, "rebuild_datasets", forbid)

    config_path = tmp_path / "refresh_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hours: 24",
                "targets: [1]",
                "p_up_min: 0.55",
                "ret_min: 0.0001",
                "dry_run: true",
            ]
        )
    )

    run_refresh_and_predict.main(["--config", str(config_path)])

    latest_path = Path("artifacts/predictions/latest.json")
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text())
    assert set(payload["predictions"].keys()) == {"1h"}

    monitoring_payload = json.loads(Path("artifacts/monitoring/latest.json").read_text())
    assert monitoring_payload["request"]["hours"] == 24
    assert monitoring_payload["request"]["targets"] == [1.0]