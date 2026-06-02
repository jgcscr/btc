from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.runtime.refresh_support import (
    base_horizon_for_target_column,
    dataset_profile_for_horizon,
    load_cli_config,
    load_probability_calibration,
    load_prepared,
    load_prepared_offline,
    normalize_refresh_args,
    periods_per_hour_for_base_horizon,
    project_price,
    resolve_prediction_inputs,
    resolve_sequence_model_dirs,
    select_dataset_candidate,
    warn_missing_thresholds,
)


@dataclass(frozen=True)
class _DatasetCandidate:
    path: Path
    target_column: str
    base_horizon: float
    offline_only: bool = False


@dataclass(frozen=True)
class _DatasetProfile:
    key: str
    candidates: tuple[_DatasetCandidate, ...]


@pytest.fixture
def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        replay_offset_bars=0,
        targets=[1.0, 4.0],
        use_local_features=False,
        features_path=None,
        dry_run=False,
        data_quality=None,
        data_quality_enabled=False,
        max_staleness_hours=2.0,
        max_missing_ratio=0.01,
        max_zero_volume_ratio=0.2,
        min_rows=120,
        intrabar_aggregation=None,
        intrabar_enabled=False,
        intrabar_interval="15m",
        intrabar_hours_multiplier=4,
        intrabar_max_rows=4000,
        trade_decision_policy=None,
        trade_decision_disabled=False,
        trade_decision_enabled=False,
        trade_decision_model=None,
        trade_decision_threshold=None,
        dir_lstm_path=None,
        dir_bilstm_path=None,
        dir_gru_path=None,
        dir_cnn_lstm_path=None,
        dir_cnn_bilstm_path=None,
        dir_garch_lstm_path=None,
        dir_transformer_path=None,
    )


def test_normalize_refresh_args_enables_dry_run_for_replay(base_args: argparse.Namespace) -> None:
    base_args.replay_offset_bars = 2

    normalize_refresh_args(base_args)

    assert base_args.dry_run is True
    assert base_args._intrabar_enabled is False
    assert base_args.trade_decision_policy == {}
    assert base_args.data_quality["max_staleness_hours"] == 2.0


def test_normalize_refresh_args_rejects_local_features_without_path(base_args: argparse.Namespace) -> None:
    base_args.use_local_features = True

    with pytest.raises(ValueError, match="--features-path is required"):
        normalize_refresh_args(base_args)


def test_resolve_sequence_model_dirs_prefers_environment(base_args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch) -> None:
    base_args.dir_lstm_path = "from-args"
    monkeypatch.setenv("DIR_LSTM_PATH", "from-env")

    resolved = resolve_sequence_model_dirs(base_args)

    assert resolved.dir_lstm_path == "from-env"
    assert resolved.has_any() is True


def test_load_cli_config_filters_unknown_keys_and_normalizes_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("hours: 24\nunknown-key: 1\nwrite-artifacts: true\n", encoding="utf-8")
    messages: list[str] = []

    payload = load_cli_config(
        str(config_path),
        config_allowed_keys=("hours", "write_artifacts"),
        normalize_config_value=lambda name, value: int(value) if name == "hours" else bool(value),
        yaml_safe_load=yaml.safe_load,
        stderr_write=messages.append,
    )

    assert payload["hours"] == 24
    assert payload["write_artifacts"] is True
    assert payload["config"] == str(config_path)
    assert messages and "Unknown config key 'unknown-key'" in messages[0]


def test_dataset_profile_and_selection_choose_existing_dataset(tmp_path: Path) -> None:
    multi_path = tmp_path / "multi.npz"
    one_h_path = tmp_path / "one_h.npz"
    subhour_path = tmp_path / "15m.npz"
    one_h_path.write_text("ok", encoding="utf-8")

    profile = dataset_profile_for_horizon(
        0.25,
        dataset_multi_path=multi_path,
        dataset_1h_path=one_h_path,
        dataset_15m_path=subhour_path,
        dataset_candidate_type=_DatasetCandidate,
        dataset_profile_type=_DatasetProfile,
    )
    selected, used_fallback = select_dataset_candidate(profile)

    assert profile.key == "15m"
    assert profile.candidates[0].offline_only is True
    assert selected.path == one_h_path
    assert used_fallback is True


def test_warn_missing_thresholds_reports_missing_horizons() -> None:
    messages: list[str] = []

    warn_missing_thresholds(
        [1.0, 4.0],
        {1.0: {"p_up_min": 0.5, "ret_min": 0.0}},
        "thresholds.json",
        normalize_horizon_value=lambda value: float(value),
        coerce_numeric_horizon=lambda value: float(value),
        format_horizon_label=lambda value: f"{int(value)}h",
        stderr_write=messages.append,
    )

    assert messages == [
        "Warning: thresholds.json is missing calibrated entries for horizons 4h; falling back to CLI defaults.\n"
    ]


def test_load_probability_calibration_parses_supported_payload_shapes(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        (
            '{'
            '"1h":{"a":1.2,"b":-0.1},'
            '"4h":{"method":"beta","a":0.9,"b":-1.1,"c":0.2},'
            '"12h":{"method":"isotonic","x":[0.3,0.7],"y":[0.4,0.8]}'
            '}'
        ),
        encoding="utf-8",
    )

    payload = load_probability_calibration(str(calibration_path), stderr_write=lambda _message: None)

    assert payload == {
        "1h": {"method": "platt", "a": 1.2, "b": -0.1},
        "4h": {"method": "beta", "a": 0.9, "b": -1.1, "c": 0.2},
        "12h": {"method": "isotonic", "x": [0.3, 0.7], "y": [0.4, 0.8]},
    }


def test_resolve_prediction_inputs_uses_runtime_calibration_loader(base_args: argparse.Namespace, tmp_path: Path) -> None:
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text("{}", encoding="utf-8")
    platt_path = tmp_path / "platt.json"
    platt_path.write_text('{"1h":{"a":1.0,"b":0.0}}', encoding="utf-8")
    direction_output_path = tmp_path / "direction_output.json"
    direction_output_path.write_text('{"1h@neutral":{"method":"platt","a":2.0,"b":0.1}}', encoding="utf-8")
    target_range_dir = tmp_path / "target_range_models"
    target_range_dir.mkdir()
    (target_range_dir / "metadata.json").write_text("{}", encoding="utf-8")

    base_args.thresholds_json = str(thresholds_path)
    base_args.platt_calibration = str(platt_path)
    base_args.direction_output_policy = {"calibration_path": str(direction_output_path)}
    base_args.target_range_models = None

    from src.runtime import refresh_support as refresh_support_module

    original_target_range_dir = refresh_support_module.TARGET_RANGE_MODEL_DIR
    original_threshold_loader = refresh_support_module.load_calibrated_thresholds
    refresh_support_module.TARGET_RANGE_MODEL_DIR = target_range_dir
    refresh_support_module.load_calibrated_thresholds = lambda _path: {1.0: {"p_up_min": 0.55, "ret_min": 0.01}}
    try:
        bundle = resolve_prediction_inputs(base_args)
    finally:
        refresh_support_module.TARGET_RANGE_MODEL_DIR = original_target_range_dir
        refresh_support_module.load_calibrated_thresholds = original_threshold_loader

    assert bundle.platt_calibration == {"1h": {"method": "platt", "a": 1.0, "b": 0.0}}
    assert bundle.direction_output_cfg["calibration_map"] == {
        "1h@neutral": {"method": "platt", "a": 2.0, "b": 0.1}
    }
    assert bundle.thresholds_by_horizon == {1.0: {"p_up_min": 0.55, "ret_min": 0.01}}
    assert base_args.target_range_models == {"enabled": True, "model_dir": str(target_range_dir)}


def test_prepared_horizon_helpers_handle_subhourly_targets() -> None:
    assert base_horizon_for_target_column("ret_15m") == 0.25
    assert base_horizon_for_target_column("ret_1h") == 1.0
    assert periods_per_hour_for_base_horizon(0.25) == 4
    assert periods_per_hour_for_base_horizon(1.0) == 1
    assert project_price(100.0, 0.01) == pytest.approx(101.0050167084)


def test_load_prepared_uses_online_preparation_callback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    prepared = SimpleNamespace(df_all=pd.DataFrame({"ts": ["2026-04-02T00:00:00Z"], "close": [123.45]}))

    result = load_prepared(
        dataset_path,
        target_column="ret_1h",
        offline=False,
        load_prepared_offline_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected offline call")),
        prepare_data_for_signals_fn=lambda path, target_column: prepared,
        format_ts_iso_fn=lambda value: str(value),
    )

    assert result == (prepared, 0, 123.45, "2026-04-02T00:00:00Z")


def test_load_prepared_offline_rehydrates_npz_and_uses_close_snapshot(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X_train=np.array([[1.0, 100.0]], dtype=float),
        X_val=np.array([[2.0, 101.0]], dtype=float),
        X_test=np.array([[3.0, 102.0]], dtype=float),
        feature_names=np.array(["feature_a", "close"]),
        close_all=np.array([200.0, 201.0, 202.0], dtype=float),
    )
    captured: dict[str, object] = {}

    def fake_prepare(df_features, *, feature_names, train_frac, expected_freq, periods_per_hour):
        captured["feature_names"] = feature_names
        captured["train_frac"] = train_frac
        captured["expected_freq"] = expected_freq
        captured["periods_per_hour"] = periods_per_hour
        captured["rows"] = len(df_features)
        return SimpleNamespace(df_all=df_features)

    prepared, index, close, ts_iso = load_prepared_offline(
        dataset_path,
        base_horizon=0.25,
        prepare_data_for_signals_from_ohlcv_fn=fake_prepare,
        format_ts_iso_fn=lambda value: value.isoformat(),
        stderr_write=lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )

    assert prepared.df_all.iloc[index]["close"] == 102.0
    assert index == 2
    assert close == 202.0
    assert ts_iso.endswith("+00:00")
    assert captured == {
        "feature_names": ["feature_a", "close"],
        "train_frac": 0.7,
        "expected_freq": pd.Timedelta(hours=0.25),
        "periods_per_hour": 4,
        "rows": 3,
    }


def test_load_prepared_offline_reconstructs_close_from_scaler_when_close_snapshot_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X_train=np.array([[1.0, -0.5]], dtype=float),
        X_val=np.array([[2.0, 0.25]], dtype=float),
        X_test=np.array([[3.0, 1.5]], dtype=float),
        feature_names=np.array(["feature_a", "close"]),
        scaler_mean=np.array([10.0, 200.0], dtype=float),
        scaler_scale=np.array([2.0, 20.0], dtype=float),
    )

    prepared, index, close, ts_iso = load_prepared_offline(
        dataset_path,
        base_horizon=1.0,
        prepare_data_for_signals_from_ohlcv_fn=lambda df_features, **kwargs: SimpleNamespace(df_all=df_features),
        format_ts_iso_fn=lambda value: value.isoformat(),
        stderr_write=lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )

    assert prepared.df_all.iloc[index]["close"] == 1.5
    assert index == 2
    assert close == pytest.approx(260.0)
    assert ts_iso.endswith("+00:00")
