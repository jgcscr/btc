from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.runtime.market_preparation import apply_replay_override, prepare_market_data
from src.runtime.models import RuntimeMode
from src.runtime.persistence import RuntimeStateStore


def _build_args(**overrides) -> argparse.Namespace:
    payload = {
        "use_local_features": False,
        "dry_run": True,
        "replay_offset_bars": 0,
        "features_path": None,
        "data_quality": {},
        "feature_coverage_policy": None,
        "hours": 360,
        "spot_provider": "binanceus",
        "targets": [1.0, 4.0],
        "_intrabar_enabled": False,
        "_intrabar_cfg": {},
        "local_feature_metadata": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_prepare_market_data_dry_run_records_cached_source(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path)
    paths = store.start_run(mode=RuntimeMode.RESEARCH, request={"targets": [1.0]})

    result = prepare_market_data(_build_args(dry_run=True), paths, store)

    assert result.prepared_override is None
    assert result.latest_close is None
    events = [json.loads(line) for line in paths.events_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "data_preparation"
    assert events[-1]["details"]["source"] == "cached_datasets"


def test_apply_replay_override_noop_when_disabled(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path)
    paths = store.start_run(mode=RuntimeMode.RESEARCH, request={"targets": [1.0]})

    result = apply_replay_override(
        _build_args(replay_offset_bars=0),
        paths,
        store,
        prepared_override=None,
        latest_close=123.45,
    )

    assert result.prepared_override is None
    assert result.latest_close == 123.45
    assert not paths.events_path.exists()


def test_apply_replay_override_uses_runtime_dataset_resolution(tmp_path: Path, monkeypatch) -> None:
    store = RuntimeStateStore(tmp_path)
    paths = store.start_run(mode=RuntimeMode.RESEARCH, request={"targets": [1.0]})
    dataset_path = tmp_path / "dataset_1h.npz"
    dataset_path.write_text("placeholder", encoding="utf-8")
    legacy_dataset_path = tmp_path / "legacy_multi.npz"

    import src.runtime.market_preparation as market_preparation

    monkeypatch.setattr(market_preparation, "DATASET_MULTI_PATH", legacy_dataset_path)
    monkeypatch.setattr(market_preparation, "DATASET_1H_PATH", dataset_path)
    monkeypatch.setattr(market_preparation, "DATASET_15M_PATH", tmp_path / "dataset_15m.npz")
    import src.runtime.refresh_support as refresh_support

    monkeypatch.setattr(
        refresh_support,
        "load_prepared_offline",
        lambda dataset_path, *, base_horizon, prepare_data_for_signals_from_ohlcv_fn, format_ts_iso_fn, stderr_write: (
            SimpleNamespace(
                df_all=pd.DataFrame(
                    {
                        "ts": pd.to_datetime(["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z", "2026-04-01T02:00:00Z"]),
                        "close": [100.0, 101.0, 102.0],
                    }
                )
            ),
            2,
            102.0,
            "2026-04-01T02:00:00+00:00",
        ),
    )

    result = apply_replay_override(
        _build_args(replay_offset_bars=1),
        paths,
        store,
        prepared_override=None,
        latest_close=123.45,
    )

    assert result.prepared_override is not None
    prepared, replay_index, replay_close, replay_ts = result.prepared_override
    assert replay_index == 1
    assert replay_close == 101.0
    assert replay_ts == "2026-04-01T01:00:00Z"
    assert list(prepared.df_all["close"]) == [100.0, 101.0, 102.0]
    assert result.latest_close == 101.0