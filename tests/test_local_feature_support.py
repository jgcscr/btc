from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.runtime.local_feature_support as local_feature_support


def test_build_ohlcv_frame_from_tidy_maps_spot_metrics() -> None:
    frame = pd.DataFrame(
        {
            "ts": ["2026-04-02T00:00:00Z", "2026-04-02T00:00:00Z", "2026-04-02T00:00:00Z"],
            "metric": ["spot_open", "spot_high", "spot_close"],
            "value": [100.0, 101.0, 100.5],
        }
    )

    result = local_feature_support.build_ohlcv_frame_from_tidy(frame)

    assert list(result.columns) == ["ts", "close", "high", "open"]
    assert result.iloc[0]["open"] == 100.0
    assert result.iloc[0]["high"] == 101.0
    assert result.iloc[0]["close"] == 100.5


def test_prepare_local_feature_bundle_merges_optional_sources_and_imputes_missing_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_path = tmp_path / "features.parquet"
    macro_path = tmp_path / "macro.parquet"
    dataset_path = tmp_path / "dataset_multi.npz"

    base = pd.DataFrame(
        {
            "ts": pd.date_range("2026-04-02T00:00:00Z", periods=4, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    macro = pd.DataFrame(
        {
            "ts": pd.date_range("2026-04-02T00:00:00Z", periods=4, freq="h", tz="UTC"),
            "macro_signal": [1.0, 2.0, 3.0, 4.0],
        }
    )
    base.to_parquet(features_path, index=False)
    macro.to_parquet(macro_path, index=False)
    np.savez(dataset_path, feature_names=np.array(["close", "volume", "macro_signal", "funding_rate"], dtype=object))

    monkeypatch.setattr(
        local_feature_support,
        "prepare_data_for_signals_from_ohlcv",
        lambda frame, feature_names, train_frac: SimpleNamespace(df_all=frame.copy()),
    )

    prepared_override, metadata = local_feature_support.prepare_local_feature_bundle(
        features_path=str(features_path),
        hours=3,
        optional_sources={"macro": str(macro_path)},
        dataset_multi_path=dataset_path,
        dataset_1h_path=tmp_path / "missing_1h.npz",
        local_feature_required_columns={"macro": ()},
        stderr_write=lambda message: None,
    )

    prepared, index, close, ts_iso = prepared_override
    assert prepared.df_all.iloc[-1]["macro_signal"] == 4.0
    assert index == 2
    assert close == 103.5
    assert ts_iso.startswith("2026-04-02T03:00:00")
    assert metadata["macro"]["added_columns"] == ["macro_signal"]
    assert "funding_rate" in metadata["feature_alignment"]["imputed_zero_columns"]


def test_compute_intrabar_features_from_15m_raises_when_aggregation_returns_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "intrabar.parquet"
    tidy = pd.DataFrame(
        {
            "ts": [
                "2026-04-02T00:00:00Z",
                "2026-04-02T00:00:00Z",
                "2026-04-02T00:00:00Z",
                "2026-04-02T00:00:00Z",
            ],
            "metric": ["spot_open", "spot_high", "spot_low", "spot_close"],
            "value": [100.0, 101.0, 99.0, 100.5],
        }
    )
    tidy.to_parquet(path, index=False)
    monkeypatch.setattr(local_feature_support, "compute_hourly_intrabar_features", lambda frame: pd.DataFrame())

    with pytest.raises(RuntimeError, match="did not produce any intrabar features"):
        local_feature_support.compute_intrabar_features_from_15m(path)