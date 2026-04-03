from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.output_support import (
    build_trade_ready_monitoring_payload,
    refresh_meta_baseline,
    write_monitoring_artifact,
    write_monitoring_payload_file,
)


def test_build_trade_ready_monitoring_payload_includes_request_and_prompt_summary() -> None:
    args = argparse.Namespace(
        targets=[1.0, 4.0],
        spot_provider="binanceus",
        hours=360,
        dry_run=True,
        confidence_min=0.2,
        position_size_floor=0.1,
        position_size_cap=0.5,
        position_size_cap_by_horizon={1.0: 0.2},
        confidence_min_by_horizon_regime={1.0: {"neutral": 0.3}},
        data_quality={"enabled": False},
        local_feature_metadata={"feature_coverage": {"ok": True}},
    )
    predictions_payload = {
        "generated_at": "2026-04-01T00:00:00Z",
        "predictions": {
            "4h": {"trade_action": "hold"},
            "1h": {"trade_action": "short"},
        },
        "prompt_ready_summary": {"market_outlook_strategy": {"selected_direction": "Long"}},
    }

    payload = build_trade_ready_monitoring_payload(
        predictions_payload,
        args,
        horizon_sort_key=lambda value: 0 if value == "1h" else 1,
        format_horizon_label=lambda value: f"{int(value)}h",
        confidence_min_default=0.0,
        position_size_floor_default=0.0,
        position_size_cap_default=1.0,
    )

    assert payload["request"]["targets"] == [1.0, 4.0]
    assert payload["request"]["position_size_cap_by_horizon"] == {"1h": 0.2}
    assert payload["request"]["confidence_min_by_horizon_regime"] == {"1h": {"neutral": 0.3}}
    assert payload["horizons"][0]["trade_action"] == "short"
    assert payload["prompt_ready_summary"]["market_outlook_strategy"]["selected_direction"] == "Long"


def test_write_monitoring_payload_file_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "monitoring" / "latest.json"

    write_monitoring_payload_file({"status": "ok"}, output_path)

    assert output_path.exists()
    assert '"status": "ok"' in output_path.read_text(encoding="utf-8")


def test_write_monitoring_artifact_builds_and_writes_payload(tmp_path: Path) -> None:
    args = argparse.Namespace(
        targets=[1.0, 4.0],
        spot_provider="binanceus",
        hours=360,
        dry_run=False,
        confidence_min=0.1,
        position_size_floor=0.0,
        position_size_cap=0.8,
        position_size_cap_by_horizon=None,
        confidence_min_by_horizon_regime=None,
        data_quality=None,
        local_feature_metadata=None,
    )
    output_path = tmp_path / "monitoring" / "latest.json"
    predictions_payload = {
        "generated_at": "2026-04-02T00:00:00Z",
        "predictions": {
            "4h": {"trade_action": "hold", "horizon_hours": 4.0},
            "1h": {"trade_action": "long", "horizon_hours": 1.0},
        },
        "blocked_trade_analytics": {"blocked_total": 1},
    }

    payload = write_monitoring_artifact(
        predictions_payload,
        args,
        output_path=output_path,
        horizon_sort_key=lambda value: 0 if value == "1h" else 1,
        format_horizon_label=lambda value: f"{int(value)}h",
        confidence_min_default=0.0,
        position_size_floor_default=0.0,
        position_size_cap_default=1.0,
    )

    assert output_path.exists()
    assert payload["horizons"][0]["trade_action"] == "long"
    assert payload["blocked_trade_analytics"]["blocked_total"] == 1
    assert '"trade_action": "long"' in output_path.read_text(encoding="utf-8")


def test_refresh_meta_baseline_skips_when_source_missing(tmp_path: Path) -> None:
    messages: list[str] = []

    refresh_meta_baseline(
        source_csv=tmp_path / "missing.csv",
        json_path=tmp_path / "meta.json",
        parquet_path=tmp_path / "meta.parquet",
        load_dataframe=lambda *args, **kwargs: None,
        compute_baseline=lambda *args, **kwargs: {},
        baseline_to_dataframe=lambda payload: None,
        append_detected_meta_columns=lambda df, columns: columns,
        default_columns=["a", "b"],
        stderr_write=messages.append,
    )

    assert messages
    assert "skipping baseline refresh" in messages[0]
