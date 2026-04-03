from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.runtime.quality_support import (
    evaluate_data_quality,
    evaluate_feature_coverage,
    resolve_data_quality_policy,
    resolve_feature_coverage_policy,
    write_data_quality_payload,
)


def test_evaluate_feature_coverage_respects_ignored_columns() -> None:
    policy = resolve_feature_coverage_policy(
        {
            "enabled": True,
            "max_imputed_zero_columns": 0,
            "ignored_columns": ["ignored_col"],
        }
    )
    payload = evaluate_feature_coverage(
        {
            "feature_alignment": {
                "imputed_zero_columns": ["ignored_col", "active_col"],
                "required_columns": 2,
            },
            "source_freshness": {},
        },
        policy,
    )

    assert payload["imputed_zero_count"] == 1
    assert payload["effective_required_columns"] == 1
    assert payload["failed_checks"] == ["imputed_zero_columns"]


def test_evaluate_data_quality_captures_validation_error() -> None:
    written = []

    class FakePolicy:
        def __init__(self, max_staleness_hours, max_missing_ratio, max_zero_volume_ratio, min_rows):
            self.max_staleness_hours = max_staleness_hours
            self.max_missing_ratio = max_missing_ratio
            self.max_zero_volume_ratio = max_zero_volume_ratio
            self.min_rows = min_rows

    class FakeError(Exception):
        pass

    payload = evaluate_data_quality(
        pd.DataFrame({"close": [1.0, 2.0]}),
        resolve_data_quality_policy({"enabled": True, "min_rows": 3}),
        data_quality_policy_type=FakePolicy,
        evaluate_ohlcv_quality=lambda frame, policy: (_ for _ in ()).throw(FakeError("too short")),
        data_quality_error_type=FakeError,
        write_data_quality_payload=written.append,
    )

    assert payload["ok"] is False
    assert payload["error"] == "too short"
    assert payload["row_count"] == 2
    assert written and written[0]["ok"] is False


def test_write_data_quality_payload_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "monitoring" / "data_quality.json"

    write_data_quality_payload({"ok": True, "rows": 10}, output_path)

    assert output_path.exists()
    assert '"ok": true' in output_path.read_text(encoding="utf-8")