from __future__ import annotations

import math

import pandas as pd

from src.monitoring.metrics import ensure_meta_metrics


def test_ensure_meta_metrics_derives_hit_rate_from_ret_net() -> None:
    df = pd.DataFrame(
        {
            "signal_meta": [1, 0, 1],
            "ret_net_fee_20_10": [0.01, 0.0, -0.02],
        }
    )

    result = ensure_meta_metrics(df)

    assert "meta_net_fee_20_10" in result.columns
    meta_net = result["meta_net_fee_20_10"].tolist()
    assert meta_net == [0.01, 0.0, -0.02]

    hit_rate = result["meta_hit_rate"].tolist()
    assert hit_rate[0] == 1.0
    assert math.isnan(hit_rate[1])
    assert hit_rate[2] == 0.0


def test_ensure_meta_metrics_respects_existing_meta_net_columns() -> None:
    df = pd.DataFrame(
        {
            "signal_meta": [1, 1],
            "meta_net_fee_20_10": [0.2, -0.1],
        }
    )

    result = ensure_meta_metrics(df)

    assert "meta_net_fee_20_10" in result.columns
    hit_rate = result["meta_hit_rate"].tolist()
    assert hit_rate == [1.0, 0.0]
