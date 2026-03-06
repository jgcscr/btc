from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class DataQualityPolicy:
    max_staleness_hours: float = 2.0
    max_missing_ratio: float = 0.01
    max_zero_volume_ratio: float = 0.2
    min_rows: int = 120


class DataQualityError(RuntimeError):
    pass


def evaluate_ohlcv_quality(frame: pd.DataFrame, policy: Optional[DataQualityPolicy] = None) -> Dict[str, float | bool | str]:
    policy = policy or DataQualityPolicy()
    if frame.empty:
        raise DataQualityError("OHLCV frame is empty")
    if "ts" not in frame.columns:
        raise DataQualityError("OHLCV frame missing ts column")

    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    ts = ts.dropna().sort_values()
    if ts.empty:
        raise DataQualityError("No valid timestamps in OHLCV frame")

    row_count = int(len(frame))
    latest_ts = ts.iloc[-1].to_pydatetime()
    now = datetime.now(timezone.utc)
    staleness_hours = max(0.0, (now - latest_ts).total_seconds() / 3600.0)

    expected = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq="1h", tz="UTC")
    missing_ratio = 0.0
    if len(expected) > 0:
        missing_ratio = float(1.0 - (ts.nunique() / len(expected)))
        missing_ratio = max(0.0, missing_ratio)

    zero_volume_ratio = 0.0
    if "volume" in frame.columns and len(frame) > 0:
        zero_volume_ratio = float((pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0) <= 0).mean())

    checks = {
        "min_rows_ok": row_count >= policy.min_rows,
        "staleness_ok": staleness_hours <= policy.max_staleness_hours,
        "missing_ok": missing_ratio <= policy.max_missing_ratio,
        "zero_volume_ok": zero_volume_ratio <= policy.max_zero_volume_ratio,
    }

    payload: Dict[str, float | bool | str] = {
        "row_count": row_count,
        "latest_ts": latest_ts.isoformat(),
        "staleness_hours": float(staleness_hours),
        "missing_ratio": float(missing_ratio),
        "zero_volume_ratio": float(zero_volume_ratio),
        **checks,
    }

    if not all(bool(value) for value in checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise DataQualityError(f"Data quality checks failed: {', '.join(failed)}")

    return payload
