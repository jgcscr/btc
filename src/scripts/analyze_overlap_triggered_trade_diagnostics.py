from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _normalize_ts(values: Iterable[Any]) -> pd.Series:
    return pd.to_datetime(pd.Series(list(values)), utc=True, errors="coerce").dt.floor("h")


def _load_overlap_timestamps(npz_path: Path) -> pd.DatetimeIndex:
    with np.load(npz_path, allow_pickle=True) as data:
        rows: List[pd.Series] = []
        for key in ("ts_all", "ts_train", "ts_val", "ts_test"):
            if key in data.files:
                rows.append(_normalize_ts(data[key].reshape(-1).tolist()))
        if not rows:
            raise KeyError(f"No timestamp arrays found in {npz_path}")
        combined = pd.concat(rows, ignore_index=True).dropna().drop_duplicates().sort_values()
        return pd.DatetimeIndex(combined)


def _normalize_regime(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text or "unknown"


def _bucketize(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(numeric >= float(threshold), "high_vol", "low_vol"), index=values.index)


def _group_summary(df: pd.DataFrame, group_cols: List[str], return_col: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    working = df.copy()
    working["_ret"] = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    grouped = working.groupby(group_cols, dropna=False)
    rows: List[Dict[str, Any]] = []
    for group_key, grp in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: Dict[str, Any] = {
            col: ("unknown" if pd.isna(value) else str(value))
            for col, value in zip(group_cols, key_values)
        }
        row.update(
            {
                "row_count": int(len(grp)),
                "net_return_total": float(grp["_ret"].sum()),
                "net_return_mean": float(grp["_ret"].mean()) if len(grp) else float("nan"),
                "hit_rate": float((grp["_ret"] > 0.0).mean()) if len(grp) else float("nan"),
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: (item["net_return_total"], -item["row_count"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze triggered overlap trades by regime and volatility.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--overlap-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ts-col", type=str, default="ts")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--return-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--regime-col", type=str, default="regime_state")
    parser.add_argument("--volatility-col", type=str, default="volatility_realized_24h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.candidate, args.overlap_dataset):
        if not path.exists():
            raise FileNotFoundError(path)

    candidate = pd.read_csv(args.candidate)
    if args.ts_col not in candidate.columns:
        raise KeyError(f"Missing timestamp column '{args.ts_col}' in {args.candidate}")
    if args.signal_col not in candidate.columns:
        raise KeyError(f"Missing signal column '{args.signal_col}' in {args.candidate}")
    if args.return_col not in candidate.columns:
        raise KeyError(f"Missing return column '{args.return_col}' in {args.candidate}")

    overlap_ts = _load_overlap_timestamps(args.overlap_dataset)
    candidate = candidate.copy()
    candidate["_ts_norm"] = pd.to_datetime(candidate[args.ts_col], utc=True, errors="coerce").dt.floor("h")
    overlap_df = candidate.loc[candidate["_ts_norm"].isin(set(overlap_ts))].copy()
    overlap_df["_signal"] = pd.to_numeric(overlap_df[args.signal_col], errors="coerce").fillna(0.0)
    triggered_df = overlap_df.loc[overlap_df["_signal"] != 0.0].copy()
    triggered_df[args.regime_col] = triggered_df.get(args.regime_col, pd.Series(index=triggered_df.index, dtype=object)).map(
        _normalize_regime
    )

    volatility_threshold = None
    if args.volatility_col in triggered_df.columns:
        numeric_vol = pd.to_numeric(triggered_df[args.volatility_col], errors="coerce")
        if numeric_vol.notna().any():
            volatility_threshold = float(numeric_vol.median())
            triggered_df["volatility_bucket"] = _bucketize(numeric_vol, volatility_threshold)

    regime_summary = _group_summary(triggered_df, [args.regime_col], args.return_col)
    volatility_summary = _group_summary(triggered_df, ["volatility_bucket"], args.return_col) if volatility_threshold is not None else []
    regime_volatility_summary = (
        _group_summary(triggered_df, [args.regime_col, "volatility_bucket"], args.return_col)
        if volatility_threshold is not None
        else []
    )

    dominant_loss_bucket = min(regime_volatility_summary, key=lambda item: item["net_return_total"]) if regime_volatility_summary else None
    payload = {
        "candidate": str(args.candidate),
        "overlap_dataset": str(args.overlap_dataset),
        "scope": {
            "ts_col": str(args.ts_col),
            "signal_col": str(args.signal_col),
            "return_col": str(args.return_col),
            "regime_col": str(args.regime_col),
            "volatility_col": str(args.volatility_col),
        },
        "overlap_scope": {
            "row_count": int(len(overlap_df)),
            "triggered_row_count": int(len(triggered_df)),
            "triggered_net_return_total": float(pd.to_numeric(triggered_df[args.return_col], errors="coerce").fillna(0.0).sum()),
            "triggered_hit_rate": float(
                (pd.to_numeric(triggered_df[args.return_col], errors="coerce").fillna(0.0) > 0.0).mean()
            ) if len(triggered_df) else float("nan"),
        },
        "volatility_threshold_median": volatility_threshold,
        "regime_summary": regime_summary,
        "volatility_summary": volatility_summary,
        "regime_volatility_summary": regime_volatility_summary,
        "dominant_loss_bucket": dominant_loss_bucket,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()