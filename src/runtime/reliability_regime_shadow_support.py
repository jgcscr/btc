from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_regime_abs_ret_pred_floor_shadow(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    signal_col: str,
    return_col: str,
    regime_col: str,
    ret_pred_col: str,
    regime_state: str,
    min_abs_ret_pred: float,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = {signal_col, return_col, regime_col, ret_pred_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    working = df.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    ret_pred = pd.to_numeric(working[ret_pred_col], errors="coerce").fillna(0.0).abs()
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    normalized_regime_state = str(regime_state).strip().lower()
    blocked_mask = (signal > 0.0) & (regimes == normalized_regime_state) & (ret_pred < float(min_abs_ret_pred))

    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        working.to_parquet(output_path, index=False)
    else:
        working.to_csv(output_path, index=False)

    final_signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    final_returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "trade_count": int((final_signal > 0.0).sum()),
        "net_return_total": float(final_returns.loc[final_signal > 0.0].sum()),
        "neutral_abs_ret_pred_floor": {
            "enabled": True,
            "regime_col": regime_col,
            "ret_pred_col": ret_pred_col,
            "regime_state": normalized_regime_state,
            "min_abs_ret_pred": float(min_abs_ret_pred),
            "blocked_rows": int(blocked_mask.sum()),
            "blocked_net_return_total": float(returns.loc[blocked_mask].sum()) if bool(blocked_mask.any()) else 0.0,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_regime_max_p_up_shadow(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    regime_state: str,
    max_p_up_exclusive: float,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = {signal_col, return_col, regime_col, p_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    working = df.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    p_up = pd.to_numeric(working[p_col], errors="coerce")
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    normalized_regime_state = str(regime_state).strip().lower()
    blocked_mask = (signal > 0.0) & (regimes == normalized_regime_state) & (p_up >= float(max_p_up_exclusive))

    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        working.to_parquet(output_path, index=False)
    else:
        working.to_csv(output_path, index=False)

    final_signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    final_returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "trade_count": int((final_signal > 0.0).sum()),
        "net_return_total": float(final_returns.loc[final_signal > 0.0].sum()),
        "neutral_p_up_cap": {
            "enabled": True,
            "regime_col": regime_col,
            "p_col": p_col,
            "regime_state": normalized_regime_state,
            "max_p_up_exclusive": float(max_p_up_exclusive),
            "blocked_rows": int(blocked_mask.sum()),
            "blocked_net_return_total": float(returns.loc[blocked_mask].sum()) if bool(blocked_mask.any()) else 0.0,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload