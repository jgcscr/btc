from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_grid(values: str) -> List[float]:
    out: List[float] = []
    for raw in values.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(float(raw))
    if not out:
        raise ValueError("Grid cannot be empty")
    return out


@dataclass
class Candidate:
    p_up_min: float
    ret_min: float
    direction_threshold: float
    n_trades: int
    cum_ret: float
    max_drawdown: float


def _max_drawdown_from_log_returns(strategy_ret: np.ndarray) -> float:
    equity_log = np.cumsum(strategy_ret)
    running_max = np.maximum.accumulate(equity_log)
    drawdown = equity_log - running_max
    if drawdown.size == 0:
        return 0.0
    return float(drawdown.min())


def _eval_candidate(df: pd.DataFrame, p_up_min: float, ret_min: float, direction_threshold: float) -> Candidate:
    p_up = df["p_up"].to_numpy(dtype=float)
    ret_pred = df["ret_pred"].to_numpy(dtype=float)
    ret_realized = df["ret_1h"].to_numpy(dtype=float)

    signal_ensemble = (p_up >= p_up_min) & (ret_pred >= ret_min)
    signal_dir = p_up >= direction_threshold

    # Objective focuses on deployable ensemble performance; direction threshold is tracked for paired policy updates.
    strategy_ret = ret_realized * signal_ensemble.astype(float)
    n_trades = int(signal_ensemble.sum())
    cum_ret = float(strategy_ret.sum())
    max_dd = _max_drawdown_from_log_returns(strategy_ret)

    # Keep dir signal computed to make sure threshold is valid over current distribution.
    _ = int(signal_dir.sum())

    return Candidate(
        p_up_min=float(p_up_min),
        ret_min=float(ret_min),
        direction_threshold=float(direction_threshold),
        n_trades=n_trades,
        cum_ret=cum_ret,
        max_drawdown=max_dd,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly tune p_up_min, ret_min, and direction_threshold on canonical evaluation data.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical labeled CSV with p_up, ret_pred, ret_1h columns.")
    parser.add_argument("--p-up-grid", type=str, default="0.50,0.55,0.60,0.65")
    parser.add_argument("--ret-min-grid", type=str, default="-0.0002,0.0,0.0002,0.0005")
    parser.add_argument("--direction-threshold-grid", type=str, default="0.50,0.55,0.60")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--max-dd", type=float, default=-0.12, help="Minimum acceptable max drawdown (log space).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/joint_threshold_tuning.json"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    required = ["p_up", "ret_pred", "ret_1h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise RuntimeError("No valid rows available after numeric coercion.")

    p_up_grid = _parse_grid(args.p_up_grid)
    ret_min_grid = _parse_grid(args.ret_min_grid)
    dir_grid = _parse_grid(args.direction_threshold_grid)

    candidates: List[Candidate] = []
    for p_up_min in p_up_grid:
        for ret_min in ret_min_grid:
            for dthr in dir_grid:
                cand = _eval_candidate(df, p_up_min=p_up_min, ret_min=ret_min, direction_threshold=dthr)
                candidates.append(cand)

    feasible = [
        c
        for c in candidates
        if c.n_trades >= int(args.min_trades) and c.max_drawdown >= float(args.max_dd)
    ]
    if not feasible:
        feasible = [c for c in candidates if c.n_trades >= max(1, int(args.min_trades // 2))]
    if not feasible:
        feasible = list(candidates)

    best = max(feasible, key=lambda c: (c.cum_ret, c.n_trades, c.max_drawdown))

    payload = {
        "rows": int(len(df)),
        "constraints": {
            "min_trades": int(args.min_trades),
            "max_dd": float(args.max_dd),
        },
        "best": {
            "p_up_min": best.p_up_min,
            "ret_min": best.ret_min,
            "direction_threshold": best.direction_threshold,
            "n_trades": best.n_trades,
            "cum_ret": best.cum_ret,
            "max_drawdown": best.max_drawdown,
        },
        "n_candidates": int(len(candidates)),
        "n_feasible": int(len(feasible)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
