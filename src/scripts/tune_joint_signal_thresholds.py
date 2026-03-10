from __future__ import annotations

import argparse
import json
import sys
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
    full_cum_ret: float
    stability_gap: float
    max_drawdown: float
    economics_score: float
    selection_value: float


def _max_drawdown_from_log_returns(strategy_ret: np.ndarray) -> float:
    equity_log = np.cumsum(strategy_ret)
    running_max = np.maximum.accumulate(equity_log)
    drawdown = equity_log - running_max
    if drawdown.size == 0:
        return 0.0
    return float(drawdown.min())


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")


def _load_overlap_timestamps(npz_path: Path) -> set[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        keys = ["ts_all", "ts_train", "ts_val", "ts_test"]
        rows: List[str] = []
        for key in keys:
            if key not in data.files:
                continue
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size == 0:
                continue
            ser = pd.Series(arr.astype(str), dtype="string")
            norm = _normalize_timestamp_series(ser).dropna()
            rows.extend(str(v) for v in norm.tolist())
    if not rows:
        raise RuntimeError(f"No overlap timestamps found in dataset: {npz_path}")
    return set(rows)


def _economics_score(cum_ret: float, n_trades: int, *, turnover_penalty: float, downside_penalty: float, min_trades: int) -> float:
    excess_turnover = max(0.0, float(n_trades) - float(min_trades))
    downside = max(0.0, -float(cum_ret))
    return float(cum_ret) - float(turnover_penalty) * excess_turnover - float(downside_penalty) * downside


def _eval_candidate(
    df: pd.DataFrame,
    p_up_min: float,
    ret_min: float,
    direction_threshold: float,
    *,
    turnover_penalty: float,
    downside_penalty: float,
    min_trades: int,
) -> Candidate:
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
    econ = _economics_score(
        cum_ret,
        n_trades,
        turnover_penalty=float(turnover_penalty),
        downside_penalty=float(downside_penalty),
        min_trades=int(min_trades),
    )

    # Keep dir signal computed to make sure threshold is valid over current distribution.
    _ = int(signal_dir.sum())

    return Candidate(
        p_up_min=float(p_up_min),
        ret_min=float(ret_min),
        direction_threshold=float(direction_threshold),
        n_trades=n_trades,
        cum_ret=cum_ret,
        full_cum_ret=cum_ret,
        stability_gap=0.0,
        max_drawdown=max_dd,
        economics_score=econ,
        selection_value=econ,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly tune p_up_min, ret_min, and direction_threshold on canonical evaluation data.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical labeled CSV with p_up, ret_pred, ret_1h columns.")
    parser.add_argument("--p-up-grid", type=str, default="0.50,0.55,0.60,0.65")
    parser.add_argument("--ret-min-grid", type=str, default="-0.0002,0.0,0.0002,0.0005")
    parser.add_argument("--direction-threshold-grid", type=str, default="0.50,0.55,0.60")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--max-dd", type=float, default=-0.12, help="Minimum acceptable max drawdown (log space).")
    parser.add_argument("--min-cum-ret", type=float, default=0.0, help="Minimum cumulative return required for deployable candidates.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="cum_ret",
        choices=["cum_ret", "economics_score"],
        help="Metric to maximize when selecting best feasible/deployable thresholds.",
    )
    parser.add_argument(
        "--economics-turnover-penalty",
        type=float,
        default=0.002,
        help="Penalty per trade above min-trades when selection metric is economics_score.",
    )
    parser.add_argument(
        "--economics-downside-penalty",
        type=float,
        default=2.0,
        help="Penalty multiplier applied to negative cumulative return when selection metric is economics_score.",
    )
    parser.add_argument(
        "--overlap-dataset",
        type=Path,
        default=None,
        help="Optional overlap NPZ dataset with ts_* keys used to filter input rows to the labeled-overlap slice.",
    )
    parser.add_argument(
        "--ts-col",
        type=str,
        default="ts",
        help="Timestamp column name in --input used for overlap filtering.",
    )
    parser.add_argument(
        "--stability-gap-penalty",
        type=float,
        default=0.0,
        help="Penalty multiplied by |cum_ret(full)-cum_ret(overlap)| and subtracted from selection metric.",
    )
    parser.add_argument(
        "--max-stability-gap",
        type=float,
        default=1e9,
        help="Maximum allowed absolute full-vs-overlap cumulative-return gap for a candidate to remain feasible.",
    )
    parser.add_argument(
        "--min-overlap-rows",
        type=int,
        default=0,
        help="Minimum rows required after overlap filtering when --overlap-dataset is used.",
    )
    parser.add_argument(
        "--strict-accept",
        action="store_true",
        help="Exit with code 2 when no deployable threshold candidate passes constraints.",
    )
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
    df_full = df.copy()

    overlap_rows_before = int(len(df))
    overlap_rows_after = int(len(df))
    overlap_enabled = args.overlap_dataset is not None
    if args.overlap_dataset is not None:
        if not args.overlap_dataset.exists():
            raise FileNotFoundError(args.overlap_dataset)
        if args.ts_col not in df.columns:
            raise ValueError(f"Input missing required timestamp column for overlap filtering: {args.ts_col}")
        overlap_ts = _load_overlap_timestamps(args.overlap_dataset)
        overlap_rows_before = int(len(df))
        ts_norm = _normalize_timestamp_series(df[args.ts_col])
        mask = ts_norm.isin(overlap_ts)
        df = df.loc[mask].copy()
        overlap_rows_after = int(len(df))
        if overlap_rows_after < int(args.min_overlap_rows):
            raise RuntimeError(
                f"Overlap filtered rows {overlap_rows_after} below min_overlap_rows={int(args.min_overlap_rows)}",
            )
    if df.empty:
        raise RuntimeError("No valid rows available after numeric coercion.")

    p_up_grid = _parse_grid(args.p_up_grid)
    ret_min_grid = _parse_grid(args.ret_min_grid)
    dir_grid = _parse_grid(args.direction_threshold_grid)

    candidates: List[Candidate] = []
    for p_up_min in p_up_grid:
        for ret_min in ret_min_grid:
            for dthr in dir_grid:
                cand = _eval_candidate(
                    df,
                    p_up_min=p_up_min,
                    ret_min=ret_min,
                    direction_threshold=dthr,
                    turnover_penalty=float(args.economics_turnover_penalty),
                    downside_penalty=float(args.economics_downside_penalty),
                    min_trades=int(args.min_trades),
                )
                candidates.append(cand)

    metric_name = str(args.selection_metric)
    stability_gap_penalty = float(args.stability_gap_penalty)

    for idx, cand in enumerate(candidates):
        full_eval = _eval_candidate(
            df_full,
            p_up_min=cand.p_up_min,
            ret_min=cand.ret_min,
            direction_threshold=cand.direction_threshold,
            turnover_penalty=float(args.economics_turnover_penalty),
            downside_penalty=float(args.economics_downside_penalty),
            min_trades=int(args.min_trades),
        )
        base_metric = cand.economics_score if metric_name == "economics_score" else cand.cum_ret
        stability_gap = abs(float(full_eval.cum_ret) - float(cand.cum_ret))
        selection_value = float(base_metric) - stability_gap_penalty * float(stability_gap)
        candidates[idx] = Candidate(
            p_up_min=cand.p_up_min,
            ret_min=cand.ret_min,
            direction_threshold=cand.direction_threshold,
            n_trades=cand.n_trades,
            cum_ret=cand.cum_ret,
            full_cum_ret=float(full_eval.cum_ret),
            stability_gap=float(stability_gap),
            max_drawdown=cand.max_drawdown,
            economics_score=cand.economics_score,
            selection_value=float(selection_value),
        )

    def _sort_key(c: Candidate) -> tuple[float, float, float, float]:
        return (c.selection_value, c.cum_ret, c.n_trades, c.max_drawdown)

    max_stability_gap = float(args.max_stability_gap)
    feasible_all = [
        c
        for c in candidates
        if c.n_trades >= int(args.min_trades)
        and c.max_drawdown >= float(args.max_dd)
        and c.stability_gap <= max_stability_gap
    ]
    feasible_deployable = [c for c in feasible_all if c.cum_ret >= float(args.min_cum_ret)]

    best_any = None
    if feasible_all:
        best_any = max(feasible_all, key=_sort_key)
    elif candidates:
        best_any = max(candidates, key=_sort_key)

    best = None
    accepted = False
    if feasible_deployable:
        best = max(feasible_deployable, key=_sort_key)
        accepted = True
    elif best_any is not None:
        best = best_any

    if best is None:
        raise RuntimeError("No threshold candidates available.")

    payload = {
        "rows": int(len(df)),
        "constraints": {
            "min_trades": int(args.min_trades),
            "max_dd": float(args.max_dd),
            "min_cum_ret": float(args.min_cum_ret),
            "selection_metric": metric_name,
            "stability_gap_penalty": stability_gap_penalty,
            "max_stability_gap": max_stability_gap,
            "min_overlap_rows": int(args.min_overlap_rows),
            "economics_turnover_penalty": float(args.economics_turnover_penalty),
            "economics_downside_penalty": float(args.economics_downside_penalty),
        },
        "overlap_filter": {
            "enabled": bool(overlap_enabled),
            "dataset": str(args.overlap_dataset) if args.overlap_dataset else None,
            "ts_col": str(args.ts_col),
            "rows_before": int(overlap_rows_before),
            "rows_after": int(overlap_rows_after),
        },
        "accepted": bool(accepted),
        "best": {
            "p_up_min": best.p_up_min,
            "ret_min": best.ret_min,
            "direction_threshold": best.direction_threshold,
            "n_trades": best.n_trades,
            "cum_ret": best.cum_ret,
            "full_cum_ret": best.full_cum_ret,
            "stability_gap": best.stability_gap,
            "max_drawdown": best.max_drawdown,
            "economics_score": best.economics_score,
            "selection_value": best.selection_value,
        },
        "n_candidates": int(len(candidates)),
        "n_feasible": int(len(feasible_all)),
        "n_deployable": int(len(feasible_deployable)),
    }

    if not accepted:
        payload["reason"] = "no_deployable_candidate"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not accepted and bool(args.strict_accept):
        print(
            "No deployable threshold candidate satisfied min_trades/max_dd/min_cum_ret constraints.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
