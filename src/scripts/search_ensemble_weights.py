import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

StrategyMetrics = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype sweep for weighting transformer/LSTM/XGB ensembles.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("artifacts/backtests/backtest_signals.csv"),
        help="CSV with baseline model metrics (LSTM, XGB).",
    )
    parser.add_argument(
        "--transformer-summary",
        type=Path,
        default=Path(
            "artifacts/analysis/backtest_signals_transformer_dir1h_optuna_v2_sweep/summary.csv"
        ),
        help="CSV with transformer sweep metrics (includes fee/slippage columns).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/ensemble_weight_sweep"),
        help="Directory where the sweep summary will be written.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.25,
        help="Step size for weight grid (defaults to quarters).",
    )
    return parser.parse_args()


def adjust_net_return(
    base_net: float,
    base_fee_bps: float,
    base_slip_bps: float,
    target_fee_bps: float,
    target_slip_bps: float,
    trades: int,
) -> float:
    """Apply a simple linear adjustment for fees/slippage deltas."""

    if trades == 0:
        return base_net
    base_total = base_fee_bps + base_slip_bps
    target_total = target_fee_bps + target_slip_bps
    delta_bps = target_total - base_total
    delta_per_trade = delta_bps / 10000.0
    return base_net - trades * delta_per_trade


def load_baseline_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {path}")
    df = pd.read_csv(path)
    expected = {"strategy", "n_trades", "cum_ret_net"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {sorted(missing)}")
    return df


def load_transformer_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transformer summary file not found: {path}")
    df = pd.read_csv(path)
    expected = {
        "combo",
        "fee_bps",
        "slippage_bps",
        "n_trades",
        "cum_ret_net",
        "ret_min",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Transformer summary CSV missing columns: {sorted(missing)}")
    return df


def get_weight_grid(step: float) -> Iterable[Tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("weight-step must be in (0, 1]")
    precision = int(round(1 / step))
    for i in range(precision + 1):
        for j in range(precision + 1 - i):
            k = precision - i - j
            wt = round(i * step, 10)
            wl = round(j * step, 10)
            wx = round(k * step, 10)
            if wt + wl + wx != 1.0:
                # Numerical guard: round again
                total = round(wt + wl + wx, 6)
                if total != 1.0:
                    continue
            yield wt, wl, wx


def pick_transformer_row(
    df: pd.DataFrame,
    fee_bps: float,
    slip_bps: float,
) -> pd.Series:
    mask = (
        df["fee_bps"].round(6) == round(fee_bps, 6)
    ) & (df["slippage_bps"].round(6) == round(slip_bps, 6))
    candidates = df[mask]
    if not candidates.empty:
        # Prefer ret_min == 0 when multiple entries exist.
        zero_ret = candidates[candidates["ret_min"].round(6) == 0.0]
        if not zero_ret.empty:
            candidates = zero_ret
        return candidates.sort_values("cum_ret_net", ascending=False).iloc[0]
    # Fallback: use the best available row (highest net) and adjust fees downstream.
    zero_ret = df[df["ret_min"].round(6) == 0.0]
    source = zero_ret if not zero_ret.empty else df
    return source.sort_values("cum_ret_net", ascending=False).iloc[0]


def build_strategy_metrics(
    baseline_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    fee_schedules: List[Tuple[float, float]],
) -> Dict[Tuple[float, float], Dict[str, Dict[str, float]]]:
    metrics: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
    for fee_bps, slip_bps in fee_schedules:
        schedule_key = (fee_bps, slip_bps)
        fee_metrics: Dict[str, Dict[str, float]] = {}
        # Baseline strategies are assumed to originate from 20/10 bps
        baseline_fee = 2.0
        baseline_slip = 1.0
        for _, row in baseline_df.iterrows():
            strategy = row["strategy"]
            trades = int(row["n_trades"])
            base_net = float(row["cum_ret_net"])
            net = adjust_net_return(base_net, baseline_fee, baseline_slip, fee_bps, slip_bps, trades)
            fee_metrics[strategy] = {
                "n_trades": trades,
                "cum_ret_net": net,
            }
        # Transformer metrics come from sweep CSV (already recorded under different fees)
        t_row = pick_transformer_row(transformer_df, fee_bps, slip_bps)
        trades = int(t_row["n_trades"])
        base_net = float(t_row["cum_ret_net"])
        net = adjust_net_return(
            base_net,
            float(t_row["fee_bps"]),
            float(t_row["slippage_bps"]),
            fee_bps,
            slip_bps,
            trades,
        )
        fee_metrics["transformer_optuna_v2"] = {
            "n_trades": trades,
            "cum_ret_net": net,
        }
        metrics[schedule_key] = fee_metrics
    return metrics


def sweep_weights(
    metrics_by_fee: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    weight_grid: Iterable[Tuple[float, float, float]],
) -> pd.DataFrame:
    records = []
    for (fee_bps, slip_bps), strat_metrics in metrics_by_fee.items():
        for wt, wl, wx in weight_grid:
            weights = {
                "transformer_optuna_v2": wt,
                "lstm_dir1h_v2": wl,
                "xgb_dir1h_optuna": wx,
            }
            # Skip combinations that exclude a strategy entirely if its weight grid omitted by construction
            if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
                continue
            net = 0.0
            trades = 0.0
            missing = False
            for name, weight in weights.items():
                if name not in strat_metrics:
                    missing = True
                    break
                metrics = strat_metrics[name]
                net += weight * metrics["cum_ret_net"]
                trades += weight * metrics["n_trades"]
            if missing:
                continue
            records.append(
                {
                    "fee_bps": fee_bps,
                    "slippage_bps": slip_bps,
                    "weight_transformer": wt,
                    "weight_lstm": wl,
                    "weight_xgb": wx,
                    "ensemble_cum_ret_net": net,
                    "ensemble_trades": trades,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline_metrics(args.baseline_csv)
    transformer_df = load_transformer_summary(args.transformer_summary)

    fee_schedules = [(2.0, 1.0), (2.5, 1.2), (3.0, 1.5)]
    metrics_by_fee = build_strategy_metrics(baseline_df, transformer_df, fee_schedules)
    weight_grid = list(get_weight_grid(args.weight_step))

    results_df = sweep_weights(metrics_by_fee, weight_grid)
    if results_df.empty:
        raise RuntimeError("No ensemble combinations were evaluated; check inputs.")

    # Rank within each fee schedule for quick inspection
    ranked = (
        results_df.sort_values(["fee_bps", "slippage_bps", "ensemble_cum_ret_net"], ascending=[True, True, False])
        .groupby(["fee_bps", "slippage_bps"])
        .head(5)
    )
    output_csv = args.output_dir / "ensemble_weight_sweep.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Wrote full sweep to {output_csv}")

    print("Top 5 combinations per fee schedule (by net log return):")
    for (fee_bps, slip_bps), group in ranked.groupby(["fee_bps", "slippage_bps"]):
        print(f"\nFee {fee_bps:.1f} bps / Slippage {slip_bps:.1f} bps")
        for _, row in group.iterrows():
            print(
                "  weights (T/L/X) = "
                f"({row['weight_transformer']:.2f}, {row['weight_lstm']:.2f}, {row['weight_xgb']:.2f})"
                f" | net={row['ensemble_cum_ret_net']:.6f} | trades={row['ensemble_trades']:.1f}"
            )


if __name__ == "__main__":
    main()
