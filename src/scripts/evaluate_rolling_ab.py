from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_policy_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"ts", "ret_ensemble_net", "signal_ensemble"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    out = df[["ts", "ret_ensemble_net", "signal_ensemble"]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["ret_ensemble_net"] = pd.to_numeric(out["ret_ensemble_net"], errors="coerce")
    out["signal_ensemble"] = pd.to_numeric(out["signal_ensemble"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["ts", "ret_ensemble_net"]).sort_values("ts")
    return out.reset_index(drop=True)


def _window_stats(ret: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    signal_mask = signal > 0
    active = ret[signal_mask]
    n = int(signal_mask.sum())
    if n == 0:
        return {
            "n_trades": 0,
            "cum_ret": 0.0,
            "avg_ret": 0.0,
            "hit_rate": 0.0,
        }
    return {
        "n_trades": n,
        "cum_ret": float(active.sum()),
        "avg_ret": float(active.mean()),
        "hit_rate": float((active > 0).mean()),
    }


def _rolling_windows(df: pd.DataFrame, window_size: int, step_size: int) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    if window_size <= 0 or step_size <= 0:
        return out
    if len(df) <= window_size:
        return [df.copy()] if len(df) > 0 else []
    for start in range(0, max(len(df) - window_size + 1, 0), step_size):
        out.append(df.iloc[start : start + window_size].copy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rolling-window A/B between baseline and candidate signal policies.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline backtest_signals.csv")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate/shadow backtest_signals.csv")
    parser.add_argument("--window-size", type=int, default=168, help="Window size in rows (default 168 ~ 1 week hourly).")
    parser.add_argument("--step-size", type=int, default=24, help="Window step in rows (default 24 ~ 1 day hourly).")
    parser.add_argument("--min-window-trades", type=int, default=5, help="Minimum trades per policy in a window to count it.")
    parser.add_argument(
        "--allow-no-trade-baseline",
        action="store_true",
        help="Treat windows with baseline no-trade as valid when candidate meets min-window-trades.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/rolling_ab_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("artifacts/monitoring/rolling_ab_report.md"))
    args = parser.parse_args()

    baseline = _load_policy_csv(args.baseline).rename(
        columns={"ret_ensemble_net": "ret_baseline", "signal_ensemble": "signal_baseline"}
    )
    candidate = _load_policy_csv(args.candidate).rename(
        columns={"ret_ensemble_net": "ret_candidate", "signal_ensemble": "signal_candidate"}
    )

    merged = baseline.merge(candidate, on="ts", how="inner")
    if merged.empty:
        payload = {
            "status": "no_overlapping_timestamps",
            "message": "No overlapping timestamps between baseline and candidate backtests",
            "rows_overlap": 0,
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "min_window_trades": int(args.min_window_trades),
            "allow_no_trade_baseline": bool(args.allow_no_trade_baseline),
            "overall": {
                "baseline": _window_stats(np.array([], dtype=float), np.array([], dtype=float)),
                "candidate": _window_stats(np.array([], dtype=float), np.array([], dtype=float)),
                "delta_cum_ret": 0.0,
            },
            "rolling_summary": {
                "windows_total": 0,
                "candidate_wins": 0,
                "baseline_wins": 0,
                "ties": 0,
            },
            "windows": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_lines = [
            "# Rolling A/B Report",
            "",
            "- Status: no_overlapping_timestamps",
            "- Message: No overlapping timestamps between baseline and candidate backtests",
            f"- Window size: {payload['window_size']}",
            f"- Step size: {payload['step_size']}",
            f"- Min window trades: {payload['min_window_trades']}",
        ]
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(md_lines), encoding="utf-8")
        print(json.dumps(payload["rolling_summary"], indent=2))
        return

    windows = _rolling_windows(merged, window_size=int(args.window_size), step_size=int(args.step_size))
    report_windows: List[Dict[str, object]] = []
    wins_candidate = 0
    wins_baseline = 0
    ties = 0

    for idx, w in enumerate(windows):
        b_stats = _window_stats(w["ret_baseline"].to_numpy(dtype=float), w["signal_baseline"].to_numpy(dtype=float))
        c_stats = _window_stats(w["ret_candidate"].to_numpy(dtype=float), w["signal_candidate"].to_numpy(dtype=float))

        min_trades = int(args.min_window_trades)
        baseline_ok = b_stats["n_trades"] >= min_trades
        candidate_ok = c_stats["n_trades"] >= min_trades
        valid = baseline_ok and candidate_ok
        if bool(args.allow_no_trade_baseline) and (not baseline_ok) and int(b_stats["n_trades"]) == 0 and candidate_ok:
            valid = True
        winner = "insufficient_trades"
        if valid:
            if c_stats["cum_ret"] > b_stats["cum_ret"]:
                winner = "candidate"
                wins_candidate += 1
            elif c_stats["cum_ret"] < b_stats["cum_ret"]:
                winner = "baseline"
                wins_baseline += 1
            else:
                winner = "tie"
                ties += 1

        report_windows.append(
            {
                "window_index": idx,
                "start_ts": w["ts"].iloc[0].isoformat(),
                "end_ts": w["ts"].iloc[-1].isoformat(),
                "valid": valid,
                "winner": winner,
                "validity_context": {
                    "baseline_ok": bool(baseline_ok),
                    "candidate_ok": bool(candidate_ok),
                    "allow_no_trade_baseline": bool(args.allow_no_trade_baseline),
                },
                "baseline": b_stats,
                "candidate": c_stats,
                "delta_cum_ret": float(c_stats["cum_ret"] - b_stats["cum_ret"]),
            }
        )

    overall_baseline = _window_stats(merged["ret_baseline"].to_numpy(dtype=float), merged["signal_baseline"].to_numpy(dtype=float))
    overall_candidate = _window_stats(merged["ret_candidate"].to_numpy(dtype=float), merged["signal_candidate"].to_numpy(dtype=float))

    payload = {
        "rows_overlap": int(len(merged)),
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "min_window_trades": int(args.min_window_trades),
        "allow_no_trade_baseline": bool(args.allow_no_trade_baseline),
        "overall": {
            "baseline": overall_baseline,
            "candidate": overall_candidate,
            "delta_cum_ret": float(overall_candidate["cum_ret"] - overall_baseline["cum_ret"]),
        },
        "rolling_summary": {
            "windows_total": int(len(report_windows)),
            "candidate_wins": int(wins_candidate),
            "baseline_wins": int(wins_baseline),
            "ties": int(ties),
        },
        "windows": report_windows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Rolling A/B Report",
        "",
        f"- Overlap rows: {payload['rows_overlap']}",
        f"- Window size: {payload['window_size']}",
        f"- Step size: {payload['step_size']}",
        f"- Candidate wins: {wins_candidate}",
        f"- Baseline wins: {wins_baseline}",
        f"- Ties: {ties}",
        "",
        "## Overall",
        f"- Baseline cum_ret: {overall_baseline['cum_ret']:.6f}",
        f"- Candidate cum_ret: {overall_candidate['cum_ret']:.6f}",
        f"- Delta cum_ret: {payload['overall']['delta_cum_ret']:.6f}",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(payload["rolling_summary"], indent=2))


if __name__ == "__main__":
    main()
