from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pair_frames(default_df: pd.DataFrame, midband_df: pd.DataFrame, incumbent_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    has_ts = "ts" in default_df.columns and "ts" in midband_df.columns and "ts" in incumbent_df.columns
    if has_ts:
        default_copy = default_df.copy()
        midband_copy = midband_df.copy()
        incumbent_copy = incumbent_df.copy()

        default_copy["_ts"] = pd.to_datetime(default_copy["ts"], utc=True, errors="coerce").dt.floor("h")
        midband_copy["_ts"] = pd.to_datetime(midband_copy["ts"], utc=True, errors="coerce").dt.floor("h")
        incumbent_copy["_ts"] = pd.to_datetime(incumbent_copy["ts"], utc=True, errors="coerce").dt.floor("h")

        default_copy = default_copy.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        midband_copy = midband_copy.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        incumbent_copy = incumbent_copy.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")

        default_cols = [col for col in default_copy.columns if col != "_ts"]
        midband_cols = [col for col in midband_copy.columns if col != "_ts"]

        paired = default_copy.rename(columns={col: f"{col}_default" for col in default_cols})
        paired = paired.merge(
            midband_copy.rename(columns={col: f"{col}_midband_shadow" for col in midband_cols}),
            on="_ts",
            how="inner",
        )
        paired = paired.merge(incumbent_copy, on="_ts", how="inner")
        return paired, "timestamp_hour"

    n_rows = int(min(len(default_df), len(midband_df), len(incumbent_df)))
    default_slice = default_df.tail(n_rows).reset_index(drop=True)
    midband_slice = midband_df.tail(n_rows).reset_index(drop=True)
    incumbent_slice = incumbent_df.tail(n_rows).reset_index(drop=True)

    paired = pd.DataFrame(index=np.arange(n_rows))
    for col in default_slice.columns:
        paired[f"{col}_default"] = default_slice[col].to_numpy()
    for col in midband_slice.columns:
        paired[f"{col}_midband_shadow"] = midband_slice[col].to_numpy()
    for col in incumbent_slice.columns:
        paired[col] = incumbent_slice[col].to_numpy()
    return paired, "tail_index"


def _bootstrap_stats(candidate: np.ndarray, incumbent: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    n_rows = int(min(candidate.size, incumbent.size))
    if n_rows <= 5:
        return {
            "mean_diff": float("nan"),
            "pvalue_one_sided": float("nan"),
            "nonzero_paired_rows": 0,
            "std_diff": float("nan"),
            "n_pairs": int(n_rows),
        }

    candidate_slice = candidate[-n_rows:]
    incumbent_slice = incumbent[-n_rows:]
    diff = candidate_slice - incumbent_slice

    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        samples[idx] = float(np.mean(diff[sample_idx]))

    return {
        "mean_diff": float(np.mean(diff)),
        "pvalue_one_sided": float(np.mean(samples <= 0.0)) if int(n_boot) > 0 else float("nan"),
        "nonzero_paired_rows": int(np.count_nonzero(np.abs(diff) > 0.0)),
        "std_diff": float(np.std(diff, ddof=1)) if n_rows > 1 else float("nan"),
        "n_pairs": int(n_rows),
    }


def _window_metrics(
    *,
    candidate_ret: np.ndarray,
    candidate_signal: np.ndarray,
    incumbent_ret: np.ndarray,
    incumbent_signal: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    stats = _bootstrap_stats(candidate_ret, incumbent_ret, n_boot=n_boot, seed=seed)
    return {
        "candidate_trade_count": int(np.count_nonzero(candidate_signal != 0.0)),
        "incumbent_trade_count": int(np.count_nonzero(incumbent_signal != 0.0)),
        "candidate_net_return_total": float(np.sum(candidate_ret)),
        "incumbent_net_return_total": float(np.sum(incumbent_ret)),
        "mean_diff": float(stats["mean_diff"]),
        "pvalue_one_sided": float(stats["pvalue_one_sided"]),
        "nonzero_paired_rows": int(stats["nonzero_paired_rows"]),
        "std_diff": float(stats["std_diff"]),
        "n_pairs": int(stats["n_pairs"]),
    }


def _build_windows(n_rows: int, window_size: int, step_size: int, min_rows: int) -> List[Tuple[int, int]]:
    if n_rows <= 0:
        return []
    if n_rows < max(int(window_size), int(min_rows)):
        return [(0, n_rows)] if n_rows >= int(min_rows) else []

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_rows:
        end = min(start + int(window_size), n_rows)
        if end - start >= int(min_rows):
            windows.append((start, end))
        if end == n_rows:
            break
        start += int(step_size)
    return windows


def _run_level_verdict(aggregate_delta_net: float, aggregate_delta_mean: float, clearly_harmed_windows: int) -> str:
    if np.isfinite(aggregate_delta_net) and np.isfinite(aggregate_delta_mean):
        if aggregate_delta_net > 0.0 and aggregate_delta_mean > 0.0 and clearly_harmed_windows == 0:
            return "midband better"
        if aggregate_delta_net < 0.0 and aggregate_delta_mean < 0.0:
            return "midband worse"
    return "inconclusive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused retrospective comparison for default vs midband shadow veto.")
    parser.add_argument("--default-candidate", type=Path, required=True)
    parser.add_argument("--midband-shadow-candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--incumbent-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=24)
    parser.add_argument("--min-rows", type=int, default=80)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--midband-shadow-meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.default_candidate, args.midband_shadow_candidate, args.incumbent):
        if not path.exists():
            raise FileNotFoundError(path)

    default_df = _read_csv_or_parquet(args.default_candidate)
    midband_df = _read_csv_or_parquet(args.midband_shadow_candidate)
    incumbent_df = _read_csv_or_parquet(args.incumbent)

    paired, pairing = _pair_frames(default_df, midband_df, incumbent_df)
    if paired.empty:
        output_payload = {
            "default_candidate": str(args.default_candidate),
            "midband_shadow_candidate": str(args.midband_shadow_candidate),
            "incumbent": str(args.incumbent),
            "pairing": pairing,
            "status": "no_paired_rows",
            "message": "No paired rows available for midband shadow comparison",
            "windowing": {
                "window_size": int(args.window_size),
                "step_size": int(args.step_size),
                "min_rows": int(args.min_rows),
                "n_boot": int(args.n_boot),
                "seed": int(args.seed),
            },
            "current_window": None,
            "retrospective_windows": [],
            "aggregate_summary": {
                "number_of_evaluated_windows": 0,
                "windows_improved_by_mean_diff": 0,
                "windows_improved_by_candidate_net_return": 0,
                "clearly_harmed_windows": [],
                "aggregate_delta_candidate_net_return_total": float("nan"),
                "aggregate_delta_mean_diff": float("nan"),
                "median_vetoed_rows": float("nan"),
                "mean_vetoed_rows": float("nan"),
            },
            "shadow_meta_vetoed_rows": {
                "midband_shadow": None,
            },
            "run_level_verdict": "inconclusive",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(json.dumps(output_payload, indent=2))
        return

    default_ret = pd.to_numeric(paired[f"{args.candidate_col}_default"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    midband_ret = pd.to_numeric(paired[f"{args.candidate_col}_midband_shadow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    incumbent_ret = pd.to_numeric(paired[args.incumbent_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    default_sig = pd.to_numeric(paired[f"{args.signal_col}_default"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    midband_sig = pd.to_numeric(paired[f"{args.signal_col}_midband_shadow"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    incumbent_sig = pd.to_numeric(paired[args.signal_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    default_current = _window_metrics(
        candidate_ret=default_ret,
        candidate_signal=default_sig,
        incumbent_ret=incumbent_ret,
        incumbent_signal=incumbent_sig,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    midband_current = _window_metrics(
        candidate_ret=midband_ret,
        candidate_signal=midband_sig,
        incumbent_ret=incumbent_ret,
        incumbent_signal=incumbent_sig,
        n_boot=int(args.n_boot),
        seed=int(args.seed) + 100,
    )

    current_vetoed_rows = int(np.count_nonzero((default_sig != 0.0) & (midband_sig == 0.0)))
    current_window = {
        "default": {**default_current, "vetoed_rows": 0},
        "midband_shadow": {**midband_current, "vetoed_rows": current_vetoed_rows},
        "delta_midband_vs_default": {
            "candidate_net_return_total": float(
                midband_current["candidate_net_return_total"] - default_current["candidate_net_return_total"]
            ),
            "mean_diff": float(midband_current["mean_diff"] - default_current["mean_diff"]),
            "candidate_trade_count": int(midband_current["candidate_trade_count"] - default_current["candidate_trade_count"]),
        },
    }

    windows = _build_windows(
        n_rows=len(default_ret),
        window_size=int(args.window_size),
        step_size=int(args.step_size),
        min_rows=int(args.min_rows),
    )

    retrospective_windows: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(windows):
        d_ret = default_ret[start:end]
        m_ret = midband_ret[start:end]
        i_ret = incumbent_ret[start:end]

        d_sig = default_sig[start:end]
        m_sig = midband_sig[start:end]
        i_sig = incumbent_sig[start:end]

        d_metrics = _window_metrics(
            candidate_ret=d_ret,
            candidate_signal=d_sig,
            incumbent_ret=i_ret,
            incumbent_signal=i_sig,
            n_boot=int(args.n_boot),
            seed=int(args.seed) + 10 + idx,
        )
        m_metrics = _window_metrics(
            candidate_ret=m_ret,
            candidate_signal=m_sig,
            incumbent_ret=i_ret,
            incumbent_signal=i_sig,
            n_boot=int(args.n_boot),
            seed=int(args.seed) + 1000 + idx,
        )
        vetoed_rows = int(np.count_nonzero((d_sig != 0.0) & (m_sig == 0.0)))

        delta_net = float(m_metrics["candidate_net_return_total"] - d_metrics["candidate_net_return_total"])
        delta_mean = float(m_metrics["mean_diff"] - d_metrics["mean_diff"])

        retrospective_windows.append(
            {
                "window_id": int(idx + 1),
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "rows": int(end - start),
                "default": {**d_metrics, "vetoed_rows": 0},
                "midband_shadow": {**m_metrics, "vetoed_rows": vetoed_rows},
                "delta_midband_vs_default": {
                    "candidate_net_return_total": delta_net,
                    "mean_diff": delta_mean,
                    "candidate_trade_count": int(m_metrics["candidate_trade_count"] - d_metrics["candidate_trade_count"]),
                },
                "flags": {
                    "improves_mean_diff": bool(np.isfinite(delta_mean) and delta_mean > 0.0),
                    "improves_candidate_net_return_total": bool(np.isfinite(delta_net) and delta_net > 0.0),
                    "clearly_harms": bool(np.isfinite(delta_net) and np.isfinite(delta_mean) and delta_net < 0.0 and delta_mean < 0.0),
                },
            }
        )

    net_deltas = np.asarray(
        [float(window["delta_midband_vs_default"]["candidate_net_return_total"]) for window in retrospective_windows],
        dtype=float,
    )
    mean_deltas = np.asarray(
        [float(window["delta_midband_vs_default"]["mean_diff"]) for window in retrospective_windows],
        dtype=float,
    )
    vetoed_rows_arr = np.asarray(
        [float(window["midband_shadow"]["vetoed_rows"]) for window in retrospective_windows],
        dtype=float,
    )

    clearly_harmed_windows = [
        {
            "window_id": int(window["window_id"]),
            "delta_candidate_net_return_total": float(window["delta_midband_vs_default"]["candidate_net_return_total"]),
            "delta_mean_diff": float(window["delta_midband_vs_default"]["mean_diff"]),
            "vetoed_rows": int(window["midband_shadow"]["vetoed_rows"]),
        }
        for window in retrospective_windows
        if bool(window["flags"]["clearly_harms"])
    ]

    aggregate_delta_net = float(np.nansum(net_deltas)) if len(net_deltas) else float("nan")
    aggregate_delta_mean = float(np.nansum(mean_deltas)) if len(mean_deltas) else float("nan")

    aggregate_summary = {
        "number_of_evaluated_windows": int(len(retrospective_windows)),
        "windows_improved_by_mean_diff": int(sum(1 for window in retrospective_windows if bool(window["flags"]["improves_mean_diff"]))),
        "windows_improved_by_candidate_net_return": int(
            sum(1 for window in retrospective_windows if bool(window["flags"]["improves_candidate_net_return_total"]))
        ),
        "clearly_harmed_windows": clearly_harmed_windows,
        "aggregate_delta_candidate_net_return_total": aggregate_delta_net,
        "aggregate_delta_mean_diff": aggregate_delta_mean,
        "median_vetoed_rows": float(np.nanmedian(vetoed_rows_arr)) if len(vetoed_rows_arr) else float("nan"),
        "mean_vetoed_rows": float(np.nanmean(vetoed_rows_arr)) if len(vetoed_rows_arr) else float("nan"),
    }

    midband_meta_vetoed_rows = None
    if args.midband_shadow_meta is not None and args.midband_shadow_meta.exists():
        try:
            meta_payload = json.loads(args.midband_shadow_meta.read_text(encoding="utf-8"))
            midband_meta = meta_payload.get("midband_candidate_only_veto", {})
            if isinstance(midband_meta, dict):
                midband_meta_vetoed_rows = int(midband_meta.get("vetoed_rows", 0))
        except Exception:
            midband_meta_vetoed_rows = None

    run_level_verdict = _run_level_verdict(
        aggregate_delta_net=aggregate_delta_net,
        aggregate_delta_mean=aggregate_delta_mean,
        clearly_harmed_windows=len(clearly_harmed_windows),
    )

    output_payload = {
        "default_candidate": str(args.default_candidate),
        "midband_shadow_candidate": str(args.midband_shadow_candidate),
        "incumbent": str(args.incumbent),
        "pairing": pairing,
        "windowing": {
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "min_rows": int(args.min_rows),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        "current_window": current_window,
        "retrospective_windows": retrospective_windows,
        "aggregate_summary": aggregate_summary,
        "shadow_meta_vetoed_rows": {
            "midband_shadow": midband_meta_vetoed_rows,
        },
        "run_level_verdict": run_level_verdict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
