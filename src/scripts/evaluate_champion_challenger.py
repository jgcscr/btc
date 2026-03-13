from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _load_frame(path: Path, col: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}")
    out = pd.DataFrame({"ret": pd.to_numeric(df[col], errors="coerce")})
    if "ts" in df.columns:
        out["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return out.dropna(subset=["ret"]).reset_index(drop=True)


def _resolve_paired_returns(
    candidate_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    if "ts" in candidate_df.columns and "ts" in baseline_df.columns:
        candidate_ts = candidate_df.dropna(subset=["ts"]).copy()
        baseline_ts = baseline_df.dropna(subset=["ts"]).copy()
        if not candidate_ts.empty and not baseline_ts.empty:
            merged = candidate_ts.loc[:, ["ts", "ret"]].merge(
                baseline_ts.loc[:, ["ts", "ret"]],
                on="ts",
                how="inner",
                suffixes=("_candidate", "_baseline"),
            )
            if not merged.empty:
                return (
                    merged["ret_candidate"].to_numpy(dtype=float),
                    merged["ret_baseline"].to_numpy(dtype=float),
                    "timestamp_inner_join",
                )

    n = int(min(len(candidate_df), len(baseline_df)))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), "tail_truncate"
    return (
        candidate_df["ret"].to_numpy(dtype=float)[-n:],
        baseline_df["ret"].to_numpy(dtype=float)[-n:],
        "tail_truncate",
    )


def _bootstrap_mean_diff(candidate: np.ndarray, baseline: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    n = int(min(candidate.size, baseline.size))
    if n <= 5:
        return {
            "mean_diff": float("nan"),
            "pvalue_one_sided": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_pairs": int(n),
            "nonzero_paired_rows": 0,
            "std_diff": float("nan"),
        }

    cand = candidate[-n:]
    base = baseline[-n:]
    diff = cand - base
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diff[idx]))

    mean_diff = float(np.mean(diff))
    pvalue = float(np.mean(samples <= 0.0))
    ci_low = float(np.quantile(samples, 0.025))
    ci_high = float(np.quantile(samples, 0.975))
    nonzero_paired_rows = int(np.count_nonzero(np.abs(diff) > 0.0))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    return {
        "mean_diff": mean_diff,
        "pvalue_one_sided": pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": int(n),
        "nonzero_paired_rows": int(nonzero_paired_rows),
        "std_diff": std_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Champion-challenger statistical gate using bootstrap mean-return difference.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--candidate-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/champion_challenger_gate.json"))
    args = parser.parse_args()

    baseline_df = _load_frame(args.baseline, args.baseline_col)
    candidate_df = _load_frame(args.candidate, args.candidate_col)
    candidate, baseline, alignment = _resolve_paired_returns(candidate_df, baseline_df)
    stats = _bootstrap_mean_diff(candidate, baseline, int(args.n_boot), int(args.seed))

    promote = bool(
        np.isfinite(stats["mean_diff"])
        and np.isfinite(stats["pvalue_one_sided"])
        and stats["mean_diff"] > 0.0
        and stats["pvalue_one_sided"] <= float(args.alpha)
    )

    payload: Dict[str, Any] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_col": args.baseline_col,
        "candidate_col": args.candidate_col,
        "alpha": float(args.alpha),
        "n_boot": int(args.n_boot),
        "alignment": alignment,
        "stats": stats,
        "promote": promote,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
