from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _load_returns(path: Path, col: str) -> np.ndarray:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}")
    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return vals


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

    baseline = _load_returns(args.baseline, args.baseline_col)
    candidate = _load_returns(args.candidate, args.candidate_col)
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
        "stats": stats,
        "promote": promote,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
