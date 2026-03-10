from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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

    diff = candidate[-n_rows:] - incumbent[-n_rows:]
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


def _extract_vetoed_rows(meta_path: Path | None) -> int | None:
    if meta_path is None or (not meta_path.exists()):
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    keys = [
        "midband_candidate_only_veto",
        "weak_band_candidate_only_veto",
        "refined_candidate_only_veto",
        "raw_ev_sign_candidate_only_veto",
        "direction_align_candidate_only_veto",
        "joint_direction_midband_candidate_only_veto",
        "regime_state_candidate_only_veto",
        "chop_high_vol_candidate_only_veto",
        "volatility_only_candidate_only_veto",
    ]
    for key in keys:
        obj = payload.get(key)
        if isinstance(obj, dict) and ("vetoed_rows" in obj):
            try:
                return int(obj.get("vetoed_rows", 0) or 0)
            except Exception:
                return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize profile candidate vs incumbent metrics for one run.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--incumbent-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--candidate-meta", type=Path, default=None)
    parser.add_argument("--profile-id", type=str, required=True)
    parser.add_argument("--profile-name", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)
    if not args.incumbent.exists():
        raise FileNotFoundError(args.incumbent)

    candidate_df = _read_csv_or_parquet(args.candidate)
    incumbent_df = _read_csv_or_parquet(args.incumbent)

    candidate_ret = pd.to_numeric(candidate_df.get(args.candidate_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    incumbent_ret = pd.to_numeric(incumbent_df.get(args.incumbent_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    candidate_sig = pd.to_numeric(candidate_df.get(args.signal_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    incumbent_sig = pd.to_numeric(incumbent_df.get(args.signal_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n_rows = int(min(candidate_ret.size, incumbent_ret.size, candidate_sig.size, incumbent_sig.size))
    candidate_ret = candidate_ret[-n_rows:]
    incumbent_ret = incumbent_ret[-n_rows:]
    candidate_sig = candidate_sig[-n_rows:]
    incumbent_sig = incumbent_sig[-n_rows:]

    stats = _bootstrap_stats(candidate_ret, incumbent_ret, n_boot=int(args.n_boot), seed=int(args.seed))
    payload: Dict[str, Any] = {
        "profile_id": str(args.profile_id),
        "profile_name": str(args.profile_name or args.profile_id),
        "run_id": str(args.run_id),
        "candidate": str(args.candidate),
        "incumbent": str(args.incumbent),
        "metrics": {
            "candidate_trade_count": int(np.count_nonzero(candidate_sig != 0.0)),
            "incumbent_trade_count": int(np.count_nonzero(incumbent_sig != 0.0)),
            "candidate_net_return_total": float(np.sum(candidate_ret)),
            "incumbent_net_return_total": float(np.sum(incumbent_ret)),
            "mean_diff": float(stats["mean_diff"]),
            "pvalue_one_sided": float(stats["pvalue_one_sided"]),
            "nonzero_paired_rows": int(stats["nonzero_paired_rows"]),
            "std_diff": float(stats["std_diff"]),
            "vetoed_row_count": _extract_vetoed_rows(args.candidate_meta),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
