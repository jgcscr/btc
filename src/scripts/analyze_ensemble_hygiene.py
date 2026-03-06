from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _find_prob_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if lower.startswith("p_up_") and col not in {"p_up", "p_up_meta"}:
            cols.append(col)
    return sorted(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ensemble member diversity and suggest pruning.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with per-model probability columns.")
    parser.add_argument("--min-std", type=float, default=0.005)
    parser.add_argument("--max-corr", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/ensemble_hygiene.json"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    prob_cols = _find_prob_columns(df)
    if len(prob_cols) < 2:
        payload = {
            "ok": False,
            "reason": "Need at least two probability component columns.",
            "detected_columns": prob_cols,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    stats: Dict[str, Dict[str, float]] = {}
    prune_candidates: List[str] = []
    for col in prob_cols:
        std = float(pd.to_numeric(df[col], errors="coerce").std(ddof=0))
        stats[col] = {"std": std}
        if std < args.min_std:
            prune_candidates.append(col)

    corr = df[prob_cols].corr().fillna(0.0)
    high_corr_pairs = []
    for i, left in enumerate(prob_cols):
        for right in prob_cols[i + 1 :]:
            c = float(corr.loc[left, right])
            if abs(c) >= args.max_corr:
                high_corr_pairs.append({"left": left, "right": right, "corr": c})
                # prune the lower-variance member in a highly collinear pair
                keep_left = stats[left]["std"] >= stats[right]["std"]
                prune_candidates.append(right if keep_left else left)

    prune_unique = sorted(set(prune_candidates))
    payload = {
        "ok": True,
        "input": str(args.input),
        "members": prob_cols,
        "member_stats": stats,
        "high_corr_pairs": high_corr_pairs,
        "suggested_prune": prune_unique,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
