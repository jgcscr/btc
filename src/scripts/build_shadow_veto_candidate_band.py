from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostics-only shadow candidate by vetoing candidate-only trades in a weak p_up band."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--ret-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--p-up-col", type=str, default="p_up")
    parser.add_argument("--band-low", type=float, default=0.55)
    parser.add_argument("--band-high", type=float, default=0.60)
    parser.add_argument("--high-inclusive", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _bool_signal(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float) != 0.0


def _pair_on_ts(candidate: pd.DataFrame, incumbent: pd.DataFrame) -> pd.DataFrame:
    cand = candidate.copy()
    inc = incumbent.copy()
    if "ts" in cand.columns and "ts" in inc.columns:
        cand["_k"] = pd.to_datetime(cand["ts"], utc=True, errors="coerce").dt.floor("h")
        inc["_k"] = pd.to_datetime(inc["ts"], utc=True, errors="coerce").dt.floor("h")
        cand = cand.dropna(subset=["_k"]).drop_duplicates(subset=["_k"], keep="last")
        inc = inc.dropna(subset=["_k"]).drop_duplicates(subset=["_k"], keep="last")
        return cand.merge(inc[["_k", "_inc_trigger"]], on="_k", how="left")

    n = int(min(len(cand), len(inc)))
    cand_tail = cand.tail(n).copy().reset_index(drop=True)
    inc_tail = inc.tail(n).copy().reset_index(drop=True)
    cand_tail["_inc_trigger"] = inc_tail["_inc_trigger"].to_numpy()
    return cand_tail


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)
    if not args.incumbent.exists():
        raise FileNotFoundError(args.incumbent)

    cand = _read(args.candidate)
    inc = _read(args.incumbent)

    cand["_cand_trigger"] = _bool_signal(cand, args.signal_col)
    inc["_inc_trigger"] = _bool_signal(inc, args.signal_col)

    paired = _pair_on_ts(cand, inc)
    if "_inc_trigger" not in paired.columns:
        paired["_inc_trigger"] = False
    paired["_inc_trigger"] = paired["_inc_trigger"].fillna(False).astype(bool)

    p_up = pd.to_numeric(paired.get(args.p_up_col, np.nan), errors="coerce")
    if args.high_inclusive:
        in_band = (p_up >= float(args.band_low)) & (p_up <= float(args.band_high))
    else:
        in_band = (p_up >= float(args.band_low)) & (p_up < float(args.band_high))

    veto_mask = paired["_cand_trigger"] & (~paired["_inc_trigger"]) & in_band.fillna(False)

    out = paired.copy()
    if args.signal_col not in out.columns:
        out[args.signal_col] = 0
    out.loc[veto_mask, args.signal_col] = 0

    if args.ret_col in out.columns:
        out.loc[veto_mask, args.ret_col] = 0.0

    if "trade_action" in out.columns:
        out.loc[veto_mask, "trade_action"] = "hold"

    out = out.drop(columns=[c for c in ["_cand_trigger", "_inc_trigger", "_k"] if c in out.columns])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    out_signal = pd.to_numeric(out.get(args.signal_col, 0), errors="coerce").fillna(0.0).astype(float) != 0.0
    out_ret = pd.to_numeric(out.get(args.ret_col, 0), errors="coerce").fillna(0.0)

    meta = {
        "candidate": str(args.candidate),
        "incumbent": str(args.incumbent),
        "output": str(args.output),
        "signal_col": args.signal_col,
        "ret_col": args.ret_col,
        "p_up_col": args.p_up_col,
        "band_low": float(args.band_low),
        "band_high": float(args.band_high),
        "high_inclusive": bool(args.high_inclusive),
        "rows": int(len(out)),
        "vetoed_rows": int(veto_mask.sum()),
        "trade_count_after": int(out_signal.sum()),
        "net_return_total_after": float(out_ret.sum()),
    }

    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
