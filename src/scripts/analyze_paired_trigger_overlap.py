from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired trigger-overlap buckets between policy-aligned candidate and incumbent artifacts."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--incumbent-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _coerce_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _strength_bands(values: pd.Series, edges: List[float], labels: List[str]) -> pd.Series:
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True)


def _bucket_metrics(df: pd.DataFrame, name: str) -> Dict[str, float | int]:
    cand_active = df["cand_trigger"]
    inc_active = df["inc_trigger"]
    cand_ret = df["cand_ret"]
    inc_ret = df["inc_ret"]
    diff = cand_ret - inc_ret

    # Hit rates are measured only on active rows for each side.
    cand_hits = ((cand_ret > 0.0) & cand_active).sum()
    inc_hits = ((inc_ret > 0.0) & inc_active).sum()
    cand_active_n = int(cand_active.sum())
    inc_active_n = int(inc_active.sum())

    out: Dict[str, float | int] = {
        "bucket": name,
        "row_count": int(len(df)),
        "candidate_trade_count": cand_active_n,
        "incumbent_trade_count": inc_active_n,
        "candidate_net_total": float(cand_ret.sum()),
        "incumbent_net_total": float(inc_ret.sum()),
        "net_diff_total": float(diff.sum()),
        "candidate_net_mean": float(cand_ret.mean()) if len(df) else float("nan"),
        "incumbent_net_mean": float(inc_ret.mean()) if len(df) else float("nan"),
        "net_diff_mean": float(diff.mean()) if len(df) else float("nan"),
        "candidate_hit_rate": float(cand_hits / cand_active_n) if cand_active_n else float("nan"),
        "incumbent_hit_rate": float(inc_hits / inc_active_n) if inc_active_n else float("nan"),
        "avg_candidate_p_up": float(df["cand_p_up"].mean()) if len(df) else float("nan"),
        "avg_incumbent_p_up": float(df["inc_p_up"].mean()) if len(df) else float("nan"),
        "avg_candidate_abs_ret_pred": float(df["cand_abs_ret_pred"].mean()) if len(df) else float("nan"),
        "avg_incumbent_abs_ret_pred": float(df["inc_abs_ret_pred"].mean()) if len(df) else float("nan"),
    }
    return out


def _band_breakdown(df: pd.DataFrame, *, by: str) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for value, grp in df.groupby(by, dropna=False):
        key = "nan" if pd.isna(value) else str(value)
        rows.append(
            {
                "band": key,
                "row_count": int(len(grp)),
                "candidate_net_total": float(grp["cand_ret"].sum()),
                "incumbent_net_total": float(grp["inc_ret"].sum()),
                "net_diff_total": float((grp["cand_ret"] - grp["inc_ret"]).sum()),
                "candidate_net_mean": float(grp["cand_ret"].mean()) if len(grp) else float("nan"),
                "incumbent_net_mean": float(grp["inc_ret"].mean()) if len(grp) else float("nan"),
            }
        )
    rows.sort(key=lambda x: x["row_count"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)
    if not args.incumbent.exists():
        raise FileNotFoundError(args.incumbent)

    cand = _read(args.candidate).copy()
    inc = _read(args.incumbent).copy()

    cand["_row_idx"] = np.arange(len(cand), dtype=int)
    inc["_row_idx"] = np.arange(len(inc), dtype=int)

    has_ts = ("ts" in cand.columns) and ("ts" in inc.columns)
    if has_ts:
        cand["_key"] = pd.to_datetime(cand["ts"], utc=True, errors="coerce").dt.floor("h")
        inc["_key"] = pd.to_datetime(inc["ts"], utc=True, errors="coerce").dt.floor("h")
        cand = cand.dropna(subset=["_key"]).drop_duplicates(subset=["_key"], keep="last")
        inc = inc.dropna(subset=["_key"]).drop_duplicates(subset=["_key"], keep="last")
        merged = cand.merge(inc, on="_key", how="inner", suffixes=("_cand", "_inc"))
        pairing = "timestamp_hour"
    else:
        n = int(min(len(cand), len(inc)))
        cand_tail = cand.tail(n).reset_index(drop=True)
        inc_tail = inc.tail(n).reset_index(drop=True)
        merged = cand_tail.join(inc_tail, lsuffix="_cand", rsuffix="_inc")
        pairing = "tail_index"

    if merged.empty:
        raise RuntimeError("No paired rows available for trigger-overlap diagnostics")

    sig_c = f"{args.signal_col}_cand"
    sig_i = f"{args.signal_col}_inc"
    cand_trigger = _coerce_numeric(merged, sig_c, 0.0).astype(float) != 0.0
    inc_trigger = _coerce_numeric(merged, sig_i, 0.0).astype(float) != 0.0

    cand_ret = _coerce_numeric(merged, f"{args.candidate_col}_cand", 0.0)
    inc_ret = _coerce_numeric(merged, f"{args.incumbent_col}_inc", 0.0)

    cand_p_up = _coerce_numeric(merged, "p_up_cand", np.nan)
    inc_p_up = _coerce_numeric(merged, "p_up_inc", np.nan)
    cand_ret_pred = _coerce_numeric(merged, "ret_pred_cand", np.nan)
    inc_ret_pred = _coerce_numeric(merged, "ret_pred_inc", np.nan)

    frame = pd.DataFrame(
        {
            "cand_trigger": cand_trigger,
            "inc_trigger": inc_trigger,
            "cand_ret": cand_ret,
            "inc_ret": inc_ret,
            "cand_p_up": cand_p_up,
            "inc_p_up": inc_p_up,
            "cand_abs_ret_pred": cand_ret_pred.abs(),
            "inc_abs_ret_pred": inc_ret_pred.abs(),
        }
    )

    both = frame[frame["cand_trigger"] & frame["inc_trigger"]]
    cand_only = frame[frame["cand_trigger"] & (~frame["inc_trigger"])]
    inc_only = frame[(~frame["cand_trigger"]) & frame["inc_trigger"]]
    neither = frame[(~frame["cand_trigger"]) & (~frame["inc_trigger"])]

    p_up_edges = [-np.inf, 0.45, 0.5, 0.55, 0.6, np.inf]
    p_up_labels = ["<0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", ">=0.60"]
    abs_ret_edges = [-np.inf, 0.0002, 0.0005, 0.001, np.inf]
    abs_ret_labels = ["<2bp", "2-5bp", "5-10bp", ">=10bp"]

    cand_only = cand_only.assign(
        cand_p_up_band=_strength_bands(cand_only["cand_p_up"], p_up_edges, p_up_labels),
        cand_abs_ret_pred_band=_strength_bands(cand_only["cand_abs_ret_pred"], abs_ret_edges, abs_ret_labels),
    )
    inc_only = inc_only.assign(
        inc_p_up_band=_strength_bands(inc_only["inc_p_up"], p_up_edges, p_up_labels),
        inc_abs_ret_pred_band=_strength_bands(inc_only["inc_abs_ret_pred"], abs_ret_edges, abs_ret_labels),
    )

    bucket_metrics = {
        "both_triggered": _bucket_metrics(both, "both_triggered"),
        "candidate_only": _bucket_metrics(cand_only, "candidate_only"),
        "incumbent_only": _bucket_metrics(inc_only, "incumbent_only"),
        "neither_triggered": _bucket_metrics(neither, "neither_triggered"),
    }

    totals = {
        "paired_rows": int(len(frame)),
        "candidate_trade_count": int(frame["cand_trigger"].sum()),
        "incumbent_trade_count": int(frame["inc_trigger"].sum()),
        "candidate_net_total": float(frame["cand_ret"].sum()),
        "incumbent_net_total": float(frame["inc_ret"].sum()),
        "net_diff_total": float((frame["cand_ret"] - frame["inc_ret"]).sum()),
        "net_diff_mean": float((frame["cand_ret"] - frame["inc_ret"]).mean()),
        "net_diff_std": float((frame["cand_ret"] - frame["inc_ret"]).std(ddof=1)) if len(frame) > 1 else float("nan"),
        "nonzero_paired_rows": int(np.count_nonzero(np.abs((frame["cand_ret"] - frame["inc_ret"]).to_numpy()) > 0.0)),
    }

    # Worst contributor is the bucket with most negative net contribution.
    worst_bucket = min(
        bucket_metrics.values(),
        key=lambda item: float(item.get("net_diff_total", 0.0)),
    )

    payload = {
        "candidate": str(args.candidate),
        "incumbent": str(args.incumbent),
        "candidate_col": args.candidate_col,
        "incumbent_col": args.incumbent_col,
        "signal_col": args.signal_col,
        "pairing": pairing,
        "totals": totals,
        "buckets": bucket_metrics,
        "candidate_only_breakdown": {
            "by_candidate_p_up_band": _band_breakdown(cand_only, by="cand_p_up_band"),
            "by_candidate_abs_ret_pred_band": _band_breakdown(cand_only, by="cand_abs_ret_pred_band"),
        },
        "incumbent_only_breakdown": {
            "by_incumbent_p_up_band": _band_breakdown(inc_only, by="inc_p_up_band"),
            "by_incumbent_abs_ret_pred_band": _band_breakdown(inc_only, by="inc_abs_ret_pred_band"),
        },
        "worst_bucket": worst_bucket,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
