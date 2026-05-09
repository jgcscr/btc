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


def _pair_frames(candidate_df: pd.DataFrame, incumbent_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    has_ts = ("ts" in candidate_df.columns) and ("ts" in incumbent_df.columns)
    if has_ts:
        candidate = candidate_df.copy()
        incumbent = incumbent_df.copy()
        candidate["_ts"] = pd.to_datetime(candidate["ts"], utc=True, errors="coerce").dt.floor("h")
        incumbent["_ts"] = pd.to_datetime(incumbent["ts"], utc=True, errors="coerce").dt.floor("h")
        candidate = candidate.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        incumbent = incumbent.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        paired = candidate.merge(incumbent[["_ts", "signal_ensemble"]], on="_ts", how="inner", suffixes=("", "_inc"))
        return paired, "timestamp_hour"

    n = int(min(len(candidate_df), len(incumbent_df)))
    candidate = candidate_df.tail(n).reset_index(drop=True)
    incumbent = incumbent_df.tail(n).reset_index(drop=True)
    paired = candidate.copy()
    paired["signal_ensemble_inc"] = incumbent["signal_ensemble"].to_numpy()
    return paired, "tail_index"


def _normalize_regime(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize weak-band candidate-only performance by regime state and volatility bucket.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--return-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--p-up-low", type=float, default=0.55)
    parser.add_argument("--p-up-high", type=float, default=0.60)
    parser.add_argument("--high-inclusive", type=int, default=0)
    parser.add_argument("--use-midband-slice", type=int, default=0)
    parser.add_argument("--min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--regime-col", type=str, default="regime_state")
    parser.add_argument("--volatility-col", type=str, default="volatility_realized_24h")
    parser.add_argument("--min-regime-rows", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)
    if not args.incumbent.exists():
        raise FileNotFoundError(args.incumbent)

    candidate_df = _read_csv_or_parquet(args.candidate)
    incumbent_df = _read_csv_or_parquet(args.incumbent)
    paired, pairing = _pair_frames(candidate_df, incumbent_df)
    if paired.empty:
        output = {
            "candidate": str(args.candidate),
            "incumbent": str(args.incumbent),
            "pairing": pairing,
            "status": "no_paired_rows",
            "message": "No paired rows available for weak-band regime diagnostics",
            "scope": {
                "signal_col": str(args.signal_col),
                "incumbent_signal_col": str(args.incumbent_signal_col),
                "return_col": str(args.return_col),
                "p_up_low": float(args.p_up_low),
                "p_up_high": float(args.p_up_high),
                "high_inclusive": bool(int(args.high_inclusive)),
                "use_midband_slice": bool(int(args.use_midband_slice)),
                "min_abs_ret_pred": float(args.min_abs_ret_pred),
                "max_abs_ret_pred": float(args.max_abs_ret_pred),
                "regime_col": str(args.regime_col),
                "volatility_col": str(args.volatility_col),
                "min_regime_rows": int(args.min_regime_rows),
            },
            "candidate_only_scope": {
                "row_count": 0,
                "net_return_total": 0.0,
                "net_return_mean": float("nan"),
                "hit_rate": float("nan"),
            },
            "regime_summary": [],
            "volatility_bucket_summary": [],
            "volatility_bucket_overall_summary": [],
            "selected_harmful_regimes": [],
            "selected_high_volatility_rule": {
                "volatility_col": str(args.volatility_col),
                "comparator": ">=",
                "threshold": float("nan"),
                "threshold_source": "median_volatility_within_selected_harmful_regimes",
            },
            "selection_rule": {
                "requires_min_regime_rows": int(args.min_regime_rows),
                "requires_negative_total_return": True,
                "requires_negative_mean_return": True,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
        return

    if str(args.signal_col) not in paired.columns:
        raise KeyError(f"Missing candidate signal column: {args.signal_col}")
    if "signal_ensemble_inc" not in paired.columns:
        raise KeyError("Missing incumbent signal column after pairing")
    if str(args.return_col) not in paired.columns:
        raise KeyError(f"Missing candidate return column: {args.return_col}")

    candidate_signal = pd.to_numeric(paired[str(args.signal_col)], errors="coerce").fillna(0.0)
    incumbent_signal = pd.to_numeric(paired["signal_ensemble_inc"], errors="coerce").fillna(0.0)
    p_up = pd.to_numeric(paired.get("p_up", paired.get("p_up_meta", np.nan)), errors="coerce")
    returns = pd.to_numeric(paired[str(args.return_col)], errors="coerce").fillna(0.0)
    abs_ret_pred = pd.to_numeric(paired.get("ret_pred", np.nan), errors="coerce").abs()
    regime_values = paired.get(str(args.regime_col), pd.Series(index=paired.index, dtype=object)).map(_normalize_regime)
    volatility = pd.to_numeric(paired.get(str(args.volatility_col), np.nan), errors="coerce")

    if bool(int(args.high_inclusive)):
        in_band = (p_up >= float(args.p_up_low)) & (p_up <= float(args.p_up_high))
    else:
        in_band = (p_up >= float(args.p_up_low)) & (p_up < float(args.p_up_high))

    candidate_only = (candidate_signal != 0.0) & (incumbent_signal == 0.0) & in_band.fillna(False)
    if bool(int(args.use_midband_slice)):
        candidate_only = candidate_only & (abs_ret_pred >= float(args.min_abs_ret_pred)).fillna(False)
        candidate_only = candidate_only & (abs_ret_pred < float(args.max_abs_ret_pred)).fillna(False)

    scoped = paired.loc[candidate_only].copy()
    if not scoped.empty:
        scoped["_regime"] = regime_values.loc[scoped.index]
        scoped["_return"] = returns.loc[scoped.index]
        scoped["_volatility"] = volatility.loc[scoped.index]
    else:
        scoped = pd.DataFrame(columns=["_regime", "_return", "_volatility"])

    regime_summary: List[Dict[str, Any]] = []
    volatility_bucket_summary: List[Dict[str, Any]] = []
    volatility_bucket_overall_summary: List[Dict[str, Any]] = []
    selected_harmful_regimes: List[str] = []
    selected_high_volatility_threshold = float("nan")
    if not scoped.empty:
        grouped = (
            scoped.groupby("_regime", dropna=False)["_return"]
            .agg([("row_count", "size"), ("net_return_total", "sum"), ("net_return_mean", "mean")])
            .reset_index()
        )
        hit_rates = scoped.groupby("_regime", dropna=False)["_return"].apply(lambda s: float((s > 0.0).mean()))
        grouped["hit_rate"] = grouped["_regime"].map(hit_rates)
        grouped = grouped.sort_values(["net_return_total", "net_return_mean", "row_count"], ascending=[True, True, False])
        regime_summary = [
            {
                "regime_state": str(row["_regime"]),
                "row_count": int(row["row_count"]),
                "net_return_total": float(row["net_return_total"]),
                "net_return_mean": float(row["net_return_mean"]),
                "hit_rate": float(row["hit_rate"]),
            }
            for _, row in grouped.iterrows()
        ]
        selected_harmful_regimes = [
            str(item["regime_state"])
            for item in regime_summary
            if int(item["row_count"]) >= int(args.min_regime_rows)
            and float(item["net_return_total"]) < 0.0
            and float(item["net_return_mean"]) < 0.0
        ]

        threshold_source = scoped[scoped["_regime"].isin(selected_harmful_regimes)].copy()
        if threshold_source.empty:
            threshold_source = scoped.copy()
        valid_vol = pd.to_numeric(threshold_source["_volatility"], errors="coerce").dropna()
        if not valid_vol.empty:
            selected_high_volatility_threshold = float(valid_vol.median())

            def _bucket_vol(value: Any) -> str:
                try:
                    numeric = float(value)
                except Exception:
                    return "unknown_vol"
                if pd.isna(numeric):
                    return "unknown_vol"
                return "high_vol" if numeric >= selected_high_volatility_threshold else "low_vol"

            scoped["_volatility_bucket"] = scoped["_volatility"].map(_bucket_vol)
            vol_grouped = (
                scoped.groupby(["_regime", "_volatility_bucket"], dropna=False)["_return"]
                .agg([("row_count", "size"), ("net_return_total", "sum"), ("net_return_mean", "mean")])
                .reset_index()
            )
            vol_hit_rates = scoped.groupby(["_regime", "_volatility_bucket"], dropna=False)["_return"].apply(
                lambda s: float((s > 0.0).mean())
            )
            vol_ranges = scoped.groupby(["_regime", "_volatility_bucket"], dropna=False)["_volatility"].agg(["min", "max"])
            for _, row in vol_grouped.sort_values(["_regime", "_volatility_bucket"]).iterrows():
                key = (row["_regime"], row["_volatility_bucket"])
                volatility_bucket_summary.append(
                    {
                        "regime_state": str(row["_regime"]),
                        "volatility_bucket": str(row["_volatility_bucket"]),
                        "row_count": int(row["row_count"]),
                        "net_return_total": float(row["net_return_total"]),
                        "net_return_mean": float(row["net_return_mean"]),
                        "hit_rate": float(vol_hit_rates.loc[key]),
                        "min_volatility": float(vol_ranges.loc[key, "min"]),
                        "max_volatility": float(vol_ranges.loc[key, "max"]),
                    }
                )

            overall_grouped = (
                scoped.groupby(["_volatility_bucket"], dropna=False)["_return"]
                .agg([("row_count", "size"), ("net_return_total", "sum"), ("net_return_mean", "mean")])
                .reset_index()
            )
            overall_hit_rates = scoped.groupby(["_volatility_bucket"], dropna=False)["_return"].apply(
                lambda s: float((s > 0.0).mean())
            )
            overall_ranges = scoped.groupby(["_volatility_bucket"], dropna=False)["_volatility"].agg(["min", "max"])
            for _, row in overall_grouped.sort_values(["_volatility_bucket"]).iterrows():
                key = row["_volatility_bucket"]
                volatility_bucket_overall_summary.append(
                    {
                        "volatility_bucket": str(row["_volatility_bucket"]),
                        "row_count": int(row["row_count"]),
                        "net_return_total": float(row["net_return_total"]),
                        "net_return_mean": float(row["net_return_mean"]),
                        "hit_rate": float(overall_hit_rates.loc[key]),
                        "min_volatility": float(overall_ranges.loc[key, "min"]),
                        "max_volatility": float(overall_ranges.loc[key, "max"]),
                    }
                )

    output = {
        "candidate": str(args.candidate),
        "incumbent": str(args.incumbent),
        "pairing": pairing,
        "scope": {
            "signal_col": str(args.signal_col),
            "incumbent_signal_col": str(args.incumbent_signal_col),
            "return_col": str(args.return_col),
            "p_up_low": float(args.p_up_low),
            "p_up_high": float(args.p_up_high),
            "high_inclusive": bool(int(args.high_inclusive)),
            "use_midband_slice": bool(int(args.use_midband_slice)),
            "min_abs_ret_pred": float(args.min_abs_ret_pred),
            "max_abs_ret_pred": float(args.max_abs_ret_pred),
            "regime_col": str(args.regime_col),
            "volatility_col": str(args.volatility_col),
            "min_regime_rows": int(args.min_regime_rows),
        },
        "candidate_only_scope": {
            "row_count": int(candidate_only.sum()),
            "net_return_total": float(returns.loc[candidate_only].sum()),
            "net_return_mean": float(returns.loc[candidate_only].mean()) if int(candidate_only.sum()) > 0 else float("nan"),
            "hit_rate": float((returns.loc[candidate_only] > 0.0).mean()) if int(candidate_only.sum()) > 0 else float("nan"),
        },
        "regime_summary": regime_summary,
        "volatility_bucket_summary": volatility_bucket_summary,
        "volatility_bucket_overall_summary": volatility_bucket_overall_summary,
        "selected_harmful_regimes": selected_harmful_regimes,
        "selected_high_volatility_rule": {
            "volatility_col": str(args.volatility_col),
            "comparator": ">=",
            "threshold": selected_high_volatility_threshold,
            "threshold_source": "median_volatility_within_selected_harmful_regimes",
        },
        "selection_rule": {
            "requires_min_regime_rows": int(args.min_regime_rows),
            "requires_negative_total_return": True,
            "requires_negative_mean_return": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()