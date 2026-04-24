from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.component_diversity_support import component_feature_column_names


FEATURE_COLUMNS: List[str] = [
    "ret_pred",
    "signal_dir_only",
    "expected_value",
    "confidence_score",
    "position_size",
    "regime_state",
    "volatility_realized_24h",
    "volatility_ewm_24h",
    "volatility_garch_like",
    "range_expansion_1h",
    "distance_from_session_high_8h",
    "distance_from_session_low_8h",
    "vwap_deviation_8h",
    "momentum_slope_2h",
    "momentum_slope_4h",
    "confluence_support_ratio",
    "confluence_short_term_ratio",
    "confluence_mid_term_ratio",
    "confluence_direction_matches_dominant",
    "horizon_consensus_support_ratio",
    "horizon_directional_agreement_ratio",
    "horizon_directional_disagreement_count",
    "horizon_short_term_alignment_ratio",
    "horizon_mid_term_alignment_ratio",
    "horizon_weighted_p_up",
    "horizon_weighted_ret_pred",
    "horizon_p_up_dispersion",
    "horizon_bias_conflict",
    "incumbent_signal_reference",
    "candidate_only_reference",
    "candidate_incumbent_disagreement",
    *component_feature_column_names(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a backtest-style CSV with live-equivalent decision features by timestamp from feature sources."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument(
        "--feature-source",
        action="append",
        default=[],
        help="CSV path used to backfill decision features by hour-aligned timestamp.",
    )
    parser.add_argument(
        "--auto-discover-sources",
        action="store_true",
        help=(
            "Also scan artifacts/reliability/*/summary/backtest_signals_meta_ensemble_enriched.csv "
            "and pick the highest-overlap source with non-null decision features."
        ),
    )
    parser.add_argument(
        "--incumbent-reference-source",
        type=Path,
        default=None,
        help="Optional incumbent backtest CSV used to derive candidate-vs-incumbent disagreement features.",
    )
    return parser.parse_args()


def _missing_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for col in cols:
        if col not in df.columns:
            out[col] = int(len(df))
            continue
        if col == "regime_state":
            out[col] = int(df[col].isna().sum())
        else:
            out[col] = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
    return out


def _rank_discovered_sources(base: pd.DataFrame) -> List[Path]:
    base_ts = pd.to_datetime(base["ts"], utc=True, errors="coerce").dt.floor("h")
    base_ts_set = set(base_ts.dropna().tolist())
    if not base_ts_set:
        return []

    candidates = sorted(
        Path("artifacts/reliability").glob("*/summary/backtest_signals_meta_ensemble_enriched.csv"),
        reverse=True,
    )
    ranked: List[tuple[int, int, Path]] = []
    for path in candidates:
        try:
            src_df = pd.read_csv(path)
            if "ts" not in src_df.columns:
                continue
            src_ts = pd.to_datetime(src_df["ts"], utc=True, errors="coerce").dt.floor("h")
            src_df = src_df.assign(_ts_norm=src_ts).dropna(subset=["_ts_norm"]).drop_duplicates("_ts_norm", keep="last")
            overlap = int(src_df["_ts_norm"].isin(base_ts_set).sum())
            if overlap <= 0:
                continue
            non_null = 0
            for col in FEATURE_COLUMNS:
                if col in src_df.columns:
                    non_null += int(src_df[col].notna().sum())
            ranked.append((overlap, non_null, path))
        except Exception:
            continue
    ranked.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    return [item[2] for item in ranked]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    base = pd.read_csv(args.input)
    if "ts" not in base.columns:
        raise ValueError("Input must include a ts column for feature alignment")

    out = base.copy()
    out["_ts_norm"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.floor("h")

    missing_before = _missing_counts(out, FEATURE_COLUMNS)
    backfill_by_col = {col: 0 for col in FEATURE_COLUMNS}
    backfill_by_source: Dict[str, Dict[str, int]] = {}

    sources: List[Path] = []
    for src_path in args.feature_source:
        sources.append(Path(str(src_path)))
    if args.auto_discover_sources:
        for discovered in _rank_discovered_sources(out):
            if discovered not in sources:
                sources.append(discovered)

    for src in sources:
        if not src.exists():
            continue
        src_df = pd.read_csv(src)
        if "ts" not in src_df.columns:
            continue
        src_df = src_df.copy()
        src_df["_ts_norm"] = pd.to_datetime(src_df["ts"], utc=True, errors="coerce").dt.floor("h")
        available = [c for c in FEATURE_COLUMNS if c in src_df.columns]
        if not available:
            continue
        src_df = src_df.loc[:, ["_ts_norm", *available]].dropna(subset=["_ts_norm"]).drop_duplicates(
            subset=["_ts_norm"], keep="last"
        )
        per_source = {col: 0 for col in FEATURE_COLUMNS}
        indexed = src_df.set_index("_ts_norm")
        for col in available:
            mapped = out["_ts_norm"].map(indexed[col])
            if col not in out.columns:
                filled = mapped.notna()
                out[col] = mapped
            else:
                current = out[col]
                filled = current.isna() & mapped.notna()
                out[col] = current.where(current.notna(), mapped)
            n = int(filled.sum())
            backfill_by_col[col] += n
            per_source[col] += n
        backfill_by_source[str(src)] = per_source

    incumbent_reference_summary = {
        "source": None,
        "rows_with_reference": 0,
        "candidate_only_rows": 0,
        "disagreement_rows": 0,
    }
    if args.incumbent_reference_source is not None and args.incumbent_reference_source.exists():
        incumbent_src = pd.read_csv(args.incumbent_reference_source)
        if "ts" in incumbent_src.columns and "signal_ensemble" in incumbent_src.columns:
            incumbent_src = incumbent_src.copy()
            incumbent_src["_ts_norm"] = pd.to_datetime(incumbent_src["ts"], utc=True, errors="coerce").dt.floor("h")
            incumbent_src = incumbent_src.loc[:, ["_ts_norm", "signal_ensemble"]].dropna(subset=["_ts_norm"]).drop_duplicates(
                subset=["_ts_norm"], keep="last"
            )
            incumbent_map = pd.to_numeric(
                incumbent_src.set_index("_ts_norm")["signal_ensemble"],
                errors="coerce",
            ).fillna(0.0)
            incumbent_signal = out["_ts_norm"].map(incumbent_map).fillna(0.0)
            candidate_signal = pd.to_numeric(out.get("signal_ensemble", 0.0), errors="coerce").fillna(0.0)
            out["incumbent_signal_reference"] = incumbent_signal.astype(float)
            out["candidate_only_reference"] = ((candidate_signal > 0.0) & (incumbent_signal <= 0.0)).astype(float)
            out["candidate_incumbent_disagreement"] = ((candidate_signal > 0.0) != (incumbent_signal > 0.0)).astype(float)
            incumbent_reference_summary = {
                "source": str(args.incumbent_reference_source),
                "rows_with_reference": int((incumbent_signal.notna()).sum()),
                "candidate_only_rows": int(out["candidate_only_reference"].sum()),
                "disagreement_rows": int(out["candidate_incumbent_disagreement"].sum()),
            }

    missing_after = _missing_counts(out, FEATURE_COLUMNS)
    out = out.drop(columns=["_ts_norm"], errors="ignore")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(out)),
        "feature_columns": FEATURE_COLUMNS,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "backfill_by_column": backfill_by_col,
        "backfill_by_source": backfill_by_source,
        "incumbent_reference": incumbent_reference_summary,
    }
    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
