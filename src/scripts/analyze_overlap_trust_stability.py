from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _normalize_ts(values: Iterable[Any]) -> pd.Series:
    return pd.to_datetime(pd.Series(list(values)), utc=True, errors="coerce").dt.floor("h")


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(["", "nan", "none", "null"])


def _enrich_from_sources(
    base: pd.DataFrame,
    ts_col: str,
    feature_sources: List[Path],
    enrich_cols: List[str],
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if base.empty or ts_col not in base.columns or not feature_sources:
        return base, []

    out = base.copy()
    out["_ts_norm"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce").dt.floor("h")
    source_stats: List[Dict[str, Any]] = []
    for src_path in feature_sources:
        if not src_path.exists():
            continue
        src_df = pd.read_csv(src_path)
        if ts_col not in src_df.columns:
            continue
        available_cols = [col for col in enrich_cols if col in src_df.columns]
        if not available_cols:
            continue
        src_df = src_df.copy()
        src_df["_ts_norm"] = pd.to_datetime(src_df[ts_col], utc=True, errors="coerce").dt.floor("h")
        src_df = src_df.loc[:, ["_ts_norm", *available_cols]].dropna(subset=["_ts_norm"])
        src_df = src_df.drop_duplicates(subset=["_ts_norm"], keep="last")
        mapped = src_df.set_index("_ts_norm")
        filled_by_column: Dict[str, int] = {}
        for col in available_cols:
            fill_values = out["_ts_norm"].map(mapped[col])
            if col not in out.columns:
                out[col] = fill_values
                filled = fill_values.notna()
            else:
                missing = _missing_mask(out[col])
                filled = missing & fill_values.notna()
                out[col] = out[col].where(~missing, fill_values)
            filled_by_column[col] = int(filled.sum())
        source_stats.append(
            {
                "path": str(src_path),
                "rows": int(len(src_df)),
                "filled_by_column": filled_by_column,
            }
        )
    return out.drop(columns=["_ts_norm"], errors="ignore"), source_stats


def _load_overlap_timestamps(npz_path: Path) -> pd.DatetimeIndex:
    with np.load(npz_path, allow_pickle=True) as data:
        rows: List[pd.Series] = []
        for key in ("ts_all", "ts_train", "ts_val", "ts_test"):
            if key in data.files:
                rows.append(_normalize_ts(data[key].reshape(-1).tolist()))
        if not rows:
            raise KeyError(f"No timestamp arrays found in {npz_path}")
        combined = pd.concat(rows, ignore_index=True).dropna().drop_duplicates().sort_values()
        return pd.DatetimeIndex(combined)


def _read_walkforward(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    folds_obj = payload.get("folds", [])
    folds = [row for row in folds_obj if isinstance(row, dict)] if isinstance(folds_obj, list) else []
    detailed_output = payload.get("detailed_output")
    return {
        "path": str(path),
        "model_kind": payload.get("model_kind"),
        "cum_ret_net_total": float(payload.get("cum_ret_net_total", 0.0) or 0.0),
        "trade_count_total": int(payload.get("trade_count_total", 0) or 0),
        "auc_mean": float(payload.get("auc_mean", float("nan"))),
        "folds": folds,
        "detailed_output": str(detailed_output) if detailed_output else None,
    }


def _fold_summary(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    negative_sum = 0.0
    for row in folds:
        cum_ret = float(row.get("cum_ret_net", 0.0) or 0.0)
        if cum_ret < 0.0:
            negative_sum += abs(cum_ret)
        out.append(
            {
                "fold": int(row.get("fold", 0) or 0),
                "n_test": int(row.get("n_test", 0) or 0),
                "trade_count": int(row.get("trade_count", 0) or 0),
                "hit_rate": float(row.get("hit_rate", float("nan"))),
                "cum_ret_net": cum_ret,
                "auc": float(row.get("auc", float("nan"))),
                "is_negative": bool(cum_ret < 0.0),
            }
        )
    out.sort(key=lambda item: item["fold"])
    worst = min(out, key=lambda item: item["cum_ret_net"]) if out else None
    worst_share = None
    if worst is not None and negative_sum > 0.0 and float(worst["cum_ret_net"]) < 0.0:
        worst_share = abs(float(worst["cum_ret_net"])) / negative_sum
    return {
        "folds": out,
        "negative_fold_count": int(sum(1 for row in out if row["is_negative"])),
        "worst_fold": worst,
        "negative_loss_concentration": {
            "negative_return_abs_total": float(negative_sum),
            "worst_fold_share": worst_share,
        },
    }


def _bucketize(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(numeric >= float(threshold), "high_vol", "low_vol"), index=values.index)


def _resolve_column(df: pd.DataFrame, requested: str, aliases: List[str]) -> str:
    candidates = [requested, *aliases]
    seen: set[str] = set()
    scored: List[tuple[int, str]] = []
    for name in candidates:
        column = str(name).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        if column not in df.columns:
            continue
        present = int((~_missing_mask(df[column])).sum())
        scored.append((present, column))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1] == requested), reverse=True)
        return scored[0][1]
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def _group_summary(df: pd.DataFrame, group_col: str, return_col: str, signal_col: str) -> List[Dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return []
    try:
        resolved_return_col = _resolve_column(df, return_col, ["ret_net", "ret_ensemble_net"])
    except KeyError:
        resolved_return_col = _resolve_column(df, return_col, ["ret_realized"])
    resolved_signal_col = _resolve_column(df, signal_col, ["signal", "signal_ensemble", "signal_meta"])
    ret = pd.to_numeric(df[resolved_return_col], errors="coerce").fillna(0.0)
    signal = pd.to_numeric(df[resolved_signal_col], errors="coerce").fillna(0.0)
    working = df.copy()
    working["_ret"] = ret
    working["_signal"] = signal
    grouped = working.groupby(group_col, dropna=False)
    rows: List[Dict[str, Any]] = []
    for name, grp in grouped:
        active = grp["_signal"] != 0.0
        rows.append(
            {
                group_col: "unknown" if pd.isna(name) else str(name),
                "row_count": int(len(grp)),
                "trade_count": int(active.sum()),
                "abstention_count": int((~active).sum()),
                "net_return_total": float(grp["_ret"].sum()),
                "net_return_mean": float(grp["_ret"].mean()) if len(grp) else float("nan"),
                "hit_rate": float((grp.loc[active, "_ret"] > 0.0).mean()) if int(active.sum()) else float("nan"),
            }
        )
    rows.sort(key=lambda item: (item["net_return_total"], -item["row_count"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze overlap-slice instability behind trust failure.")
    parser.add_argument("--full-walkforward", type=Path, required=True)
    parser.add_argument("--overlap-walkforward", type=Path, required=True)
    parser.add_argument("--overlap-dataset", type=Path, required=True)
    parser.add_argument("--labeled-csv", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path, action="append", default=[])
    parser.add_argument("--ts-col", type=str, default="ts")
    parser.add_argument("--return-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--regime-col", type=str, default="regime_state")
    parser.add_argument("--volatility-col", type=str, default="volatility_realized_24h")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.full_walkforward, args.overlap_walkforward, args.overlap_dataset, args.labeled_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    full = _read_walkforward(args.full_walkforward)
    overlap = _read_walkforward(args.overlap_walkforward)
    overlap_ts = _load_overlap_timestamps(args.overlap_dataset)

    auto_feature_sources: List[Path] = []
    for item in (full, overlap):
        detailed_output = item.get("detailed_output")
        if not detailed_output:
            continue
        detailed_path = Path(str(detailed_output))
        if detailed_path.exists() and detailed_path not in auto_feature_sources:
            auto_feature_sources.append(detailed_path)
    feature_sources = auto_feature_sources + [path for path in args.feature_source if path not in auto_feature_sources]

    labeled = pd.read_csv(args.labeled_csv)
    if args.ts_col not in labeled.columns:
        raise KeyError(f"Missing timestamp column '{args.ts_col}' in {args.labeled_csv}")
    labeled = labeled.copy()
    enrich_cols = list(
        dict.fromkeys(
            [
                args.return_col,
                "ret_net",
                "ret_realized",
                args.signal_col,
                "signal",
                "signal_meta",
                args.regime_col,
                args.volatility_col,
                "volatility_ewm_24h",
                "volatility_garch_like",
                "ret_pred",
                "expected_value",
            ]
        )
    )
    labeled, source_stats = _enrich_from_sources(labeled, args.ts_col, feature_sources, enrich_cols)
    labeled["_ts_norm"] = pd.to_datetime(labeled[args.ts_col], utc=True, errors="coerce").dt.floor("h")
    overlap_df = labeled.loc[labeled["_ts_norm"].isin(set(overlap_ts))].copy()

    volatility_threshold = None
    if args.volatility_col in overlap_df.columns:
        numeric_vol = pd.to_numeric(overlap_df[args.volatility_col], errors="coerce")
        if numeric_vol.notna().any():
            volatility_threshold = float(numeric_vol.median())
            overlap_df["volatility_bucket"] = _bucketize(numeric_vol, volatility_threshold)

    regime_summary = _group_summary(overlap_df, args.regime_col, args.return_col, args.signal_col)
    vol_summary = _group_summary(overlap_df, "volatility_bucket", args.return_col, args.signal_col) if volatility_threshold is not None else []

    dominant_regime_loss = min(regime_summary, key=lambda item: item["net_return_total"]) if regime_summary else None
    dominant_vol_loss = min(vol_summary, key=lambda item: item["net_return_total"]) if vol_summary else None

    payload = {
        "full": {
            "model_kind": full["model_kind"],
            "cum_ret_net_total": full["cum_ret_net_total"],
            "trade_count_total": full["trade_count_total"],
            "auc_mean": full["auc_mean"],
            "fold_summary": _fold_summary(full["folds"]),
        },
        "overlap": {
            "model_kind": overlap["model_kind"],
            "cum_ret_net_total": overlap["cum_ret_net_total"],
            "trade_count_total": overlap["trade_count_total"],
            "auc_mean": overlap["auc_mean"],
            "fold_summary": _fold_summary(overlap["folds"]),
            "rows_overlap": int(len(overlap_df)),
        },
        "overlap_slice_characteristics": {
            "labeled_csv": str(args.labeled_csv),
            "feature_sources": [str(path) for path in feature_sources],
            "feature_source_enrichment": source_stats,
            "overlap_dataset": str(args.overlap_dataset),
            "volatility_threshold_median": volatility_threshold,
            "regime_summary": regime_summary,
            "volatility_summary": vol_summary,
            "dominant_regime_loss_bucket": dominant_regime_loss,
            "dominant_volatility_loss_bucket": dominant_vol_loss,
        },
        "diagnosis": {
            "losses_concentrated_in_negative_overlap_folds": bool(
                overlap["cum_ret_net_total"] < 0.0 and _fold_summary(overlap["folds"])["negative_fold_count"] > 0
            ),
            "overlap_worse_than_full": bool(overlap["cum_ret_net_total"] < full["cum_ret_net_total"]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()