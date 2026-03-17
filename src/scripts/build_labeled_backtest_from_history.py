from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _format_horizon_label(horizon: float) -> str:
    if float(horizon).is_integer() and horizon >= 1.0:
        return f"{int(round(horizon))}h"
    if horizon < 1.0:
        return f"{int(round(horizon * 60))}m"
    return f"{horizon:g}h"


def _parse_horizon_hours(horizon: str) -> float:
    value = str(horizon).strip().lower()
    if value.endswith("h"):
        return float(value[:-1])
    if value.endswith("m"):
        return float(value[:-1]) / 60.0
    return float(value)


def _realized_column_names(horizon: str) -> Tuple[str, str]:
    label = _format_horizon_label(_parse_horizon_hours(horizon))
    return (f"close_next_{label}", f"ret_{label}_realized")


def _resolve_spot_ohlcv_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Spot OHLCV path not found: {path}")
    candidates = sorted(path.glob("*.parquet"))
    if not candidates:
        candidates = sorted(path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return candidates


def _load_prediction_entries(path: Path, *, include_reliability_snapshots: bool = False) -> List[Dict[str, object]]:
    sources: List[Path] = []
    if include_reliability_snapshots:
        sources.extend(sorted(Path("artifacts/reliability").glob("*/summary/live_predictions_snapshot.json")))
    sources.append(path)

    entries: List[Dict[str, object]] = []
    for source in sources:
        if not source.exists():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            entries.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            entries.append(payload)
    if not entries:
        raise RuntimeError(f"No prediction entries found from configured sources rooted at {path}")
    return entries


def _load_history_rows(
    path: Path,
    horizon: str,
    *,
    include_reliability_snapshots: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"History not found: {path}")
    payload = _load_prediction_entries(path, include_reliability_snapshots=include_reliability_snapshots)

    rows: List[Dict[str, object]] = []
    for entry in payload:
        predictions = entry.get("predictions") if isinstance(entry, dict) else {}
        if not isinstance(predictions, dict) and isinstance(entry, dict):
            predictions = entry.get("horizons")
        if not isinstance(predictions, dict) and isinstance(entry, dict):
            predictions = entry
        horizon_pred = predictions.get(horizon, {}) if isinstance(predictions, dict) else {}
        if not isinstance(horizon_pred, dict):
            continue
        ts_value = horizon_pred.get("timestamp")
        p_up = horizon_pred.get("p_up")
        if ts_value is None or p_up is None:
            continue

        row: Dict[str, object] = {
            "ts": ts_value,
            "generated_at": entry.get("generated_at") if isinstance(entry, dict) else None,
            "horizon": _format_horizon_label(_parse_horizon_hours(horizon)),
            "horizon_hours": _parse_horizon_hours(horizon),
            "p_up": p_up,
            "ret_pred": horizon_pred.get("ret_pred"),
            "signal_dir_only": horizon_pred.get("signal_dir_only"),
            "expected_value": horizon_pred.get("expected_value"),
            "regime_state": horizon_pred.get("regime_state"),
        }
        volatility = horizon_pred.get("volatility", {})
        if isinstance(volatility, dict):
            snapshot = volatility.get("snapshot", {})
            if isinstance(snapshot, dict):
                row["volatility_realized_24h"] = snapshot.get("volatility_realized_24h")
                row["volatility_ewm_24h"] = snapshot.get("volatility_ewm_24h")
                row["volatility_garch_like"] = snapshot.get("volatility_garch_like")
        components = horizon_pred.get("p_up_components", {})
        if isinstance(components, dict):
            for name, value in components.items():
                row[f"p_up_{name}"] = value
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No prediction rows found for horizon '{horizon}'")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["generated_at"] = pd.to_datetime(df["generated_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["ts_hour"] = df["ts"].dt.floor("h")
    df = df.sort_values(["ts", "generated_at"]).drop_duplicates(subset=["ts_hour"], keep="last")
    return df.reset_index(drop=True)


def _load_ohlcv(path: Path, horizons: Optional[List[str]] = None) -> pd.DataFrame:
    source_files = _resolve_spot_ohlcv_files(path)
    frames: List[pd.DataFrame] = []
    for source_file in source_files:
        if source_file.suffix.lower() == ".csv":
            frames.append(pd.read_csv(source_file))
        else:
            frames.append(pd.read_parquet(source_file))
    df = pd.concat(frames, ignore_index=True)
    if "ts" not in df.columns or "close" not in df.columns:
        raise ValueError("OHLCV input must include 'ts' and 'close' columns")
    out = df[["ts", "close"]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["ts", "close"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    out["ts_hour"] = out["ts"].dt.floor("h")

    target_horizons = horizons or ["1h"]
    for horizon in target_horizons:
        horizon_steps = int(round(_parse_horizon_hours(horizon)))
        if horizon_steps <= 0:
            raise ValueError(f"Unsupported horizon for OHLCV labeling: {horizon}")
        close_next_col, ret_col = _realized_column_names(horizon)
        out[close_next_col] = out["close"].shift(-horizon_steps)
        out[ret_col] = np.log(out[close_next_col] / out["close"])

    default_close_next_col, default_ret_col = _realized_column_names("1h")
    out["close_next_1h"] = out[default_close_next_col]
    out["ret_1h_realized"] = out[default_ret_col]
    out["y_true"] = (out["ret_1h_realized"] > 0).astype(int)
    return out.reset_index(drop=True)


def _find_latest_backtest_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if path.exists() and path.is_dir():
        direct = path / "backtest_signals.csv"
        if direct.exists():
            return direct

    candidates = sorted(Path("artifacts/backtests").glob("**/backtest_signals.csv"))
    if not candidates:
        raise FileNotFoundError("No backtest_signals.csv files found under artifacts/backtests")
    return candidates[-1]


def _estimate_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        return max(rows - 1, 0)
    except OSError:
        return 0


def _select_best_backtest_candidate(preferred: Path, min_rows_hint: int) -> Path:
    candidates = list(Path("artifacts/backtests").glob("**/backtest_signals.csv"))
    if preferred.exists() and preferred.is_file():
        candidates.append(preferred)
    if preferred.exists() and preferred.is_dir() and (preferred / "backtest_signals.csv").exists():
        candidates.append(preferred / "backtest_signals.csv")

    if not candidates:
        raise FileNotFoundError("No backtest_signals.csv files available for canonical labeling.")

    unique = {c.resolve(): c for c in candidates}.values()
    ranked = sorted(
        ((int(_estimate_csv_rows(p)), p) for p in unique),
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )
    for rows, path in ranked:
        if rows >= min_rows_hint:
            return path
    return ranked[0][1]


def _load_backtest_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"ts", "p_up"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest CSV is missing required columns: {missing}")

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    if "ret_1h" in out.columns and "y_true" not in out.columns:
        ret = pd.to_numeric(out["ret_1h"], errors="coerce")
        out["y_true"] = (ret > 0).astype(float)

    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    return out.reset_index(drop=True)


def _enrich_with_history_decision_features(
    base: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    if base.empty or history_df.empty:
        return base

    enrich_cols = [
        "ret_pred",
        "signal_dir_only",
        "expected_value",
        "regime_state",
        "volatility_realized_24h",
        "volatility_ewm_24h",
        "volatility_garch_like",
    ]
    available_cols = [c for c in enrich_cols if c in history_df.columns]
    if not available_cols:
        return base

    out = base.copy()
    out["ts_hour"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.floor("h")
    hist = history_df.copy()
    hist["ts_hour"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce").dt.floor("h")
    hist = hist.dropna(subset=["ts_hour"]).drop_duplicates(subset=["ts_hour"], keep="last")
    mapped = hist.loc[:, ["ts_hour", *available_cols]]
    merged = out.merge(mapped, on="ts_hour", how="left", suffixes=("", "_hist"))
    for col in available_cols:
        hist_col = f"{col}_hist"
        if hist_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[hist_col]
        else:
            merged[col] = merged[col].where(merged[col].notna(), merged[hist_col])
        merged = merged.drop(columns=[hist_col])
    return merged.drop(columns=["ts_hour"], errors="ignore")


def _apply_time_filters(
    df: pd.DataFrame,
    lookback_rows: Optional[int],
    lookback_hours: Optional[int],
) -> pd.DataFrame:
    out = df.sort_values("ts").copy()
    if lookback_hours is not None and lookback_hours > 0 and not out.empty:
        cutoff = out["ts"].max() - pd.Timedelta(hours=int(lookback_hours))
        out = out.loc[out["ts"] >= cutoff].copy()
    if lookback_rows is not None and lookback_rows > 0 and len(out) > lookback_rows:
        out = out.tail(int(lookback_rows)).copy()
    return out.reset_index(drop=True)


def _assign_fold(df: pd.DataFrame, fold_size: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = (np.arange(len(out)) // fold_size).astype(int)
    return out


def _build_from_backtest(
    backtest_csv: Path,
    history_path: Path,
    horizon: str,
    fold_size: int,
    lookback_rows: Optional[int],
    lookback_hours: Optional[int],
    include_reliability_snapshots: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    backtest = _load_backtest_rows(backtest_csv)
    backtest = _apply_time_filters(backtest, lookback_rows=lookback_rows, lookback_hours=lookback_hours)
    if "y_true" not in backtest.columns:
        raise ValueError(
            "Backtest CSV must include y_true or ret_1h to derive labels."
        )
    labeled = backtest.dropna(subset=["y_true"]).copy()
    try:
        history_df = _load_history_rows(
            history_path,
            horizon,
            include_reliability_snapshots=include_reliability_snapshots,
        )
        labeled = _enrich_with_history_decision_features(labeled, history_df)
    except Exception:
        # Keep canonical backtest behavior even when history enrichment is unavailable.
        pass
    labeled["y_true"] = labeled["y_true"].astype(int)
    labeled["horizon"] = "1h"
    labeled["horizon_hours"] = 1.0
    labeled = _assign_fold(labeled, fold_size)
    meta = {
        "source": "backtest_csv",
        "source_path": str(backtest_csv),
        "history_rows": int(len(backtest)),
        "labeled_rows": int(len(labeled)),
    }
    return labeled, meta


def _build_from_history(
    history_path: Path,
    horizon: str,
    spot_ohlcv_path: Path,
    fold_size: int,
    lookback_rows: Optional[int],
    lookback_hours: Optional[int],
    include_reliability_snapshots: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    history_df = _load_history_rows(
        history_path,
        horizon,
        include_reliability_snapshots=include_reliability_snapshots,
    )
    history_df = _apply_time_filters(history_df, lookback_rows=lookback_rows, lookback_hours=lookback_hours)
    ohlcv_df = _load_ohlcv(spot_ohlcv_path, horizons=[horizon])

    close_next_col, ret_col = _realized_column_names(horizon)

    merged = history_df.merge(
        ohlcv_df[["ts_hour", "close", close_next_col, ret_col]],
        on="ts_hour",
        how="left",
    )
    merged = merged.rename(
        columns={
            close_next_col: "close_target",
            ret_col: "ret_realized",
        }
    )
    merged["y_true"] = (pd.to_numeric(merged["ret_realized"], errors="coerce") > 0).astype(float)
    merged = merged.dropna(subset=["y_true", "ret_realized"]).copy()
    merged["y_true"] = merged["y_true"].astype(int)
    merged = _assign_fold(merged, fold_size)
    meta = {
        "source": "history_plus_ohlcv",
        "source_path": str(history_path),
        "history_rows": int(len(history_df)),
        "labeled_rows": int(len(merged)),
        "spot_ohlcv_file": str(spot_ohlcv_path),
    }
    return merged, meta


def _build_multi_horizon_from_history(
    history_path: Path,
    horizons: List[str],
    spot_ohlcv_path: Path,
    fold_size: int,
    lookback_rows: Optional[int],
    lookback_hours: Optional[int],
    include_reliability_snapshots: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    normalized_horizons = [_format_horizon_label(_parse_horizon_hours(horizon)) for horizon in horizons]
    ohlcv_df = _load_ohlcv(spot_ohlcv_path, horizons=normalized_horizons)

    labeled_frames: List[pd.DataFrame] = []
    horizon_rows: Dict[str, int] = {}
    for horizon in normalized_horizons:
        history_df = _load_history_rows(
            history_path,
            horizon,
            include_reliability_snapshots=include_reliability_snapshots,
        )
        history_df = _apply_time_filters(history_df, lookback_rows=lookback_rows, lookback_hours=lookback_hours)
        close_next_col, ret_col = _realized_column_names(horizon)
        merged = history_df.merge(
            ohlcv_df[["ts_hour", "close", close_next_col, ret_col]],
            on="ts_hour",
            how="left",
        )
        merged = merged.rename(
            columns={
                close_next_col: "close_target",
                ret_col: "ret_realized",
            }
        )
        merged["y_true"] = (pd.to_numeric(merged["ret_realized"], errors="coerce") > 0).astype(float)
        merged = merged.dropna(subset=["y_true", "ret_realized"]).copy()
        if merged.empty:
            continue
        merged["y_true"] = merged["y_true"].astype(int)
        labeled_frames.append(merged)
        horizon_rows[horizon] = int(len(merged))

    if not labeled_frames:
        raise RuntimeError("No labeled rows found for requested horizons from history+OHLCV.")

    combined = pd.concat(labeled_frames, ignore_index=True)
    combined = combined.sort_values(["ts", "horizon"]).reset_index(drop=True)
    combined = _assign_fold(combined, fold_size)
    meta = {
        "source": "history_plus_ohlcv_multi_horizon",
        "source_path": str(history_path),
        "history_rows": int(len(combined)),
        "labeled_rows": int(len(combined)),
        "spot_ohlcv_file": str(spot_ohlcv_path),
        "horizons": normalized_horizons,
        "rows_by_horizon": horizon_rows,
    }
    return combined, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical labeled backtest CSV for model-quality evaluation. "
            "Prefers backtest_signals.csv when available, with history+OHLCV fallback."
        )
    )
    parser.add_argument("--history-path", type=Path, default=Path("artifacts/predictions/history.json"))
    parser.add_argument(
        "--backtest-csv",
        type=Path,
        default=Path("artifacts/backtests/latest/backtest_signals.csv"),
        help="Backtest CSV path or directory. If present, used as canonical source.",
    )
    parser.add_argument(
        "--prefer-backtest",
        action="store_true",
        help="Prefer backtest CSV as source when it exists (default: true).",
    )
    parser.add_argument(
        "--no-prefer-backtest",
        dest="prefer_backtest",
        action="store_false",
        help="Disable backtest-source preference and force history+OHLCV path.",
    )
    parser.set_defaults(prefer_backtest=True)
    parser.add_argument(
        "--spot-ohlcv-path",
        type=Path,
        default=Path("data/spot_klines"),
        help="Parquet/CSV file or directory (latest parquet will be used).",
    )
    parser.add_argument(
        "--include-reliability-snapshots",
        action="store_true",
        help="Also merge archived artifacts/reliability/*/summary/live_predictions_snapshot.json entries into the history source.",
    )
    parser.add_argument("--horizon", type=str, default="1h")
    parser.add_argument(
        "--horizons",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of horizons to label from prediction history, e.g. 1h 4h 8h 12h.",
    )
    parser.add_argument("--fold-size", type=int, default=6)
    parser.add_argument(
        "--lookback-rows",
        type=int,
        default=2000,
        help="Keep only the most recent N rows before labeling (0 disables).",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=0,
        help="Keep only rows in the last N hours before labeling (0 disables).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum labeled rows required for reliable quality evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/monitoring/labeled_backtest_1h.csv"),
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("artifacts/monitoring/labeled_backtest_1h_meta.json"),
        help="JSON metadata for source, row counts, and filters.",
    )
    args = parser.parse_args()

    if args.fold_size <= 0:
        raise ValueError("--fold-size must be > 0")

    lookback_rows = int(args.lookback_rows) if int(args.lookback_rows) > 0 else None
    lookback_hours = int(args.lookback_hours) if int(args.lookback_hours) > 0 else None

    labeled: pd.DataFrame
    meta: Dict[str, object]
    requested_horizons = args.horizons or [args.horizon]
    normalized_horizons = [_format_horizon_label(_parse_horizon_hours(horizon)) for horizon in requested_horizons]
    multi_horizon_requested = len(normalized_horizons) > 1

    if multi_horizon_requested:
        labeled, meta = _build_multi_horizon_from_history(
            history_path=args.history_path,
            horizons=normalized_horizons,
            spot_ohlcv_path=args.spot_ohlcv_path,
            fold_size=args.fold_size,
            lookback_rows=lookback_rows,
            lookback_hours=lookback_hours,
            include_reliability_snapshots=args.include_reliability_snapshots,
        )
    elif args.prefer_backtest:
        try:
            _ = _find_latest_backtest_csv(args.backtest_csv)
            backtest_csv = _select_best_backtest_candidate(args.backtest_csv, min_rows_hint=int(args.min_rows))
            labeled, meta = _build_from_backtest(
                backtest_csv,
                history_path=args.history_path,
                horizon=normalized_horizons[0],
                fold_size=args.fold_size,
                lookback_rows=lookback_rows,
                lookback_hours=lookback_hours,
                include_reliability_snapshots=args.include_reliability_snapshots,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Warning: backtest-source build unavailable ({exc}); falling back to history+OHLCV.")
            labeled, meta = _build_from_history(
                history_path=args.history_path,
                horizon=normalized_horizons[0],
                spot_ohlcv_path=args.spot_ohlcv_path,
                fold_size=args.fold_size,
                lookback_rows=lookback_rows,
                lookback_hours=lookback_hours,
                include_reliability_snapshots=args.include_reliability_snapshots,
            )
    else:
        labeled, meta = _build_from_history(
            history_path=args.history_path,
            horizon=normalized_horizons[0],
            spot_ohlcv_path=args.spot_ohlcv_path,
            fold_size=args.fold_size,
            lookback_rows=lookback_rows,
            lookback_hours=lookback_hours,
            include_reliability_snapshots=args.include_reliability_snapshots,
        )

    if len(labeled) < int(args.min_rows):
        raise RuntimeError(
            f"Labeled rows {len(labeled)} below --min-rows={args.min_rows}; "
            "expand history/backtest window before quality gating."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)

    meta_payload = {
        **meta,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "output": str(args.output),
        "lookback_rows": lookback_rows,
        "lookback_hours": lookback_hours,
        "min_rows": int(args.min_rows),
        "requested_horizons": normalized_horizons,
        "include_reliability_snapshots": bool(args.include_reliability_snapshots),
        "columns": [str(c) for c in labeled.columns],
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"source={meta_payload.get('source')}")
    print(f"history_rows={meta_payload.get('history_rows')}")
    print(f"labeled_rows={len(labeled)}")
    print(f"output={args.output}")
    print(f"meta_output={args.meta_output}")


if __name__ == "__main__":
    main()
