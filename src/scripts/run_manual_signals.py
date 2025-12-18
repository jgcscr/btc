"""Manual multi-horizon signal CLI leveraging local feature sources."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from data.ingestors.binance_us_spot import ingest_binance_us_spot
from data.processed.compute_cryptoquant_resampled import process_cryptoquant_resampled
from data.processed.compute_macro_features import process_macro_features
from src.trading.signals import (
    compute_signal_for_index,
    load_models,
    prepare_data_for_signals_from_ohlcv,
)
from src.data.dataset_preparation import enforce_unique_hourly_index
from src.trading.thresholds import load_calibrated_thresholds

DEFAULT_HOURS = 12
DEFAULT_TARGETS = (1, 2, 3, 4, 8, 12)
DEFAULT_P_UP_MIN = 0.45
DEFAULT_RET_MIN = 0.0
MODEL_ROOT = Path("artifacts/models")
RET_MODEL_META_1H = MODEL_ROOT / "xgb_ret1h_v1" / "model_metadata.json"
DIR_MODEL_META_1H = MODEL_ROOT / "xgb_dir1h_v1" / "model_metadata_direction.json"
DATASET_PATH = Path("artifacts/datasets/btc_features_multi_horizon_splits.npz")
OUTPUT_JSON = Path("artifacts/predictions/manual/latest.json")
DEFAULT_THRESHOLDS_PATH = Path("artifacts/predictions/manual/thresholds.json")
FEATURE_FALLBACK = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
]


def _augment_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def _safe_diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _safe_pct(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for base in ("close", "volume", "fut_close", "fut_volume"):
        if base not in result.columns:
            continue
        result[f"{base}_delta_1h"] = _safe_diff(result[base])
        result[f"{base}_pct_change_1h"] = _safe_pct(result[base])

    if "close" in result.columns:
        std_7 = result["close"].rolling(window=7, min_periods=3).std(ddof=0)
        std_24 = result["close"].rolling(window=24, min_periods=6).std(ddof=0)
        if "ma_close_7h" in result.columns:
            denom = std_7.replace(0.0, np.nan)
            result["close_zscore_7h"] = ((result["close"] - result["ma_close_7h"]) / denom).fillna(0.0)
        if "ma_close_24h" in result.columns:
            denom = std_24.replace(0.0, np.nan)
            result["close_zscore_24h"] = ((result["close"] - result["ma_close_24h"]) / denom).fillna(0.0)

    if "fut_close" in result.columns:
        rolling_mean = result["fut_close"].rolling(window=7, min_periods=3).mean()
        rolling_std = result["fut_close"].rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
        result["fut_close_zscore_7h"] = ((result["fut_close"] - rolling_mean) / rolling_std).fillna(0.0)

    return result


def parse_targets(value: str) -> List[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one horizon must be provided.")
    targets: List[int] = []
    for part in parts:
        try:
            horizon = int(part)
        except ValueError as exc:  # pragma: no cover - CLI validation guard
            raise argparse.ArgumentTypeError(f"Invalid horizon: {part}") from exc
        if horizon <= 0:
            raise argparse.ArgumentTypeError("Horizons must be positive integers.")
        targets.append(horizon)
    return targets


def load_feature_names(dataset_path: Path) -> List[str]:
    dataset_features: List[str] = []
    if dataset_path.exists():
        with np.load(dataset_path, allow_pickle=True) as data:
            if "feature_names" in data:
                dataset_features = data["feature_names"].tolist()

    meta_features: List[str] = []
    for meta_path in (RET_MODEL_META_1H, DIR_MODEL_META_1H):
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        names = payload.get("feature_names")
        if isinstance(names, list) and names:
            meta_features = [str(name) for name in names]
            break

    if meta_features:
        if dataset_features and any(name not in dataset_features for name in meta_features):
            missing = [name for name in meta_features if name not in dataset_features]
            print(
                f"Warning: dataset is missing model feature columns {missing}; using combined feature list.",
                file=sys.stderr,
            )
            combined = list(dataset_features)
            for name in meta_features:
                if name not in combined:
                    combined.append(name)
            return combined
        else:
            return meta_features

    if dataset_features:
        return dataset_features
    return FEATURE_FALLBACK
def _ensure_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _build_spot_features(hours: int, feature_names: List[str]) -> tuple[pd.DataFrame, Dict[str, object]]:
    limit = max(hours + 24, 48)
    path = ingest_binance_us_spot(limit=limit, interval="1h")
    tidy = pd.read_parquet(path)
    tidy["ts"] = pd.to_datetime(tidy["ts"], utc=True)

    pivot = tidy.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()

    rename_map = {
        "spot_open": "open",
        "spot_high": "high",
        "spot_low": "low",
        "spot_close": "close",
        "spot_volume": "volume",
        "spot_quote_volume": "quote_volume",
        "spot_num_trades": "num_trades",
    }
    pivot = pivot.rename(columns=rename_map)

    _ensure_numeric(pivot, rename_map.values())

    for fut_col, source in (
        ("fut_open", "open"),
        ("fut_high", "high"),
        ("fut_low", "low"),
        ("fut_close", "close"),
        ("fut_volume", "volume"),
    ):
        if fut_col not in pivot or pivot[fut_col].isna().all():
            pivot[fut_col] = pivot[source]
        else:
            pivot[fut_col] = pd.to_numeric(pivot[fut_col], errors="coerce")

    for base_col in ("open_interest", "funding_rate"):
        if base_col not in pivot:
            pivot[base_col] = 0.0
        pivot[base_col] = pd.to_numeric(pivot[base_col], errors="coerce").fillna(0.0)

    pivot["ma_close_7h"] = pivot["close"].rolling(window=7, min_periods=7).mean()
    pivot["ma_close_24h"] = pivot["close"].rolling(window=24, min_periods=24).mean()
    pivot["ma_ratio_7_24"] = pivot["ma_close_7h"] / pivot["ma_close_24h"]
    pivot["vol_24h"] = pivot["volume"].rolling(window=24, min_periods=24).sum()

    pivot = pivot.dropna(subset=["ma_close_7h", "ma_close_24h", "ma_ratio_7_24", "vol_24h"])
    if pivot.empty:
        raise RuntimeError("Spot pivot is empty after computing derived metrics; increase --hours")

    pivot = pivot.reset_index()
    pivot = _augment_price_features(pivot)

    latest_row = pivot.iloc[-1]
    missing_columns = [col for col in feature_names if col not in pivot or pivot[col].isna().all()]
    info = {
        "latest_ts": latest_row["ts"],
        "latest_close": float(latest_row["close"]),
        "row_count": len(pivot),
        "missing_columns": missing_columns,
    }

    return pivot, info


def _load_processor_frame(processor_fn, description: str) -> Path | None:
    try:
        path = processor_fn()
        if isinstance(path, tuple):
            path = path[0]
        if path is None:
            return None
        return Path(path)
    except Exception as exc:  # pragma: no cover - safety against missing raw data
        print(f"Warning: failed to run {description} ({exc}); continuing with cached data if available.", file=sys.stderr)
        return None


def _load_recent_parquet(path: Path | None, hours: int) -> tuple[pd.DataFrame, Dict[str, object]]:
    if path is None or not path.exists():
        return pd.DataFrame(), {"row_count": 0, "latest_ts": None, "missing_columns": []}

    frame = pd.read_parquet(path)
    ts_column = "timestamp" if "timestamp" in frame.columns else "ts"
    frame[ts_column] = pd.to_datetime(frame[ts_column], utc=True)
    latest_ts = frame[ts_column].max()
    cutoff = latest_ts - timedelta(hours=hours)
    recent = frame.loc[frame[ts_column] >= cutoff].copy()
    recent = recent.rename(columns={ts_column: "ts"})
    recent = recent.sort_values("ts").reset_index(drop=True)

    summary = {
        "row_count": int(len(recent)),
        "latest_ts": recent["ts"].max() if not recent.empty else latest_ts,
        "missing_columns": [],
    }
    if not recent.empty:
        last_row = recent.iloc[-1]
        summary["missing_columns"] = [col for col in recent.columns if col != "ts" and pd.isna(last_row[col])]

    return recent, summary


def _merge_optional(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return base
    merged = pd.merge_asof(
        base.sort_values("ts"),
        extra.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    return merged


def _project_price(close: float, log_return: float) -> float:
    return close * math.exp(log_return)


def format_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, header in enumerate(headers):
            widths[idx] = max(widths[idx], len(row[header]))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(row[header].ljust(widths[idx]) for idx, header in enumerate(headers)) for row in rows]
    return "\n".join([header_line, divider, *body])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch local feature slices and emit manual multi-horizon predictions without BigQuery."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Number of recent hourly bars to inspect (default: 12).",
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=list(DEFAULT_TARGETS),
        help="Comma-separated prediction horizons in hours (default: 1,2,3,4,8,12).",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Probability threshold for ensemble activation (default: 0.45).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Return threshold for ensemble activation (default: 0.0).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_PATH,
        help="NPZ dataset containing feature_names (default: artifacts/datasets/btc_features_multi_horizon_splits.npz).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help="Path to write manual prediction JSON (default: artifacts/predictions/manual/latest.json).",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help=(
            "Optional JSON file with per-horizon thresholds (default: "
            "artifacts/predictions/manual/thresholds.json)."
        ),
    )
    parser.add_argument(
        "--disable-calibrated-thresholds",
        action="store_true",
        help="Ignore calibrated per-horizon thresholds even if present.",
    )
    parser.add_argument(
        "--run-without-funding",
        action="store_true",
        help="Set RUN_WITHOUT_FUNDING=1 while executing this CLI (useful when funding data is unavailable).",
    )
    return parser.parse_args(argv)


def load_models_for_horizon(horizon: int) -> tuple[Path, Path] | None:
    reg_dir = MODEL_ROOT / f"xgb_ret{horizon}h_v1"
    dir_dir = MODEL_ROOT / f"xgb_dir{horizon}h_v1"
    reg_path = reg_dir / f"xgb_ret{horizon}h_model.json"
    dir_path = dir_dir / f"xgb_dir{horizon}h_model.json"
    if not reg_path.exists() or not dir_path.exists():
        print(
            f"Warning: skipping {horizon}h horizon because model files are missing ({reg_path} or {dir_path}).",
            file=sys.stderr,
        )
        return None
    return reg_path, dir_path


def prepare_feature_frame(combined: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    clean, _, _ = enforce_unique_hourly_index(combined, label="manual_features")
    base_columns = ["ts", *(col for col in feature_names if col in clean.columns)]
    frame = clean[base_columns].copy()
    for column in feature_names:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("ts").reset_index(drop=True)
    frame[feature_names] = frame[feature_names].ffill().bfill().fillna(0.0)
    frame = frame[["ts", *feature_names]]
    return frame


def run_predictions(
    combined: pd.DataFrame,
    feature_names: List[str],
    targets: Iterable[int],
    default_p_up_min: float,
    default_ret_min: float,
    thresholds_by_horizon: Dict[int, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, float | str | int]]:
    feature_frame = prepare_feature_frame(combined, feature_names)
    prepared = prepare_data_for_signals_from_ohlcv(feature_frame, feature_names=feature_names)

    index = len(prepared.df_all) - 1
    if index < 0:
        raise RuntimeError("Prepared dataset has no rows for inference.")

    ts_value = prepared.df_all["ts"].iloc[index]
    ts_iso = ts_value.isoformat() if isinstance(ts_value, datetime) else str(ts_value)
    close = float(prepared.df_all["close"].iloc[index])

    summary: Dict[str, Dict[str, float | str | int]] = {}

    for horizon in sorted(set(targets)):
        paths = load_models_for_horizon(horizon)
        if paths is None:
            continue
        reg_path, dir_path = paths
        models = load_models(str(reg_path), str(dir_path))
        horizon_thresholds = (thresholds_by_horizon or {}).get(horizon, {})
        p_up_min = float(horizon_thresholds.get("p_up_min", default_p_up_min))
        ret_min = float(horizon_thresholds.get("ret_min", default_ret_min))
        signal = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=p_up_min,
            ret_min=ret_min,
        )
        ret_pred = float(signal.get("ret_pred", 0.0))
        p_up = float(signal.get("p_up", 0.0))
        summary[f"{horizon}h"] = {
            "timestamp": signal.get("ts", ts_iso),
            "horizon_hours": horizon,
            "close": close,
            "p_up": p_up,
            "ret_pred": ret_pred,
            "projected_price": _project_price(close, ret_pred),
            "signal_ensemble": int(signal.get("signal_ensemble", 0)),
            "signal_dir_only": int(signal.get("signal_dir_only", 0)),
            "thresholds": {
                "p_up_min": p_up_min,
                "ret_min": ret_min,
            },
        }

    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.run_without_funding:
        os.environ["RUN_WITHOUT_FUNDING"] = "1"

    feature_names = load_feature_names(args.dataset_path)

    thresholds_by_horizon: Dict[int, Dict[str, float]] = {}
    if not args.disable_calibrated_thresholds:
        thresholds_by_horizon = load_calibrated_thresholds(args.thresholds_path)
        if thresholds_by_horizon:
            print(
                f"Loaded calibrated thresholds for horizons {sorted(thresholds_by_horizon.keys())} from {args.thresholds_path}.",
                file=sys.stderr,
            )

    spot_frame, spot_info = _build_spot_features(args.hours, feature_names)

    macro_path = _load_processor_frame(process_macro_features, "process_macro_features")
    macro_frame, macro_info = _load_recent_parquet(macro_path, args.hours)

    cq_path = _load_processor_frame(process_cryptoquant_resampled, "process_cryptoquant_resampled")
    cq_frame, cq_info = _load_recent_parquet(cq_path, args.hours)

    combined = spot_frame.copy()
    combined = _merge_optional(combined, macro_frame)
    combined = _merge_optional(combined, cq_frame)

    predictions = run_predictions(
        combined,
        feature_names,
        args.targets,
        default_p_up_min=args.p_up_min,
        default_ret_min=args.ret_min,
        thresholds_by_horizon=thresholds_by_horizon,
    )

    if not predictions:
        print("No predictions produced; verify model coverage.", file=sys.stderr)
        sys.exit(1)

    generated_at = datetime.now(timezone.utc).isoformat()

    payload = {
        "generated_at": generated_at,
        "source": "run_manual_signals",
        "predictions": predictions,
    }
    if thresholds_by_horizon:
        payload["thresholds"] = {
            "source": str(args.thresholds_path),
            "values": {f"{h}h": thresholds_by_horizon[h] for h in sorted(thresholds_by_horizon)},
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    rows = []
    for horizon_key in sorted(predictions.keys(), key=lambda x: int(x.rstrip("h"))):
        entry = predictions[horizon_key]
        rows.append(
            {
                "Horizon": horizon_key,
                "Timestamp": str(entry["timestamp"]),
                "Close": f"{entry['close']:.2f}",
                "p_up": f"{entry['p_up']:.3f}",
                "ret_pred": f"{entry['ret_pred']:.4f}",
                "projected": f"{entry['projected_price']:.2f}",
                "p_up_min": f"{entry['thresholds']['p_up_min']:.3f}",
                "ret_min": f"{entry['thresholds']['ret_min']:.4f}",
                "Trade": "ON" if entry["signal_ensemble"] else "OFF",
            }
        )

    headers = [
        "Horizon",
        "Timestamp",
        "Close",
        "p_up",
        "ret_pred",
        "projected",
        "p_up_min",
        "ret_min",
        "Trade",
    ]
    print(format_table(rows, headers))

    def _fmt_ts(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    print(
        "\nData coverage:"\
        f"\n  Binance spot -> rows={spot_info['row_count']} latest_ts={_fmt_ts(spot_info['latest_ts'])} close={spot_info['latest_close']:.2f} missing={','.join(spot_info['missing_columns']) if spot_info['missing_columns'] else 'none'}"
        f"\n  Macro       -> rows={macro_info['row_count']} latest_ts={_fmt_ts(macro_info['latest_ts'])} missing={','.join(macro_info['missing_columns']) if macro_info['missing_columns'] else 'none'}"
        f"\n  CryptoQuant -> rows={cq_info['row_count']} latest_ts={_fmt_ts(cq_info['latest_ts'])} missing={','.join(cq_info['missing_columns']) if cq_info['missing_columns'] else 'none'}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
