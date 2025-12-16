import argparse
import json
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.trading.signals import (
    compute_signal_for_index,
    load_models,
    prepare_data_for_signals,
)

DEFAULT_DATASET = Path("artifacts/datasets/btc_features_multi_horizon_splits.npz")
DEFAULT_OUTPUT = Path("artifacts/predictions/manual/thresholds.json")
DEFAULT_HORIZONS = (1, 2, 3, 4, 8, 12)
MODEL_ROOT = Path("artifacts/models")


def _parse_float_grid(value: str) -> List[float]:
    parts = [segment.strip() for segment in value.split(",") if segment.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    try:
        return [float(item) for item in parts]
    except ValueError as exc:  # pragma: no cover - CLI validation guard
        raise argparse.ArgumentTypeError(f"Invalid float in grid: {value}") from exc


def _load_return_series(dataset_path: Path, horizons: Iterable[int]) -> Dict[int, np.ndarray]:
    with np.load(dataset_path, allow_pickle=True) as data:
        series: Dict[int, np.ndarray] = {
            1: np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0),
        }
        for horizon in horizons:
            if horizon == 1:
                continue
            prefix = f"y_ret{horizon}h"
            components = [f"{prefix}_train", f"{prefix}_val", f"{prefix}_test"]
            if not all(name in data for name in components):
                raise KeyError(f"Missing return series for {horizon}h horizon in dataset.")
            series[horizon] = np.concatenate([data[name] for name in components], axis=0)
    return series


def _model_paths(horizon: int) -> tuple[Path, Path] | None:
    reg_path = MODEL_ROOT / f"xgb_ret{horizon}h_v1" / f"xgb_ret{horizon}h_model.json"
    dir_path = MODEL_ROOT / f"xgb_dir{horizon}h_v1" / f"xgb_dir{horizon}h_model.json"
    if not reg_path.exists() or not dir_path.exists():
        return None
    return reg_path, dir_path


def _prepare_time_slice(prepared_df: pd.DataFrame, start_idx: int, n_total: int) -> pd.Series:
    window = prepared_df.iloc[start_idx:start_idx + n_total].reset_index(drop=True)
    ts_series = pd.to_datetime(window["ts"], utc=True)
    if hasattr(ts_series.dt, "tz_convert"):
        ts_series = ts_series.dt.tz_convert("UTC")
        ts_series = ts_series.dt.tz_localize(None)
    return ts_series


def _score_combo(avg_ret: float, max_drawdown: float, n_trades: int, min_trades: int) -> float:
    if n_trades == 0:
        return 0.0
    scaled_avg = avg_ret if n_trades >= min_trades else avg_ret * (n_trades / max(min_trades, 1))
    return scaled_avg


def _evaluate_grid(
    p_up_scores: np.ndarray,
    ret_preds: np.ndarray,
    realized: np.ndarray,
    p_up_grid: Sequence[float],
    ret_grid: Sequence[float],
    min_trades: int,
) -> tuple[Dict[str, float], Dict[str, float]]:
    best_positive: tuple[Dict[str, float], Dict[str, float]] | None = None
    best_positive_rank = (-float("inf"), -float("inf"), -float("inf"))
    best_overall: tuple[Dict[str, float], Dict[str, float]] | None = None
    best_overall_rank = (-float("inf"), -float("inf"), -float("inf"))

    for p_up_min in p_up_grid:
        for ret_min in ret_grid:
            mask = (p_up_scores >= p_up_min) & (ret_preds >= ret_min)
            n_trades = int(mask.sum())
            if n_trades == 0:
                avg_ret = float("nan")
                hit_rate = float("nan")
                max_drawdown = 0.0
                score = 0.0
            else:
                realized_active = realized[mask]
                hit_rate = float((realized_active > 0.0).mean())
                avg_ret = float(realized_active.mean())
                eq_curve = np.cumsum(realized * mask)
                peak = np.maximum.accumulate(eq_curve)
                drawdown = eq_curve - peak
                max_drawdown = float(drawdown.min())
                score = _score_combo(avg_ret, max_drawdown, n_trades, min_trades)

            if n_trades == 0 and not np.isfinite(score):
                score = 0.0

            metrics_dict = {
                "n_trades": float(n_trades),
                "hit_rate": float(hit_rate) if np.isfinite(hit_rate) else float("nan"),
                "avg_return": float(avg_ret) if np.isfinite(avg_ret) else float("nan"),
                "max_drawdown": float(max_drawdown),
                "score": float(score),
            }
            thresholds_dict = {"p_up_min": float(p_up_min), "ret_min": float(ret_min)}

            candidate = (score, max_drawdown, float(n_trades))

            if np.isfinite(avg_ret) and avg_ret > 0.0 and n_trades > 0:
                if candidate > best_positive_rank:
                    best_positive = (thresholds_dict, metrics_dict)
                    best_positive_rank = candidate

            if candidate > best_overall_rank:
                best_overall = (thresholds_dict, metrics_dict)
                best_overall_rank = candidate

    chosen = best_positive if best_positive is not None else best_overall
    if chosen is None:
        raise RuntimeError("Failed to determine optimal thresholds; check grid or dataset alignment.")
    return chosen


def calibrate(
    dataset_path: Path,
    horizons: Sequence[int],
    window_days: int,
    p_up_grid: Sequence[float],
    ret_grid: Sequence[float],
    min_trades: int,
) -> dict:
    prepared = prepare_data_for_signals(str(dataset_path), target_column="ret_1h")
    ret_series_map = _load_return_series(dataset_path, horizons)

    n_total = len(ret_series_map[1])
    offset = len(prepared.df_all) - n_total
    if offset < 0:
        raise RuntimeError("Return series longer than prepared dataframe; cannot align.")

    ts_series = _prepare_time_slice(prepared.df_all, offset, n_total)
    period_end = ts_series.iloc[-1]
    cutoff = period_end - timedelta(days=window_days)
    start_idx = int(np.searchsorted(ts_series.to_numpy(), np.datetime64(cutoff), side="left"))

    summary: dict = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "window": {
            "start": ts_series.iloc[start_idx].isoformat() if start_idx < len(ts_series) else None,
            "end": period_end.isoformat(),
            "hours": int(len(ts_series) - start_idx),
        },
        "grid": {
            "p_up_min": list(p_up_grid),
            "ret_min": list(ret_grid),
            "min_trades": int(min_trades),
        },
        "horizons": OrderedDict(),
    }

    for horizon in horizons:
        paths = _model_paths(horizon)
        if paths is None:
            print(f"Skipping {horizon}h: missing model artifacts.")
            continue
        reg_path, dir_path = paths
        models = load_models(str(reg_path), str(dir_path))

        realized_full = ret_series_map[horizon]
        realized = realized_full[start_idx:n_total]

        p_up_scores: List[float] = []
        ret_preds: List[float] = []
        for idx in range(start_idx, n_total):
            row_idx = offset + idx
            signal = compute_signal_for_index(
                prepared=prepared,
                index=row_idx,
                models=models,
                p_up_min=0.0,
                ret_min=-1.0,
            )
            p_up_scores.append(float(signal.get("p_up", 0.0)))
            ret_preds.append(float(signal.get("ret_pred", 0.0)))

        p_up_arr = np.asarray(p_up_scores, dtype=float)
        ret_arr = np.asarray(ret_preds, dtype=float)
        realized_arr = np.asarray(realized[: len(p_up_arr)], dtype=float)

        thresholds, metrics = _evaluate_grid(
            p_up_scores=p_up_arr,
            ret_preds=ret_arr,
            realized=realized_arr,
            p_up_grid=p_up_grid,
            ret_grid=ret_grid,
            min_trades=min_trades,
        )

        metrics.update({
            "p_up_mean": float(p_up_arr.mean()) if p_up_arr.size else float("nan"),
            "ret_pred_mean": float(ret_arr.mean()) if ret_arr.size else float("nan"),
        })

        summary["horizons"][str(horizon)] = {
            "p_up_min": thresholds["p_up_min"],
            "ret_min": thresholds["ret_min"],
            "metrics": metrics,
        }

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search calibrated per-horizon ensemble thresholds using recent data.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to multi-horizon dataset NPZ (default: artifacts/datasets/btc_features_multi_horizon_splits.npz).",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated horizons to calibrate (default: 1,2,3,4,8,12).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling window length in days for calibration (default: 30).",
    )
    parser.add_argument(
        "--p-up-grid",
        type=_parse_float_grid,
        default=list(np.linspace(0.45, 0.75, 7)),
        help="Comma-separated grid for p_up_min thresholds (default: 0.45..0.75 step 0.05).",
    )
    parser.add_argument(
        "--ret-min-grid",
        type=_parse_float_grid,
        default=[-0.0005, 0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025],
        help="Comma-separated grid for ret_min thresholds.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trade count for an unpenalized score (default: 10).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path for calibrated thresholds (default: artifacts/predictions/manual/thresholds.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    horizons = [int(part.strip()) for part in args.horizons.split(",") if part.strip()]

    summary = calibrate(
        dataset_path=args.dataset_path,
        horizons=horizons,
        window_days=args.window_days,
        p_up_grid=args.p_up_grid,
        ret_grid=args.ret_min_grid,
        min_trades=args.min_trades,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
