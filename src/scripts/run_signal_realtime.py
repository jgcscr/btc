import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.config_trading import DEFAULT_P_UP_MIN, DEFAULT_RET_MIN
from src.trading.signals import PreparedData, compute_signal_for_index, find_row_index_for_ts, load_models, prepare_data_for_signals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a live-like trading signal using the latest curated row "
            "and append it to a paper-trade log, suitable for hourly cron runs."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_splits.npz",
        help="Path to the regression NPZ file (used for feature names and scaler reconstruction).",
    )
    parser.add_argument(
        "--reg-model-dir",
        type=str,
        default="artifacts/models/xgb_ret1h_v1",
        help="Directory containing regression model JSON (xgb_ret1h_model.json).",
    )
    parser.add_argument(
        "--dir-model-dir",
        type=str,
        default="artifacts/models/xgb_dir1h_v1",
        help="Directory containing direction model JSON (xgb_dir1h_model.json).",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Ensemble threshold for P(up).",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Ensemble threshold for predicted ret_1h.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="artifacts/live/paper_trade_realtime.csv",
        help="Path to the CSV log file for live/paper trading signals.",
    )
    parser.add_argument(
        "--ts",
        type=str,
        default=None,
        help=(
            "Optional timestamp to evaluate instead of the latest bar (ISO8601/RFC3339). "
            "If omitted, uses the most recent available row."
        ),
    )
    return parser.parse_args()


def _now_utc_iso() -> str:
    dt = datetime.now(timezone.utc)
    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def _load_last_logged_ts(log_path: str) -> Optional[str]:
    if not os.path.exists(log_path):
        return None

    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None

    if df.empty or "ts" not in df.columns:
        return None

    return str(df["ts"].iloc[-1])


def _append_to_log(log_path: str, row: Dict[str, Any]) -> bool:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_exists = os.path.exists(log_path)

    df_row = pd.DataFrame([row])

    if not file_exists:
        df_row.to_csv(log_path, index=False)
        return True

    # Append without rewriting the whole file header
    df_row.to_csv(log_path, mode="a", header=False, index=False)
    return True


def run_signal_realtime(args: argparse.Namespace) -> None:
    # Prepare data and models using the same helpers as run_signal_once/backtest
    prepared: PreparedData = prepare_data_for_signals(args.dataset_path, target_column="ret_1h")

    if args.ts is None:
        index = len(prepared.df_all) - 1
    else:
        index = find_row_index_for_ts(prepared.df_all, args.ts)

    models = load_models(
        reg_model_path=os.path.join(args.reg_model_dir, "xgb_ret1h_model.json"),
        dir_model_path=os.path.join(args.dir_model_dir, "xgb_dir1h_model.json"),
    )

    sig = compute_signal_for_index(
        prepared=prepared,
        index=index,
        models=models,
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
    )

    sig["thresholds"] = {
        "p_up_min": float(args.p_up_min),
        "ret_min": float(args.ret_min),
    }

    # Print signal JSON-like summary
    print(json.dumps(sig, indent=2))

    current_ts = sig["ts"]
    last_ts = _load_last_logged_ts(args.log_path)

    if last_ts is not None and last_ts == current_ts:
        print(
            f"No new bar; last ts={last_ts} equal to current ts={current_ts}; skipping append.",
        )
        return

    log_row = {
        "ts": current_ts,
        "p_up": sig["p_up"],
        "ret_pred": sig["ret_pred"],
        "signal_ensemble": sig["signal_ensemble"],
        "signal_dir_only": sig["signal_dir_only"],
        "created_at": _now_utc_iso(),
        "notes": "",
    }

    appended = _append_to_log(args.log_path, log_row)
    if appended:
        print(f"Appended signal for ts={current_ts} to {args.log_path}")


def main() -> None:
    args = _parse_args()
    run_signal_realtime(args)


if __name__ == "__main__":
    main()
