import argparse
import json
from typing import Any, Dict, Optional

from src.config_trading import DEFAULT_P_UP_MIN, DEFAULT_RET_MIN
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    find_row_index_for_ts,
    load_models,
    prepare_data_for_signals,
)


REG_MODEL_PATH = "artifacts/models/xgb_ret1h_v1/xgb_ret1h_model.json"
DIR_MODEL_PATH = "artifacts/models/xgb_dir1h_v1/xgb_dir1h_model.json"
DATASET_NPZ_PATH = "artifacts/datasets/btc_features_1h_splits.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single live trading signal using the latest row "
            "from the curated btc_features_1h table."
        ),
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=DEFAULT_P_UP_MIN,
        help="Ensemble threshold for P(up). Defaults to config_trading.DEFAULT_P_UP_MIN.",
    )
    parser.add_argument(
        "--ret-min",
        type=float,
        default=DEFAULT_RET_MIN,
        help="Ensemble threshold for predicted ret_1h. Defaults to config_trading.DEFAULT_RET_MIN.",
    )
    parser.add_argument(
        "--ts",
        type=str,
        default=None,
        help=(
            "Optional timestamp to evaluate (BigQuery TIMESTAMP-compatible string). "
            "If omitted, the latest available row by ts is used."
        ),
    )
    return parser.parse_args()


def run_signal_once(p_up_min: float, ret_min: float, ts_str: Optional[str]) -> Dict[str, Any]:
    # Prepare full dataset and scaler using shared helpers
    prepared: PreparedData = prepare_data_for_signals(DATASET_NPZ_PATH, target_column="ret_1h")

    # Determine which row index to evaluate
    if ts_str is None:
        index = len(prepared.df_all) - 1
    else:
        index = find_row_index_for_ts(prepared.df_all, ts_str)

    # Load models and compute signal for the selected index
    models = load_models(REG_MODEL_PATH, DIR_MODEL_PATH)
    sig = compute_signal_for_index(
        prepared=prepared,
        index=index,
        models=models,
        p_up_min=p_up_min,
        ret_min=ret_min,
    )

    # Attach thresholds for reporting
    sig["thresholds"] = {
        "p_up_min": float(p_up_min),
        "ret_min": float(ret_min),
    }
    return sig


def main() -> None:
    args = _parse_args()
    result = run_signal_once(p_up_min=args.p_up_min, ret_min=args.ret_min, ts_str=args.ts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
