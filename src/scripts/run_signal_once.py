import argparse
import json
import os
from typing import Any, Dict, List, Optional

from src.config_trading import (
    DEFAULT_DIR_MODEL_DIR_1H,
    DEFAULT_DIR_MODEL_WEIGHTS_1H,
    DEFAULT_DIR_MODELS_1H,
    DEFAULT_P_UP_MIN,
    DEFAULT_REG_MODEL_DIR_1H,
    DEFAULT_RET_MIN,
)
from src.trading.direction_config import (
    direction_configs_to_weight_map,
    log_direction_model_configs,
    resolve_direction_model_configs,
)
from src.trading.signals import (
    PreparedData,
    compute_signal_for_index,
    find_row_index_for_ts,
    load_models,
    prepare_data_for_signals,
)


REG_MODEL_PATH = os.path.join(DEFAULT_REG_MODEL_DIR_1H, "xgb_ret1h_model.json")
DIR_MODEL_PATH = os.path.join(DEFAULT_DIR_MODEL_DIR_1H, "xgb_dir1h_model.json")
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
    parser.add_argument(
        "--dir-model-config-json",
        type=str,
        default=None,
        help="Optional JSON file describing direction-model entries (type/path/weight list).",
    )
    parser.add_argument(
        "--dir-model-weights",
        type=str,
        default=DEFAULT_DIR_MODEL_WEIGHTS_1H,
        help="Optional comma-separated weights for direction models (e.g. transformer:2,lstm:1,xgb:1).",
    )
    return parser.parse_args()


def _build_direction_model_bundle(
    config_json_path: str | None,
    weight_spec: str | None,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    overrides = {"xgb": DIR_MODEL_PATH}
    configs = resolve_direction_model_configs(
        DEFAULT_DIR_MODELS_1H,
        config_json_path=config_json_path or None,
        weight_spec=weight_spec,
        path_overrides=overrides,
    )
    log_direction_model_configs(configs, label="[run_signal_once] direction models")
    weight_map = direction_configs_to_weight_map(configs)
    return configs, weight_map


def run_signal_once(
    p_up_min: float,
    ret_min: float,
    ts_str: Optional[str],
    *,
    dir_model_config_json: Optional[str] = None,
    dir_model_weights: Optional[str] = None,
) -> Dict[str, Any]:
    # Prepare full dataset and scaler using shared helpers
    prepared: PreparedData = prepare_data_for_signals(DATASET_NPZ_PATH, target_column="ret_1h")

    # Determine which row index to evaluate
    if ts_str is None:
        index = len(prepared.df_all) - 1
    else:
        index = find_row_index_for_ts(prepared.df_all, ts_str)

    direction_configs, dir_weight_map = _build_direction_model_bundle(
        dir_model_config_json,
        dir_model_weights,
    )
    # Load models and compute signal for the selected index
    models = load_models(REG_MODEL_PATH, direction_model_configs=direction_configs)
    sig = compute_signal_for_index(
        prepared=prepared,
        index=index,
        models=models,
        p_up_min=p_up_min,
        ret_min=ret_min,
        dir_model_weights=dir_weight_map,
    )

    # Attach thresholds for reporting
    sig["thresholds"] = {
        "p_up_min": float(p_up_min),
        "ret_min": float(ret_min),
    }
    return sig


def main() -> None:
    args = _parse_args()
    result = run_signal_once(
        p_up_min=args.p_up_min,
        ret_min=args.ret_min,
        ts_str=args.ts,
        dir_model_config_json=args.dir_model_config_json,
        dir_model_weights=args.dir_model_weights,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
