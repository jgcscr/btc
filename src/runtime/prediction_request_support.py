from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.runtime.prediction_defaults import CONFIDENCE_MIN_DEFAULT, POSITION_SIZE_CAP_DEFAULT, POSITION_SIZE_FLOOR_DEFAULT
from src.runtime.refresh_support import PredictionInputBundle, SequenceModelDirs


@dataclass(frozen=True)
class PredictionRuntimeRequest:
    payload: dict[str, Any]

    def to_kwargs(self) -> dict[str, Any]:
        return dict(self.payload)


def build_prediction_runtime_request(
    args: argparse.Namespace,
    *,
    prepared_override: Any,
    latest_close: float | None,
    prediction_inputs: PredictionInputBundle,
    sequence_dirs: SequenceModelDirs,
) -> PredictionRuntimeRequest:
    return PredictionRuntimeRequest(
        payload={
            "targets": args.targets,
            "p_up_min": args.p_up_min,
            "ret_min": args.ret_min,
            "direction_threshold": args.direction_threshold,
            "auto_direction_threshold": args.auto_direction_threshold,
            "offline": args.dry_run,
            "dir_lstm_path": sequence_dirs.dir_lstm_path,
            "dir_bilstm_path": sequence_dirs.dir_bilstm_path,
            "dir_gru_path": sequence_dirs.dir_gru_path,
            "dir_cnn_lstm_path": sequence_dirs.dir_cnn_lstm_path,
            "dir_cnn_bilstm_path": sequence_dirs.dir_cnn_bilstm_path,
            "dir_garch_lstm_path": sequence_dirs.dir_garch_lstm_path,
            "dir_transformer_path": sequence_dirs.dir_transformer_path,
            "dir_model_config_json": args.dir_model_config_json or None,
            "dir_model_weights": args.dir_model_weights,
            "thresholds_by_horizon": prediction_inputs.thresholds_by_horizon,
            "prepared_override": prepared_override,
            "trend_ignition": getattr(args, "trend_ignition", None),
            "direction_only_fallback": getattr(args, "direction_only_fallback", None),
            "adaptive_thresholds": getattr(args, "adaptive_thresholds", None),
            "target_range_models": getattr(args, "target_range_models", None),
            "platt_calibration": prediction_inputs.platt_calibration,
            "abstention_policy": getattr(args, "abstention_policy", None),
            "uncertainty_policy": getattr(args, "uncertainty_policy", None),
            "trade_decision_policy": getattr(args, "trade_decision_policy", None),
            "regime_model_weights": getattr(args, "regime_model_weights", None),
            "regime_model_dirs": getattr(args, "regime_model_dirs", None),
            "confluence_policy": getattr(args, "confluence_policy", None),
            "execution_policy": getattr(args, "execution_policy", None),
            "forecast_coherence_policy": getattr(args, "forecast_coherence_policy", None),
            "direction_output_policy": prediction_inputs.direction_output_cfg,
            "direction_ensemble_policy": getattr(args, "direction_ensemble_policy", None),
            "trust_hardening_policy": getattr(args, "trust_hardening_policy", None),
            "latest_close": latest_close,
            "confidence_min": float(getattr(args, "confidence_min", CONFIDENCE_MIN_DEFAULT)),
            "confidence_min_by_horizon_regime": getattr(args, "confidence_min_by_horizon_regime", None),
            "position_size_floor": float(getattr(args, "position_size_floor", POSITION_SIZE_FLOOR_DEFAULT)),
            "position_size_cap": float(getattr(args, "position_size_cap", POSITION_SIZE_CAP_DEFAULT)),
            "position_size_cap_by_horizon": getattr(args, "position_size_cap_by_horizon", None),
            "disabled_horizons": getattr(args, "disabled_horizons", None),
        }
    )