from __future__ import annotations

import argparse

from src.runtime.prediction_request_support import build_prediction_runtime_request
from src.runtime.refresh_support import PredictionInputBundle, SequenceModelDirs


def test_build_prediction_runtime_request_collects_runtime_kwargs() -> None:
    args = argparse.Namespace(
        targets=[1.0, 4.0],
        p_up_min=0.45,
        ret_min=0.0,
        direction_threshold=0.5,
        auto_direction_threshold=False,
        dry_run=True,
        dir_model_config_json=None,
        dir_model_weights="xgb:1.0",
        trend_ignition={},
        direction_only_fallback={},
        adaptive_thresholds={},
        target_range_models={},
        abstention_policy={},
        uncertainty_policy={},
        trade_decision_policy={},
        regime_model_weights={},
        regime_model_dirs={},
        confluence_policy={},
        execution_policy={},
        forecast_coherence_policy={},
        direction_ensemble_policy={},
        trust_hardening_policy={},
        confidence_min=0.2,
        confidence_min_by_horizon_regime=None,
        position_size_floor=0.0,
        position_size_cap=1.0,
        position_size_cap_by_horizon=None,
        disabled_horizons=None,
    )
    bundle = PredictionInputBundle(direction_output_cfg={"enabled": True}, thresholds_by_horizon={1.0: {}}, platt_calibration={})
    dirs = SequenceModelDirs(None, None, None, None, None, None, None)

    request = build_prediction_runtime_request(
        args,
        prepared_override="prepared",
        latest_close=123.45,
        prediction_inputs=bundle,
        sequence_dirs=dirs,
    )

    assert request.to_kwargs()["prepared_override"] == "prepared"
    assert request.to_kwargs()["latest_close"] == 123.45
    assert request.to_kwargs()["direction_output_policy"] == {"enabled": True}