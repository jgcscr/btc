from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping

from src.runtime.horizon_support import format_horizon_label, horizon_sort_key
from src.runtime.market_preparation import apply_replay_override, prepare_market_data
from src.runtime.models import PipelineExecutionResult, RuntimeMode, RuntimeRunPaths
from src.runtime.output_support import write_monitoring_artifact as runtime_write_monitoring_artifact, refresh_meta_baseline as runtime_refresh_meta_baseline
from src.runtime.persistence import RuntimeStateStore
from src.runtime.prediction_execution import run_predictions as runtime_run_predictions
from src.runtime.refresh_support import (
    PredictionInputBundle,
    normalize_refresh_args,
    resolve_prediction_inputs,
    resolve_sequence_model_dirs,
)
from src.runtime.summary_support import (
    build_blocked_trade_analytics,
    build_runtime_degradation_monitoring,
    build_runtime_prompt_ready_summary,
    write_prediction_summary as runtime_write_prediction_summary,
)
from src.scripts import run_refresh_and_predict as legacy
from src.scripts.build_signal_baseline import _append_detected_meta_columns
from src.trading.signals import PreparedData


PreparedOverride = tuple[PreparedData, int, float, str] | None


def execute_refresh_pipeline(args: argparse.Namespace, *, mode: RuntimeMode) -> PipelineExecutionResult:
    store = RuntimeStateStore()
    normalize_refresh_args(args)
    run_paths = store.start_run(mode=mode, request=vars(args))
    store.append_event(run_paths, stage="pipeline", status="started", details={"mode": mode.value})

    try:
        market_result = prepare_market_data(args, run_paths, store)
        replay_result = apply_replay_override(
            args,
            run_paths,
            store,
            prepared_override=market_result.prepared_override,
            latest_close=market_result.latest_close,
        )
        prediction_inputs = _load_prediction_inputs(
            args,
            run_paths,
            store,
        )
        summary = _run_prediction_stage(
            args,
            run_paths,
            store,
            prepared_override=replay_result.prepared_override,
            latest_close=replay_result.latest_close,
            prediction_inputs=prediction_inputs,
        )
        result = _persist_outputs(args, mode, summary, run_paths, store)
        store.append_event(run_paths, stage="pipeline", status="completed", details={"run_id": run_paths.run_id})
        return result
    except Exception as exc:
        store.append_event(run_paths, stage="pipeline", status="failed", details={"error": str(exc)})
        store.finalize(
            run_paths,
            mode=mode,
            status="failed",
            summary={"error": str(exc)},
        )
        raise


def _load_prediction_inputs(
    args: argparse.Namespace,
    run_paths: RuntimeRunPaths,
    store: RuntimeStateStore,
) -> PredictionInputBundle:
    store.append_event(run_paths, stage="model_input_resolution", status="started")
    prediction_inputs = resolve_prediction_inputs(args)
    store.append_event(
        run_paths,
        stage="model_input_resolution",
        status="completed",
        details={"threshold_horizons": sorted(str(key) for key in prediction_inputs.thresholds_by_horizon.keys())},
    )
    return prediction_inputs


def _run_prediction_stage(
    args: argparse.Namespace,
    run_paths: RuntimeRunPaths,
    store: RuntimeStateStore,
    *,
    prepared_override: PreparedOverride,
    latest_close: float | None,
    prediction_inputs: PredictionInputBundle,
) -> Mapping[str, Mapping[str, Any]]:
    store.append_event(run_paths, stage="prediction", status="started")
    sequence_dirs = resolve_sequence_model_dirs(args)
    if sequence_dirs.has_any():
        print(
            "Sequence ensemble directories:"
            f" LSTM={sequence_dirs.dir_lstm_path or 'None'}"
            f", BiLSTM={sequence_dirs.dir_bilstm_path or 'None'}"
            f", GRU={sequence_dirs.dir_gru_path or 'None'}"
            f", CNN-LSTM={sequence_dirs.dir_cnn_lstm_path or 'None'}"
            f", CNN-BiLSTM={sequence_dirs.dir_cnn_bilstm_path or 'None'}"
            f", GARCH-LSTM={sequence_dirs.dir_garch_lstm_path or 'None'}"
            f", transformer={sequence_dirs.dir_transformer_path or 'None'}",
        )

    summary = runtime_run_predictions(
        args.targets,
        args.p_up_min,
        args.ret_min,
        direction_threshold=args.direction_threshold,
        auto_direction_threshold=args.auto_direction_threshold,
        offline=args.dry_run,
        dir_lstm_path=sequence_dirs.dir_lstm_path,
        dir_bilstm_path=sequence_dirs.dir_bilstm_path,
        dir_gru_path=sequence_dirs.dir_gru_path,
        dir_cnn_lstm_path=sequence_dirs.dir_cnn_lstm_path,
        dir_cnn_bilstm_path=sequence_dirs.dir_cnn_bilstm_path,
        dir_garch_lstm_path=sequence_dirs.dir_garch_lstm_path,
        dir_transformer_path=sequence_dirs.dir_transformer_path,
        dir_model_config_json=args.dir_model_config_json or None,
        dir_model_weights=args.dir_model_weights,
        thresholds_by_horizon=prediction_inputs.thresholds_by_horizon,
        prepared_override=prepared_override,
        trend_ignition=getattr(args, "trend_ignition", None),
        direction_only_fallback=getattr(args, "direction_only_fallback", None),
        adaptive_thresholds=getattr(args, "adaptive_thresholds", None),
        target_range_models=getattr(args, "target_range_models", None),
        platt_calibration=prediction_inputs.platt_calibration,
        abstention_policy=getattr(args, "abstention_policy", None),
        uncertainty_policy=getattr(args, "uncertainty_policy", None),
        trade_decision_policy=getattr(args, "trade_decision_policy", None),
        regime_model_weights=getattr(args, "regime_model_weights", None),
        regime_model_dirs=getattr(args, "regime_model_dirs", None),
        confluence_policy=getattr(args, "confluence_policy", None),
        execution_policy=getattr(args, "execution_policy", None),
        forecast_coherence_policy=getattr(args, "forecast_coherence_policy", None),
        direction_output_policy=prediction_inputs.direction_output_cfg,
        direction_ensemble_policy=getattr(args, "direction_ensemble_policy", None),
        trust_hardening_policy=getattr(args, "trust_hardening_policy", None),
        latest_close=latest_close,
        confidence_min=float(getattr(args, "confidence_min", legacy.CONFIDENCE_MIN_DEFAULT)),
        confidence_min_by_horizon_regime=getattr(args, "confidence_min_by_horizon_regime", None),
        position_size_floor=float(getattr(args, "position_size_floor", legacy.POSITION_SIZE_FLOOR_DEFAULT)),
        position_size_cap=float(getattr(args, "position_size_cap", legacy.POSITION_SIZE_CAP_DEFAULT)),
        position_size_cap_by_horizon=getattr(args, "position_size_cap_by_horizon", None),
        disabled_horizons=getattr(args, "disabled_horizons", None),
    )
    store.append_event(run_paths, stage="prediction", status="completed", details={"horizons": list(summary.keys())})
    return summary


def _persist_outputs(
    args: argparse.Namespace,
    mode: RuntimeMode,
    summary: Mapping[str, Mapping[str, Any]],
    run_paths: RuntimeRunPaths,
    store: RuntimeStateStore,
) -> PipelineExecutionResult:
    store.append_event(run_paths, stage="artifact_writing", status="started")
    predictions_payload = runtime_write_prediction_summary(
        summary,
        degradation_policy=getattr(args, "degradation_monitoring", None),
        latest_prediction_path=legacy.LATEST_PREDICTION_PATH,
        history_prediction_path=legacy.HISTORY_PREDICTION_PATH,
        build_prompt_ready_summary_fn=lambda payload: build_runtime_prompt_ready_summary(
            payload,
            horizon_sort_key=horizon_sort_key,
        ),
        build_blocked_trade_analytics_fn=build_blocked_trade_analytics,
        build_degradation_monitoring_fn=lambda history, policy: build_runtime_degradation_monitoring(
            history,
            policy,
            horizon_sort_key=horizon_sort_key,
        ),
        print_fn=print,
    )
    store.write_predictions(run_paths, predictions_payload)

    monitoring_payload: dict[str, Any] | None = None
    if not args.disable_monitoring_latest:
        monitoring_payload = runtime_write_monitoring_artifact(
            predictions_payload,
            args,
            output_path=legacy.MONITORING_LATEST_PATH,
            horizon_sort_key=horizon_sort_key,
            format_horizon_label=format_horizon_label,
            confidence_min_default=legacy.CONFIDENCE_MIN_DEFAULT,
            position_size_floor_default=legacy.POSITION_SIZE_FLOOR_DEFAULT,
            position_size_cap_default=legacy.POSITION_SIZE_CAP_DEFAULT,
        )
        store.write_monitoring(run_paths, monitoring_payload)

    trade_ready_payload: dict[str, Any] | None = None
    if args.write_artifacts:
        trade_ready_payload = runtime_write_monitoring_artifact(
            predictions_payload,
            args,
            output_path=legacy.TRADE_READY_MONITOR_PATH,
            horizon_sort_key=horizon_sort_key,
            format_horizon_label=format_horizon_label,
            confidence_min_default=legacy.CONFIDENCE_MIN_DEFAULT,
            position_size_floor_default=legacy.POSITION_SIZE_FLOOR_DEFAULT,
            position_size_cap_default=legacy.POSITION_SIZE_CAP_DEFAULT,
            payload=monitoring_payload,
        )
        store.write_trade_ready(run_paths, trade_ready_payload)
        runtime_refresh_meta_baseline(
            source_csv=legacy.META_BASELINE_SOURCE_CSV,
            json_path=legacy.META_BASELINE_JSON_PATH,
            parquet_path=legacy.META_BASELINE_PARQUET_PATH,
            load_dataframe=legacy.load_dataframe,
            compute_baseline=legacy.compute_baseline,
            baseline_to_dataframe=legacy.baseline_to_dataframe,
            append_detected_meta_columns=_append_detected_meta_columns,
            default_columns=list(legacy.BASELINE_DEFAULT_COLUMNS),
            stderr_write=legacy.sys.stderr.write,
        )

    prompt_summary = predictions_payload.get("prompt_ready_summary", {}) if isinstance(predictions_payload, Mapping) else {}
    market_outlook = prompt_summary.get("market_outlook_strategy", {}) if isinstance(prompt_summary, Mapping) else {}
    store.finalize(
        run_paths,
        mode=mode,
        status="succeeded",
        summary={
            "selected_direction": market_outlook.get("selected_direction"),
            "preferred_horizon": market_outlook.get("preferred_horizon"),
            "tradeable": market_outlook.get("tradeable"),
        },
    )
    store.append_event(run_paths, stage="artifact_writing", status="completed")
    return PipelineExecutionResult(
        run_id=run_paths.run_id,
        mode=mode,
        run_root=run_paths.root,
        predictions_payload=predictions_payload,
        monitoring_payload=monitoring_payload,
        trade_ready_payload=trade_ready_payload,
        metadata={"summary_path": run_paths.summary_path.as_posix()},
    )
