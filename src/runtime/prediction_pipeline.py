from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from src.runtime.trade_decision_support import apply_derivatives_shadow_adjustment
from src.trading.signals import DEFAULT_RESIDUAL_STD, PreparedData, compute_signal_for_index, load_models, load_residual_std_from_dataset, populate_sequence_cache_from_prepared
from src.trading.volatility import latest_volatility_snapshot


PreparedBundle = tuple[PreparedData, int, float, str]
SummaryPayload = Dict[str, Dict[str, Any]]


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _prediction_timestamp_is_stale(
    signal_ts: str | None,
    reference_ts: str | None,
    *,
    horizon_hours: float,
    min_allowed_lag_hours: float = 24.0,
    horizon_lag_multiplier: float = 3.0,
) -> bool:
    signal_dt = _parse_iso_timestamp(signal_ts)
    reference_dt = _parse_iso_timestamp(reference_ts)
    if signal_dt is None or reference_dt is None:
        return False
    lag_hours = (reference_dt - signal_dt).total_seconds() / 3600.0
    if lag_hours <= 0.0:
        return False
    allowed_lag_hours = max(float(min_allowed_lag_hours), float(horizon_hours) * float(horizon_lag_multiplier))
    return lag_hours > allowed_lag_hours


def _active_direction_model_names(direction_configs: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for entry in direction_configs:
        for raw_name in (entry.get("name"), entry.get("type")):
            if raw_name is None:
                continue
            normalized = str(raw_name).strip().lower()
            if normalized and normalized not in ordered:
                ordered.append(normalized)
    return ordered


def _filter_direction_weight_map_to_active_models(
    weight_map: Mapping[str, float],
    *,
    active_model_names: Sequence[str],
) -> Dict[str, float]:
    active = {str(name).strip().lower() for name in active_model_names}
    return {
        str(name): float(value)
        for name, value in weight_map.items()
        if str(name).strip().lower() in active
    }


def _scope_direction_policy_to_active_models(
    policy: Mapping[str, Any],
    *,
    active_model_names: Sequence[str],
) -> Dict[str, Any]:
    scoped = dict(policy)
    active = {str(name).strip().lower() for name in active_model_names}
    model_groups = {
        str(name): str(group)
        for name, group in dict(policy.get("model_groups") or {}).items()
        if str(name).strip().lower() in active
    }
    active_groups = {str(group) for group in model_groups.values()}
    priority_order: list[str] = []
    for raw_name in list(policy.get("priority_order") or []):
        normalized = str(raw_name).strip().lower()
        if normalized in active and normalized not in priority_order:
            priority_order.append(normalized)
    for raw_name in active_model_names:
        normalized = str(raw_name).strip().lower()
        if normalized not in priority_order:
            priority_order.append(normalized)
    scoped["model_groups"] = model_groups
    scoped["priority_order"] = priority_order
    scoped["preferred_groups"] = [
        str(group)
        for group in list(policy.get("preferred_groups") or [])
        if str(group) in active_groups
    ]
    scoped["max_models_per_group"] = {
        str(group): int(limit)
        for group, limit in dict(policy.get("max_models_per_group") or {}).items()
        if str(group) in active_groups
    }
    return scoped


@dataclass(frozen=True)
class PredictionPipelineConfig:
    targets: Iterable[float]
    p_up_min: float
    ret_min: float
    direction_threshold: float
    auto_direction_threshold: bool
    offline: bool
    dir_lstm_path: str | None
    dir_bilstm_path: str | None
    dir_gru_path: str | None
    dir_cnn_lstm_path: str | None
    dir_cnn_bilstm_path: str | None
    dir_garch_lstm_path: str | None
    dir_transformer_path: str | None
    dir_model_config_json: str | None
    dir_model_weights: str | None
    thresholds_by_horizon: Mapping[int | float | str, Dict[str, float]] | None
    prepared_override: PreparedBundle | None
    trend_ignition: Mapping[str, Any] | None
    direction_only_fallback: Mapping[str, Any] | None
    adaptive_thresholds: Mapping[str, Any] | None
    target_range_models: Mapping[str, Any] | None
    platt_calibration: Mapping[str, Mapping[str, Any]] | None
    abstention_policy: Mapping[str, Any] | None
    uncertainty_policy: Mapping[str, Any] | None
    trade_decision_policy: Mapping[str, Any] | None
    regime_model_weights: Mapping[str, Any] | None
    regime_model_dirs: Mapping[str, Any] | None
    regression_model_dirs: Mapping[str, Any] | None
    confluence_policy: Mapping[str, Any] | None
    execution_policy: Mapping[str, Any] | None
    forecast_coherence_policy: Mapping[str, Any] | None
    direction_output_policy: Mapping[str, Any] | None
    direction_ensemble_policy: Mapping[str, Any] | None
    trust_hardening_policy: Mapping[str, Any] | None
    latest_close: float | None
    confidence_min: float
    confidence_min_by_horizon_regime: Mapping[float | int | str, Mapping[str, float]] | None
    position_size_floor: float
    position_size_cap: float
    position_size_cap_by_horizon: Mapping[float | int | str, float] | None
    disabled_horizons: Sequence[float] | None


@dataclass(frozen=True)
class PredictionPipelineDependencies:
    normalize_horizon_value: Callable[[float | int | str], float]
    normalize_threshold_overrides: Callable[[Mapping[int | float | str, Dict[str, float]] | None], Dict[float, Dict[str, float]]]
    normalize_horizon_regime_float_map: Callable[..., Dict[float, Dict[str, float]]]
    normalize_horizon_float_map: Callable[..., Dict[float, float]]
    dataset_profile_for_horizon: Callable[[float], Any]
    select_dataset_candidate: Callable[[Any], tuple[Any, bool]]
    prepare_base_direction_configs: Callable[..., list[Any]]
    load_prepared: Callable[..., PreparedBundle]
    resolve_trend_ignition_payload: Callable[[Mapping[str, Any] | None], Any]
    resolve_direction_fallback_policy: Callable[[Mapping[str, Any] | None], Any]
    resolve_adaptive_thresholds_policy: Callable[[Mapping[str, Any] | None], Any]
    resolve_target_range_policy: Callable[[Mapping[str, Any] | None], Any]
    resolve_abstention_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_uncertainty_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_trade_decision_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_regime_model_weights_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any] | None]
    resolve_regime_model_dirs_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_regression_model_dirs_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_confluence_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_execution_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_forecast_coherence_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_direction_output_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_direction_ensemble_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    resolve_trust_hardening_policy: Callable[[Mapping[str, Any] | None], Dict[str, Any]]
    compute_breakout_scores: Callable[[Mapping[str, PreparedBundle], Mapping[str, Mapping[str, Any]]], Dict[str, float]]
    load_target_range_models: Callable[[Mapping[str, Any], Sequence[float]], Dict[float, Dict[str, Any]]]
    format_horizon_label: Callable[[float], str]
    model_paths_for_horizon: Callable[[float], tuple[Any, Any]]
    classify_regime_from_score: Callable[[float, Mapping[str, Any]], str]
    resolve_regime_specific_dir_path: Callable[..., Any]
    resolve_regression_dir_path: Callable[..., Any]
    direction_configs_for_horizon: Callable[..., tuple[list[Any], Dict[str, float]]]
    resolve_thresholds_for_horizon: Callable[[float, float, float, Mapping[float, Dict[str, float]] | None], Dict[str, float]]
    apply_adaptive_thresholds: Callable[[Mapping[str, Any], float, float, str], tuple[float, float, float]]
    apply_regime_weight_overrides: Callable[..., Dict[str, float]]
    scope_direction_ensemble_policy: Callable[[Mapping[str, Any], float], Dict[str, Any]]
    resolve_direction_threshold_for_horizon: Callable[..., float]
    project_price: Callable[[float, float], float]
    resolve_trade_probability_for_horizon: Callable[..., tuple[float, str | None, bool, Mapping[str, Any] | None]]
    resolve_direction_signal_for_horizon: Callable[..., int]
    compute_directional_stop_take_prices: Callable[..., tuple[float, float]]
    resolve_confidence_min_for_horizon: Callable[..., tuple[float, str]]
    lookup_horizon_value: Callable[[Mapping[float, float] | None, float, float], float]
    compute_position_size: Callable[..., float]
    parse_iso_timestamp: Callable[[str], Any]
    predict_target_range_prices: Callable[..., Dict[str, float]]
    build_prediction_result: Callable[..., tuple[Dict[str, Any], bool]]
    get_active_regime_weight_override: Callable[..., Dict[str, float] | None]
    derive_probability_alignment_features: Callable[..., Mapping[str, Any]]
    build_direction_output: Callable[..., Mapping[str, Any]]
    apply_target_range_overrides: Callable[..., tuple[Mapping[str, Any], float, float]]
    evaluate_direction_only_fallback: Callable[..., tuple[Mapping[str, Any], bool]]
    finite_float_or_none: Callable[[Any], float | None]
    coerce_row_value: Callable[[Any], float | None]
    write_trend_ignition_state: Callable[[str], None]
    write_direction_fallback_state: Callable[[str], None]
    apply_post_prediction_policies: Callable[..., SummaryPayload]
    apply_forecast_coherence_policy: Callable[..., SummaryPayload]
    apply_trust_hardening_stage: Callable[..., SummaryPayload]
    apply_confluence_policy: Callable[..., SummaryPayload]
    apply_trade_decision_stage: Callable[..., SummaryPayload]
    apply_post_trade_gates: Callable[..., SummaryPayload]
    apply_execution_policy: Callable[..., SummaryPayload]
    build_stub_summary: Callable[..., SummaryPayload]
    stderr_write: Callable[[str], None]
    regime_neutral: str
    target_range_default_confidence_scale: float


@dataclass(frozen=True)
class PredictionPreparationState:
    normalized_targets: list[float]
    normalized_threshold_overrides: Mapping[float, Dict[str, float]]
    trend_payload: Mapping[str, Any] | None
    direction_fallback_policy: Mapping[str, Any] | None
    adaptive_policy: Mapping[str, Any] | None
    target_range_policy: Mapping[str, Any] | None
    abstention_policy_resolved: Mapping[str, Any]
    uncertainty_policy_resolved: Mapping[str, Any]
    trade_decision_policy_resolved: Mapping[str, Any]
    regime_weight_policy: Mapping[str, Any] | None
    regime_model_dirs_policy: Mapping[str, Any]
    regression_model_dirs_policy: Mapping[str, Any]
    confluence_policy_resolved: Mapping[str, Any]
    execution_policy_resolved: Mapping[str, Any]
    forecast_coherence_policy_resolved: Mapping[str, Any]
    direction_output_policy_resolved: Mapping[str, Any]
    direction_ensemble_policy_resolved: Mapping[str, Any]
    trust_hardening_policy_resolved: Mapping[str, Any]
    confidence_min: float
    confidence_min_by_horizon_regime_resolved: Mapping[float, Dict[str, float]]
    position_size_floor: float
    position_size_cap: float
    position_size_cap_by_horizon_resolved: Mapping[float, float]
    resolved_profiles: Mapping[str, Any]
    target_profiles: Mapping[float, str]
    horizons_by_profile: Mapping[str, Sequence[float]]
    base_direction_configs: Sequence[Any]
    prepared_bundles: Mapping[str, PreparedBundle]
    volatility_snapshots: Mapping[str, Mapping[str, Any]]
    breakout_scores: Mapping[str, float]
    target_range_bundles: Mapping[float, Dict[str, Any]]
    residual_std_by_horizon: Mapping[float, float]
    stub_close: float
    stub_ts: str


def _apply_derivatives_shadow_probability_adjustment(
    *,
    probability: float,
    close: float,
    row_features: Any,
    horizon_label: str,
    regime_state: str,
    ret_pred: float,
    signal_dir_only: int,
    trade_decision_policy: Mapping[str, Any] | None,
    coerce_row_value: Callable[[Any], float | None],
) -> tuple[float, Dict[str, Any]]:
    shadow_result = {"close": float(close)}
    for field in ("funding_rate_zscore_24h", "fut_close", "fut_close_zscore_7h", "open_interest"):
        if hasattr(row_features, "index") and field in row_features.index:
            value = coerce_row_value(row_features.get(field))
            if value is not None:
                shadow_result[field] = float(value)
    return apply_derivatives_shadow_adjustment(
        float(probability),
        result=shadow_result,
        horizon_label=horizon_label,
        regime_state=regime_state,
        ret_pred=float(ret_pred),
        signal_dir_only=int(signal_dir_only),
        policy=trade_decision_policy or {},
        finite_float_or_none=lambda value: None if value is None else float(value),
    )
    horizons_by_profile: Mapping[str, list[float]]
    base_direction_configs: list[Any]
    prepared_bundles: Mapping[str, PreparedBundle]
    volatility_snapshots: Mapping[str, Mapping[str, Any]]
    breakout_scores: Mapping[str, float]
    target_range_bundles: Mapping[float, Dict[str, Any]]
    residual_std_by_horizon: Mapping[float, float]
    stub_close: float
    stub_ts: str


def execute_prediction_pipeline(
    config: PredictionPipelineConfig,
    deps: PredictionPipelineDependencies,
) -> SummaryPayload:
    state = _prepare_prediction_state(config, deps)
    summary, execution_contexts, pending_trend_ts, pending_direction_fallback_ts = _build_prediction_summary(
        config,
        deps,
        state,
    )
    if state.trend_payload and pending_trend_ts:
        deps.write_trend_ignition_state(pending_trend_ts)
    if state.direction_fallback_policy and pending_direction_fallback_ts:
        deps.write_direction_fallback_state(pending_direction_fallback_ts)

    summary = deps.apply_post_prediction_policies(
        summary,
        execution_contexts,
        forecast_coherence_policy=state.forecast_coherence_policy_resolved,
        confluence_policy=state.confluence_policy_resolved,
        trust_hardening_policy=state.trust_hardening_policy_resolved,
        trade_decision_policy=state.trade_decision_policy_resolved,
        confidence_min=state.confidence_min,
        abstention_policy=state.abstention_policy_resolved,
        uncertainty_policy=state.uncertainty_policy_resolved,
        execution_policy=state.execution_policy_resolved,
        apply_forecast_coherence_policy=deps.apply_forecast_coherence_policy,
        apply_trust_hardening_stage=deps.apply_trust_hardening_stage,
        apply_confluence_policy=deps.apply_confluence_policy,
        apply_trade_decision_stage=deps.apply_trade_decision_stage,
        apply_post_trade_gates=lambda payload, threshold, abstention_cfg, uncertainty_cfg: deps.apply_post_trade_gates(
            payload,
            confidence_min=threshold,
            abstention_policy=abstention_cfg,
            uncertainty_policy=uncertainty_cfg,
        ),
        apply_execution_policy=deps.apply_execution_policy,
    )

    if not summary:
        if config.offline:
            return deps.build_stub_summary(
                config.targets,
                config.p_up_min,
                config.ret_min,
                close=state.stub_close,
                ts_iso=state.stub_ts,
                thresholds_by_horizon=config.thresholds_by_horizon,
            )
        raise RuntimeError("No predictions were produced; ensure model artifacts exist.")
    return summary


def _prepare_prediction_state(
    config: PredictionPipelineConfig,
    deps: PredictionPipelineDependencies,
) -> PredictionPreparationState:
    normalized_targets = sorted({deps.normalize_horizon_value(h) for h in config.targets})
    disabled_horizon_set = {deps.normalize_horizon_value(h) for h in (config.disabled_horizons or [])}
    if disabled_horizon_set:
        normalized_targets = [h for h in normalized_targets if h not in disabled_horizon_set]
    if not normalized_targets:
        return PredictionPreparationState(
            normalized_targets=[],
            normalized_threshold_overrides={},
            trend_payload=None,
            direction_fallback_policy=None,
            adaptive_policy=None,
            target_range_policy=None,
            abstention_policy_resolved={},
            uncertainty_policy_resolved={},
            trade_decision_policy_resolved={},
            regime_weight_policy=None,
            regime_model_dirs_policy={},
            regression_model_dirs_policy={},
            confluence_policy_resolved={},
            execution_policy_resolved={},
            forecast_coherence_policy_resolved={},
            direction_output_policy_resolved={},
            direction_ensemble_policy_resolved={},
            trust_hardening_policy_resolved={},
            confidence_min=0.0,
            confidence_min_by_horizon_regime_resolved={},
            position_size_floor=0.0,
            position_size_cap=0.0,
            position_size_cap_by_horizon_resolved={},
            resolved_profiles={},
            target_profiles={},
            horizons_by_profile={},
            base_direction_configs=[],
            prepared_bundles={},
            volatility_snapshots={},
            breakout_scores={},
            target_range_bundles={},
            residual_std_by_horizon={},
            stub_close=0.0,
            stub_ts=datetime.now(timezone.utc).isoformat(),
        )

    normalized_threshold_overrides = deps.normalize_threshold_overrides(config.thresholds_by_horizon)
    trend_payload = deps.resolve_trend_ignition_payload(config.trend_ignition)
    direction_fallback_policy = deps.resolve_direction_fallback_policy(config.direction_only_fallback)
    adaptive_policy = deps.resolve_adaptive_thresholds_policy(config.adaptive_thresholds)
    target_range_policy = deps.resolve_target_range_policy(config.target_range_models)
    abstention_policy_resolved = deps.resolve_abstention_policy(config.abstention_policy)
    uncertainty_policy_resolved = deps.resolve_uncertainty_policy(config.uncertainty_policy)
    trade_decision_policy_resolved = deps.resolve_trade_decision_policy(config.trade_decision_policy)
    regime_weight_policy = deps.resolve_regime_model_weights_policy(config.regime_model_weights)
    regime_model_dirs_policy = deps.resolve_regime_model_dirs_policy(config.regime_model_dirs)
    regression_model_dirs_policy = deps.resolve_regression_model_dirs_policy(config.regression_model_dirs)
    confluence_policy_resolved = deps.resolve_confluence_policy(config.confluence_policy)
    execution_policy_resolved = deps.resolve_execution_policy(config.execution_policy)
    forecast_coherence_policy_resolved = deps.resolve_forecast_coherence_policy(config.forecast_coherence_policy)
    direction_output_policy_resolved = deps.resolve_direction_output_policy(config.direction_output_policy)
    direction_ensemble_policy_resolved = deps.resolve_direction_ensemble_policy(config.direction_ensemble_policy)
    trust_hardening_policy_resolved = deps.resolve_trust_hardening_policy(config.trust_hardening_policy)
    confidence_min = max(0.0, min(1.0, float(config.confidence_min)))
    confidence_min_by_horizon_regime_resolved = deps.normalize_horizon_regime_float_map(
        config.confidence_min_by_horizon_regime,
        minimum=0.0,
        maximum=1.0,
    )
    position_size_floor = max(0.0, float(config.position_size_floor))
    position_size_cap = max(position_size_floor, float(config.position_size_cap))
    position_size_cap_by_horizon_resolved = deps.normalize_horizon_float_map(
        config.position_size_cap_by_horizon,
        minimum=position_size_floor,
        maximum=position_size_cap,
    )

    target_profiles: Dict[float, str] = {}
    horizons_by_profile: Dict[str, list[float]] = defaultdict(list)
    profiles: Dict[str, Any] = {}
    for horizon in normalized_targets:
        profile = deps.dataset_profile_for_horizon(horizon)
        profiles.setdefault(profile.key, profile)
        target_profiles[horizon] = profile.key
        horizons_by_profile[profile.key].append(horizon)

    resolved_profiles: Dict[str, Any] = {}
    for key, profile in profiles.items():
        candidate, used_fallback = deps.select_dataset_candidate(profile)
        resolved_profiles[key] = candidate
        if used_fallback:
            deps.stderr_write(
                f"Info: using {candidate.path.name} for {key} horizon group (fallback dataset).\n"
            )

    base_direction_configs = deps.prepare_base_direction_configs(
        config_json_path=config.dir_model_config_json,
        weight_spec=config.dir_model_weights,
        dir_lstm_path=config.dir_lstm_path,
        dir_bilstm_path=config.dir_bilstm_path,
        dir_gru_path=config.dir_gru_path,
        dir_cnn_lstm_path=config.dir_cnn_lstm_path,
        dir_cnn_bilstm_path=config.dir_cnn_bilstm_path,
        dir_garch_lstm_path=config.dir_garch_lstm_path,
        dir_transformer_path=config.dir_transformer_path,
    )

    prepared_bundles, volatility_snapshots, stub_close, stub_ts = _load_prepared_bundles(
        config,
        deps,
        resolved_profiles,
    )

    breakout_scores: Dict[str, float] = {}
    if adaptive_policy and adaptive_policy.get("enabled"):
        breakout_scores = deps.compute_breakout_scores(prepared_bundles, volatility_snapshots)

    target_range_bundles: Dict[float, Dict[str, Any]] = {}
    if target_range_policy and target_range_policy.get("enabled"):
        target_range_bundles = deps.load_target_range_models(target_range_policy, normalized_targets)

    residual_std_by_horizon = _load_residual_stds(
        resolved_profiles,
        horizons_by_profile,
        deps,
    )

    return PredictionPreparationState(
        normalized_targets=normalized_targets,
        normalized_threshold_overrides=normalized_threshold_overrides,
        trend_payload=trend_payload,
        direction_fallback_policy=direction_fallback_policy,
        adaptive_policy=adaptive_policy,
        target_range_policy=target_range_policy,
        abstention_policy_resolved=abstention_policy_resolved,
        uncertainty_policy_resolved=uncertainty_policy_resolved,
        trade_decision_policy_resolved=trade_decision_policy_resolved,
        regime_weight_policy=regime_weight_policy,
        regime_model_dirs_policy=regime_model_dirs_policy,
        regression_model_dirs_policy=regression_model_dirs_policy,
        confluence_policy_resolved=confluence_policy_resolved,
        execution_policy_resolved=execution_policy_resolved,
        forecast_coherence_policy_resolved=forecast_coherence_policy_resolved,
        direction_output_policy_resolved=direction_output_policy_resolved,
        direction_ensemble_policy_resolved=direction_ensemble_policy_resolved,
        trust_hardening_policy_resolved=trust_hardening_policy_resolved,
        confidence_min=confidence_min,
        confidence_min_by_horizon_regime_resolved=confidence_min_by_horizon_regime_resolved,
        position_size_floor=position_size_floor,
        position_size_cap=position_size_cap,
        position_size_cap_by_horizon_resolved=position_size_cap_by_horizon_resolved,
        resolved_profiles=resolved_profiles,
        target_profiles=target_profiles,
        horizons_by_profile=horizons_by_profile,
        base_direction_configs=base_direction_configs,
        prepared_bundles=prepared_bundles,
        volatility_snapshots=volatility_snapshots,
        breakout_scores=breakout_scores,
        target_range_bundles=target_range_bundles,
        residual_std_by_horizon=residual_std_by_horizon,
        stub_close=stub_close,
        stub_ts=stub_ts,
    )


def _load_prepared_bundles(
    config: PredictionPipelineConfig,
    deps: PredictionPipelineDependencies,
    resolved_profiles: Mapping[str, Any],
) -> tuple[Dict[str, PreparedBundle], Dict[str, Dict[str, Any]], float, str]:
    prepared_bundles: Dict[str, PreparedBundle] = {}
    volatility_snapshots: Dict[str, Dict[str, Any]] = {}
    stub_close = 0.0
    stub_ts = datetime.now(timezone.utc).isoformat()
    for key, candidate in resolved_profiles.items():
        if config.prepared_override is not None and not candidate.offline_only:
            bundle = config.prepared_override
        else:
            dataset_path = candidate.path
            if not dataset_path.exists():
                if config.offline:
                    deps.stderr_write(
                        f"Dry run: dataset not found for {key} group (expected {dataset_path}).\n"
                    )
                    continue
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            bundle = deps.load_prepared(
                dataset_path,
                target_column=candidate.target_column,
                offline=config.offline or candidate.offline_only,
            )
        prepared_bundles[key] = bundle
        prepared, index, close_snapshot, ts_snapshot = bundle
        stub_close = close_snapshot
        stub_ts = ts_snapshot
        volatility_snapshots[key] = latest_volatility_snapshot(
            prepared.df_all,
            prepared.volatility_columns or [],
            index=index,
        )
    return prepared_bundles, volatility_snapshots, stub_close, stub_ts


def _load_residual_stds(
    resolved_profiles: Mapping[str, Any],
    horizons_by_profile: Mapping[str, list[float]],
    deps: PredictionPipelineDependencies,
) -> Dict[float, float]:
    residual_std_by_horizon: Dict[float, float] = {}
    for key, candidate in resolved_profiles.items():
        horizons = horizons_by_profile.get(key, [])
        if not horizons:
            continue
        dataset_path = candidate.path
        if not dataset_path.exists():
            continue
        try:
            residuals = load_residual_std_from_dataset(
                str(dataset_path),
                horizons,
                base_horizon=candidate.base_horizon,
            )
            residual_std_by_horizon.update(residuals)
        except FileNotFoundError:
            deps.stderr_write(
                f"Warning: residual std dataset missing at {dataset_path}; using default {DEFAULT_RESIDUAL_STD:.4f}.\n"
            )
            for horizon in horizons:
                residual_std_by_horizon[horizon] = DEFAULT_RESIDUAL_STD
    return residual_std_by_horizon


def _build_prediction_summary(
    config: PredictionPipelineConfig,
    deps: PredictionPipelineDependencies,
    state: PredictionPreparationState,
) -> tuple[SummaryPayload, Dict[str, Dict[str, Any]], Optional[str], Optional[str]]:
    summary: SummaryPayload = {}
    execution_contexts: Dict[str, Dict[str, Any]] = {}
    pending_trend_ts: Optional[str] = None
    pending_direction_fallback_ts: Optional[str] = None
    prepared_timestamps = [bundle[3] for bundle in state.prepared_bundles.values() if len(bundle) >= 4]
    freshest_prepared_ts = max(
        prepared_timestamps,
        key=lambda value: _parse_iso_timestamp(value) or datetime.min.replace(tzinfo=timezone.utc),
        default=state.stub_ts,
    )

    for horizon in state.normalized_targets:
        profile_key = state.target_profiles.get(horizon)
        if profile_key is None:
            continue
        if profile_key not in state.prepared_bundles:
            deps.stderr_write(
                f"Warning: skipping {deps.format_horizon_label(horizon)} horizon because prepared data is missing.\n"
            )
            continue

        candidate = state.resolved_profiles[profile_key]
        prepared, index, close, ts_iso = state.prepared_bundles[profile_key]
        if _prediction_timestamp_is_stale(ts_iso, freshest_prepared_ts, horizon_hours=float(horizon)):
            deps.stderr_write(
                f"Warning: skipping {deps.format_horizon_label(horizon)} horizon because prepared timestamp {ts_iso} is stale versus freshest bundle {freshest_prepared_ts}.\n"
            )
            continue
        volatility_snapshot = state.volatility_snapshots.get(profile_key, {})
        row_features = prepared.df_all.iloc[index]
        label = deps.format_horizon_label(horizon)
        reg_path, dir_path_default = deps.model_paths_for_horizon(horizon)
        if not reg_path.exists() or not dir_path_default.exists():
            deps.stderr_write(
                f"Warning: skipping {label} horizon because model files are missing\n"
            )
            continue
        reg_path = deps.resolve_regression_dir_path(
            reg_path,
            horizon_label=label,
            policy=state.regression_model_dirs_policy,
        )

        regime_state = deps.regime_neutral
        regime_score = None
        adaptive_scale = 1.0
        if state.adaptive_policy and state.adaptive_policy.get("enabled"):
            profile_score = state.breakout_scores.get(profile_key)
            if profile_score is not None:
                regime_score = profile_score
                regime_state = deps.classify_regime_from_score(profile_score, state.adaptive_policy)

        dir_path = deps.resolve_regime_specific_dir_path(
            dir_path_default,
            regime_state=regime_state,
            horizon_label=label,
            policy=state.regime_model_dirs_policy,
        )
        direction_configs, base_dir_weight_map = deps.direction_configs_for_horizon(
            state.base_direction_configs,
            dir_model_path=str(dir_path),
            horizon=horizon,
            horizon_label=label,
        )
        models = load_models(str(reg_path), direction_model_configs=direction_configs)
        if state.trend_payload:
            models["trend_ignition"] = state.trend_payload
        populate_sequence_cache_from_prepared(prepared, models)

        horizon_thresholds = deps.resolve_thresholds_for_horizon(
            horizon,
            config.p_up_min,
            config.ret_min,
            state.normalized_threshold_overrides,
        )
        horizon_p_up = horizon_thresholds["p_up_min"]
        horizon_ret = horizon_thresholds["ret_min"]
        if state.adaptive_policy and state.adaptive_policy.get("enabled"):
            profile_score = state.breakout_scores.get(profile_key)
            if profile_score is not None:
                horizon_p_up, horizon_ret, adaptive_scale = deps.apply_adaptive_thresholds(
                    state.adaptive_policy,
                    horizon_p_up,
                    horizon_ret,
                    regime_state,
                )

        dir_weight_map = deps.apply_regime_weight_overrides(
            base_dir_weight_map,
            regime_state=regime_state,
            horizon=horizon,
            policy=state.regime_weight_policy,
        )
        active_model_names = _active_direction_model_names(direction_configs)
        dir_weight_map = _filter_direction_weight_map_to_active_models(
            dir_weight_map,
            active_model_names=active_model_names,
        )
        scoped_direction_ensemble_policy = deps.scope_direction_ensemble_policy(
            state.direction_ensemble_policy_resolved,
            horizon,
        )
        scoped_direction_ensemble_policy = _scope_direction_policy_to_active_models(
            scoped_direction_ensemble_policy,
            active_model_names=active_model_names,
        )
        signal = compute_signal_for_index(
            prepared=prepared,
            index=index,
            models=models,
            p_up_min=horizon_p_up,
            ret_min=horizon_ret,
            horizon=horizon,
            dir_model_weights=dir_weight_map,
            direction_ensemble_policy=scoped_direction_ensemble_policy,
            volatility_snapshot=volatility_snapshot,
            volatility_policy=horizon_thresholds,
            p_up_calibration=None,
        )

        p_val = _safe_float(signal.get("p_up", 0.0))
        thresh = deps.resolve_direction_threshold_for_horizon(
            direction_threshold=float(config.direction_threshold),
            auto_direction_threshold=bool(config.auto_direction_threshold),
            horizon_p_up=float(horizon_p_up),
        )
        signal["signal_dir_only"] = int(p_val >= thresh)
        if config.latest_close is not None:
            signal["close"] = config.latest_close
            close = config.latest_close

        ret_pred = float(signal.get("ret_pred", 0.0))
        raw_p_up = float(signal.get("p_up", 0.0))
        p_up = float(raw_p_up)
        signal_ts = str(signal.get("ts", ts_iso))
        signal_dir_only = int(signal.get("signal_dir_only", 0))
        signal_ensemble = int(signal.get("signal_ensemble", 0))
        residual_std = float(state.residual_std_by_horizon.get(horizon, DEFAULT_RESIDUAL_STD))
        expected_value = _compute_expected_value(p_up, ret_pred, residual_std, horizon_thresholds)
        confidence_score = deps.compute_position_size.__self__ if False else _compute_confidence_score(p_up, expected_value, residual_std)
        calibration_key = None
        calibration_used_regime_key = False
        probability_guard = None

        projected_price = deps.project_price(close, ret_pred)
        if config.platt_calibration:
            p_up, calibration_key, calibration_used_regime_key, probability_guard = deps.resolve_trade_probability_for_horizon(
                platt_calibration=config.platt_calibration,
                label=label,
                regime_state=regime_state,
                raw_probability=raw_p_up,
                close=close,
                projected_price=projected_price,
                ret_pred=ret_pred,
            )

        p_up, derivatives_shadow_adjustment = _apply_derivatives_shadow_probability_adjustment(
            probability=p_up,
            close=close,
            row_features=row_features,
            horizon_label=label,
            regime_state=regime_state,
            ret_pred=ret_pred,
            signal_dir_only=signal_dir_only,
            trade_decision_policy=state.trade_decision_policy_resolved,
            coerce_row_value=deps.coerce_row_value,
        )
        signal["derivatives_shadow_adjustment"] = derivatives_shadow_adjustment
        signal_dir_only = deps.resolve_direction_signal_for_horizon(
            raw_probability=raw_p_up,
            calibrated_probability=p_up,
            threshold=thresh,
            close=close,
            projected_price=projected_price,
            ret_pred=ret_pred,
            calibration_key=calibration_key,
            calibration_used_regime_key=calibration_used_regime_key,
        )
        signal_ensemble = int((p_up >= horizon_p_up) and (ret_pred >= horizon_ret) and (not bool(signal.get("volatility_flag"))))
        expected_value = _compute_expected_value(p_up, ret_pred, residual_std, horizon_thresholds)
        confidence_score = _compute_confidence_score(p_up, expected_value, residual_std)

        stop_loss_price, take_profit_price = deps.compute_directional_stop_take_prices(
            close=close,
            ret_pred=ret_pred,
            residual_std=residual_std,
            direction_signal=signal_dir_only,
        )
        effective_confidence_min, confidence_min_source = deps.resolve_confidence_min_for_horizon(
            state.confidence_min,
            state.confidence_min_by_horizon_regime_resolved,
            horizon=horizon,
            regime_state=regime_state,
        )
        effective_position_size_cap = deps.lookup_horizon_value(
            state.position_size_cap_by_horizon_resolved,
            horizon,
            state.position_size_cap,
        )
        position_size = deps.compute_position_size(
            confidence_score,
            confidence_min=effective_confidence_min,
            size_floor=state.position_size_floor,
            size_cap=effective_position_size_cap,
        )

        trend_prob, ignition_state, cooldown_active, pending_trend_ts = _resolve_trend_state(
            signal=signal,
            trend_payload=state.trend_payload,
            signal_ts=signal_ts,
            pending_trend_ts=pending_trend_ts,
            parse_iso_timestamp=deps.parse_iso_timestamp,
        )
        target_projection = _resolve_target_projection(
            state.target_range_policy,
            state.target_range_bundles,
            horizon,
            row_features,
            close,
            label,
            deps,
        )
        direction_output_scoped = horizon in set(state.direction_output_policy_resolved.get("horizons", []))
        result, fallback_triggered = deps.build_prediction_result(
            signal=signal,
            label=label,
            horizon=horizon,
            signal_ts=signal_ts,
            close=close,
            p_up=p_up,
            raw_p_up=raw_p_up,
            ret_pred=ret_pred,
            trend_prob=trend_prob,
            ignition_state=ignition_state,
            cooldown_active=cooldown_active,
            signal_ensemble=signal_ensemble,
            signal_dir_only=signal_dir_only,
            confidence_score=confidence_score,
            position_size=position_size,
            confidence_min=effective_confidence_min,
            confidence_min_source=confidence_min_source,
            position_size_cap=effective_position_size_cap,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            expected_value=expected_value,
            horizon_thresholds=horizon_thresholds,
            regime_state=regime_state,
            calibration_key=calibration_key,
            calibration_used_regime_key=calibration_used_regime_key,
            probability_guard=probability_guard if isinstance(probability_guard, Mapping) else None,
            regime_weight_policy=state.regime_weight_policy,
            target_projection=target_projection,
            volatility_payload=signal.get(
                "volatility",
                {
                    "snapshot": volatility_snapshot,
                    "metric": None,
                    "ceiling": horizon_thresholds.get("volatility_ceiling"),
                    "triggered": False,
                },
            ),
            volatility_flag=bool(signal.get("volatility_flag")),
            forecast_coherence_policy=state.forecast_coherence_policy_resolved,
            direction_output_policy=state.direction_output_policy_resolved,
            direction_output_scoped=direction_output_scoped,
            trade_decision_policy=state.trade_decision_policy_resolved,
            abstention_policy=state.abstention_policy_resolved,
            direction_fallback_policy=state.direction_fallback_policy,
            trend_payload=state.trend_payload,
            target_range_policy=state.target_range_policy,
            regime_score=regime_score,
            adaptive_scale=adaptive_scale,
            horizon_p_up=horizon_p_up,
            horizon_ret=horizon_ret,
            row_features=row_features,
            optional_feature_fields=(
                "funding_rate_zscore_24h",
                "open_interest",
                "fut_close",
                "fut_close_zscore_7h",
                "range_expansion_1h",
                "distance_from_session_high_8h",
                "distance_from_session_low_8h",
                "vwap_deviation_8h",
                "momentum_slope_2h",
                "momentum_slope_4h",
            ),
            project_price=deps.project_price,
            get_active_regime_weight_override=lambda regime_state, horizon, policy: deps.get_active_regime_weight_override(
                regime_state=regime_state,
                horizon=horizon,
                policy=policy,
            ),
            derive_probability_alignment_features=deps.derive_probability_alignment_features,
            build_direction_output=deps.build_direction_output,
            apply_target_range_overrides=lambda stop_loss, take_profit, target_projection, override_ratio, direction: deps.apply_target_range_overrides(
                stop_loss,
                take_profit,
                target_projection,
                override_ratio=override_ratio,
                direction=direction,
            ),
            evaluate_direction_only_fallback=deps.evaluate_direction_only_fallback,
            finite_float_or_none=deps.finite_float_or_none,
            coerce_row_value=deps.coerce_row_value,
        )
        if fallback_triggered:
            pending_direction_fallback_ts = signal_ts

        summary[label] = result
        execution_contexts[label] = {
            "prepared": prepared,
            "index": index,
            "horizon": horizon,
            "residual_std": residual_std,
        }

    return summary, execution_contexts, pending_trend_ts, pending_direction_fallback_ts


def _resolve_trend_state(
    *,
    signal: Mapping[str, Any],
    trend_payload: Mapping[str, Any] | None,
    signal_ts: str,
    pending_trend_ts: str | None,
    parse_iso_timestamp: Callable[[str], Any],
) -> tuple[float, int, bool, str | None]:
    trend_prob = float(signal.get("p_trend_ignition", 0.0))
    ignition_state = 0
    cooldown_active = False
    if trend_payload:
        threshold_value = float(trend_payload.get("threshold", 0.6))
        cooldown_hours = float(trend_payload.get("cooldown_hours", 0.0))
        last_trigger_ts = trend_payload.get("last_trigger_ts")
        if cooldown_hours > 0 and isinstance(last_trigger_ts, str) and last_trigger_ts.strip():
            try:
                elapsed_hours = (parse_iso_timestamp(signal_ts) - parse_iso_timestamp(last_trigger_ts)).total_seconds() / 3600.0
                cooldown_active = elapsed_hours < cooldown_hours
            except ValueError:
                cooldown_active = False
        if trend_prob >= threshold_value and not cooldown_active:
            ignition_state = 1
            if pending_trend_ts is None:
                pending_trend_ts = signal_ts
    return trend_prob, ignition_state, cooldown_active, pending_trend_ts


def _resolve_target_projection(
    target_range_policy: Mapping[str, Any] | None,
    target_range_bundles: Mapping[float, Dict[str, Any]],
    horizon: float,
    row_features: Any,
    close: float,
    label: str,
    deps: PredictionPipelineDependencies,
) -> Dict[str, float] | None:
    if not target_range_policy or not target_range_policy.get("enabled"):
        return None
    bundle = target_range_bundles.get(horizon)
    if not bundle:
        return None
    try:
        return deps.predict_target_range_prices(
            bundle,
            row_features,
            close=close,
            confidence_scale=float(
                target_range_policy.get("confidence_rmse_scale", deps.target_range_default_confidence_scale)
            ),
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        deps.stderr_write(f"Warning: failed to compute target-range projection for {label}: {exc}\n")
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_expected_value(
    p_up: float,
    ret_pred: float,
    residual_std: float,
    horizon_thresholds: Mapping[str, Any],
) -> float:
    expected_value = p_up * ret_pred - (1 - p_up) * residual_std
    return expected_value * float(horizon_thresholds.get("expected_value_multiplier", 1.0))


def _compute_confidence_score(p_up: float, expected_value: float, residual_std: float) -> float:
    denominator = abs(residual_std) + 1e-6
    probability_edge = abs(float(p_up) - 0.5) * 2.0
    value_edge = abs(float(expected_value)) / denominator
    raw_score = 0.6 * probability_edge + 0.4 * min(value_edge, 1.0)
    return max(0.0, min(1.0, raw_score))