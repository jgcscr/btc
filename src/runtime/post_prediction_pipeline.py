from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


SummaryPayload = Dict[str, Dict[str, Any]]
ExecutionContexts = Mapping[str, Dict[str, Any]]


def apply_post_prediction_policies(
    summary: SummaryPayload,
    execution_contexts: ExecutionContexts,
    *,
    forecast_coherence_policy: Mapping[str, Any],
    confluence_policy: Mapping[str, Any],
    trade_decision_policy: Mapping[str, Any],
    confidence_min: float,
    abstention_policy: Mapping[str, Any],
    uncertainty_policy: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    apply_forecast_coherence_policy: Callable[[SummaryPayload, Mapping[str, Any]], SummaryPayload],
    apply_confluence_policy: Callable[[SummaryPayload, Mapping[str, Any]], SummaryPayload],
    apply_trade_decision_stage: Callable[[SummaryPayload, ExecutionContexts, Mapping[str, Any]], SummaryPayload],
    apply_post_trade_gates: Callable[[SummaryPayload, float, Mapping[str, Any], Mapping[str, Any]], SummaryPayload],
    apply_execution_policy: Callable[[SummaryPayload, ExecutionContexts, Mapping[str, Any]], SummaryPayload],
) -> SummaryPayload:
    if forecast_coherence_policy.get("enabled"):
        summary = apply_forecast_coherence_policy(summary, forecast_coherence_policy)

    if confluence_policy.get("enabled"):
        summary = apply_confluence_policy(summary, confluence_policy)

    summary = apply_trade_decision_stage(summary, execution_contexts, trade_decision_policy)
    summary = apply_post_trade_gates(
        summary,
        confidence_min,
        abstention_policy,
        uncertainty_policy,
    )

    if execution_policy.get("enabled"):
        summary = apply_execution_policy(summary, execution_contexts, execution_policy)

    return summary
