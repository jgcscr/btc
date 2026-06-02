from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from src.scripts.run_4h_trade_decision_feature_sanity_diagnostic import (
    DEFAULT_MODEL_PATH,
    DEFAULT_REFERENCE_CSV,
    _behavior_label,
    _best_reference_subset,
    _extract_horizon_entry,
    _extract_last_json_object,
    _read_json,
    _safe_float,
    _stats,
    _subset_masks,
    _top_negative_features,
    _percentile_rank,
)


DEFAULT_LOG_GLOB = "artifacts/tmp/shadow_4h*_live_replay*.log"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_trade_decision_feature_sanity_sweep_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_trade_decision_feature_sanity_sweep_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 4h trade-decision feature-sanity diagnostic across a batch of replay logs."
    )
    parser.add_argument("--log-glob", default=DEFAULT_LOG_GLOB)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-csv", default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--horizon-label", default="4h")
    parser.add_argument("--signal-col", default="signal_ensemble")
    parser.add_argument("--target-col", default="ret_ensemble_net")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def _load_log_paths(pattern: str) -> List[Path]:
    return sorted(Path(".").glob(pattern))


def _feature_checks_for_log(
    *,
    log_path: Path,
    reference_df: pd.DataFrame,
    feature_columns: List[str],
    coefficients: List[float],
    signal_col: str,
    target_col: str,
    horizon_label: str,
    min_samples: int,
    top_k: int,
) -> Dict[str, Any]:
    replay_payload = _extract_last_json_object(log_path.read_text(encoding="utf-8"))
    horizon = _extract_horizon_entry(replay_payload, horizon_label)
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    feature_snapshot = trade_decision.get("feature_snapshot") if isinstance(trade_decision.get("feature_snapshot"), Mapping) else {}
    regime_name = None
    for candidate in ("trend", "neutral", "chop"):
        if _safe_float(feature_snapshot.get(f"regime_is_{candidate}"), 0.0) > 0.5:
            regime_name = candidate
            break

    subset_masks = _subset_masks(
        reference_df,
        signal_col=signal_col,
        target_col=target_col,
        regime_name=regime_name,
    )

    feature_checks: List[Dict[str, Any]] = []
    for item in _top_negative_features(feature_snapshot, feature_columns, coefficients, top_k=top_k):
        feature_name = str(item["feature"])
        subset_results: Dict[str, Dict[str, Any]] = {}
        if feature_name in reference_df.columns:
            for subset_name, mask in subset_masks.items():
                subset_series = reference_df.loc[mask, feature_name]
                stats = _stats(subset_series)
                percentile_rank = None
                if int(stats["samples"] or 0) >= int(min_samples):
                    percentile_rank = _percentile_rank(subset_series, float(item["feature_value"]))
                subset_results[subset_name] = {**stats, "percentile_rank": percentile_rank}
        profitable_subset = _best_reference_subset(subset_results)
        classification = _behavior_label(
            float(item["coefficient"]),
            {
                **profitable_subset,
                "current_value": float(item["feature_value"]),
            },
        )
        feature_checks.append({**item, "classification": classification, "subsets": subset_results})

    trade_probability = _safe_float(trade_decision.get("trade_probability"), 0.0)
    threshold = _safe_float(trade_decision.get("threshold"), 0.0)
    threshold_gap = float(trade_probability - threshold)
    return {
        "log_path": str(log_path),
        "timestamp": horizon.get("timestamp"),
        "config_label": log_path.name.replace("_live_replay.log", ""),
        "trade_probability": trade_probability,
        "threshold": threshold,
        "threshold_gap": threshold_gap,
        "triggered": bool(trade_decision.get("triggered")),
        "blocking_reason": trade_decision.get("blocking_reason"),
        "current_regime": regime_name,
        "feature_checks": feature_checks,
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h Trade Decision Feature Sanity Sweep", ""]
    lines.append("## Runs")
    for run in payload.get("runs", []):
        top_features = ", ".join(
            f"{item.get('feature')}={item.get('classification')}" for item in run.get("feature_checks", [])
        )
        lines.append(
            f"- {run.get('config_label')}: trade_probability={run.get('trade_probability')}, "
            f"threshold_gap={run.get('threshold_gap')}, triggered={run.get('triggered')}, top_negative={top_features}"
        )
    lines.append("")
    lines.append("## Aggregate Feature Outcomes")
    for item in payload.get("aggregate_feature_summary", []):
        lines.append(
            f"- {item.get('feature')}: seen={item.get('seen_count')}, dominant_classification={item.get('dominant_classification')}, "
            f"classifications={item.get('classifications')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    log_paths = _load_log_paths(str(args.log_glob))
    if not log_paths:
        raise FileNotFoundError(f"No replay logs matched: {args.log_glob}")

    model_payload = _read_json(Path(args.model_path))
    feature_columns = [str(value) for value in model_payload.get("feature_columns", [])]
    coefficients = [float(value) for value in model_payload.get("coefficients", [])]
    if not feature_columns or len(feature_columns) != len(coefficients):
        raise ValueError("Trade decision model is missing aligned feature_columns/coefficients")

    reference_df = pd.read_csv(args.reference_csv)
    runs: List[Dict[str, Any]] = []
    feature_classifications: dict[str, Counter[str]] = defaultdict(Counter)
    feature_occurrences: Counter[str] = Counter()

    for log_path in log_paths:
        run = _feature_checks_for_log(
            log_path=log_path,
            reference_df=reference_df,
            feature_columns=feature_columns,
            coefficients=coefficients,
            signal_col=str(args.signal_col),
            target_col=str(args.target_col),
            horizon_label=str(args.horizon_label),
            min_samples=int(args.min_samples),
            top_k=int(args.top_k),
        )
        runs.append(run)
        for item in run["feature_checks"]:
            feature_name = str(item["feature"])
            feature_occurrences[feature_name] += 1
            feature_classifications[feature_name][str(item["classification"])] += 1

    aggregate_feature_summary: List[Dict[str, Any]] = []
    for feature_name, seen_count in feature_occurrences.most_common():
        counts = feature_classifications[feature_name]
        dominant_classification = counts.most_common(1)[0][0] if counts else None
        aggregate_feature_summary.append(
            {
                "feature": feature_name,
                "seen_count": int(seen_count),
                "dominant_classification": dominant_classification,
                "classifications": dict(counts),
            }
        )

    consistently_adverse = [
        item["feature"]
        for item in aggregate_feature_summary
        if item["classifications"].get("current_value_above_profitable_p90", 0) >= max(2, len(runs) // 2)
    ]
    consistently_weighting_sensitive = [
        item["feature"]
        for item in aggregate_feature_summary
        if (
            item["classifications"].get("penalizing_value_common_in_profitable_rows", 0)
            + item["classifications"].get("penalizing_value_seen_in_profitable_rows_but_not_typical", 0)
            + item["classifications"].get("negative_coefficient_on_low_profitable_value", 0)
        )
        >= max(2, len(runs) // 2)
    ]

    recommendations: List[str] = []
    if consistently_adverse and consistently_weighting_sensitive:
        recommendations.append(
            "Across recent 4h replays, the negative trade-decision picture is persistent and mixed: some features repeatedly land in adverse profitable-row tails while others recur often enough to question model weighting."
        )
    elif consistently_adverse:
        recommendations.append(
            "Across recent 4h replays, the main negative features repeatedly land in adverse profitable-row tails, so a simple 4h threshold reduction still looks under-justified."
        )
    elif consistently_weighting_sensitive:
        recommendations.append(
            "Across recent 4h replays, several recurring negative contributors remain historically common enough to justify reviewing trade-decision weighting before tightening policy further."
        )
    else:
        recommendations.append(
            "Recent 4h replays do not show one stable negative-feature pattern, so the next decision should come from a larger replay batch rather than this small sweep alone."
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_glob": str(args.log_glob),
            "model_path": str(args.model_path),
            "reference_csv": str(args.reference_csv),
            "horizon_label": str(args.horizon_label),
        },
        "run_count": len(runs),
        "runs": runs,
        "aggregate_feature_summary": aggregate_feature_summary,
        "recommendations": recommendations,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote diagnostic JSON: {output_json}")
    print(f"Wrote diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()