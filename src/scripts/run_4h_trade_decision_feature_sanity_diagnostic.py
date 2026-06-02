from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_12h_beta_confluence075_live_replay.log"
DEFAULT_MODEL_PATH = "artifacts/models/trade_decision_model.json"
DEFAULT_REFERENCE_CSV = "artifacts/reliability/20260515T023232Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_reference_feature_ablation.csv"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_trade_decision_feature_sanity_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_trade_decision_feature_sanity_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current 4h trade-decision feature snapshot against historical decision-feature distributions."
        )
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-csv", default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--horizon-label", default="4h")
    parser.add_argument("--signal-col", default="signal_ensemble")
    parser.add_argument("--target-col", default="ret_ensemble_net")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def _extract_last_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    last: Dict[str, Any] | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            trailing = text[idx + end :].strip()
            if not trailing or trailing.startswith("@"):
                last = payload
    if last is None:
        raise ValueError("Could not find a terminal JSON payload in the replay log.")
    return last


def _extract_horizon_entry(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), Mapping) else {}
    entry = predictions.get(label)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Horizon '{label}' not found in replay payload")
    return dict(entry)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _percentile_rank(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    le = float((numeric <= value).sum())
    return float(le / len(numeric))


def _stats(series: pd.Series) -> Dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {
            "samples": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    values = numeric.to_numpy(dtype=float)
    return {
        "samples": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _subset_masks(
    df: pd.DataFrame,
    *,
    signal_col: str,
    target_col: str,
    regime_name: str | None,
) -> Dict[str, pd.Series]:
    signal = pd.to_numeric(df.get(signal_col, 0.0), errors="coerce").fillna(0.0)
    target = pd.to_numeric(df.get(target_col, 0.0), errors="coerce").fillna(0.0)
    candidate_mask = signal > 0.0
    profitable_mask = target > 0.0

    subsets: Dict[str, pd.Series] = {
        "all_rows": pd.Series(True, index=df.index),
        "candidate_rows": candidate_mask,
        "profitable_candidate_rows": candidate_mask & profitable_mask,
    }

    if regime_name:
        regime_key = f"regime_is_{regime_name.strip().lower()}"
        if regime_key in df.columns:
            regime_mask = pd.to_numeric(df[regime_key], errors="coerce").fillna(0.0) > 0.5
            subsets["same_regime_candidate_rows"] = candidate_mask & regime_mask
            subsets["same_regime_profitable_candidate_rows"] = candidate_mask & profitable_mask & regime_mask
    return subsets


def _behavior_label(coefficient: float, reference_stats: Mapping[str, Any]) -> str:
    profitable_percentile = reference_stats.get("percentile_rank")
    p10 = reference_stats.get("p10")
    p25 = reference_stats.get("p25")
    median = reference_stats.get("median")
    p75 = reference_stats.get("p75")
    p90 = reference_stats.get("p90")
    value = reference_stats.get("current_value")
    if profitable_percentile is None or value is None:
        return "insufficient_reference_samples"
    if coefficient < 0.0:
        if p75 is not None and value <= p75:
            return "penalizing_value_common_in_profitable_rows"
        if p90 is not None and value <= p90:
            return "penalizing_value_seen_in_profitable_rows_but_not_typical"
        if median is not None and value > median:
            return "current_value_above_profitable_p90"
        if p10 is not None and value <= p10:
            return "negative_coefficient_on_low_profitable_value"
        return "current_value_somewhat_adverse_vs_profitable_rows"
    if coefficient > 0.0:
        if p25 is not None and value >= p25:
            return "current_value_within_profitable_range_for_positive_feature"
        if p10 is not None and value >= p10:
            return "current_value_below_profitable_center"
        if p10 is not None and value < p10:
            return "missing_value_that_profitable_rows_usually_have"
        if profitable_percentile <= 0.50:
            return "current_value_below_profitable_center"
        if profitable_percentile >= 0.90:
            return "current_value_strong_for_positive_feature"
        return "current_value_acceptable_for_positive_feature"
    return "neutral_coefficient"


def _best_reference_subset(subsets: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    preferred = subsets.get("same_regime_profitable_candidate_rows")
    if isinstance(preferred, Mapping) and preferred.get("percentile_rank") is not None:
        return preferred
    fallback = subsets.get("profitable_candidate_rows")
    if isinstance(fallback, Mapping):
        return fallback
    return {}


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h Trade Decision Feature Sanity Diagnostic", ""]
    lines.append("## Snapshot")
    lines.append(f"- trade_probability: {payload.get('trade_probability')}")
    lines.append(f"- threshold: {payload.get('threshold')}")
    lines.append(f"- threshold_gap: {payload.get('threshold_gap')}")
    lines.append(f"- reference_csv: {payload.get('inputs', {}).get('reference_csv')}")
    lines.append("")
    lines.append("## Top Negative Features")
    for item in payload.get("feature_checks", []):
        profitable = item.get("subsets", {}).get("profitable_candidate_rows", {})
        same_regime = item.get("subsets", {}).get("same_regime_profitable_candidate_rows", {})
        lines.append(
            "- "
            f"{item.get('feature')}: value={item.get('feature_value')}, contribution={item.get('contribution')}, "
            f"profitable_candidate_percentile={profitable.get('percentile_rank')}, "
            f"same_regime_profitable_percentile={same_regime.get('percentile_rank')}, "
            f"classification={item.get('classification')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _top_negative_features(
    feature_snapshot: Mapping[str, Any],
    feature_columns: Iterable[str],
    coefficients: Iterable[float],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for feature_name, coefficient in zip(feature_columns, coefficients):
        feature_value = _safe_float(feature_snapshot.get(feature_name), 0.0)
        contribution = float(feature_value * float(coefficient))
        if contribution >= 0.0:
            continue
        features.append(
            {
                "feature": str(feature_name),
                "feature_value": feature_value,
                "coefficient": float(coefficient),
                "contribution": contribution,
            }
        )
    return sorted(features, key=lambda item: item["contribution"])[: max(int(top_k), 1)]


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path)
    model_path = Path(args.model_path)
    reference_csv = Path(args.reference_csv)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    replay_payload = _extract_last_json_object(log_path.read_text(encoding="utf-8"))
    horizon = _extract_horizon_entry(replay_payload, str(args.horizon_label))
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    feature_snapshot = trade_decision.get("feature_snapshot") if isinstance(trade_decision.get("feature_snapshot"), Mapping) else {}
    model_payload = _read_json(model_path)
    feature_columns = [str(value) for value in model_payload.get("feature_columns", [])]
    coefficients = [float(value) for value in model_payload.get("coefficients", [])]
    if not feature_columns or len(feature_columns) != len(coefficients):
        raise ValueError("Trade decision model is missing aligned feature_columns/coefficients")

    reference_df = pd.read_csv(reference_csv)
    regime_name = None
    for candidate in ("trend", "neutral", "chop"):
        if _safe_float(feature_snapshot.get(f"regime_is_{candidate}"), 0.0) > 0.5:
            regime_name = candidate
            break
    subset_masks = _subset_masks(
        reference_df,
        signal_col=str(args.signal_col),
        target_col=str(args.target_col),
        regime_name=regime_name,
    )

    feature_checks: List[Dict[str, Any]] = []
    for item in _top_negative_features(feature_snapshot, feature_columns, coefficients, top_k=int(args.top_k)):
        feature_name = str(item["feature"])
        subset_results: Dict[str, Dict[str, Any]] = {}
        if feature_name in reference_df.columns:
            for subset_name, mask in subset_masks.items():
                subset_series = reference_df.loc[mask, feature_name]
                stats = _stats(subset_series)
                percentile_rank = None
                if int(stats["samples"] or 0) >= int(args.min_samples):
                    percentile_rank = _percentile_rank(subset_series, float(item["feature_value"]))
                subset_results[subset_name] = {
                    **stats,
                    "percentile_rank": percentile_rank,
                }
        else:
            subset_results["all_rows"] = {
                "samples": 0,
                "mean": None,
                "median": None,
                "std": None,
                "p10": None,
                "p25": None,
                "p75": None,
                "p90": None,
                "percentile_rank": None,
            }

        profitable_subset = _best_reference_subset(subset_results)
        classification = _behavior_label(
            float(item["coefficient"]),
            {
                **profitable_subset,
                "current_value": float(item["feature_value"]),
            },
        )
        feature_checks.append(
            {
                **item,
                "classification": classification,
                "subsets": subset_results,
            }
        )

    sign_pathology_count = sum(
        1
        for item in feature_checks
        if item["classification"] in {
            "penalizing_value_common_in_profitable_rows",
            "penalizing_value_seen_in_profitable_rows_but_not_typical",
            "negative_coefficient_on_low_profitable_value",
        }
    )
    genuine_adverse_count = sum(
        1
        for item in feature_checks
        if item["classification"] == "current_value_above_profitable_p90"
    )

    recommendations: List[str] = []
    if sign_pathology_count >= 2 and genuine_adverse_count >= 2:
        recommendations.append(
            "The largest negative features are mixed: some are historically common enough to question trade-decision weighting, while others are clearly in adverse profitable-row tails, so the current 4h hold should not be explained by threshold policy alone."
        )
    elif sign_pathology_count >= 2:
        recommendations.append(
            "Multiple large negative contributors sit in ordinary profitable-row percentiles, which points more toward trade-decision weighting/sign sanity than uniquely bad live 4h conditions."
        )
    elif genuine_adverse_count >= 2:
        recommendations.append(
            "Multiple large negative contributors are in adverse profitable-row tails, so the current 4h hold still looks structurally justified by live feature state."
        )
    if not recommendations:
        recommendations.append(
            "The largest negative features are mixed: some look historically ordinary and some look adverse, so threshold changes alone remain weakly justified without either a broader replay sweep or trade-decision retraining evidence."
        )

    profitable_reference = int(subset_masks.get("profitable_candidate_rows", pd.Series(dtype=bool)).sum())
    same_regime_profitable_reference = int(
        subset_masks.get("same_regime_profitable_candidate_rows", pd.Series(dtype=bool)).sum()
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "model_path": str(model_path),
            "reference_csv": str(reference_csv),
            "horizon_label": str(args.horizon_label),
            "signal_col": str(args.signal_col),
            "target_col": str(args.target_col),
        },
        "trade_probability": _safe_float(trade_decision.get("trade_probability"), 0.0),
        "threshold": _safe_float(trade_decision.get("threshold"), 0.0),
        "threshold_gap": _safe_float(trade_decision.get("trade_probability"), 0.0)
        - _safe_float(trade_decision.get("threshold"), 0.0),
        "current_regime": regime_name,
        "reference_profitable_samples": profitable_reference,
        "reference_same_regime_profitable_samples": same_regime_profitable_reference,
        "feature_checks": feature_checks,
        "recommendations": recommendations,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote diagnostic JSON: {output_json}")
    print(f"Wrote diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()