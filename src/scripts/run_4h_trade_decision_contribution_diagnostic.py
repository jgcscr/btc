from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_12h_beta_confluence075_live_replay.log"
DEFAULT_MODEL_PATH = "artifacts/models/trade_decision_model.json"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_trade_decision_contribution_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_trade_decision_contribution_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain the 4h trade-decision score by feature contribution from a mixed replay log."
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--horizon-label", default="4h")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--top-k", type=int, default=10)
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


def _extract_horizon_entry(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), Mapping) else {}
    entry = predictions.get(label)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Horizon '{label}' not found in replay payload")
    return dict(entry)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h Trade Decision Contribution Diagnostic", ""]
    lines.append("## Score")
    lines.append(f"- trade_probability: {payload.get('trade_probability')}")
    lines.append(f"- threshold: {payload.get('threshold')}")
    lines.append(f"- replay_threshold_gap: {payload.get('replay_threshold_gap')}")
    lines.append(f"- reconstructed_threshold_gap: {payload.get('reconstructed_threshold_gap')}")
    lines.append(f"- threshold_gap: {payload.get('threshold_gap')}")
    lines.append(f"- reconstructed_probability: {payload.get('reconstructed_probability')}")
    lines.append(f"- intercept: {payload.get('intercept')}")
    lines.append("")
    lines.append("## Largest Negative Contributions")
    for item in payload.get("top_negative_contributions", []):
        lines.append(
            f"- {item.get('feature')}: contribution={item.get('contribution')}, value={item.get('feature_value')}, coefficient={item.get('coefficient')}"
        )
    lines.append("")
    lines.append("## Largest Positive Contributions")
    for item in payload.get("top_positive_contributions", []):
        lines.append(
            f"- {item.get('feature')}: contribution={item.get('contribution')}, value={item.get('feature_value')}, coefficient={item.get('coefficient')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path)
    model_path = Path(args.model_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    replay_payload = _extract_last_json_object(log_path.read_text(encoding="utf-8"))
    horizon = _extract_horizon_entry(replay_payload, str(args.horizon_label))
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    feature_snapshot = trade_decision.get("feature_snapshot") if isinstance(trade_decision.get("feature_snapshot"), Mapping) else {}
    model_payload = _read_json(model_path)
    feature_columns = [str(value) for value in model_payload.get("feature_columns", [])]
    coefficients = [float(value) for value in model_payload.get("coefficients", [])]
    intercept = float(model_payload.get("intercept", 0.0))
    if not feature_columns or len(feature_columns) != len(coefficients):
        raise ValueError("Trade decision model is missing aligned feature_columns/coefficients")

    contributions: List[Dict[str, Any]] = []
    logit = intercept
    for feature_name, coefficient in zip(feature_columns, coefficients):
        feature_value = _safe_float(feature_snapshot.get(feature_name), 0.0)
        contribution = float(feature_value * coefficient)
        logit += contribution
        contributions.append(
            {
                "feature": feature_name,
                "feature_value": feature_value,
                "coefficient": float(coefficient),
                "contribution": contribution,
            }
        )

    reconstructed_probability = _sigmoid(logit)
    trade_probability = _safe_float(trade_decision.get("trade_probability"), 0.0)
    threshold = _safe_float(trade_decision.get("threshold"), 0.0)
    replay_threshold_gap = float(trade_probability - threshold)
    reconstructed_threshold_gap = float(reconstructed_probability - threshold)
    threshold_gap = reconstructed_threshold_gap
    top_k = max(int(args.top_k), 1)
    top_negative = sorted((item for item in contributions if item["contribution"] < 0.0), key=lambda item: item["contribution"])[:top_k]
    top_positive = sorted((item for item in contributions if item["contribution"] > 0.0), key=lambda item: item["contribution"], reverse=True)[:top_k]

    recommendations: list[str] = []
    if reconstructed_threshold_gap < 0.0:
        recommendations.append(
            "The downstream hold is consistent with the reconstructed model score; focus on the largest negative feature contributions before considering any threshold change."
        )
    elif reconstructed_threshold_gap >= 0.0:
        recommendations.append(
            "The reconstructed model score clears the configured threshold on this replay, so this model artifact is a plausible candidate for targeted 4h counterfactual evaluation."
        )
    if top_negative:
        recommendations.append(
            f"The largest negative contribution is `{top_negative[0]['feature']}`, so the next targeted diagnostic should explain why that feature is currently unfavorable for 4h."
        )
    if not recommendations:
        recommendations.append("No material trade-decision contribution issue was detected.")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "model_path": str(model_path),
            "horizon_label": str(args.horizon_label),
        },
        "trade_probability": trade_probability,
        "threshold": threshold,
        "replay_threshold_gap": replay_threshold_gap,
        "reconstructed_threshold_gap": reconstructed_threshold_gap,
        "threshold_gap": threshold_gap,
        "intercept": intercept,
        "reconstructed_logit": logit,
        "reconstructed_probability": reconstructed_probability,
        "probability_reconstruction_error": float(reconstructed_probability - trade_probability),
        "top_negative_contributions": top_negative,
        "top_positive_contributions": top_positive,
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