from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_12h_beta_confluence075_live_replay.log"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_trade_decision_threshold_counterfactual_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_trade_decision_threshold_counterfactual_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 4h trade-decision threshold overrides against a fixed replay payload."
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--horizon-label", default="4h")
    parser.add_argument("--thresholds", nargs="+", default=["0.33", "0.35", "0.4", "0.45", "0.5495"])
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# 4h Threshold Counterfactual", ""]
    lines.append("## Fixed Replay")
    lines.append(f"- trade_probability: {payload.get('trade_probability')}")
    lines.append(f"- current_threshold: {payload.get('current_threshold')}")
    lines.append(f"- upstream_blocking_reason: {payload.get('upstream_blocking_reason')}")
    lines.append("")
    lines.append("## Threshold Sweep")
    for item in payload.get("scenarios", []):
        lines.append(
            f"- threshold={item.get('threshold')}: passes_threshold={item.get('passes_threshold')}, gap={item.get('threshold_gap')}, would_trigger_without_upstream_change={item.get('would_trigger_without_upstream_change')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path)
    payload = _extract_last_json_object(log_path.read_text(encoding="utf-8"))
    horizon = _extract_horizon_entry(payload, str(args.horizon_label))
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    trade_probability = _safe_float(trade_decision.get("trade_probability"))
    current_threshold = _safe_float(trade_decision.get("threshold"))
    upstream_blocking_reason = trade_decision.get("blocking_reason")

    scenarios: List[Dict[str, Any]] = []
    for raw_threshold in args.thresholds:
        threshold = _safe_float(raw_threshold)
        passes_threshold = bool(trade_probability >= threshold)
        scenarios.append(
            {
                "threshold": threshold,
                "passes_threshold": passes_threshold,
                "threshold_gap": float(trade_probability - threshold),
                "would_trigger_without_upstream_change": bool(passes_threshold and not upstream_blocking_reason),
            }
        )

    recommendations: list[str] = []
    passing = [item for item in scenarios if item["would_trigger_without_upstream_change"]]
    if passing:
        recommendations.append(
            f"On this fixed replay payload, the smallest tested threshold that would trigger 4h without any upstream change is {passing[0]['threshold']}."
        )
    else:
        recommendations.append(
            "None of the tested thresholds would trigger 4h on this fixed replay payload without additional upstream or score changes."
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "horizon_label": str(args.horizon_label),
            "thresholds": [str(value) for value in args.thresholds],
        },
        "trade_probability": trade_probability,
        "current_threshold": current_threshold,
        "upstream_blocking_reason": upstream_blocking_reason,
        "scenarios": scenarios,
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