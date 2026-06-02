from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_live_replay_with_candidate_calibration.log"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_forecast_coherence_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_forecast_coherence_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and summarize the 4h forecast coherence blocker from a mixed replay log."
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--horizon-label", default="4h")
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


def _read_json_payload(path: Path) -> Dict[str, Any]:
    return _extract_last_json_object(path.read_text(encoding="utf-8"))


def _extract_horizon_entry(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), Mapping) else {}
    entry = predictions.get(label)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Horizon '{label}' not found in replay payload")
    return dict(entry)


def _render_markdown(payload: Mapping[str, Any]) -> str:
    horizon = payload["horizon"]
    lines = ["# 4h Forecast Coherence Diagnostic", ""]
    lines.append("## Calibration")
    lines.append(f"- requested_key: {horizon['probability_calibration'].get('requested_key')}")
    lines.append(f"- applied_key: {horizon['probability_calibration'].get('applied_key')}")
    lines.append(f"- used_regime_key: {horizon['probability_calibration'].get('used_regime_key')}")
    lines.append(f"- raw_p_up: {horizon.get('raw_p_up')}")
    lines.append(f"- p_up: {horizon.get('p_up')}")
    lines.append(f"- probability_alignment_gap: {horizon.get('probability_alignment_gap')}")
    lines.append("")
    lines.append("## Direction Alignment")
    lines.append(f"- direction_next: {horizon.get('direction_next')}")
    lines.append(f"- raw_p_up_side: {horizon.get('raw_p_up_side')}")
    lines.append(f"- resolved_p_up_side: {horizon.get('resolved_p_up_side')}")
    lines.append(f"- ret_pred_side: {horizon.get('ret_pred_side')}")
    lines.append(f"- projected_price_side: {horizon.get('projected_price_side')}")
    lines.append("")
    lines.append("## Gates")
    lines.append(f"- trust_status: {horizon.get('trust_status')}")
    lines.append(f"- trust_reasons: {', '.join(horizon.get('trust_reasons') or []) or 'none'}")
    lines.append(f"- trust_hardening_action: {horizon.get('trust_hardening_action')}")
    coherence = horizon.get("forecast_coherence") if isinstance(horizon.get("forecast_coherence"), Mapping) else {}
    lines.append(f"- forecast_coherence_triggered: {coherence.get('triggered')}")
    lines.append(f"- forecast_coherence_reasons: {', '.join(coherence.get('reasons') or []) or 'none'}")
    lines.append(f"- forecast_coherence_advisory_reasons: {', '.join(coherence.get('advisory_reasons') or []) or 'none'}")
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    lines.append(f"- trade_decision_blocking_reason: {trade_decision.get('blocking_reason')}")
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    payload = _read_json_payload(log_path)
    horizon = _extract_horizon_entry(payload, str(args.horizon_label))
    calibration = horizon.get("probability_calibration") if isinstance(horizon.get("probability_calibration"), Mapping) else {}
    coherence = horizon.get("forecast_coherence") if isinstance(horizon.get("forecast_coherence"), Mapping) else {}
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}

    recommendations: list[str] = []
    if str(horizon.get("trust_status")) == "trusted" and bool(coherence.get("triggered")):
        recommendations.append(
            "4h trust is cleared in this replay; the remaining blocker is forecast coherence rather than trust hardening."
        )
    if calibration.get("applied_key") and not bool(calibration.get("used_regime_key", False)):
        recommendations.append(
            "The replay fell back from the regime-specific 4h calibration key to the base 4h calibration; the next 4h calibration improvement should target regime-aware alignment, not trust metadata."
        )
    if "p_up_ret_mismatch" in (coherence.get("reasons") or []):
        recommendations.append(
            "Resolved 4h probability remains on the opposite side of the positive return forecast, so the next remediation target is probability/return coherence for the neutral regime."
        )
    if not recommendations:
        recommendations.append("No 4h forecast coherence blocker was detected in this replay.")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "horizon_label": str(args.horizon_label),
        },
        "horizon": {
            "timestamp": horizon.get("timestamp"),
            "p_up": horizon.get("p_up"),
            "raw_p_up": horizon.get("raw_p_up"),
            "probability_alignment_gap": horizon.get("probability_alignment_gap"),
            "direction_next": horizon.get("direction_next"),
            "trade_action": horizon.get("trade_action"),
            "raw_p_up_side": horizon.get("raw_p_up_side"),
            "resolved_p_up_side": horizon.get("resolved_p_up_side"),
            "ret_pred_side": horizon.get("ret_pred_side"),
            "projected_price_side": horizon.get("projected_price_side"),
            "trust_status": horizon.get("trust_status"),
            "trust_reasons": horizon.get("trust_reasons") or [],
            "trust_hardening_action": horizon.get("trust_hardening_action"),
            "voting_weight_after_trust": horizon.get("voting_weight_after_trust"),
            "probability_calibration": {
                "requested_key": calibration.get("requested_key"),
                "applied_key": calibration.get("applied_key"),
                "used_regime_key": calibration.get("used_regime_key"),
                "fallback_to_base": calibration.get("fallback_to_base"),
                "raw_side": calibration.get("raw_side"),
                "resolved_side": calibration.get("resolved_side"),
            },
            "forecast_coherence": {
                "triggered": coherence.get("triggered"),
                "reasons": coherence.get("reasons") or [],
                "advisory_reasons": coherence.get("advisory_reasons") or [],
                "consensus_relief_applied": coherence.get("consensus_relief_applied"),
            },
            "trade_decision": {
                "blocking_reason": trade_decision.get("blocking_reason"),
                "triggered": trade_decision.get("triggered"),
            },
        },
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