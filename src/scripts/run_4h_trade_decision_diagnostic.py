from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_12h_beta_confluence075_live_replay.log"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_trade_decision_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_trade_decision_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and summarize the 4h downstream trade-decision state from a mixed replay log."
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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_markdown(payload: Mapping[str, Any]) -> str:
    horizon = payload["horizon"]
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    envelope = trade_decision.get("positive_oof_envelope") if isinstance(trade_decision.get("positive_oof_envelope"), Mapping) else {}
    lines = ["# 4h Trade Decision Diagnostic", ""]
    lines.append("## Decision")
    lines.append(f"- trade_probability: {trade_decision.get('trade_probability')}")
    lines.append(f"- threshold: {trade_decision.get('threshold')}")
    lines.append(f"- threshold_gap: {payload.get('threshold_gap')}")
    lines.append(f"- triggered: {trade_decision.get('triggered')}")
    lines.append(f"- blocking_reason: {trade_decision.get('blocking_reason')}")
    lines.append(f"- proposed_trade_action: {trade_decision.get('proposed_trade_action')}")
    lines.append("")
    lines.append("## Envelope")
    lines.append(f"- available: {envelope.get('available')}")
    lines.append(f"- has_positive_bin: {envelope.get('has_positive_bin')}")
    lines.append(f"- in_positive_bin: {envelope.get('in_positive_bin')}")
    lines.append(f"- matched_bin_mean_ret_net: {envelope.get('matched_bin_mean_ret_net')}")
    lines.append("")
    lines.append("## EV")
    lines.append(f"- expected_net: {trade_decision.get('expected_net')}")
    lines.append(f"- expected_net_raw: {trade_decision.get('expected_net_raw')}")
    lines.append(f"- expected_net_raw_calibrated: {trade_decision.get('expected_net_raw_calibrated')}")
    lines.append(f"- edge_over_fee: {trade_decision.get('edge_over_fee')}")
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
    trade_decision = horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {}
    weak_band = trade_decision.get("weak_band_veto") if isinstance(trade_decision.get("weak_band_veto"), Mapping) else {}
    midband = trade_decision.get("midband_veto") if isinstance(trade_decision.get("midband_veto"), Mapping) else {}
    trade_probability = _safe_float(trade_decision.get("trade_probability"))
    threshold = _safe_float(trade_decision.get("threshold"))
    threshold_gap = None if trade_probability is None or threshold is None else float(trade_probability - threshold)

    recommendations: list[str] = []
    if threshold_gap is not None and threshold_gap < 0.0:
        recommendations.append(
            "The 4h hold is downstream: trade probability remains below the configured trade-decision threshold even after upstream gates clear."
        )
    envelope = trade_decision.get("positive_oof_envelope") if isinstance(trade_decision.get("positive_oof_envelope"), Mapping) else {}
    if envelope.get("available") and envelope.get("has_positive_bin") and envelope.get("in_positive_bin"):
        recommendations.append(
            "The positive OOF envelope is not the blocker in this replay; the downstream hold is dominated by the trade-decision score itself."
        )
    if bool(midband.get("triggered")) or bool(weak_band.get("triggered")):
        recommendations.append(
            "A band veto triggered in this replay, so threshold-only interpretation would be incomplete."
        )
    if not recommendations:
        recommendations.append("No 4h trade-decision blocker was detected in this replay.")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "horizon_label": str(args.horizon_label),
        },
        "threshold_gap": threshold_gap,
        "horizon": {
            "timestamp": horizon.get("timestamp"),
            "direction_next": horizon.get("direction_next"),
            "trade_action": horizon.get("trade_action"),
            "trust_status": horizon.get("trust_status"),
            "trade_decision": {
                "enabled": trade_decision.get("enabled"),
                "triggered": trade_decision.get("triggered"),
                "blocking_reason": trade_decision.get("blocking_reason"),
                "proposed_signal_ensemble": trade_decision.get("proposed_signal_ensemble"),
                "proposed_trade_action": trade_decision.get("proposed_trade_action"),
                "trade_probability": trade_probability,
                "threshold": threshold,
                "threshold_source": trade_decision.get("threshold_source"),
                "expected_net": trade_decision.get("expected_net"),
                "expected_net_valid": trade_decision.get("expected_net_valid"),
                "expected_net_raw": trade_decision.get("expected_net_raw"),
                "expected_net_raw_calibrated": trade_decision.get("expected_net_raw_calibrated"),
                "expected_net_oof": trade_decision.get("expected_net_oof"),
                "edge_over_fee": trade_decision.get("edge_over_fee"),
                "direction_ret_aligned": trade_decision.get("direction_ret_aligned"),
                "positive_oof_envelope": trade_decision.get("positive_oof_envelope") if isinstance(trade_decision.get("positive_oof_envelope"), Mapping) else {},
                "weak_band_veto": weak_band,
                "midband_veto": midband,
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