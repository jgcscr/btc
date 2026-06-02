from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from src.runtime.confluence_support import resolve_confluence_policy


DEFAULT_LOG_PATH = "artifacts/tmp/shadow_4h_ultra_conservative_regime_calibration_live_replay.log"
DEFAULT_CONFIG_PATH = "configs/run_refresh_and_predict.shadow_4h_ultra_conservative_regime_calibration.yaml"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_confluence_diagnostic_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_confluence_diagnostic_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and summarize the 4h confluence blocker from a mixed replay log."
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
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


def _coerce_numeric_horizon(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw.endswith("h"):
                return float(raw[:-1])
            if raw.endswith("m"):
                return float(raw[:-1]) / 60.0
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_horizon_value(value: float) -> float:
    return float(value)


def _lookup_horizon_value(map_by_horizon: Mapping[float, float], horizon: float, default: float) -> float:
    return float(map_by_horizon.get(float(horizon), default))


def _load_confluence_policy(config_path: Path) -> Dict[str, Any]:
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return resolve_confluence_policy(
        config_payload.get("confluence_policy"),
        coerce_numeric_horizon=_coerce_numeric_horizon,
        normalize_horizon_value=_normalize_horizon_value,
    )


def _extract_horizon_entry(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), Mapping) else {}
    entry = predictions.get(label)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Horizon '{label}' not found in replay payload")
    return dict(entry)


def _direction(entry: Mapping[str, Any]) -> str:
    value = str(entry.get("direction_next") or "").strip().lower()
    if value in {"up", "down", "neutral"}:
        return value
    try:
        ret_pred = float(entry.get("ret_pred"))
    except (TypeError, ValueError):
        return "neutral"
    if ret_pred > 0.0:
        return "up"
    if ret_pred < 0.0:
        return "down"
    return "neutral"


def _weight(entry: Mapping[str, Any]) -> float:
    try:
        return max(float(entry.get("voting_weight_after_trust") or 1.0), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _peer_summary(payload: Mapping[str, Any], current_label: str, current_direction: str) -> list[Dict[str, Any]]:
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), Mapping) else {}
    peers: list[Dict[str, Any]] = []
    for label, raw_entry in predictions.items():
        if not isinstance(raw_entry, Mapping):
            continue
        forecast_coherence = raw_entry.get("forecast_coherence") if isinstance(raw_entry.get("forecast_coherence"), Mapping) else {}
        excluded_from_voting = bool(raw_entry.get("excluded_from_voting", False)) or bool(
            forecast_coherence.get("exclude_from_voting")
        )
        peers.append(
            {
                "horizon": str(label),
                "direction_next": _direction(raw_entry),
                "trade_action": raw_entry.get("trade_action"),
                "voting_weight_after_trust": _weight(raw_entry),
                "trust_status": raw_entry.get("trust_status"),
                "forecast_coherence_triggered": (
                    bool(forecast_coherence.get("triggered")) if forecast_coherence else None
                ),
                "forecast_coherence_excluded": bool(forecast_coherence.get("exclude_from_voting")) if forecast_coherence else False,
                "excluded_from_voting": excluded_from_voting,
                "counts_for_confluence": not excluded_from_voting,
                "aligned_with_target": bool(_direction(raw_entry) == current_direction and str(label) != current_label),
                "aligned_and_counted": bool(
                    _direction(raw_entry) == current_direction and str(label) != current_label and not excluded_from_voting
                ),
            }
        )
    peers.sort(key=lambda item: item["horizon"])
    return peers


def _render_markdown(payload: Mapping[str, Any]) -> str:
    horizon = payload["horizon"]
    thresholds = payload["thresholds"]
    confluence = horizon.get("confluence") if isinstance(horizon.get("confluence"), Mapping) else {}
    lines = ["# 4h Confluence Diagnostic", ""]
    lines.append("## Thresholds")
    lines.append(f"- min_aligned_horizons: {thresholds.get('min_aligned_horizons')}")
    lines.append(f"- min_support_ratio: {thresholds.get('min_support_ratio')}")
    lines.append(f"- min_mid_term_ratio: {thresholds.get('min_mid_term_ratio')}")
    lines.append("")
    lines.append("## Replay")
    lines.append(f"- direction_next: {horizon.get('direction_next')}")
    lines.append(f"- trade_action: {horizon.get('trade_action')}")
    lines.append(f"- trust_status: {horizon.get('trust_status')}")
    lines.append(f"- confluence_triggered: {confluence.get('triggered')}")
    lines.append(f"- confluence_reasons: {', '.join(confluence.get('reasons') or []) or 'none'}")
    lines.append(f"- aligned_horizons: {confluence.get('aligned_horizons')}")
    lines.append(f"- total_horizons: {confluence.get('total_horizons')}")
    lines.append(f"- support_ratio: {confluence.get('support_ratio')}")
    lines.append(f"- short_term_ratio: {confluence.get('short_term_ratio')}")
    lines.append(f"- mid_term_ratio: {confluence.get('mid_term_ratio')}")
    lines.append("")
    lines.append("## Peer Horizons")
    for peer in payload.get("peer_horizons", []):
        lines.append(
            "- "
            f"{peer.get('horizon')}: direction={peer.get('direction_next')}, "
            f"aligned={peer.get('aligned_with_target')}, "
            f"counts_for_confluence={peer.get('counts_for_confluence')}, "
            f"weight={peer.get('voting_weight_after_trust')}, "
            f"trade_action={peer.get('trade_action')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path)
    config_path = Path(args.config_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    payload = _read_json_payload(log_path)
    horizon = _extract_horizon_entry(payload, str(args.horizon_label))
    policy = _load_confluence_policy(config_path)
    horizon_hours = _coerce_numeric_horizon(horizon.get("horizon_hours") or args.horizon_label)
    if horizon_hours is None:
        raise ValueError(f"Could not resolve numeric horizon from {args.horizon_label!r}")

    thresholds = {
        "min_aligned_horizons": int(
            round(
                _lookup_horizon_value(
                    policy.get("min_aligned_horizons_by_horizon", {}) if isinstance(policy.get("min_aligned_horizons_by_horizon"), Mapping) else {},
                    horizon_hours,
                    float(policy.get("min_aligned_horizons", 2)),
                )
            )
        ),
        "min_support_ratio": _lookup_horizon_value(
            policy.get("min_support_ratio_by_horizon", {}) if isinstance(policy.get("min_support_ratio_by_horizon"), Mapping) else {},
            horizon_hours,
            float(policy.get("min_support_ratio", 0.6)),
        ),
        "min_mid_term_ratio": float(policy.get("min_mid_term_ratio", 0.5)),
        "min_short_term_ratio": float(policy.get("min_short_term_ratio", 0.5)),
        "require_mid_term_alignment": bool(policy.get("require_mid_term_alignment", True)),
        "require_short_term_alignment": bool(policy.get("require_short_term_alignment", False)),
    }

    confluence = horizon.get("confluence") if isinstance(horizon.get("confluence"), Mapping) else {}
    support_ratio = confluence.get("support_ratio")
    aligned_horizons = confluence.get("aligned_horizons")
    peer_horizons = _peer_summary(payload, str(args.horizon_label), _direction(horizon))

    recommendations: list[str] = []
    if isinstance(aligned_horizons, int) and aligned_horizons < thresholds["min_aligned_horizons"]:
        recommendations.append(
            "The 4h entry misses the aligned-horizon requirement; the next diagnostic target is which peer horizon failed to align with the 4h direction."
        )
    if isinstance(support_ratio, (float, int)) and float(support_ratio) < thresholds["min_support_ratio"]:
        recommendations.append(
            "The 4h weighted support ratio is below the configured floor; compare the 1h/12h voting weights before relaxing any confluence threshold."
        )
    if not recommendations:
        recommendations.append("No 4h confluence blocker was detected in this replay.")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "log_path": str(log_path),
            "config_path": str(config_path),
            "horizon_label": str(args.horizon_label),
        },
        "thresholds": thresholds,
        "horizon": {
            "timestamp": horizon.get("timestamp"),
            "direction_next": horizon.get("direction_next"),
            "trade_action": horizon.get("trade_action"),
            "trust_status": horizon.get("trust_status"),
            "trust_reasons": horizon.get("trust_reasons") or [],
            "confluence": {
                "triggered": confluence.get("triggered"),
                "reasons": confluence.get("reasons") or [],
                "dominant_direction": confluence.get("dominant_direction"),
                "dominant_ratio": confluence.get("dominant_ratio"),
                "aligned_horizons": confluence.get("aligned_horizons"),
                "total_horizons": confluence.get("total_horizons"),
                "support_ratio": confluence.get("support_ratio"),
                "short_term_ratio": confluence.get("short_term_ratio"),
                "mid_term_ratio": confluence.get("mid_term_ratio"),
            },
            "trade_decision": horizon.get("trade_decision") if isinstance(horizon.get("trade_decision"), Mapping) else {},
        },
        "peer_horizons": peer_horizons,
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