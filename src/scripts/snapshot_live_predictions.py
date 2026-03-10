from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _format_horizon_label(value: Any) -> Optional[str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    if float(numeric).is_integer():
        return f"{int(numeric)}h"
    return f"{numeric:g}h"


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_horizon_payloads(predictions_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    predictions_obj = predictions_payload.get("predictions")
    if isinstance(predictions_obj, dict):
        for label, obj in predictions_obj.items():
            if isinstance(obj, dict):
                out[str(label)] = obj

    horizons_obj = predictions_payload.get("horizons")
    if isinstance(horizons_obj, list):
        for row in horizons_obj:
            if not isinstance(row, dict):
                continue
            label = _format_horizon_label(row.get("horizon_hours"))
            if label:
                out[label] = row

    return out


def _extract_monitoring_horizon_payloads(monitoring_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    horizons_obj = monitoring_payload.get("horizons")
    if not isinstance(horizons_obj, list):
        return out

    for row in horizons_obj:
        if not isinstance(row, dict):
            continue
        label = _format_horizon_label(row.get("horizon_hours"))
        if label:
            out[label] = row
    return out


def _normalize_horizon_row(
    *,
    label: str,
    row: Dict[str, Any],
    monitoring_row: Optional[Dict[str, Any]],
    generated_at_dt: Optional[datetime],
) -> Dict[str, Any]:
    trade_decision = row.get("trade_decision") if isinstance(row.get("trade_decision"), dict) else {}
    abstention = row.get("abstention") if isinstance(row.get("abstention"), dict) else {}
    volatility = row.get("volatility") if isinstance(row.get("volatility"), dict) else {}
    vol_snapshot = volatility.get("snapshot") if isinstance(volatility.get("snapshot"), dict) else {}

    timestamp = row.get("timestamp")
    ts_dt = _parse_ts(timestamp)
    age_seconds: Optional[float] = None
    if generated_at_dt is not None and ts_dt is not None:
        age_seconds = max(0.0, float((generated_at_dt - ts_dt).total_seconds()))

    freshness = None
    if monitoring_row and isinstance(monitoring_row, dict):
        freshness = monitoring_row.get("fresh")

    expected_value = row.get("expected_value")
    expected_net = trade_decision.get("expected_net")
    if expected_net is None:
        expected_net = row.get("expected_net")

    normalized = {
        "horizon": label,
        "timestamp": timestamp,
        "age_seconds": age_seconds,
        "fresh": _to_bool(freshness),
        "signal_ensemble": row.get("signal_ensemble"),
        "trade_action": row.get("trade_action"),
        "trade_decision": {
            "triggered": _to_bool(trade_decision.get("triggered")),
            "trade_probability": trade_decision.get("trade_probability"),
            "threshold": trade_decision.get("threshold"),
            "expected_net": expected_net,
        },
        "abstention": {
            "triggered": _to_bool(abstention.get("triggered")),
            "reason": abstention.get("reason"),
        },
        "p_up": row.get("p_up"),
        "expected_value": expected_value,
        "regime_state": row.get("regime_state"),
        "volatility": {
            "volatility_realized_24h": vol_snapshot.get("volatility_realized_24h"),
            "volatility_ewm_24h": vol_snapshot.get("volatility_ewm_24h"),
            "volatility_garch_like": vol_snapshot.get("volatility_garch_like"),
            "volatility_triggered": _to_bool(volatility.get("triggered")),
        },
    }
    return normalized


def _compute_summary(horizons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted(horizons.keys(), key=lambda x: (float(x[:-1]) if x.endswith("h") else float("inf"), x))
    actionable = []
    for label in labels:
        row = horizons[label]
        trade_action = str(row.get("trade_action") or "").lower()
        if trade_action not in {"", "hold"}:
            actionable.append(label)

    return {
        "horizons": labels,
        "any_actionable": bool(actionable),
        "actionable_horizons": actionable,
        "all_hold": len(actionable) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot live prediction + monitoring fields into run summary artifact.")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--profile-id", type=str, required=True)
    parser.add_argument("--profile-name", type=str, required=True)
    parser.add_argument("--predictions-latest", type=Path, default=Path("artifacts/predictions/latest.json"))
    parser.add_argument("--monitoring-latest", type=Path, default=Path("artifacts/monitoring/latest.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions_latest.exists():
        raise FileNotFoundError(args.predictions_latest)

    predictions_payload = _load_json(args.predictions_latest)
    monitoring_payload: Dict[str, Any] = {}
    if args.monitoring_latest.exists():
        monitoring_payload = _load_json(args.monitoring_latest)

    generated_at = predictions_payload.get("generated_at")
    generated_at_dt = _parse_ts(generated_at)

    prediction_rows = _extract_horizon_payloads(predictions_payload)
    monitoring_rows = _extract_monitoring_horizon_payloads(monitoring_payload)

    horizons_out: Dict[str, Dict[str, Any]] = {}
    for label, row in prediction_rows.items():
        monitoring_row = monitoring_rows.get(label)
        horizons_out[label] = _normalize_horizon_row(
            label=label,
            row=row,
            monitoring_row=monitoring_row,
            generated_at_dt=generated_at_dt,
        )

    summary = _compute_summary(horizons_out)

    output_payload = {
        "generated_at": _to_iso_utc(datetime.now(timezone.utc)),
        "run_id": str(args.run_id),
        "profile": {
            "id": str(args.profile_id),
            "name": str(args.profile_name),
        },
        "sources": {
            "predictions_latest_path": str(args.predictions_latest),
            "monitoring_latest_path": str(args.monitoring_latest),
            "predictions_generated_at": generated_at,
            "monitoring_generated_at": monitoring_payload.get("generated_at"),
            "monitoring_source": monitoring_payload.get("source"),
        },
        "monitoring_context": {
            "request": monitoring_payload.get("request") if isinstance(monitoring_payload.get("request"), dict) else {},
            "regime": monitoring_payload.get("regime") if isinstance(monitoring_payload.get("regime"), dict) else {},
        },
        "horizons": horizons_out,
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
