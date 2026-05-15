from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from src.runtime.reliability_workflow_common import load_yaml


def parse_weight_spec(spec: str | None, *, allowed_components: Sequence[str] | None = None) -> Dict[str, float]:
    if not spec:
        return {}
    allowed = {str(value) for value in allowed_components} if allowed_components else None
    parsed: Dict[str, float] = {}
    for raw_chunk in str(spec).split(","):
        chunk = raw_chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        raw_name, raw_value = chunk.split(":", 1)
        name = raw_name.strip()
        if allowed is not None and name not in allowed:
            continue
        try:
            parsed[name] = float(raw_value.strip())
        except ValueError:
            continue
    return parsed


def format_weight_spec(weights: Dict[str, float]) -> str:
    ordered = [
        "transformer",
        "transformer_large",
        "lstm",
        "bilstm",
        "gru",
        "cnn_lstm",
        "cnn_bilstm",
        "garch_lstm",
        "xgb",
        "lgbm",
        "regime_logit",
    ]
    seen = set()
    parts: List[str] = []
    for name in ordered:
        if name in weights:
            parts.append(f"{name}:{weights[name]:.1f}")
            seen.add(name)
    for name in sorted(weights):
        if name not in seen:
            parts.append(f"{name}:{weights[name]:.1f}")
    return ",".join(parts)


def extract_audit_weight_spec(
    audit_payload: Dict[str, Any],
    *,
    allowed_components: Sequence[str] | None = None,
) -> str | None:
    recs = audit_payload.get("weight_recommendations") if isinstance(audit_payload, dict) else None
    if not isinstance(recs, dict):
        return None
    weights = recs.get("recommended_weights")
    if isinstance(weights, dict):
        filtered = {
            str(name): float(value)
            for name, value in weights.items()
            if allowed_components is None or str(name) in set(str(v) for v in allowed_components)
        }
        if any(float(value) > 0.0 for value in filtered.values()):
            return format_weight_spec(filtered)
    spec = recs.get("recommended_weight_spec_1h")
    parsed = parse_weight_spec(str(spec) if spec is not None else None, allowed_components=allowed_components)
    positive = {name: value for name, value in parsed.items() if float(value) > 0.0}
    return format_weight_spec(positive) if positive else None


def build_audit_weighted_runtime_config(
    *,
    base_config_path: Path,
    audit_payload: Dict[str, Any],
    output_path: Path,
) -> bool:
    payload = load_yaml(base_config_path)
    recs = audit_payload.get("weight_recommendations") if isinstance(audit_payload, dict) else None
    if not isinstance(recs, dict):
        return False

    regime_specs = recs.get("recommended_regime_weights_1h")
    fallback_spec = recs.get("recommended_weight_spec_1h")
    apply_fallback_for_missing_regimes = bool(recs.get("apply_fallback_for_missing_regimes", True))
    if not isinstance(regime_specs, dict) and not fallback_spec:
        return False

    regime_model_weights = payload.get("regime_model_weights")
    if not isinstance(regime_model_weights, dict):
        regime_model_weights = {}
    regime_model_weights["enabled"] = True

    for regime in ("trend_ignition", "neutral", "chop"):
        spec = None
        if isinstance(regime_specs, dict):
            raw = regime_specs.get(regime)
            if raw is not None:
                spec = str(raw)
        if spec is None and apply_fallback_for_missing_regimes and fallback_spec is not None:
            spec = str(fallback_spec)
        if spec is None:
            continue

        current = regime_model_weights.get(regime)
        if isinstance(current, dict):
            updated = dict(current)
            horizon_key: Any = "1"
            for existing_key in updated:
                try:
                    if float(existing_key) == 1.0:
                        horizon_key = existing_key
                        break
                except (TypeError, ValueError):
                    continue
            updated[horizon_key] = spec
            regime_model_weights[regime] = updated
        else:
            regime_model_weights[regime] = {"1": spec}

    payload["regime_model_weights"] = regime_model_weights
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return True


def join_horizons(horizons: Sequence[float | int]) -> str:
    return ",".join(str(v) for v in horizons)


def format_horizon_label(horizon: float | int) -> str:
    value = float(horizon)
    if value.is_integer() and value >= 1.0:
        return f"{int(round(value))}h"
    if value < 1.0:
        return f"{int(round(value * 60))}m"
    return f"{value:g}h"


def load_prediction_targets(config_path: Path | None) -> List[float]:
    if config_path is None or not config_path.exists():
        return []
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return []
    resolved: List[float] = []
    for item in targets:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            resolved.append(value)
    return resolved