from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _extract_marginal_rerank_policy(
    audit_payload: Mapping[str, Any],
    *,
    horizons: Sequence[float],
) -> Dict[str, Any] | None:
    recs = audit_payload.get("weight_recommendations") if isinstance(audit_payload, Mapping) else None
    if not isinstance(recs, Mapping):
        return None

    marginal_band = audit_payload.get("marginal_band") if isinstance(audit_payload, Mapping) else None
    lower = 0.5
    upper = 0.6
    if isinstance(marginal_band, Mapping):
        try:
            lower = float(marginal_band.get("lower", lower))
        except (TypeError, ValueError):
            lower = 0.5
        try:
            upper = float(marginal_band.get("upper", upper))
        except (TypeError, ValueError):
            upper = 0.6
    if upper < lower:
        lower, upper = upper, lower

    regime_specs = recs.get("recommended_regime_weights_1h")
    fallback_spec = recs.get("recommended_weight_spec_1h")
    weight_specs: Dict[str, str] = {}
    if fallback_spec is not None:
        weight_specs["default"] = str(fallback_spec)
    if isinstance(regime_specs, Mapping):
        for regime, spec in regime_specs.items():
            if spec is None:
                continue
            weight_specs[str(regime)] = str(spec)
    if not weight_specs:
        return None

    return {
        "enabled": True,
        "horizons": [float(horizon) for horizon in horizons],
        "lower": float(lower),
        "upper": float(upper),
        "min_component_count": 2,
        "use_raw_probability_gate": True,
        "weight_specs": weight_specs,
    }


def build_shadow_config(
    *,
    base_config_path: Path,
    direction_output_calibration_path: Path,
    output_path: Path,
    marginal_audit_path: Path | None = None,
    neutral_band: float = 0.02,
    horizons: Sequence[float] = (1.0,),
) -> Dict[str, Any]:
    config = _load_yaml(base_config_path)
    marginal_rerank_policy = None
    if marginal_audit_path is not None and marginal_audit_path.exists():
        marginal_rerank_policy = _extract_marginal_rerank_policy(
            _load_json(marginal_audit_path),
            horizons=horizons,
        )

    direction_output_policy: Dict[str, Any] = {
        "enabled": True,
        "horizons": [float(horizon) for horizon in horizons],
        "neutral_band": float(neutral_band),
        "calibration_path": str(direction_output_calibration_path),
        "use_trade_probability_fallback": True,
    }
    if marginal_rerank_policy is not None:
        direction_output_policy["marginal_rerank"] = marginal_rerank_policy

    config["direction_output_policy"] = direction_output_policy
    config["write_artifacts"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "base_config": str(base_config_path),
        "direction_output_calibration_path": str(direction_output_calibration_path),
        "marginal_audit_path": None if marginal_audit_path is None else str(marginal_audit_path),
        "audit_weights_applied": bool(marginal_rerank_policy is not None),
        "marginal_rerank_applied": bool(marginal_rerank_policy is not None),
        "marginal_rerank": marginal_rerank_policy,
        "neutral_band": float(neutral_band),
        "horizons": [float(horizon) for horizon in horizons],
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shadow runtime config that applies direction-output calibration and optional marginal 1h rerank weights.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--direction-output-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marginal-audit", type=Path, default=None)
    parser.add_argument("--neutral-band", type=float, default=0.02)
    parser.add_argument("--horizons", type=float, nargs="+", default=[1.0])
    args = parser.parse_args()

    payload = build_shadow_config(
        base_config_path=args.base_config,
        direction_output_calibration_path=args.direction_output_calibration,
        output_path=args.output,
        marginal_audit_path=args.marginal_audit,
        neutral_band=float(args.neutral_band),
        horizons=args.horizons,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()