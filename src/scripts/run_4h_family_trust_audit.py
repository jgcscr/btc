from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from src.runtime.trust_hardening_support import _evaluate_metadata_risk, resolve_trust_hardening_policy


DEFAULT_CONFIG = "configs/run_refresh_and_predict.live_conservative_binance_only.yaml"
DEFAULT_MODELS_ROOT = "artifacts/models"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_family_trust_audit_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_family_trust_audit_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all existing 4h direction family summaries against trust metadata rules and a predictive floor."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models-root", default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--min-test-accuracy", type=float, default=0.55)
    parser.add_argument("--min-test-auc", type=float, default=0.52)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _normalize_horizon_value(value: Any) -> float:
    return float(value)


def _coerce_numeric_horizon(value: Any) -> float | None:
    return _safe_float(value)


def _train_val_gap(summary: Mapping[str, Any]) -> float | None:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    train = metrics.get("train") if isinstance(metrics.get("train"), Mapping) else {}
    val = metrics.get("val") if isinstance(metrics.get("val"), Mapping) else {}
    train_acc = _safe_float(train.get("accuracy"))
    val_acc = _safe_float(val.get("accuracy"))
    if train_acc is None or val_acc is None:
        return None
    return abs(train_acc - val_acc)


def _test_quality(summary: Mapping[str, Any]) -> Dict[str, float | None]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), Mapping) else {}
    return {
        "test_accuracy": _safe_float(test.get("accuracy")),
        "test_auc": _safe_float(test.get("roc_auc") if test.get("roc_auc") is not None else test.get("auc")),
        "test_f1": _safe_float(test.get("f1")),
    }


def _family_name_from_path(path: Path) -> str:
    return path.parent.name


def _scan_4h_summaries(models_root: Path) -> List[Path]:
    out: List[Path] = []
    for path in models_root.rglob("summary.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if str(payload.get("target")) != "direction_4h":
            continue
        out.append(path)
    return sorted(out)


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = ["# 4h Family Trust Audit", ""]
    lines.append("## Recommendation")
    lines.append(f"- replacement_candidate_found: {payload.get('replacement_candidate_found')}")
    best = payload.get("best_candidate") or {}
    lines.append(f"- best_candidate: {best.get('family') or 'none'}")
    lines.append("")
    lines.append("## Candidates")
    for item in payload.get("rows", []):
        if not isinstance(item, Mapping):
            continue
        lines.append(f"- {item.get('family')}: trust_pass={item.get('trust_pass')}, quality_pass={item.get('quality_pass')}, overall_pass={item.get('overall_pass')}, train_val_gap={item.get('train_val_gap')}, test_accuracy={item.get('test_accuracy')}, test_auc={item.get('test_auc')}, metadata_reasons={item.get('metadata_reasons')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    models_root = Path(args.models_root)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    trust_policy = resolve_trust_hardening_policy(
        config_payload.get("trust_hardening_policy"),
        normalize_horizon_value=_normalize_horizon_value,
        coerce_numeric_horizon=_coerce_numeric_horizon,
    )
    metadata_cfg = trust_policy.get("metadata_checks") if isinstance(trust_policy.get("metadata_checks"), Mapping) else {}

    rows: List[Dict[str, Any]] = []
    for summary_path in _scan_4h_summaries(models_root):
        summary = _read_json(summary_path)
        metadata_reasons, metadata_missing = _evaluate_metadata_risk(4.0, {4.0: str(summary_path)}, metadata_cfg)
        quality = _test_quality(summary)
        gap = _train_val_gap(summary)
        effective_metadata_reasons = list(metadata_reasons)
        if gap is None:
            effective_metadata_reasons.append("metadata_incomplete_train_val_accuracy")
        trust_pass = not effective_metadata_reasons and not metadata_missing
        quality_pass = (
            (quality["test_accuracy"] is not None and quality["test_accuracy"] >= float(args.min_test_accuracy))
            and (
                quality["test_auc"] is None
                or quality["test_auc"] >= float(args.min_test_auc)
            )
        )
        rows.append(
            {
                "family": _family_name_from_path(summary_path),
                "summary_path": str(summary_path),
                "train_val_gap": gap,
                "test_accuracy": quality["test_accuracy"],
                "test_auc": quality["test_auc"],
                "test_f1": quality["test_f1"],
                "metadata_reasons": effective_metadata_reasons,
                "metadata_missing": bool(metadata_missing),
                "trust_pass": bool(trust_pass),
                "quality_pass": bool(quality_pass),
                "overall_pass": bool(trust_pass and quality_pass),
            }
        )

    rows.sort(
        key=lambda item: (
            0 if item.get("overall_pass") else 1,
            0 if item.get("trust_pass") else 1,
            0 if item.get("quality_pass") else 1,
            item.get("train_val_gap") if item.get("train_val_gap") is not None else 999.0,
            -(item.get("test_accuracy") or -999.0),
        )
    )

    best_candidate = next((row for row in rows if row.get("overall_pass")), None)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "models_root": str(models_root),
            "min_test_accuracy": float(args.min_test_accuracy),
            "min_test_auc": float(args.min_test_auc),
        },
        "replacement_candidate_found": best_candidate is not None,
        "best_candidate": best_candidate,
        "rows": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote 4h family audit JSON: {output_json}")
    print(f"Wrote 4h family audit memo: {output_md}")


if __name__ == "__main__":
    main()