from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.runtime.signal_program_support import (
    DEFAULT_MODELS_ROOT,
    build_derivatives_family_audit,
    build_derivatives_shadow_candidate_config,
    build_derivatives_shadow_scaffold,
    render_derivatives_audit_markdown,
    render_derivatives_scaffold_markdown,
)


BASE_CONFIG = Path("configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
OUTPUT_CONFIG = Path("configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml")
OUTPUT_JSON = Path("artifacts/analysis/derivatives_shadow_rollout_package.json")
OUTPUT_MD = Path("artifacts/analysis/derivatives_shadow_rollout_package.md")
AUDIT_JSON = Path("artifacts/analysis/derivatives_family_audit_latest.json")
AUDIT_MD = Path("artifacts/analysis/derivatives_family_audit_latest.md")
SCAFFOLD_JSON = Path("artifacts/analysis/derivatives_shadow_scaffold_latest.json")
SCAFFOLD_MD = Path("artifacts/analysis/derivatives_shadow_scaffold_latest.md")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_markdown(payload: Mapping[str, Any]) -> str:
    readiness = payload.get("readiness", {}) if isinstance(payload.get("readiness"), Mapping) else {}
    lines = [
        "# Derivatives Shadow Rollout Package",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "## Recommendation",
        "",
        "Promote derivatives into a dedicated shadow-validation candidate config before any live promotion.",
        "",
        "## Package",
        "",
        f"- Base config: {payload.get('base_config_path')}",
        f"- Shadow candidate config: {payload.get('config_path')}",
        f"- Audit JSON: {payload.get('audit_json_path')}",
        f"- Scaffold JSON: {payload.get('scaffold_json_path')}",
        "",
        "## Readiness",
        "",
        f"- Decision: {readiness.get('decision')}",
        f"- Next action: {readiness.get('next_action')}",
        f"- Runner status: {payload.get('runner_status')}",
        f"- Training derivatives feature count: {payload.get('training_derivatives_feature_count')}",
        f"- Dataset derivatives feature count: {payload.get('dataset_derivatives_feature_count')}",
        "",
        "## Blockers",
        "",
    ]
    blockers = readiness.get("blockers", []) if isinstance(readiness.get("blockers"), list) else []
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a derivatives-first shadow rollout candidate from the approved live profile.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--models-root", type=Path, default=Path(DEFAULT_MODELS_ROOT))
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON)
    parser.add_argument("--audit-markdown", type=Path, default=AUDIT_MD)
    parser.add_argument("--scaffold-json", type=Path, default=SCAFFOLD_JSON)
    parser.add_argument("--scaffold-markdown", type=Path, default=SCAFFOLD_MD)
    args = parser.parse_args()

    base_config = _load_yaml(args.base_config)
    audit = build_derivatives_family_audit(config=base_config, models_root=args.models_root)
    scaffold = build_derivatives_shadow_scaffold(audit)
    candidate_config = build_derivatives_shadow_candidate_config(base_config, audit=audit)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(candidate_config, sort_keys=False), encoding="utf-8")

    generated_at = datetime.now(UTC).isoformat()
    audit["generated_at"] = generated_at
    scaffold["generated_at"] = generated_at

    _write_json(args.audit_json, audit)
    _write_text(args.audit_markdown, render_derivatives_audit_markdown(audit))
    _write_json(args.scaffold_json, scaffold)
    _write_text(args.scaffold_markdown, render_derivatives_scaffold_markdown(scaffold))

    payload = {
        "generated_at": generated_at,
        "base_config_path": str(args.base_config),
        "config_path": str(args.output_config),
        "audit_json_path": str(args.audit_json),
        "scaffold_json_path": str(args.scaffold_json),
        "readiness": audit.get("readiness", {}),
        "runner_status": scaffold.get("runner_status"),
        "training_derivatives_feature_count": int(audit.get("training_derivatives_family_count", 0) or 0),
        "dataset_derivatives_feature_count": int(audit.get("dataset_derivatives_family_count", 0) or 0),
        "notes": [
            "This package removes funding-derived columns from the ignored live coverage list for shadow validation.",
            "Macro and on-chain remain ignored here; this candidate is intentionally derivatives-first.",
            "Use the emitted config for dry-run or shadow evaluation only.",
        ],
    }
    _write_json(args.output_json, payload)
    _write_text(args.output_markdown, _render_markdown(payload))

    print(f"Wrote shadow candidate config to {args.output_config}")
    print(f"Wrote rollout package JSON to {args.output_json}")
    print(f"Wrote rollout package markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()