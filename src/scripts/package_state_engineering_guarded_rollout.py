from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


NARROW_SCOPE_ARTIFACT = Path("artifacts/analysis/state_engineering_narrow_scope_latest.json")
GUARDED_JSON = Path("artifacts/analysis/state_engineering_guarded_shadow_4h_latest.json")
GUARDED_MD = Path("artifacts/analysis/state_engineering_guarded_shadow_4h_latest.md")
OUTPUT_JSON = Path("artifacts/analysis/state_engineering_guarded_rollout_package.json")
OUTPUT_MD = Path("artifacts/analysis/state_engineering_guarded_rollout_package.md")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# State-Engineering Guarded Rollout Package",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "## Recommendation",
        "",
        "Keep state-engineering constrained to the guarded 4h-only shadow runner instead of widening it into a broader profile change.",
        "",
        "## Package",
        "",
        f"- Narrow-scope artifact: {payload.get('narrow_scope_artifact')}",
        f"- Guarded shadow JSON: {payload.get('guarded_shadow_json')}",
        f"- Guarded shadow markdown: {payload.get('guarded_shadow_markdown')}",
        f"- Recommended command: {payload.get('recommended_command')}",
        "",
        "## Readiness",
        "",
        f"- Narrow-scope decision: {payload.get('narrow_scope_decision')}",
        f"- Narrow-scope reason: {payload.get('narrow_scope_reason')}",
        f"- Guarded runner readiness: {payload.get('guarded_shadow_readiness')}",
        f"- Guarded runner assessment: {payload.get('guarded_shadow_assessment')}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reusable rollout package for the guarded 4h-only state-engineering shadow lane.")
    parser.add_argument("--narrow-scope-artifact", type=Path, default=NARROW_SCOPE_ARTIFACT)
    parser.add_argument("--guarded-json", type=Path, default=GUARDED_JSON)
    parser.add_argument("--guarded-markdown", type=Path, default=GUARDED_MD)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    args = parser.parse_args()

    narrow_scope = _load_json(args.narrow_scope_artifact)
    guarded_shadow = _load_json(args.guarded_json)
    narrow_final = narrow_scope.get("final_recommendation") if isinstance(narrow_scope.get("final_recommendation"), Mapping) else {}
    guarded_readiness = guarded_shadow.get("readiness") if isinstance(guarded_shadow.get("readiness"), Mapping) else {}
    guarded_summary = guarded_shadow.get("summary") if isinstance(guarded_shadow.get("summary"), Mapping) else {}

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "narrow_scope_artifact": str(args.narrow_scope_artifact),
        "guarded_shadow_json": str(args.guarded_json),
        "guarded_shadow_markdown": str(args.guarded_markdown),
        "recommended_command": "/workspaces/btc/.venv/bin/python -m src.scripts.run_state_engineering_guarded_shadow",
        "narrow_scope_decision": str(narrow_final.get("decision") or "unknown"),
        "narrow_scope_reason": str(narrow_final.get("reason") or "unknown"),
        "guarded_shadow_readiness": str(guarded_readiness.get("decision") or "unknown"),
        "guarded_shadow_assessment": str(guarded_summary.get("assessment") or "unknown"),
        "notes": [
            "This package is shadow-only and does not authorize broader state-engineering rollout.",
            "Use it as the state/interaction follow-through lane parallel to the 4h feature-lift package.",
            "If the guarded shadow artifact is stale or missing, rerun src.scripts.run_state_engineering_guarded_shadow before review.",
        ],
    }

    _write_json(args.output_json, payload)
    _write_text(args.output_markdown, _render_markdown(payload))

    print(f"Wrote guarded rollout package JSON to {args.output_json}")
    print(f"Wrote guarded rollout package markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()