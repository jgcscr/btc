from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.runtime.signal_program_support import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_MODELS_ROOT,
    build_derivatives_family_audit,
    build_derivatives_shadow_candidate_config,
    build_derivatives_shadow_scaffold,
    build_signal_expansion_rollout_summary,
    build_signal_program_dispositions,
)


BASE_CONFIG = Path("configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
DERIVATIVES_CONFIG = Path("configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml")
OUTPUT_JSON = Path("artifacts/analysis/signal_expansion_rollout_latest.json")
OUTPUT_MD = Path("artifacts/analysis/signal_expansion_rollout_latest.md")
FEATURELIFT_CONFIG = Path("configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml")
FEATURELIFT_MD = Path("artifacts/analysis/featurelift_20260331_rerun/shadow_rollout_4h_package.md")
STATE_GUARDED_JSON = Path("artifacts/analysis/state_engineering_guarded_shadow_4h_latest.json")
STATE_GUARDED_MD = Path("artifacts/analysis/state_engineering_guarded_shadow_4h_latest.md")
STATE_GUARDED_PACKAGE_MD = Path("artifacts/analysis/state_engineering_guarded_rollout_package.md")


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
    direction = payload.get("program_direction", {}) if isinstance(payload.get("program_direction"), Mapping) else {}
    derivatives = direction.get("derivatives", {}) if isinstance(direction.get("derivatives"), Mapping) else {}
    featurelift = direction.get("featurelift_4h", {}) if isinstance(direction.get("featurelift_4h"), Mapping) else {}
    state = direction.get("state_engineering", {}) if isinstance(direction.get("state_engineering"), Mapping) else {}
    macro = direction.get("macro", {}) if isinstance(direction.get("macro"), Mapping) else {}
    lines = [
        "# Signal Expansion Rollout",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "## Direction",
        "",
        f"- Next priority family: {payload.get('next_priority_family')}",
        f"- Derivatives readiness: {derivatives.get('status')}",
        f"- Derivatives candidate config: {derivatives.get('candidate_config')}",
        f"- 4h feature-lift config: {featurelift.get('candidate_config')}",
        f"- State guarded shadow artifact: {state.get('guarded_shadow_json')}",
        f"- Macro posture: {macro.get('recommended_action')}",
        "",
        "## Commands",
        "",
        "- `/workspaces/btc/.venv/bin/python -m src.scripts.package_derivatives_shadow_rollout`",
        "- `/workspaces/btc/.venv/bin/python -m src.scripts.package_featurelift_4h_shadow_rollout`",
        "- `/workspaces/btc/.venv/bin/python -m src.scripts.package_state_engineering_guarded_rollout`",
        "- `/workspaces/btc/.venv/bin/python -m src.scripts.run_state_engineering_guarded_shadow`",
        "- `/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict --config configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml --dry-run`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in payload.get("implementation_notes", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a single rollout summary for derivatives, 4h feature-lift, and guarded state-engineering validation.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--analysis-dir", type=Path, default=Path(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--models-root", type=Path, default=Path(DEFAULT_MODELS_ROOT))
    parser.add_argument("--derivatives-config", type=Path, default=DERIVATIVES_CONFIG)
    parser.add_argument("--featurelift-config", type=Path, default=FEATURELIFT_CONFIG)
    parser.add_argument("--featurelift-package-markdown", type=Path, default=FEATURELIFT_MD)
    parser.add_argument("--state-guarded-json", type=Path, default=STATE_GUARDED_JSON)
    parser.add_argument("--state-guarded-markdown", type=Path, default=STATE_GUARDED_MD)
    parser.add_argument("--state-guarded-package-markdown", type=Path, default=STATE_GUARDED_PACKAGE_MD)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MD)
    args = parser.parse_args()

    base_config = _load_yaml(args.base_config)
    signal_payload = build_signal_program_dispositions(args.analysis_dir)
    derivatives_audit = build_derivatives_family_audit(config=base_config, models_root=args.models_root)
    derivatives_scaffold = build_derivatives_shadow_scaffold(derivatives_audit)
    derivatives_candidate = build_derivatives_shadow_candidate_config(base_config, audit=derivatives_audit)

    args.derivatives_config.parent.mkdir(parents=True, exist_ok=True)
    args.derivatives_config.write_text(yaml.safe_dump(derivatives_candidate, sort_keys=False), encoding="utf-8")

    payload = build_signal_expansion_rollout_summary(
        signal_payload=signal_payload,
        derivatives_audit=derivatives_audit,
        derivatives_scaffold=derivatives_scaffold,
        derivatives_config_path=str(args.derivatives_config),
        featurelift_config_path=str(args.featurelift_config),
        featurelift_package_path=str(args.featurelift_package_markdown),
        state_guarded_json_path=str(args.state_guarded_json),
        state_guarded_md_path=str(args.state_guarded_markdown),
    )
    payload["generated_at"] = datetime.now(UTC).isoformat()
    payload["program_direction"]["state_engineering"]["guarded_rollout_package_markdown"] = str(
        args.state_guarded_package_markdown
    )

    _write_json(args.output_json, payload)
    _write_text(args.output_markdown, _render_markdown(payload))

    print(f"Wrote derivatives candidate config to {args.derivatives_config}")
    print(f"Wrote rollout summary JSON to {args.output_json}")
    print(f"Wrote rollout summary markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()