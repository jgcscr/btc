from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.runtime.signal_program_support import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_MODELS_ROOT,
    build_derivatives_family_audit,
    build_derivatives_shadow_scaffold,
    build_signal_program_dispositions,
    render_derivatives_audit_markdown,
    render_derivatives_scaffold_markdown,
    render_signal_program_markdown,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Config must be a YAML mapping.")
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize current signal-program dispositions and audit derivatives-family readiness.",
    )
    parser.add_argument("--config", default="configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--signal-json", default="artifacts/analysis/signal_program_dispositions_latest.json")
    parser.add_argument("--signal-md", default="artifacts/analysis/signal_program_dispositions_latest.md")
    parser.add_argument("--derivatives-json", default="artifacts/analysis/derivatives_family_audit_latest.json")
    parser.add_argument("--derivatives-md", default="artifacts/analysis/derivatives_family_audit_latest.md")
    parser.add_argument("--scaffold-json", default="artifacts/analysis/derivatives_shadow_scaffold_latest.json")
    parser.add_argument("--scaffold-md", default="artifacts/analysis/derivatives_shadow_scaffold_latest.md")
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config))

    signal_payload = build_signal_program_dispositions(Path(args.analysis_dir))
    signal_payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    derivatives_payload = build_derivatives_family_audit(
        config=config,
        models_root=Path(args.models_root),
    )
    derivatives_payload["generated_at"] = signal_payload["generated_at"]
    derivatives_payload["config_path"] = str(args.config)

    next_priority = derivatives_payload.get("next_priority_confirmation", {}) if isinstance(derivatives_payload.get("next_priority_confirmation"), Mapping) else {}
    state_payload = signal_payload.get("families", {}).get("state_engineering", {}) if isinstance(signal_payload.get("families"), Mapping) else {}
    state_closed = str(state_payload.get("status") or "") == "closed"
    if next_priority and not state_closed:
        signal_payload["next_priority_family"] = str(next_priority.get("family") or signal_payload.get("next_priority_family"))
        signal_payload.setdefault("notes", []).append(str(next_priority.get("reason") or ""))
    elif state_closed:
        signal_payload["next_priority_family"] = "derivatives"
        signal_payload.setdefault("notes", []).append(
            "No active replay candidate remains; only derivatives data-and-training enablement is left as a next engineering lane."
        )

    scaffold_payload = build_derivatives_shadow_scaffold(derivatives_payload)
    scaffold_payload["generated_at"] = signal_payload["generated_at"]

    _write_json(Path(args.signal_json), signal_payload)
    _write_text(Path(args.signal_md), render_signal_program_markdown(signal_payload))

    _write_json(Path(args.derivatives_json), derivatives_payload)
    _write_text(Path(args.derivatives_md), render_derivatives_audit_markdown(derivatives_payload))

    _write_json(Path(args.scaffold_json), scaffold_payload)
    _write_text(Path(args.scaffold_md), render_derivatives_scaffold_markdown(scaffold_payload))

    print(f"Wrote signal dispositions JSON: {args.signal_json}")
    print(f"Wrote signal dispositions markdown: {args.signal_md}")
    print(f"Wrote derivatives audit JSON: {args.derivatives_json}")
    print(f"Wrote derivatives audit markdown: {args.derivatives_md}")
    print(f"Wrote derivatives scaffold JSON: {args.scaffold_json}")
    print(f"Wrote derivatives scaffold markdown: {args.scaffold_md}")


if __name__ == "__main__":
    main()