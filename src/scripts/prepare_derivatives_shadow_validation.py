from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.runtime.signal_program_support import (
    DEFAULT_MODELS_ROOT,
    build_derivatives_family_audit,
    build_derivatives_shadow_scaffold,
    render_derivatives_audit_markdown,
    render_derivatives_scaffold_markdown,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Config must be a YAML mapping.")
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare derivatives-family shadow-validation scaffold and readiness artifact.",
    )
    parser.add_argument("--config", default="configs/run_refresh_and_predict.live_conservative_binance_only.yaml")
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--audit-json", default="artifacts/analysis/derivatives_family_audit_latest.json")
    parser.add_argument("--audit-md", default="artifacts/analysis/derivatives_family_audit_latest.md")
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
    generated_at = datetime.now(timezone.utc).isoformat()

    audit = build_derivatives_family_audit(
        config=config,
        models_root=Path(args.models_root),
    )
    audit["generated_at"] = generated_at
    scaffold = build_derivatives_shadow_scaffold(audit)
    scaffold["generated_at"] = generated_at

    _write_json(Path(args.audit_json), audit)
    _write_text(Path(args.audit_md), render_derivatives_audit_markdown(audit))
    _write_json(Path(args.scaffold_json), scaffold)
    _write_text(Path(args.scaffold_md), render_derivatives_scaffold_markdown(scaffold))

    print(f"Wrote derivatives audit JSON: {args.audit_json}")
    print(f"Wrote derivatives audit markdown: {args.audit_md}")
    print(f"Wrote derivatives scaffold JSON: {args.scaffold_json}")
    print(f"Wrote derivatives scaffold markdown: {args.scaffold_md}")


if __name__ == "__main__":
    main()