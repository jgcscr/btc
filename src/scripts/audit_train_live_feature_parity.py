from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.runtime.feature_parity_audit import build_parity_audit, render_markdown_report
from src.scripts import run_live_inference


def _parse_targets(raw: Any) -> list[float]:
    if isinstance(raw, list):
        return [float(v) for v in raw]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return [float(v) for v in parts]
    return [0.25, 1.0, 4.0, 12.0]


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Config must be a YAML mapping.")
    return dict(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit train/live feature family parity for live conservative runtime.")
    parser.add_argument(
        "--config",
        default=run_live_inference.DEFAULT_LIVE_CONFIG,
        help="Live runtime config to inspect.",
    )
    parser.add_argument(
        "--models-root",
        default="artifacts/models",
        help="Root directory containing model metadata artifacts.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/analysis/train_live_feature_parity_latest.json",
        help="Machine-readable audit output path.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/analysis/train_live_feature_parity_latest.md",
        help="Markdown audit report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = _load_config(config_path)

    targets = _parse_targets(cfg.get("targets"))
    coverage = cfg.get("feature_coverage_policy") if isinstance(cfg.get("feature_coverage_policy"), Mapping) else {}
    ignored_columns = coverage.get("ignored_columns") if isinstance(coverage.get("ignored_columns"), list) else []
    ignored_sources = coverage.get("ignored_sources") if isinstance(coverage.get("ignored_sources"), list) else []
    max_source_lag_hours = float(coverage.get("max_source_lag_hours", 1e9))

    audit = build_parity_audit(
        horizons=targets,
        models_root=Path(args.models_root),
        ignored_columns=[str(v) for v in ignored_columns],
        ignored_sources=[str(v) for v in ignored_sources],
        max_source_lag_hours=max_source_lag_hours,
    )

    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["config_path"] = str(config_path)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    md_payload = render_markdown_report(audit)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_payload, encoding="utf-8")

    print(f"Wrote parity audit JSON: {json_path}")
    print(f"Wrote parity audit report: {md_path}")


if __name__ == "__main__":
    main()
