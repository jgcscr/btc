from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from src.runtime.family_shadow_simulator import (
    default_family_policy_variants,
    load_prediction_history,
    load_spot_feature_frame,
    render_family_shadow_markdown_report,
    run_state_order_flow_shadow_validation,
)


DEFAULT_CONFIG = "configs/run_refresh_and_predict.live_conservative_binance_only.yaml"
DEFAULT_HISTORY = "artifacts/predictions/history.json"
DEFAULT_SPOT_DIR = "data/spot_klines"
DEFAULT_JSON_OUT = "artifacts/analysis/state_orderflow_shadow_enforcement_latest.json"
DEFAULT_MD_OUT = "artifacts/analysis/state_orderflow_shadow_enforcement_latest.md"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay prediction history with shadow-only state_engineering/order_flow policy sweeps.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Live profile config path.")
    parser.add_argument("--history-path", default=DEFAULT_HISTORY, help="Prediction history JSON path.")
    parser.add_argument("--spot-dir", default=DEFAULT_SPOT_DIR, help="Directory of spot kline parquet files.")
    parser.add_argument("--recent-window", type=int, default=2000, help="Number of most recent snapshots to replay.")
    parser.add_argument("--max-staleness-hours", type=float, default=6.0, help="Feature freshness threshold.")
    parser.add_argument("--output-json", default=DEFAULT_JSON_OUT, help="JSON output path.")
    parser.add_argument("--output-md", default=DEFAULT_MD_OUT, help="Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = _load_config(Path(args.config))
    history = load_prediction_history(Path(args.history_path))
    if not history:
        raise SystemExit("No prediction history snapshots found.")
    if args.recent_window > 0:
        history = history[-args.recent_window :]

    feature_frame = load_spot_feature_frame(Path(args.spot_dir))
    sweep = run_state_order_flow_shadow_validation(
        snapshots=history,
        feature_frame=feature_frame,
        policies=default_family_policy_variants(),
        max_staleness_hours=float(args.max_staleness_hours),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "history_path": str(args.history_path),
        "spot_dir": str(args.spot_dir),
        "recent_window": int(args.recent_window),
        "max_staleness_hours": float(args.max_staleness_hours),
        "live_profile_context": {
            "targets": cfg.get("targets", []),
            "feature_coverage_policy": cfg.get("feature_coverage_policy", {}),
        },
        "sweep": sweep,
        "conclusion": {
            "best_family": sweep.get("overall_recommendation", {}).get("best_family"),
            "best_policy": sweep.get("overall_recommendation", {}).get("best_policy"),
            "move_to_next_validation_stage": bool(
                sweep.get("overall_recommendation", {}).get("advance_to_deeper_validation", False)
            ),
            "macro_disposition": "remain_deprioritized",
            "note": "Shadow replay only; production behavior unchanged.",
        },
    }

    json_out = Path(args.output_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_out = Path(args.output_md)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_family_shadow_markdown_report(payload), encoding="utf-8")

    print(f"Wrote family shadow JSON: {json_out}")
    print(f"Wrote family shadow report: {md_out}")


if __name__ == "__main__":
    main()
