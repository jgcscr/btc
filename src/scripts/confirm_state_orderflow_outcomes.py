from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.runtime.family_outcome_confirmation import (
    render_confirmation_markdown,
    run_confirmation_pass,
)

DEFAULT_SHADOW_ARTIFACT = "artifacts/analysis/state_orderflow_shadow_enforcement_latest.json"
DEFAULT_HISTORY = "artifacts/predictions/history.json"
DEFAULT_SPOT_DIR = "data/spot_klines"
DEFAULT_JSON_OUT = "artifacts/analysis/state_orderflow_outcome_confirmation_latest.json"
DEFAULT_MD_OUT = "artifacts/analysis/state_orderflow_outcome_confirmation_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm top state/order-flow shadow variants against realized outcome proxies.",
    )
    parser.add_argument("--shadow-artifact", default=DEFAULT_SHADOW_ARTIFACT)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY)
    parser.add_argument("--spot-dir", default=DEFAULT_SPOT_DIR)
    parser.add_argument("--recent-window", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--max-staleness-hours", type=float, default=6.0)
    parser.add_argument("--output-json", default=DEFAULT_JSON_OUT)
    parser.add_argument("--output-md", default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    payload = run_confirmation_pass(
        shadow_artifact_path=Path(args.shadow_artifact),
        history_path=Path(args.history_path),
        spot_dir=Path(args.spot_dir),
        recent_window=int(args.recent_window),
        max_staleness_hours=float(args.max_staleness_hours),
        top_n=int(args.top_n),
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["inputs"] = {
        "shadow_artifact": str(args.shadow_artifact),
        "history_path": str(args.history_path),
        "spot_dir": str(args.spot_dir),
        "recent_window": int(args.recent_window),
        "top_n": int(args.top_n),
        "max_staleness_hours": float(args.max_staleness_hours),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_confirmation_markdown(payload), encoding="utf-8")

    print(f"Wrote outcome confirmation JSON: {out_json}")
    print(f"Wrote outcome confirmation report: {out_md}")


if __name__ == "__main__":
    main()
