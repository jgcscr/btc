from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.runtime.family_outcome_confirmation import (
    render_state_engineering_narrow_scope_memo,
    run_state_engineering_narrow_scope_followup,
)

DEFAULT_HISTORY = "artifacts/predictions/history.json"
DEFAULT_SPOT_DIR = "data/spot_klines"
DEFAULT_JSON_OUT = "artifacts/analysis/state_engineering_narrow_scope_latest.json"
DEFAULT_MD_OUT = "artifacts/analysis/state_engineering_narrow_scope_latest.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run narrow-scope follow-up validation for state_engineering weak_signal_veto_only.",
    )
    parser.add_argument("--history-path", default=DEFAULT_HISTORY)
    parser.add_argument("--spot-dir", default=DEFAULT_SPOT_DIR)
    parser.add_argument("--recent-window", type=int, default=0, help="0 means use the full available history.")
    parser.add_argument("--max-staleness-hours", type=float, default=6.0)
    parser.add_argument("--output-json", default=DEFAULT_JSON_OUT)
    parser.add_argument("--output-md", default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_state_engineering_narrow_scope_followup(
        history_path=Path(args.history_path),
        spot_dir=Path(args.spot_dir),
        recent_window=int(args.recent_window),
        max_staleness_hours=float(args.max_staleness_hours),
        family="state_engineering",
        variant="weak_signal_veto_only",
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["inputs"] = {
        "history_path": str(args.history_path),
        "spot_dir": str(args.spot_dir),
        "recent_window": int(args.recent_window),
        "max_staleness_hours": float(args.max_staleness_hours),
        "family": "state_engineering",
        "variant": "weak_signal_veto_only",
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_state_engineering_narrow_scope_memo(payload), encoding="utf-8")

    print(f"Wrote state narrow-scope JSON: {out_json}")
    print(f"Wrote state narrow-scope memo: {out_md}")


if __name__ == "__main__":
    main()