from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concise comparison artifact for default vs midband-paper profile.")
    parser.add_argument("--default-summary", type=Path, required=True)
    parser.add_argument("--midband-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.default_summary.exists():
        raise FileNotFoundError(args.default_summary)
    if not args.midband_summary.exists():
        raise FileNotFoundError(args.midband_summary)

    default_payload = _load(args.default_summary)
    midband_payload = _load(args.midband_summary)

    default_metrics = default_payload.get("metrics", {}) if isinstance(default_payload.get("metrics"), dict) else {}
    midband_metrics = midband_payload.get("metrics", {}) if isinstance(midband_payload.get("metrics"), dict) else {}

    comparison = {
        "default_profile": {
            "profile_id": default_payload.get("profile_id"),
            "profile_name": default_payload.get("profile_name"),
            "run_id": default_payload.get("run_id"),
            "metrics": default_metrics,
        },
        "midband_paper_profile": {
            "profile_id": midband_payload.get("profile_id"),
            "profile_name": midband_payload.get("profile_name"),
            "run_id": midband_payload.get("run_id"),
            "metrics": midband_metrics,
        },
        "deltas_midband_minus_default": {
            "candidate_trade_count": int(midband_metrics.get("candidate_trade_count", 0) or 0)
            - int(default_metrics.get("candidate_trade_count", 0) or 0),
            "incumbent_trade_count": int(midband_metrics.get("incumbent_trade_count", 0) or 0)
            - int(default_metrics.get("incumbent_trade_count", 0) or 0),
            "candidate_net_return_total": float(midband_metrics.get("candidate_net_return_total", 0.0) or 0.0)
            - float(default_metrics.get("candidate_net_return_total", 0.0) or 0.0),
            "incumbent_net_return_total": float(midband_metrics.get("incumbent_net_return_total", 0.0) or 0.0)
            - float(default_metrics.get("incumbent_net_return_total", 0.0) or 0.0),
            "mean_diff": float(midband_metrics.get("mean_diff", 0.0) or 0.0)
            - float(default_metrics.get("mean_diff", 0.0) or 0.0),
            "pvalue_one_sided": float(midband_metrics.get("pvalue_one_sided", 0.0) or 0.0)
            - float(default_metrics.get("pvalue_one_sided", 0.0) or 0.0),
            "nonzero_paired_rows": int(midband_metrics.get("nonzero_paired_rows", 0) or 0)
            - int(default_metrics.get("nonzero_paired_rows", 0) or 0),
            "std_diff": float(midband_metrics.get("std_diff", 0.0) or 0.0)
            - float(default_metrics.get("std_diff", 0.0) or 0.0),
            "vetoed_row_count": (midband_metrics.get("vetoed_row_count") or 0) - (default_metrics.get("vetoed_row_count") or 0),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
