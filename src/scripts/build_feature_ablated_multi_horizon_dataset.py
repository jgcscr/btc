from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


RAW_PRICE_LEVELS_PROFILE = [
    "open",
    "high",
    "low",
    "close",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "intrabar_open_first",
    "intrabar_close_last",
    "intrabar_path_high",
    "intrabar_path_low",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feature-ablated copy of the multi-horizon NPZ dataset."
    )
    parser.add_argument(
        "--input-path",
        default="artifacts/datasets/btc_features_multi_horizon_splits.npz",
    )
    parser.add_argument(
        "--output-path",
        required=True,
    )
    parser.add_argument(
        "--report-path",
        default=None,
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Feature name to remove. Repeatable.",
    )
    parser.add_argument(
        "--profile",
        choices=("raw_price_levels",),
        default=None,
    )
    return parser.parse_args()


def _resolve_exclusions(profile: str | None, explicit: Iterable[str]) -> List[str]:
    names = {str(value).strip() for value in explicit if str(value).strip()}
    if profile == "raw_price_levels":
        names.update(RAW_PRICE_LEVELS_PROFILE)
    return sorted(names)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".json")

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    exclusions = _resolve_exclusions(args.profile, args.exclude_feature)
    if not exclusions:
        raise ValueError("No features were selected for exclusion.")

    with np.load(input_path, allow_pickle=True) as data:
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        keep_indices = [idx for idx, name in enumerate(feature_names) if name not in exclusions]
        removed_features = [name for name in feature_names if name in exclusions]
        missing_features = [name for name in exclusions if name not in feature_names]
        if not removed_features:
            raise ValueError("Requested exclusions did not match any dataset features.")

        payload: Dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if key in {"X_train", "X_val", "X_test"}:
                payload[key] = value[:, keep_indices]
            elif key in {"feature_names", "scaler_mean", "scaler_scale"}:
                payload[key] = value[keep_indices]
            else:
                payload[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "profile": args.profile,
        "requested_exclusions": exclusions,
        "removed_features": removed_features,
        "missing_features": missing_features,
        "original_feature_count": len(feature_names),
        "ablated_feature_count": len(keep_indices),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote ablated dataset: {output_path}")
    print(f"Removed {len(removed_features)} features; {len(keep_indices)} remain.")
    print(f"Wrote ablation report: {report_path}")


if __name__ == "__main__":
    main()