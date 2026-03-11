from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_dataset(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        feature_names = [str(name) for name in data["feature_names"].tolist()]
        X = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
        ts = np.concatenate([data["ts_train"], data["ts_val"], data["ts_test"]])
        scaler_mean = np.asarray(data["scaler_mean"], dtype=float) if "scaler_mean" in data.files else None
        scaler_scale = np.asarray(data["scaler_scale"], dtype=float) if "scaler_scale" in data.files else None

    rows: List[Dict[str, Any]] = []
    for idx, ts_value in enumerate(ts):
        timestamp = np.datetime_as_string(ts_value, unit="s")
        feature_values = {feature_names[col]: float(X[idx, col]) for col in range(len(feature_names))}
        raw_values: Dict[str, float] = {}
        if scaler_mean is not None and scaler_scale is not None:
            for col, feature_name in enumerate(feature_names):
                raw_values[feature_name] = float(X[idx, col] * scaler_scale[col] + scaler_mean[col])
        rows.append(
            {
                "timestamp": timestamp,
                "row_index": int(idx),
                "features": feature_values,
                "raw_features": raw_values,
            }
        )

    scaler_stats = {
        feature_names[idx]: {
            "mean": float(scaler_mean[idx]),
            "scale": float(scaler_scale[idx]),
        }
        for idx in range(len(feature_names))
    } if scaler_mean is not None and scaler_scale is not None and len(scaler_mean) == len(feature_names) else {}

    return {
        "feature_names": feature_names,
        "rows": rows,
        "rows_by_ts": {row["timestamp"]: row for row in rows},
        "scaler_stats": scaler_stats,
    }


def _monitored_features(
    feature_names: Iterable[str],
    feature_prefixes: Iterable[str],
    explicit_features: Iterable[str],
) -> List[str]:
    prefixes = [prefix for prefix in feature_prefixes if prefix]
    explicit = {feature for feature in explicit_features if feature}
    selected = []
    for feature_name in feature_names:
        if feature_name in explicit or any(feature_name.startswith(prefix) for prefix in prefixes):
            selected.append(str(feature_name))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag large overlap tail feature shifts versus a trusted baseline pack before they silently degrade trust.",
    )
    parser.add_argument("--baseline-pack", type=Path, required=True, help="Trusted baseline pack manifest path.")
    parser.add_argument("--current-overlap-dataset", type=Path, required=True, help="Current run overlap dataset path.")
    parser.add_argument("--tail-rows", type=int, default=24, help="Number of latest overlap rows to monitor.")
    parser.add_argument(
        "--feature-prefix",
        action="append",
        default=[],
        help="Feature prefix to monitor. May be supplied multiple times.",
    )
    parser.add_argument(
        "--feature-name",
        action="append",
        default=[],
        help="Explicit feature name to monitor. May be supplied multiple times.",
    )
    parser.add_argument(
        "--warn-abs-train-std-shift",
        type=float,
        default=1.5,
        help="Warning threshold in trusted train std units.",
    )
    parser.add_argument(
        "--fail-abs-train-std-shift",
        type=float,
        default=2.5,
        help="Failure threshold in trusted train std units.",
    )
    parser.add_argument(
        "--min-failed-features",
        type=int,
        default=2,
        help="Minimum number of failed monitored features needed to trip the guard.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON artifact path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_pack = _load_json(args.baseline_pack)
    replay_inputs = baseline_pack.get("replay_inputs") if isinstance(baseline_pack.get("replay_inputs"), dict) else {}
    baseline_overlap = Path(str(replay_inputs.get("overlap_dataset", "")))
    if not baseline_overlap.exists():
        raise FileNotFoundError(
            f"Baseline pack is missing overlap dataset path or file does not exist: {baseline_overlap}"
        )
    if not args.current_overlap_dataset.exists():
        raise FileNotFoundError(args.current_overlap_dataset)

    baseline = _load_dataset(baseline_overlap)
    current = _load_dataset(args.current_overlap_dataset)
    if not baseline["scaler_stats"] or not current["scaler_stats"]:
        payload = {
            "generated_at": _utc_now(),
            "baseline_pack": str(args.baseline_pack),
            "current_overlap_dataset": str(args.current_overlap_dataset),
            "status": "unavailable",
            "reason": "scaler_stats_missing",
            "guard_failed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    monitored = _monitored_features(
        current["feature_names"],
        args.feature_prefix,
        args.feature_name,
    )
    current_tail = current["rows"][-max(int(args.tail_rows), 1) :]
    common_tail = [
        row for row in current_tail if row["timestamp"] in baseline["rows_by_ts"]
    ]
    feature_summaries: List[Dict[str, Any]] = []
    failed_features = 0
    warned_features = 0

    for feature_name in monitored:
        baseline_stats = baseline["scaler_stats"].get(feature_name)
        if not baseline_stats:
            continue
        denom = float(baseline_stats.get("scale", 0.0) or 0.0)
        shifts: List[Dict[str, Any]] = []
        for row in common_tail:
            ts = row["timestamp"]
            baseline_row = baseline["rows_by_ts"][ts]
            current_raw = row["raw_features"].get(feature_name)
            baseline_raw = baseline_row["raw_features"].get(feature_name)
            if current_raw is None or baseline_raw is None:
                continue
            abs_raw_delta = abs(float(current_raw) - float(baseline_raw))
            shift = abs_raw_delta / denom if denom > 1e-12 else None
            shifts.append(
                {
                    "timestamp": ts,
                    "baseline_raw": float(baseline_raw),
                    "current_raw": float(current_raw),
                    "abs_raw_delta": float(abs_raw_delta),
                    "abs_delta_in_trusted_train_std": float(shift) if shift is not None else None,
                }
            )
        if not shifts:
            continue
        shifts.sort(
            key=lambda item: item["abs_delta_in_trusted_train_std"] if item["abs_delta_in_trusted_train_std"] is not None else -1.0,
            reverse=True,
        )
        max_shift = shifts[0]["abs_delta_in_trusted_train_std"]
        mean_shift = float(
            np.mean([
                float(item["abs_delta_in_trusted_train_std"])
                for item in shifts
                if item["abs_delta_in_trusted_train_std"] is not None
            ])
        )
        warn_flag = bool(max_shift is not None and max_shift >= float(args.warn_abs_train_std_shift))
        fail_flag = bool(max_shift is not None and max_shift >= float(args.fail_abs_train_std_shift))
        warned_features += 1 if warn_flag else 0
        failed_features += 1 if fail_flag else 0
        feature_summaries.append(
            {
                "feature": feature_name,
                "trusted_train_mean_raw": float(baseline_stats["mean"]),
                "trusted_train_std_raw": float(baseline_stats["scale"]),
                "tail_common_rows": int(len(shifts)),
                "max_abs_train_std_shift": max_shift,
                "mean_abs_train_std_shift": mean_shift,
                "warn_flag": warn_flag,
                "fail_flag": fail_flag,
                "top_shift_rows": shifts[:5],
            }
        )

    feature_summaries.sort(
        key=lambda item: item["max_abs_train_std_shift"] if item["max_abs_train_std_shift"] is not None else -1.0,
        reverse=True,
    )
    guard_failed = failed_features >= int(args.min_failed_features)
    payload = {
        "generated_at": _utc_now(),
        "baseline_pack": str(args.baseline_pack),
        "current_overlap_dataset": str(args.current_overlap_dataset),
        "status": "ok",
        "guard_failed": bool(guard_failed),
        "tail_rows_requested": int(args.tail_rows),
        "tail_rows_compared": int(len(common_tail)),
        "warn_abs_train_std_shift": float(args.warn_abs_train_std_shift),
        "fail_abs_train_std_shift": float(args.fail_abs_train_std_shift),
        "min_failed_features": int(args.min_failed_features),
        "monitored_feature_count": int(len(monitored)),
        "warned_feature_count": int(warned_features),
        "failed_feature_count": int(failed_features),
        "monitored_features": monitored,
        "feature_summaries": feature_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
