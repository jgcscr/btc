from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from src.training.time_series_cv import build_time_series_folds


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_csv_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = str(row.get("ts", "")).strip()
            if not timestamp:
                continue
            rows[timestamp] = dict(row)
    return rows


def _load_overlap_rows(npz_path: Path) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    data = np.load(npz_path, allow_pickle=True)
    feature_names = [str(name) for name in data["feature_names"].tolist()]
    X = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_val"], data["y_test"]]).astype(int)
    y_ret = np.concatenate([data["y_ret_train"], data["y_ret_val"], data["y_ret_test"]]).astype(float)
    ts = np.concatenate([data["ts_train"], data["ts_val"], data["ts_test"]])

    rows: Dict[str, Dict[str, Any]] = {}
    for idx, ts_value in enumerate(ts):
        timestamp = np.datetime_as_string(ts_value, unit="s")
        feature_values = {feature_names[col]: float(X[idx, col]) for col in range(len(feature_names))}
        rows[timestamp] = {
            "timestamp": timestamp,
            "row_index": int(idx),
            "y_true": int(y[idx]),
            "y_ret": float(y_ret[idx]),
            "features": feature_values,
        }
    return rows, feature_names


def _resolved_walkforward_context(compare_summary_path: Path, n_rows: int, fold_number: int) -> Dict[str, Any]:
    summary = _load_json(compare_summary_path)
    resolved = summary.get("resolved_walkforward", {}) if isinstance(summary.get("resolved_walkforward"), dict) else {}
    folds = build_time_series_folds(
        n_rows,
        n_splits=int(resolved.get("folds", 1)),
        train_size=int(resolved.get("train_size", 1)),
        val_size=int(resolved.get("val_size", 1)),
        test_size=int(resolved.get("test_size", 1)),
        gap=int(resolved.get("gap", 0)),
        purge_size=int(resolved.get("purge_size", 0)),
        embargo_size=int(resolved.get("embargo_size", 0)),
        mode=str(resolved.get("mode", "rolling")),
    )
    if fold_number < 1 or fold_number > len(folds):
        raise ValueError(f"Fold {fold_number} is out of range for {compare_summary_path}")
    fold = folds[fold_number - 1]
    return {
        "compare_summary": str(compare_summary_path),
        "resolved_walkforward": resolved,
        "fold": fold,
    }


def _train_feature_context(
    rows_by_ts: Dict[str, Dict[str, Any]],
    fold_info: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    fold = fold_info["fold"]
    ordered_rows = sorted(rows_by_ts.values(), key=lambda item: int(item["row_index"]))
    train_rows = ordered_rows[fold.train_start:fold.train_end]
    if not train_rows:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for feature_name in feature_names:
        values = np.asarray([float(row["features"][feature_name]) for row in train_rows], dtype=float)
        out[feature_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return out


def _window_summary(rows_by_ts: Dict[str, Dict[str, Any]], fold_info: Dict[str, Any]) -> Dict[str, Any]:
    fold = fold_info["fold"]
    ordered_rows = sorted(rows_by_ts.values(), key=lambda item: int(item["row_index"]))
    train_rows = ordered_rows[fold.train_start:fold.train_end]
    val_rows = ordered_rows[fold.val_start:fold.val_end]
    test_rows = ordered_rows[fold.test_start:fold.test_end]
    return {
        "train_start_index": int(fold.train_start),
        "train_end_index": int(fold.train_end),
        "val_start_index": int(fold.val_start),
        "val_end_index": int(fold.val_end),
        "test_start_index": int(fold.test_start),
        "test_end_index": int(fold.test_end),
        "train_start_ts": str(train_rows[0]["timestamp"]) if train_rows else None,
        "train_end_ts": str(train_rows[-1]["timestamp"]) if train_rows else None,
        "val_start_ts": str(val_rows[0]["timestamp"]) if val_rows else None,
        "val_end_ts": str(val_rows[-1]["timestamp"]) if val_rows else None,
        "test_start_ts": str(test_rows[0]["timestamp"]) if test_rows else None,
        "test_end_ts": str(test_rows[-1]["timestamp"]) if test_rows else None,
        "train_row_count": int(len(train_rows)),
        "val_row_count": int(len(val_rows)),
        "test_row_count": int(len(test_rows)),
    }


def _collect_timestamps(detail_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    worst_fold = detail_payload.get("worst_fold_bar_deltas", {})
    return {
        "common_changed": [str(row["timestamp"]) for row in worst_fold.get("common_changed_rows", [])],
        "trusted_only": [str(row["timestamp"]) for row in worst_fold.get("trusted_only_rows", [])],
        "drift_only": [str(row["timestamp"]) for row in worst_fold.get("drift_only_rows", [])],
    }


def _top_feature_deltas(
    trusted_features: Dict[str, float],
    drift_features: Dict[str, float],
    feature_names: Iterable[str],
    limit: int,
) -> List[Dict[str, Any]]:
    deltas: List[Dict[str, Any]] = []
    for feature_name in feature_names:
        trusted_value = float(trusted_features.get(feature_name, float("nan")))
        drift_value = float(drift_features.get(feature_name, float("nan")))
        delta = drift_value - trusted_value
        deltas.append(
            {
                "feature": feature_name,
                "trusted": trusted_value,
                "drift": drift_value,
                "delta": float(delta),
                "abs_delta": float(abs(delta)),
            }
        )
    deltas.sort(key=lambda item: item["abs_delta"], reverse=True)
    return deltas[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export feature-level diffs for the exact overlap bars that flipped trust between a trusted and drifting run.",
    )
    parser.add_argument("--trusted-run-id", required=True, help="Trusted default reliability run id.")
    parser.add_argument("--drift-run-id", required=True, help="Drifting default reliability run id.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Reliability workflow run root.",
    )
    parser.add_argument(
        "--detail-analysis",
        type=Path,
        default=Path("artifacts/analysis/overlap_trust_flip_detailed_latest.json"),
        help="Detailed trust-flip analysis artifact containing the worst-fold bar timestamps.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=12,
        help="Number of largest absolute feature deltas to emit for common changed rows.",
    )
    parser.add_argument(
        "--trusted-compare-summary",
        type=Path,
        default=Path("artifacts/analysis/trusted_overlap_compare.json"),
        help="Trusted overlap compare summary with resolved walkforward settings.",
    )
    parser.add_argument(
        "--drift-compare-summary",
        type=Path,
        default=Path("artifacts/analysis/drift_overlap_compare.json"),
        help="Drifting overlap compare summary with resolved walkforward settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/overlap_feature_drift_latest.json"),
        help="Output JSON artifact path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trusted_summary = args.run_root / args.trusted_run_id / "summary"
    drift_summary = args.run_root / args.drift_run_id / "summary"
    trusted_npz = trusted_summary / "btc_features_1h_direction_splits.labeled_overlap.npz"
    drift_npz = drift_summary / "btc_features_1h_direction_splits.labeled_overlap.npz"
    if not trusted_npz.exists():
        raise FileNotFoundError(trusted_npz)
    if not drift_npz.exists():
        raise FileNotFoundError(drift_npz)
    if not args.detail_analysis.exists():
        raise FileNotFoundError(args.detail_analysis)
    if not args.trusted_compare_summary.exists():
        raise FileNotFoundError(args.trusted_compare_summary)
    if not args.drift_compare_summary.exists():
        raise FileNotFoundError(args.drift_compare_summary)

    detail_payload = _load_json(args.detail_analysis)
    worst_fold_payload = detail_payload.get("worst_fold_bar_deltas", {})
    worst_fold_number = int(worst_fold_payload.get("fold", 0) or 0)
    trusted_detail_path = Path(str(worst_fold_payload.get("trusted_detail_path", "")))
    drift_detail_path = Path(str(worst_fold_payload.get("drift_detail_path", "")))
    timestamps = _collect_timestamps(detail_payload)
    trusted_rows, feature_names = _load_overlap_rows(trusted_npz)
    drift_rows, _ = _load_overlap_rows(drift_npz)
    trusted_detail_rows = _load_csv_rows(trusted_detail_path) if trusted_detail_path.exists() else {}
    drift_detail_rows = _load_csv_rows(drift_detail_path) if drift_detail_path.exists() else {}
    trusted_fold_info = _resolved_walkforward_context(args.trusted_compare_summary, len(trusted_rows), worst_fold_number)
    drift_fold_info = _resolved_walkforward_context(args.drift_compare_summary, len(drift_rows), worst_fold_number)
    trusted_train_context = _train_feature_context(trusted_rows, trusted_fold_info, feature_names)
    drift_train_context = _train_feature_context(drift_rows, drift_fold_info, feature_names)

    common_changed_rows: List[Dict[str, Any]] = []
    for timestamp in timestamps["common_changed"]:
        trusted_row = trusted_rows.get(timestamp)
        drift_row = drift_rows.get(timestamp)
        if trusted_row is None or drift_row is None:
            continue
        trusted_detail = trusted_detail_rows.get(timestamp, {})
        drift_detail = drift_detail_rows.get(timestamp, {})
        top_feature_deltas = _top_feature_deltas(
            trusted_row["features"],
            drift_row["features"],
            feature_names,
            limit=int(args.top_features),
        )
        for feature_delta in top_feature_deltas:
            feature_name = str(feature_delta["feature"])
            trusted_stats = trusted_train_context.get(feature_name, {})
            drift_stats = drift_train_context.get(feature_name, {})
            feature_delta["trusted_train_mean"] = trusted_stats.get("mean")
            feature_delta["trusted_train_std"] = trusted_stats.get("std")
            feature_delta["drift_train_mean"] = drift_stats.get("mean")
            feature_delta["drift_train_std"] = drift_stats.get("std")
        common_changed_rows.append(
            {
                "timestamp": timestamp,
                "trusted_row_index": trusted_row["row_index"],
                "drift_row_index": drift_row["row_index"],
                "trusted_y_true": trusted_row["y_true"],
                "drift_y_true": drift_row["y_true"],
                "trusted_y_ret": trusted_row["y_ret"],
                "drift_y_ret": drift_row["y_ret"],
                "trusted_p_up": float(trusted_detail["p_up"]) if trusted_detail.get("p_up") else None,
                "drift_p_up": float(drift_detail["p_up"]) if drift_detail.get("p_up") else None,
                "trusted_signal": int(trusted_detail["signal"]) if trusted_detail.get("signal") else 0,
                "drift_signal": int(drift_detail["signal"]) if drift_detail.get("signal") else 0,
                "trusted_ret_net": float(trusted_detail["ret_net"]) if trusted_detail.get("ret_net") else None,
                "drift_ret_net": float(drift_detail["ret_net"]) if drift_detail.get("ret_net") else None,
                "top_feature_deltas": top_feature_deltas,
            }
        )

    trusted_only_rows: List[Dict[str, Any]] = []
    for timestamp in timestamps["trusted_only"]:
        trusted_row = trusted_rows.get(timestamp)
        if trusted_row is None:
            continue
        trusted_only_rows.append(
            {
                "timestamp": timestamp,
                "trusted_row_index": trusted_row["row_index"],
                "trusted_y_true": trusted_row["y_true"],
                "trusted_y_ret": trusted_row["y_ret"],
                "feature_snapshot": trusted_row["features"],
            }
        )

    drift_only_rows: List[Dict[str, Any]] = []
    for timestamp in timestamps["drift_only"]:
        drift_row = drift_rows.get(timestamp)
        if drift_row is None:
            continue
        drift_only_rows.append(
            {
                "timestamp": timestamp,
                "drift_row_index": drift_row["row_index"],
                "drift_y_true": drift_row["y_true"],
                "drift_y_ret": drift_row["y_ret"],
                "feature_snapshot": drift_row["features"],
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "trusted_run_id": args.trusted_run_id,
        "drift_run_id": args.drift_run_id,
        "detail_analysis": str(args.detail_analysis),
        "trusted_compare_summary": str(args.trusted_compare_summary),
        "drift_compare_summary": str(args.drift_compare_summary),
        "worst_fold": int(worst_fold_number),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "trusted_train_window": _window_summary(trusted_rows, trusted_fold_info),
        "drift_train_window": _window_summary(drift_rows, drift_fold_info),
        "common_changed_rows": common_changed_rows,
        "trusted_only_rows": trusted_only_rows,
        "drift_only_rows": drift_only_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()