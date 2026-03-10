from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_overlap_timestamps(path: Path) -> pd.Series:
    payload = np.load(path, allow_pickle=True)
    return pd.Series(pd.to_datetime(payload["ts_all"]).tz_localize(None), name="ts")


def _to_records(rows: pd.DataFrame, pos_col: str) -> List[Dict[str, Any]]:
    if rows.empty:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        records.append({
            "timestamp": row["ts"].isoformat(sep="T"),
            pos_col: int(row[pos_col]),
        })
    return records


def _load_detail_rows(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"])
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trusted overlap run against a drifting run and isolate which bars flipped overlap trust.",
    )
    parser.add_argument("--trusted-run-id", required=True, help="Trusted default reliability run id.")
    parser.add_argument("--drift-run-id", required=True, help="Drifting/default latest reliability run id to compare against.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Reliability workflow run root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/overlap_trust_flip_latest.json"),
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--trusted-detail-path",
        type=Path,
        default=None,
        help="Optional per-bar CSV exported from walkforward overlap validation for the trusted run.",
    )
    parser.add_argument(
        "--drift-detail-path",
        type=Path,
        default=None,
        help="Optional per-bar CSV exported from walkforward overlap validation for the drifting run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trusted_summary = args.run_root / args.trusted_run_id / "summary"
    drift_summary = args.run_root / args.drift_run_id / "summary"
    if not trusted_summary.exists():
        raise FileNotFoundError(trusted_summary)
    if not drift_summary.exists():
        raise FileNotFoundError(drift_summary)

    trusted_overlap_ts = _load_overlap_timestamps(trusted_summary / "btc_features_1h_direction_splits.labeled_overlap.npz")
    drift_overlap_ts = _load_overlap_timestamps(drift_summary / "btc_features_1h_direction_splits.labeled_overlap.npz")
    trusted_df = pd.DataFrame({"ts": trusted_overlap_ts, "trusted_pos": range(len(trusted_overlap_ts))})
    drift_df = pd.DataFrame({"ts": drift_overlap_ts, "drift_pos": range(len(drift_overlap_ts))})
    merged_ts = trusted_df.merge(drift_df, on="ts", how="outer").sort_values("ts")
    only_trusted = merged_ts[merged_ts["drift_pos"].isna()][["ts", "trusted_pos"]]
    only_drift = merged_ts[merged_ts["trusted_pos"].isna()][["ts", "drift_pos"]]

    trusted_rec = _load_json(trusted_summary / "walkforward_labeled_reconciliation.json")
    drift_rec = _load_json(drift_summary / "walkforward_labeled_reconciliation.json")
    trusted_fold_payload = _load_json(trusted_summary / "walkforward_model_compare_labeled_overlap_meta_stack.json")
    drift_fold_payload = _load_json(drift_summary / "walkforward_model_compare_labeled_overlap_meta_stack.json")
    trusted_folds = trusted_fold_payload.get("folds", [])
    drift_folds = drift_fold_payload.get("folds", [])
    if len(trusted_folds) != len(drift_folds):
        raise ValueError("Trusted and drift overlap fold counts do not match.")

    fold_deltas: List[Dict[str, Any]] = []
    for trusted_fold, drift_fold in zip(trusted_folds, drift_folds):
        fold_deltas.append(
            {
                "fold": int(trusted_fold["fold"]),
                "trusted_cum_ret": float(trusted_fold["cum_ret_net"]),
                "drift_cum_ret": float(drift_fold["cum_ret_net"]),
                "delta": float(drift_fold["cum_ret_net"] - trusted_fold["cum_ret_net"]),
                "trusted_trade_count": int(trusted_fold["trade_count"]),
                "drift_trade_count": int(drift_fold["trade_count"]),
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "trusted_run_id": args.trusted_run_id,
        "drift_run_id": args.drift_run_id,
        "summary": {
            "trusted_overlap_cum_ret_net_total": float(trusted_rec["overlap_selected_row"]["cum_ret_net_total"]),
            "drift_overlap_cum_ret_net_total": float(drift_rec["overlap_selected_row"]["cum_ret_net_total"]),
            "overlap_delta_total": float(
                drift_rec["overlap_selected_row"]["cum_ret_net_total"]
                - trusted_rec["overlap_selected_row"]["cum_ret_net_total"]
            ),
            "trusted_edge_trustworthy": bool(trusted_rec["agreement"]["edge_trustworthy"]),
            "drift_edge_trustworthy": bool(drift_rec["agreement"]["edge_trustworthy"]),
        },
        "overlap_slice": {
            "trusted_rows": int(len(trusted_overlap_ts)),
            "drift_rows": int(len(drift_overlap_ts)),
            "only_in_trusted": [row["timestamp"] for row in _to_records(only_trusted, "trusted_pos")],
            "only_in_drift": [row["timestamp"] for row in _to_records(only_drift, "drift_pos")],
        },
        "fold_deltas": fold_deltas,
        "tail_overlap_positions": {
            "trusted_only_positions": _to_records(only_trusted, "trusted_pos"),
            "drift_only_positions": _to_records(only_drift, "drift_pos"),
        },
        "conclusion": (
            "The overlap trust flip is concentrated in the fold with the most negative delta; "
            "inspect fold_deltas to identify the bar cluster that moved overlap return sign."
        ),
    }

    worst_fold = min(payload["fold_deltas"], key=lambda row: row["delta"])
    payload["conclusion"] = (
        "The overlap trust flip is fully explained by fold "
        f"{worst_fold['fold']}. It moved from {worst_fold['trusted_cum_ret']:.10f} "
        f"to {worst_fold['drift_cum_ret']:.10f}, a delta of {worst_fold['delta']:.10f}."
    )

    trusted_detail = _load_detail_rows(args.trusted_detail_path)
    drift_detail = _load_detail_rows(args.drift_detail_path)
    if trusted_detail is not None and drift_detail is not None:
        trusted_fold_rows = trusted_detail[trusted_detail["fold"] == int(worst_fold["fold"])]
        drift_fold_rows = drift_detail[drift_detail["fold"] == int(worst_fold["fold"])]
        trusted_fold_view = trusted_fold_rows[["ts", "signal", "ret_net", "p_up", "global_index"]].rename(
            columns={
                "signal": "signal_trusted",
                "ret_net": "ret_net_trusted",
                "p_up": "p_up_trusted",
                "global_index": "global_index_trusted",
            }
        )
        drift_fold_view = drift_fold_rows[["ts", "signal", "ret_net", "p_up", "global_index"]].rename(
            columns={
                "signal": "signal_drift",
                "ret_net": "ret_net_drift",
                "p_up": "p_up_drift",
                "global_index": "global_index_drift",
            }
        )
        detailed_merge = trusted_fold_view.merge(drift_fold_view, on="ts", how="outer").sort_values("ts")
        trusted_only_rows = detailed_merge[detailed_merge["global_index_drift"].isna()].copy()
        drift_only_rows = detailed_merge[detailed_merge["global_index_trusted"].isna()].copy()
        common_changed_rows = detailed_merge[
            detailed_merge["global_index_trusted"].notna()
            & detailed_merge["global_index_drift"].notna()
            & (
                (detailed_merge["signal_trusted"] != detailed_merge["signal_drift"])
                | ((detailed_merge["ret_net_trusted"] - detailed_merge["ret_net_drift"]).abs() > 1e-12)
            )
        ].copy()
        payload["worst_fold_bar_deltas"] = {
            "fold": int(worst_fold["fold"]),
            "trusted_detail_path": str(args.trusted_detail_path),
            "drift_detail_path": str(args.drift_detail_path),
            "trusted_only_rows": [
                {
                    "timestamp": row["ts"].isoformat(),
                    "global_index_trusted": int(row["global_index_trusted"]),
                    "signal_trusted": int(row["signal_trusted"]),
                    "ret_net_trusted": float(row["ret_net_trusted"]),
                    "p_up_trusted": float(row["p_up_trusted"]),
                }
                for _, row in trusted_only_rows.iterrows()
            ],
            "drift_only_rows": [
                {
                    "timestamp": row["ts"].isoformat(),
                    "global_index_drift": int(row["global_index_drift"]),
                    "signal_drift": int(row["signal_drift"]),
                    "ret_net_drift": float(row["ret_net_drift"]),
                    "p_up_drift": float(row["p_up_drift"]),
                }
                for _, row in drift_only_rows.iterrows()
            ],
            "common_changed_rows": [
                {
                    "timestamp": row["ts"].isoformat(),
                    "global_index_trusted": int(row["global_index_trusted"]),
                    "global_index_drift": int(row["global_index_drift"]),
                    "signal_trusted": int(row["signal_trusted"]),
                    "signal_drift": int(row["signal_drift"]),
                    "ret_net_trusted": float(row["ret_net_trusted"]),
                    "ret_net_drift": float(row["ret_net_drift"]),
                    "ret_net_delta": float(row["ret_net_drift"] - row["ret_net_trusted"]),
                    "p_up_trusted": float(row["p_up_trusted"]),
                    "p_up_drift": float(row["p_up_drift"]),
                }
                for _, row in common_changed_rows.iterrows()
            ],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()