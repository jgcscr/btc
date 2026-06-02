from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_TRADE_READY_PATH = "artifacts/monitoring/trade_ready_summary.json"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/meta_stack_shadow_rollout_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/meta_stack_shadow_rollout_latest.md"
DEFAULT_HORIZONS = ("4h", "12h")
HORIZON_TO_TARGET_KEY = {
    "4h": "y_dir4h",
    "12h": "y_dir12h",
}
HORIZON_TO_HOURS = {
    "4h": 4.0,
    "12h": 12.0,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a shadow-only meta-stack promotion check for the 4h and 12h horizons "
            "without changing the live ensemble path."
        )
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--trade-ready-path", default=DEFAULT_TRADE_READY_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--horizon", action="append", dest="horizons", default=[])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--purge-size", type=int, default=0)
    parser.add_argument("--embargo-size", type=int, default=0)
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--meta-margin", type=float, default=0.0)
    parser.add_argument("--meta-min-rolling-trades", type=int, default=25)
    parser.add_argument("--min-auc", type=float, default=0.5)
    parser.add_argument(
        "--max-ece-delta",
        type=float,
        default=0.01,
        help="Maximum calibration degradation the meta stack may show versus XGB.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _find_row(rows: Iterable[Mapping[str, Any]], model_kind: str) -> Dict[str, Any] | None:
    for row in rows:
        if str(row.get("model_kind")) == model_kind:
            return dict(row)
    return None


def _compare_output_path(output_root: Path, horizon_label: str) -> Path:
    safe = horizon_label.replace(".", "p")
    return output_root / f"meta_stack_shadow_compare_{safe}.json"


def _run_compare_for_horizon(
    *,
    repo_root: Path,
    dataset_path: Path,
    output_root: Path,
    horizon_label: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    y_key = HORIZON_TO_TARGET_KEY[horizon_label]
    output_path = _compare_output_path(output_root, horizon_label)
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.compare_walkforward_models",
        "--dataset-path",
        str(dataset_path),
        "--y-key",
        y_key,
        "--folds",
        str(int(args.folds)),
        "--train-size",
        str(int(args.train_size)),
        "--val-size",
        str(int(args.val_size)),
        "--test-size",
        str(int(args.test_size)),
        "--gap",
        str(int(args.gap)),
        "--purge-size",
        str(int(args.purge_size)),
        "--embargo-size",
        str(int(args.embargo_size)),
        "--signal-threshold",
        str(float(args.signal_threshold)),
        "--fee-bps",
        str(float(args.fee_bps)),
        "--slippage-bps",
        str(float(args.slippage_bps)),
        "--rolling-guard",
        "--meta-margin",
        str(float(args.meta_margin)),
        "--meta-min-rolling-trades",
        str(int(args.meta_min_rolling_trades)),
        "--selection-policy",
        "incumbent_guarded",
        "--min-auc",
        str(float(args.min_auc)),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    summary = _read_json(output_path)
    return {
        "compare_path": str(output_path),
        "summary": summary,
    }


def _load_model_payloads(compare_summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for row in compare_summary.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        model_kind = str(row.get("model_kind"))
        path = row.get("path")
        if model_kind and path:
            payloads[model_kind] = _read_json(Path(str(path)))
    return payloads


def _load_rolling_model_payloads(compare_summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    rolling = compare_summary.get("rolling_guard")
    if not isinstance(rolling, Mapping):
        return payloads
    for row in rolling.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        model_kind = str(row.get("model_kind"))
        path = row.get("path")
        if model_kind and path:
            payloads[model_kind] = _read_json(Path(str(path)))
    return payloads


def _promotion_checks(
    *,
    compare_summary: Mapping[str, Any],
    model_payloads: Mapping[str, Mapping[str, Any]],
    rolling_payloads: Mapping[str, Mapping[str, Any]],
    max_ece_delta: float,
    meta_min_rolling_trades: int,
) -> Dict[str, Any]:
    rows = compare_summary.get("rows", [])
    rolling = compare_summary.get("rolling_guard", {})
    rolling_rows = rolling.get("rows", []) if isinstance(rolling, Mapping) else []
    meta = _find_row(rows, "meta_stack")
    xgb = _find_row(rows, "xgb")
    meta_rolling = _find_row(rolling_rows, "meta_stack")
    xgb_rolling = _find_row(rolling_rows, "xgb")
    meta_payload = model_payloads.get("meta_stack", {})
    xgb_payload = model_payloads.get("xgb", {})
    meta_rolling_payload = rolling_payloads.get("meta_stack", {})
    xgb_rolling_payload = rolling_payloads.get("xgb", {})

    checks = {
        "selected_by_guarded_policy": str(compare_summary.get("selected_model_kind")) == "meta_stack",
        "expanding_net_return": bool(meta and xgb)
        and _as_float(meta.get("cum_ret_net_total")) >= _as_float(xgb.get("cum_ret_net_total")),
        "expanding_auc": bool(meta and xgb)
        and _as_float(meta.get("auc_mean")) >= _as_float(xgb.get("auc_mean")),
        "expanding_ece": bool(meta_payload and xgb_payload)
        and _as_float(meta_payload.get("ece_10_mean")) <= _as_float(xgb_payload.get("ece_10_mean")) + float(max_ece_delta),
        "rolling_net_return": bool(meta_rolling and xgb_rolling)
        and _as_float(meta_rolling.get("cum_ret_net_total")) >= _as_float(xgb_rolling.get("cum_ret_net_total")),
        "rolling_auc": bool(meta_rolling and xgb_rolling)
        and _as_float(meta_rolling.get("auc_mean")) >= _as_float(xgb_rolling.get("auc_mean")),
        "rolling_ece": bool(meta_rolling_payload and xgb_rolling_payload)
        and _as_float(meta_rolling_payload.get("ece_10_mean")) <= _as_float(xgb_rolling_payload.get("ece_10_mean")) + float(max_ece_delta),
        "rolling_trade_count": bool(meta_rolling)
        and _as_int(meta_rolling.get("trade_count_total")) >= int(meta_min_rolling_trades),
    }
    return {
        "checks": checks,
        "promote_meta_stack": all(bool(value) for value in checks.values()),
        "expanding": {
            "meta_stack": meta_payload,
            "xgb": xgb_payload,
        },
        "rolling": {
            "meta_stack": meta_rolling_payload,
            "xgb": xgb_rolling_payload,
        },
    }


def _horizon_label_from_entry(entry: Mapping[str, Any]) -> str | None:
    hours = _as_float(entry.get("horizon_hours"))
    for label, expected in HORIZON_TO_HOURS.items():
        if abs(hours - expected) < 1e-9:
            return label
    return None


def _extract_live_horizon_snapshots(trade_ready_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for entry in trade_ready_payload.get("horizons", []):
        if not isinstance(entry, Mapping):
            continue
        label = _horizon_label_from_entry(entry)
        if label is None:
            continue
        snapshots[label] = {
            "direction_next": entry.get("direction_next"),
            "trade_action": entry.get("trade_action"),
            "confidence_score": entry.get("confidence_score"),
            "trust_status": entry.get("trust_status"),
            "trust_reasons": entry.get("trust_reasons") or [],
            "trust_hardening_action": entry.get("trust_hardening_action"),
            "voting_weight_after_trust": entry.get("voting_weight_after_trust"),
            "p_up": entry.get("p_up"),
            "expected_value": entry.get("expected_value"),
            "regime_state": entry.get("regime_state"),
            "execution_status": ((entry.get("execution_plan") or {}).get("status") if isinstance(entry.get("execution_plan"), Mapping) else None),
        }
    return snapshots


def _build_overall_recommendation(
    *,
    horizon_reports: Mapping[str, Mapping[str, Any]],
    live_snapshots: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    meta_promoted = all(bool((report.get("promotion") or {}).get("promote_meta_stack")) for report in horizon_reports.values())
    four_h_trust = live_snapshots.get("4h", {})
    four_h_needs_fix = str(four_h_trust.get("trust_status")) != "trusted"
    return {
        "keep_current_live_ensemble": True,
        "run_meta_stack_in_shadow": True,
        "meta_stack_ready_for_live_consideration": bool(meta_promoted) and not four_h_needs_fix,
        "move_to_more_complex_boosting": False,
        "prioritize_4h_trust_calibration_fix": four_h_needs_fix,
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Meta Stack Shadow Rollout")
    lines.append("")
    overall = payload.get("overall_recommendation", {})
    lines.append("## Summary")
    lines.append(f"- Keep current live ensemble: {overall.get('keep_current_live_ensemble')}")
    lines.append(f"- Run meta stack in shadow: {overall.get('run_meta_stack_in_shadow')}")
    lines.append(
        f"- Meta stack ready for live consideration: {overall.get('meta_stack_ready_for_live_consideration')}"
    )
    lines.append(f"- Move to more complex boosting: {overall.get('move_to_more_complex_boosting')}")
    lines.append(
        f"- Prioritize 4h trust/calibration fix: {overall.get('prioritize_4h_trust_calibration_fix')}"
    )
    lines.append("")
    live_summary = payload.get("live_summary", {})
    lines.append("## Live Snapshot")
    lines.append(f"- Selected direction: {((live_summary.get('market_outlook_strategy') or {}).get('selected_direction'))}")
    lines.append(f"- Preferred horizon: {((live_summary.get('market_outlook_strategy') or {}).get('preferred_horizon'))}")
    lines.append(f"- Confidence: {((live_summary.get('market_outlook_strategy') or {}).get('confidence_level'))}")
    lines.append("")
    horizon_reports = payload.get("horizons", {})
    for horizon_label, report in horizon_reports.items():
        if not isinstance(report, Mapping):
            continue
        promotion = report.get("promotion", {})
        live = report.get("live_snapshot", {})
        lines.append(f"## {horizon_label} Shadow Decision")
        lines.append(f"- Promote meta stack: {promotion.get('promote_meta_stack')}")
        lines.append(f"- Live trust status: {live.get('trust_status')}")
        lines.append(f"- Live trust reasons: {', '.join(str(item) for item in (live.get('trust_reasons') or [])) or 'none'}")
        checks = promotion.get("checks", {})
        for check_name, check_value in checks.items():
            lines.append(f"- {check_name}: {check_value}")
        lines.append("")
    lines.append("## Promotion Bar")
    lines.append("- Meta stack must beat XGB on expanding net return and AUC.")
    lines.append("- Meta stack must beat XGB on rolling net return and AUC.")
    lines.append("- Meta stack must not degrade ECE beyond the configured tolerance.")
    lines.append("- 4h trust and calibration issues should be fixed before any live promotion.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    dataset_path = Path(args.dataset_path)
    trade_ready_path = Path(args.trade_ready_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_root = output_json.parent

    horizons = [str(h).lower() for h in (args.horizons or list(DEFAULT_HORIZONS))]
    for horizon in horizons:
        if horizon not in HORIZON_TO_TARGET_KEY:
            raise ValueError(f"Unsupported horizon: {horizon}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not trade_ready_path.exists():
        raise FileNotFoundError(f"Trade-ready summary not found: {trade_ready_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    trade_ready_payload = _read_json(trade_ready_path)
    live_snapshots = _extract_live_horizon_snapshots(trade_ready_payload)
    live_summary = trade_ready_payload.get("prompt_ready_summary", {})

    horizon_reports: Dict[str, Any] = {}
    for horizon_label in horizons:
        compare_result = _run_compare_for_horizon(
            repo_root=repo_root,
            dataset_path=dataset_path,
            output_root=output_root,
            horizon_label=horizon_label,
            args=args,
        )
        compare_summary = compare_result["summary"]
        model_payloads = _load_model_payloads(compare_summary)
        rolling_payloads = _load_rolling_model_payloads(compare_summary)
        promotion = _promotion_checks(
            compare_summary=compare_summary,
            model_payloads=model_payloads,
            rolling_payloads=rolling_payloads,
            max_ece_delta=float(args.max_ece_delta),
            meta_min_rolling_trades=int(args.meta_min_rolling_trades),
        )
        horizon_reports[horizon_label] = {
            "compare": compare_result,
            "promotion": promotion,
            "live_snapshot": live_snapshots.get(horizon_label, {}),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(dataset_path),
            "trade_ready_path": str(trade_ready_path),
            "horizons": horizons,
            "folds": int(args.folds),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "gap": int(args.gap),
            "purge_size": int(args.purge_size),
            "embargo_size": int(args.embargo_size),
            "signal_threshold": float(args.signal_threshold),
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "meta_margin": float(args.meta_margin),
            "meta_min_rolling_trades": int(args.meta_min_rolling_trades),
            "min_auc": float(args.min_auc),
            "max_ece_delta": float(args.max_ece_delta),
        },
        "live_summary": live_summary,
        "live_horizon_snapshots": live_snapshots,
        "horizons": horizon_reports,
        "overall_recommendation": _build_overall_recommendation(
            horizon_reports=horizon_reports,
            live_snapshots=live_snapshots,
        ),
    }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote shadow rollout JSON: {output_json}")
    print(f"Wrote shadow rollout memo: {output_md}")


if __name__ == "__main__":
    main()