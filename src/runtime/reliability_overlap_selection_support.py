from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


def selected_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
    selected_kind = str(payload.get("selected_model_kind", ""))
    rows_obj = payload.get("rows", [])
    if not isinstance(rows_obj, list):
        return {}
    for item in rows_obj:
        if isinstance(item, dict) and str(item.get("model_kind", "")) == selected_kind:
            return item
    return {}


def row_by_model_kind(payload: Mapping[str, Any], model_kind: str) -> Dict[str, Any]:
    rows_obj = payload.get("rows", [])
    if not isinstance(rows_obj, list):
        return {}
    for item in rows_obj:
        if isinstance(item, dict) and str(item.get("model_kind", "")) == str(model_kind):
            return item
    return {}


def apply_overlap_model_selection_override(
    *,
    overlap_payload: Dict[str, Any],
    reconcile_cfg: Mapping[str, Any],
) -> Dict[str, Any] | None:
    overlap_selection_cfg_obj = reconcile_cfg.get("overlap_model_selection", {})
    overlap_selection_cfg = overlap_selection_cfg_obj if isinstance(overlap_selection_cfg_obj, dict) else {}
    overlap_selection_enabled = bool(overlap_selection_cfg.get("enabled", False))
    if not overlap_selection_enabled:
        return None

    overlap_selection_primary = str(overlap_selection_cfg.get("primary_model", "xgb"))
    overlap_selection_fallback = str(overlap_selection_cfg.get("fallback_model", "meta_stack"))
    overlap_selection_min_ret_improvement = float(overlap_selection_cfg.get("min_ret_improvement", 0.0))
    overlap_selection_only_when_primary_negative = bool(
        overlap_selection_cfg.get("only_when_primary_negative", True)
    )
    overlap_selection_require_auc_non_worse = bool(
        overlap_selection_cfg.get("require_fallback_auc_non_worse", False)
    )
    overlap_selection_min_fallback_trades = int(overlap_selection_cfg.get("min_fallback_trades", 0))

    primary_row = row_by_model_kind(overlap_payload, overlap_selection_primary)
    fallback_row = row_by_model_kind(overlap_payload, overlap_selection_fallback)
    if not (primary_row and fallback_row):
        return {
            "enabled": True,
            "primary_model": overlap_selection_primary,
            "fallback_model": overlap_selection_fallback,
            "selected_override": False,
            "reason": "missing_model_rows",
        }

    primary_ret = float(primary_row.get("cum_ret_net_total", 0.0) or 0.0)
    fallback_ret = float(fallback_row.get("cum_ret_net_total", 0.0) or 0.0)
    primary_auc = float(primary_row.get("auc_mean", float("nan")))
    fallback_auc = float(fallback_row.get("auc_mean", float("nan")))
    fallback_trades = int(fallback_row.get("trade_count_total", 0) or 0)

    ret_improvement_ok = fallback_ret >= (primary_ret + overlap_selection_min_ret_improvement)
    primary_negative_ok = (primary_ret < 0.0) if overlap_selection_only_when_primary_negative else True
    auc_ok = (fallback_auc >= primary_auc) if overlap_selection_require_auc_non_worse else True
    trades_ok = fallback_trades >= overlap_selection_min_fallback_trades
    selected_override = bool(ret_improvement_ok and primary_negative_ok and auc_ok and trades_ok)

    if selected_override:
        overlap_payload["selected_model_kind"] = overlap_selection_fallback

    return {
        "enabled": True,
        "primary_model": overlap_selection_primary,
        "fallback_model": overlap_selection_fallback,
        "selected_override": bool(selected_override),
        "constraints": {
            "min_ret_improvement": overlap_selection_min_ret_improvement,
            "only_when_primary_negative": overlap_selection_only_when_primary_negative,
            "require_fallback_auc_non_worse": overlap_selection_require_auc_non_worse,
            "min_fallback_trades": overlap_selection_min_fallback_trades,
        },
        "metrics": {
            "primary_ret": primary_ret,
            "fallback_ret": fallback_ret,
            "primary_auc": primary_auc,
            "fallback_auc": fallback_auc,
            "fallback_trade_count": fallback_trades,
        },
        "checks": {
            "ret_improvement_ok": bool(ret_improvement_ok),
            "primary_negative_ok": bool(primary_negative_ok),
            "auc_ok": bool(auc_ok),
            "trades_ok": bool(trades_ok),
        },
    }


@dataclass
class OverlapModelPruningResult:
    pruning_payload: Dict[str, Any]
    selected_row: Dict[str, Any]
    allows_tuning: bool


def apply_overlap_model_pruning(
    *,
    overlap_payload: Dict[str, Any],
    overlap_compare_output: Path,
    overlap_pre_tuning_cfg: Mapping[str, Any],
) -> OverlapModelPruningResult:
    overlap_rows_obj = overlap_payload.get("rows", [])
    overlap_rows = overlap_rows_obj if isinstance(overlap_rows_obj, list) else []

    overlap_pruning_cfg_obj = overlap_pre_tuning_cfg.get("model_pruning", {})
    overlap_pruning_cfg = overlap_pruning_cfg_obj if isinstance(overlap_pruning_cfg_obj, dict) else {}
    overlap_pruning_enabled = bool(overlap_pre_tuning_cfg.get("enabled", True)) and bool(
        overlap_pruning_cfg.get("enabled", True)
    )
    overlap_min_model_cum_ret = float(overlap_pruning_cfg.get("min_cum_ret", 0.0))
    overlap_min_model_trades = int(overlap_pruning_cfg.get("min_trade_count", 10))

    pruned_rows: list[Dict[str, Any]] = []
    rejected_rows: list[Dict[str, Any]] = []
    for row in overlap_rows:
        if not isinstance(row, dict):
            continue
        row_ret = float(row.get("cum_ret_net_total", float("nan")))
        row_trades = int(row.get("trade_count_total", 0) or 0)
        reasons: list[str] = []
        if row_ret < overlap_min_model_cum_ret:
            reasons.append("min_cum_ret")
        if row_trades < overlap_min_model_trades:
            reasons.append("min_trade_count")
        if reasons:
            rejected_rows.append(
                {
                    "model_kind": row.get("model_kind"),
                    "cum_ret_net_total": row_ret,
                    "trade_count_total": row_trades,
                    "reasons": reasons,
                }
            )
        else:
            pruned_rows.append(row)

    pruned_selected_model_kind = None
    pruned_selected_row: Dict[str, Any] | None = None
    if overlap_pruning_enabled and pruned_rows:
        pruned_rows_sorted = sorted(
            pruned_rows,
            key=lambda row: (
                float(row.get("cum_ret_net_total", float("-inf"))),
                int(row.get("trade_count_total", 0) or 0),
                float(row.get("auc_mean", float("-inf"))),
            ),
            reverse=True,
        )
        pruned_selected_row = pruned_rows_sorted[0]
        pruned_selected_model_kind = str(pruned_selected_row.get("model_kind", ""))
        overlap_payload["selected_model_kind"] = pruned_selected_model_kind

    require_viable_for_tuning = bool(overlap_pruning_cfg.get("require_viable_model_for_tuning", True))
    allows_tuning = (not overlap_pruning_enabled) or bool(pruned_rows) or (not require_viable_for_tuning)
    selected = pruned_selected_row if pruned_selected_row is not None else selected_row(overlap_payload)

    pruning_payload = {
        "enabled": bool(overlap_pruning_enabled),
        "source_compare_path": str(overlap_compare_output),
        "constraints": {
            "min_cum_ret": overlap_min_model_cum_ret,
            "min_trade_count": overlap_min_model_trades,
            "require_viable_model_for_tuning": require_viable_for_tuning,
        },
        "selected_model_from_compare": overlap_payload.get("selected_model_kind"),
        "pruned_selected_model": pruned_selected_model_kind,
        "pruned_selected_row": pruned_selected_row,
        "viable_rows": pruned_rows,
        "rejected_rows": rejected_rows,
        "allows_tuning": bool(allows_tuning),
    }
    return OverlapModelPruningResult(
        pruning_payload=pruning_payload,
        selected_row=selected,
        allows_tuning=bool(allows_tuning),
    )