from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class OverlapReconciliationAssessment:
    trustworthy: bool
    reconciliation_payload: Dict[str, Any]
    failed_checks: list[str]


def assess_overlap_reconciliation(
    *,
    compare_output: Path,
    overlap_compare_output: Path,
    labeled_overlap_meta: Path | None,
    overlap_model_pruning_path: Path,
    overlap_selection_override_payload: Dict[str, Any] | None,
    full_payload: Mapping[str, Any],
    overlap_payload: Mapping[str, Any],
    full_row: Mapping[str, Any],
    overlap_row: Mapping[str, Any],
    full_depth: Mapping[str, Any],
    overlap_depth: Mapping[str, Any],
    reconcile_cfg: Mapping[str, Any],
) -> OverlapReconciliationAssessment:
    full_ret = float(full_row.get("cum_ret_net_total", 0.0) or 0.0)
    overlap_ret = float(overlap_row.get("cum_ret_net_total", 0.0) or 0.0)
    same_sign = bool((full_ret == 0.0 and overlap_ret == 0.0) or (full_ret * overlap_ret > 0.0))
    abs_gap = float(abs(full_ret - overlap_ret))
    min_overlap = float(reconcile_cfg.get("min_overlap_cum_ret", -0.005))
    require_non_negative_overlap = bool(reconcile_cfg.get("require_non_negative_overlap_selected", True))
    overlap_non_negative_ok = (overlap_ret >= 0.0) if require_non_negative_overlap else True
    max_abs_gap = float(reconcile_cfg.get("max_abs_cum_ret_gap", 0.03))
    min_overlap_folds = int(reconcile_cfg.get("min_overlap_folds", 2))
    min_overlap_test_rows = int(reconcile_cfg.get("min_overlap_test_rows", 40))
    overlap_folds_ok = int(overlap_depth.get("n_folds", 0) or 0) >= min_overlap_folds
    overlap_test_rows_ok = int(overlap_depth.get("test_rows_total", 0) or 0) >= min_overlap_test_rows
    overlap_depth_ok = bool(overlap_folds_ok and overlap_test_rows_ok)
    trustworthy = bool(
        same_sign
        and overlap_ret >= min_overlap
        and overlap_non_negative_ok
        and abs_gap <= max_abs_gap
        and overlap_depth_ok
    )

    failed_checks: list[str] = []
    if not same_sign:
        failed_checks.append("same_return_sign")
    if overlap_ret < min_overlap:
        failed_checks.append("min_overlap_cum_ret")
    if not overlap_non_negative_ok:
        failed_checks.append("overlap_selected_non_negative")
    if abs_gap > max_abs_gap:
        failed_checks.append("max_abs_cum_ret_gap")
    if not overlap_folds_ok:
        failed_checks.append("min_overlap_folds")
    if not overlap_test_rows_ok:
        failed_checks.append("min_overlap_test_rows")

    reconciliation_payload = {
        "full_compare_path": str(compare_output),
        "labeled_overlap_compare_path": str(overlap_compare_output),
        "labeled_overlap_meta_path": str(labeled_overlap_meta) if labeled_overlap_meta else None,
        "overlap_model_pruning_path": str(overlap_model_pruning_path),
        "overlap_model_selection_override": overlap_selection_override_payload,
        "full_selected_model": full_payload.get("selected_model_kind"),
        "overlap_selected_model": overlap_payload.get("selected_model_kind"),
        "full_selected_row": dict(full_row),
        "overlap_selected_row": dict(overlap_row),
        "full_depth": dict(full_depth),
        "overlap_depth": dict(overlap_depth),
        "agreement": {
            "same_return_sign": same_sign,
            "abs_cum_ret_gap": abs_gap,
            "min_overlap_cum_ret": min_overlap,
            "require_non_negative_overlap_selected": bool(require_non_negative_overlap),
            "overlap_non_negative_ok": bool(overlap_non_negative_ok),
            "max_abs_cum_ret_gap": max_abs_gap,
            "min_overlap_folds": int(min_overlap_folds),
            "min_overlap_test_rows": int(min_overlap_test_rows),
            "overlap_folds_ok": bool(overlap_folds_ok),
            "overlap_test_rows_ok": bool(overlap_test_rows_ok),
            "overlap_depth_ok": bool(overlap_depth_ok),
            "edge_trustworthy": trustworthy,
            "failed_checks": failed_checks,
        },
    }
    return OverlapReconciliationAssessment(
        trustworthy=trustworthy,
        reconciliation_payload=reconciliation_payload,
        failed_checks=failed_checks,
    )