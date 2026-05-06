from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_reconciliation_support import assess_overlap_reconciliation


def test_assess_overlap_reconciliation_marks_edge_trustworthy_when_checks_pass() -> None:
    result = assess_overlap_reconciliation(
        compare_output=Path("summary/full.json"),
        overlap_compare_output=Path("summary/overlap.json"),
        labeled_overlap_meta=Path("summary/meta.json"),
        overlap_model_pruning_path=Path("summary/pruning.json"),
        overlap_selection_override_payload={"enabled": False},
        full_payload={"selected_model_kind": "xgb"},
        overlap_payload={"selected_model_kind": "meta_stack"},
        full_row={"cum_ret_net_total": 0.02},
        overlap_row={"cum_ret_net_total": 0.015},
        full_depth={"n_folds": 3, "test_rows_total": 60},
        overlap_depth={"n_folds": 3, "test_rows_total": 60},
        reconcile_cfg={},
    )

    assert result.trustworthy is True
    assert result.reconciliation_payload["agreement"]["edge_trustworthy"] is True
    assert result.failed_checks == []


def test_assess_overlap_reconciliation_reports_failed_checks() -> None:
    result = assess_overlap_reconciliation(
        compare_output=Path("summary/full.json"),
        overlap_compare_output=Path("summary/overlap.json"),
        labeled_overlap_meta=None,
        overlap_model_pruning_path=Path("summary/pruning.json"),
        overlap_selection_override_payload=None,
        full_payload={"selected_model_kind": "xgb"},
        overlap_payload={"selected_model_kind": "meta_stack"},
        full_row={"cum_ret_net_total": 0.03},
        overlap_row={"cum_ret_net_total": -0.02},
        full_depth={"n_folds": 3, "test_rows_total": 60},
        overlap_depth={"n_folds": 1, "test_rows_total": 10},
        reconcile_cfg={"min_overlap_cum_ret": 0.0, "max_abs_cum_ret_gap": 0.01, "min_overlap_folds": 2, "min_overlap_test_rows": 40},
    )

    assert result.trustworthy is False
    assert "same_return_sign" in result.failed_checks
    assert "min_overlap_cum_ret" in result.failed_checks
    assert "max_abs_cum_ret_gap" in result.failed_checks
    assert "min_overlap_folds" in result.failed_checks
    assert "min_overlap_test_rows" in result.failed_checks
