from __future__ import annotations

from pathlib import Path

from src.runtime.reliability_overlap_selection_support import (
    apply_overlap_model_pruning,
    apply_overlap_model_selection_override,
    row_by_model_kind,
    selected_row,
)


def test_apply_overlap_model_selection_override_promotes_fallback_when_constraints_pass() -> None:
    overlap_payload = {
        "selected_model_kind": "xgb",
        "rows": [
            {"model_kind": "xgb", "cum_ret_net_total": -0.01, "auc_mean": 0.53, "trade_count_total": 12},
            {"model_kind": "meta_stack", "cum_ret_net_total": 0.02, "auc_mean": 0.55, "trade_count_total": 15},
        ],
    }

    override_payload = apply_overlap_model_selection_override(
        overlap_payload=overlap_payload,
        reconcile_cfg={
            "overlap_model_selection": {
                "enabled": True,
                "primary_model": "xgb",
                "fallback_model": "meta_stack",
                "min_ret_improvement": 0.01,
                "only_when_primary_negative": True,
                "require_fallback_auc_non_worse": True,
                "min_fallback_trades": 10,
            }
        },
    )

    assert overlap_payload["selected_model_kind"] == "meta_stack"
    assert override_payload is not None
    assert bool(override_payload["selected_override"]) is True
    assert bool(override_payload["checks"]["auc_ok"]) is True


def test_apply_overlap_model_selection_override_reports_missing_rows() -> None:
    overlap_payload = {"selected_model_kind": "xgb", "rows": [{"model_kind": "xgb"}]}

    override_payload = apply_overlap_model_selection_override(
        overlap_payload=overlap_payload,
        reconcile_cfg={"overlap_model_selection": {"enabled": True, "fallback_model": "meta_stack"}},
    )

    assert override_payload == {
        "enabled": True,
        "primary_model": "xgb",
        "fallback_model": "meta_stack",
        "selected_override": False,
        "reason": "missing_model_rows",
    }


def test_apply_overlap_model_pruning_selects_best_viable_model_and_allows_tuning() -> None:
    overlap_payload = {
        "selected_model_kind": "xgb",
        "rows": [
            {"model_kind": "xgb", "cum_ret_net_total": 0.01, "trade_count_total": 12, "auc_mean": 0.54},
            {"model_kind": "meta_stack", "cum_ret_net_total": 0.015, "trade_count_total": 15, "auc_mean": 0.56},
            {"model_kind": "lstm", "cum_ret_net_total": -0.02, "trade_count_total": 20, "auc_mean": 0.58},
        ],
    }

    result = apply_overlap_model_pruning(
        overlap_payload=overlap_payload,
        overlap_compare_output=Path("summary/overlap.json"),
        overlap_pre_tuning_cfg={"enabled": True, "model_pruning": {"enabled": True, "min_cum_ret": 0.0, "min_trade_count": 10}},
    )

    assert overlap_payload["selected_model_kind"] == "meta_stack"
    assert result.selected_row["model_kind"] == "meta_stack"
    assert result.allows_tuning is True
    assert result.pruning_payload["pruned_selected_model"] == "meta_stack"
    assert len(result.pruning_payload["rejected_rows"]) == 1


def test_apply_overlap_model_pruning_blocks_tuning_when_no_viable_rows_required() -> None:
    overlap_payload = {
        "selected_model_kind": "xgb",
        "rows": [{"model_kind": "xgb", "cum_ret_net_total": -0.01, "trade_count_total": 1, "auc_mean": 0.54}],
    }

    result = apply_overlap_model_pruning(
        overlap_payload=overlap_payload,
        overlap_compare_output=Path("summary/overlap.json"),
        overlap_pre_tuning_cfg={
            "enabled": True,
            "model_pruning": {
                "enabled": True,
                "min_cum_ret": 0.0,
                "min_trade_count": 10,
                "require_viable_model_for_tuning": True,
            },
        },
    )

    assert result.allows_tuning is False
    assert result.pruning_payload["viable_rows"] == []
    assert selected_row(overlap_payload)["model_kind"] == "xgb"
    assert row_by_model_kind(overlap_payload, "missing") == {}