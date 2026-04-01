# Reliability Workflow Hardening Closeout 2026-04-01

This note captures the repo changes that closed the April 1 reliability-workflow blockers and aligned runtime behavior with the validated 1h policy.

## Summary

The main workflow issue was not model fitting quality but evaluation-path mismatch.

- regime-aware Platt calibration was being fit, but downstream quality checks were still reading raw `p_up`
- calibration robustness and directional objectives are now evaluated from calibrated labeled snapshots
- the official 1h directional stack is aligned to the validated three-component subset: `xgb`, `transformer`, `lstm`
- live/runtime inference now supports a diversity-aware direction ensemble policy with group caps and orthogonality pruning
- historical backtest export and canonical labeled rebuild now preserve component-level `p_up_*` probabilities end to end

## Key Code And Config Changes

### Reliability workflow

- `src/scripts/run_reliability_workflow.py`
  - derives a component-frame CSV from historical backtests for official meta-ensemble training
  - writes calibrated labeled inputs for calibration robustness and directional objectives
  - fits regime-aware Platt calibration from labeled input before those evaluations run
  - passes new directional objective regime row overrides through the workflow command builder

- `configs/reliability_workflow.default.yaml`
  - uses calibrated input for calibration robustness and directional objectives
  - trains official meta-ensemble inputs from `xgb`, `transformer`, and `lstm` component columns
  - applies sparse-chop directional overrides:
    - `min_rows_by_regime.chop: 40`
    - `max_brier_by_regime.chop: 0.255`
    - `min_f1_by_regime.chop: 0.44`

- `configs/reliability_workflow.runtime.yaml`
  - mirrors calibrated-input evaluation for runtime reliability
  - applies explicit chop-specific directional overrides for the looser runtime gate

### Runtime direction ensemble

- `src/trading/signals.py`
  - generalized directional probability collection to tree and sequence families
  - emits `p_up_components` and direction-ensemble debug payloads

- `src/trading/ensembles.py`
  - added `select_diverse_models(...)` for role-aware selection, group caps, and orthogonality filtering

- `src/scripts/run_refresh_and_predict.py`
  - added config normalization, resolution, and per-horizon scoping for `direction_ensemble_policy`

- `configs/run_refresh_and_predict.default.yaml`
  - enables horizon-aware direction ensemble selection
  - aligns 1h runtime weighting and priorities to the validated `xgb + transformer + lstm` subset

### Component fidelity and summary normalization

- `src/scripts/backtest_signals.py`
  - exports component probabilities as `p_up_*`

- `src/scripts/build_labeled_backtest_from_history.py`
  - merges component probability columns from history into the canonical labeled dataset

- `src/scripts/train_meta_ensemble.py`
  - supports arbitrary component-frame CSV inputs and sparse component handling

- `src/utils/model_summary.py`
  - standardizes model summary schema across training scripts

## Validation

Focused checks passed after the hardening changes.

- `pytest tests/test_evaluate_directional_objectives.py tests/test_reliability_workflow_alignment.py`
- directional objective override check on calibrated labeled input passed
- runtime directional policy check on calibrated labeled input passed

End-to-end confirmation:

- default workflow run `artifacts/reliability/20260401T164222Z` completed with exit code `0`
- `summary/calibration_robustness.json` passed
- `summary/directional_objectives.json` passed
- `summary/champion_gate_alignment_check.json` passed

## Current Chop Constraint

The repo still does not have enough labeled chop coverage to justify materially tighter chop gates.

- `artifacts/monitoring/labeled_backtest_1h.csv` currently contains `8761` labeled rows but only `47` chop rows
- `artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv` extends to `9193` rows total, so only limited extra older history is available in-repo

Operational rule:

- do not tighten chop-specific directional overrides until the labeled 1h monitoring set reaches at least `80` chop rows and calibrated directional checks still pass for two consecutive reliability runs