# Agent System Handoff

This document is the shortest safe handoff for an agent taking over this repository.

It is not a changelog. Use it to understand which runtime path to use, which artifacts are authoritative, and which mistakes to avoid when switching between live-style refreshes, cadence operations, and reliability work.

## 1. Operating Split

There are three distinct operating contexts in this workspace:

1. Direct runtime refreshes for current market state.
2. Cadence refreshes driven by the latest trustworthy reliability run.
3. Reliability workflows that rebuild and validate deployable artifacts.

Do not treat these as interchangeable.

Core runtime references:

- `configs/run_refresh_and_predict.default.yaml`: trusted research and comparison baseline.
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`: approved conservative live profile for current operations.
- `configs/run_refresh_and_predict.shadow_simplified.yaml`: artifact-writing cadence refresh profile used by `daily`.
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`: active left-hand shadow comparison profile.
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`: active right-hand shadow comparison profile.
- `.github/workflows/validation-guards.yml`: parity, leakage, and report-freshness guard workflow.

Backward compatibility note:

- `configs/run_refresh_and_predict.live_conservative.yaml` remains available as a legacy-equivalent alias.

## 2. Source Of Truth

For the current runtime state, read these artifacts first:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

For March 31 model-improvement context, also read:

- `artifacts/analysis/featurelift_20260331_rerun/comparison_report.md`
- `artifacts/analysis/featurelift_20260331_rerun/comparison_report.json`

Treat those artifacts, not this handoff note, as the source of truth for:

- current deploy lineage
- latest trustworthy run usage
- preferred horizon
- recommended action
- current blockers and execution state

## 3. Read Order

Read these first in order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/trade_decision_post_fix_trust_basis_20260319.md`
4. `docs/live_trading_rollout_20260320.md`
5. `docs/live_operator_checklist_20260320.md`
6. `docs/trade_decision_8h_hardening_memo_20260320.md`
7. `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
8. `artifacts/analysis/featurelift_20260331_rerun/comparison_report.md` when the task touches features, datasets, or model quality

If older promotion context is needed, then read:

- `docs/trade_decision_operator_handoff_20260316.md`

## 4. Golden Paths

### Direct conservative live-style refresh

Use this when the task is to read the current market under the approved live profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Inspect immediately after the run:

1. `artifacts/predictions/latest.json`
2. `artifacts/monitoring/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` only if that run path refreshed it

### Standard cadence refresh

Use this for scheduled or operational daily refreshes:

```bash
bash ./scripts/run_cadence.sh daily
```

Current behavior:

- resolves the latest trustworthy reliability run
- refreshes with `configs/run_refresh_and_predict.shadow_simplified.yaml`
- is for cadence operations, not for discretionary live execution
- local feature builders can also refresh macro and on-chain bundles opportunistically during runtime rebuilds

### Shadow comparison cadence

Use this for observational profile comparison only:

```bash
bash ./scripts/run_cadence.sh shadow
```

Current behavior:

- compares `shadow_direction_enhanced_relaxed_chop` against `shadow_chop_suppression`
- records `source_reliability_run_id` separately from the comparison run id
- determines `latest` by `generated_at`, not by lexicographic run id order

Read after the run:

1. `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
2. `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`
3. `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`

### Reliability workflow

Runtime pass:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Default pass:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail
```

### Replay validation

Use this when validating policy changes without touching active live logic:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

## 5. Files That Matter Most

Runtime config and policy:

- `configs/run_refresh_and_predict.default.yaml`
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`
- `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`
- `src/trading/feature_engineering.py`
- `src/trading/intrabar_features.py`
- `src/scripts/audit_feature_parity.py`
- `src/scripts/generate_featurelift_comparison_report.py`
- `src/scripts/check_featurelift_report_references.py`
- `src/scripts/refresh_onchain_features.py`

Reliability control:

- `configs/reliability_workflow.runtime.yaml`
- `configs/reliability_workflow.default.yaml`
- `scripts/run_cadence.sh`

Runtime outputs:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

Shadow comparison outputs:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Reliability outputs:

- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/directional_objectives.json`

Local processed-source outputs worth checking when feature coverage or rebuild quality is in scope:

- `data/processed/macro/daily_features.parquet`
- `data/processed/macro/source_manifest.json`
- `data/processed/onchain/hourly_features.parquet`
- `data/processed/onchain/source_manifest.json`

## 6. Safe Operating Rules

1. Do not weaken promotion gates to make a candidate pass.
2. Do not replace the trusted default with an unvalidated policy change.
3. Do not use shadow comparison outputs as live authorization inputs.
4. Do not force entries when `execution_plan.status` is `waiting_pullback`, `bias_only_ready`, or `rejected`.
5. Treat runtime config validation failures as correctness failures, not warnings.
6. Keep scratch replay and validation work under `artifacts/tmp_validation/`.
7. Do not rely on the superseded `featurelift_20260331` comparison report for current model-quality claims.
8. Regenerate local processed macro or on-chain bundles rather than assuming they exist on a fresh clone.

## 7. Checks Before Trusting A Change

For reliability and deployment changes, read in this order:

1. `summary/champion_gate_alignment_check.json`
2. `summary/promotion_gate.json`
3. `summary/directional_objectives.json`
4. `summary/trade_decision_model_shift_guard.json`
5. `summary/overlap_triggered_trade_diagnostics.json` when present
6. `summary/calibration_robustness.json`
7. `summary/rolling_ab_report.json`

For runtime policy changes, verify:

1. replay behavior is consistent with the intended profile change
2. snapshot and manifest lineage are correct
3. feature parity holds if local-feature behavior changed
4. focused regression tests pass when code changed

For feature or dataset changes, also verify:

1. `tests/test_intrabar_feature_parity.py`
2. `tests/test_onchain_loader_and_integration.py`
3. `tests/test_feature_leakage_guards.py`
4. `tests/test_featurelift_report_reference_check.py`
5. `python -m src.scripts.generate_featurelift_comparison_report`
6. `python -m src.scripts.check_featurelift_report_references`

## 8. Minimal First Session

If a new agent has to pick up the workspace quickly, the minimum safe sequence is:

1. Read the documents listed above.
2. Inspect the current deploy manifest.
3. Run a fresh conservative live-style refresh if the task is market-state related.
4. Inspect `latest.json` and `monitoring/latest.json`.
5. Decide whether the task belongs to live operations, cadence, reliability, or replay validation.

The main failure mode in this repository is mixing up direct live-style refreshes, cadence refreshes, and reliability workflows. Keep those paths separate.
