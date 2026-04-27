# Operations Runbook

This is the current operator reference for this workspace.

Use it to answer four practical questions:

1. Which command path should be used for each job?
2. Which artifacts are authoritative after the run?
3. Which shell cadence behavior is local-only versus automated in GitHub Actions?
4. Which current runtime checks should block promotion, deployment, or trade action?

## 1. Pick The Right Path First

There are five distinct operating contexts in the current codebase:

1. direct research refresh
2. direct live inference
3. reliability workflow execution
4. shell cadence execution
5. service/API execution through a subprocess worker boundary

Do not mix them.

Recommended command paths:

- research refresh: `python -m src.scripts.run_research_refresh`
- live inference: `python -m src.scripts.run_live_inference`
- reliability workflow: `python -m src.scripts.run_reliability_pipeline`
- shell cadence: `bash ./scripts/run_cadence.sh <daily|weekly|monthly|shadow>`
- service/API: `src/service/main.py`

Legacy full-surface note:

- `src.scripts.run_refresh_and_predict` is still the full CLI and prediction executor, but it is no longer the simplest operator entrypoint for normal use.

## 2. Exact Commands By Mode

### Research Refresh

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12
```

### Live Inference

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference
```

Optional explicit live config:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Current wrapper note:

- when `0.25` is in the target set, `src.scripts.run_live_inference` automatically forwards `--intrabar-enabled`

### Reliability Workflow

Runtime profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Default profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail
```

Current workflow note:

- the checked-in runtime and default reliability profiles already use `quality.lookback_rows: 50000` and `quality.lookback_hours: 8760`, so the workflow path operates on the hardened long-window labeled slice rather than the older short 2000-row manual rebuild

### Shell Cadence

```bash
bash ./scripts/run_cadence.sh daily
bash ./scripts/run_cadence.sh weekly
bash ./scripts/run_cadence.sh monthly
bash ./scripts/run_cadence.sh shadow
```

### Replay Validation

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run \
  --targets 1,4,8,12 \
  --replay-offset-bars 24
```

## 3. What Each Path Actually Does

### Research And Live Wrappers

Both wrappers parse through the legacy CLI surface and then execute:

- `src.runtime.refresh_pipeline.execute_refresh_pipeline(...)`

That runtime pipeline currently owns:

- runtime telemetry under `artifacts/runtime_runs/<run-id>/`
- market preparation
- prediction input resolution
- runtime-owned prediction dispatch via `src.runtime.prediction_execution`
- summary writing
- monitoring artifact writing

The remaining broad CLI/config compatibility surface still lives in:

- `src/scripts/run_refresh_and_predict.py`

### Reliability Wrapper

The reliability wrapper executes:

- `src.runtime.reliability_pipeline.execute_reliability_pipeline(...)`

That runtime layer writes runtime telemetry, but the actual workflow steps still come from:

- `src.scripts.run_reliability_workflow`

### Shell Cadence

The shell wrapper in `scripts/run_cadence.sh` currently:

- resolves the latest trustworthy reliability run through `src.scripts.resolve_latest_trustworthy_reliability_run`
- uses `configs/run_refresh_and_predict.shadow_simplified.yaml` for the daily refresh path
- runs reliability through `src.scripts.run_reliability_pipeline`
- runs `shadow` comparison locally through `src.scripts.run_shadow_profile_comparison`

### GitHub Actions Cadence

The workflow in `.github/workflows/cadence.yml` is narrower than the shell wrapper:

- scheduled/manual modes supported: `daily`, `weekly`, `monthly`
- `shadow` is not scheduled or wired into workflow dispatch
- workflow preflights and bootstraps cadence artifacts using `src.scripts.bootstrap_cadence_artifacts`
- workflow runs a deployment-gating live wrapper smoke check using `src.scripts.run_live_wrapper_smoke_check` before cadence execution
- workflow rejects remote artifact URIs and expects local filesystem paths visible to the self-hosted runner

Smoke check blocking semantics in cadence workflow:

1. wrapper process exits non-zero
2. prediction stage does not complete
3. `predictions.json` / `monitoring.json` missing
4. required trust telemetry fields missing
5. `4h` trust-action contract mismatch or unexpected `8h` live emission

## 4. Current Source Of Truth Artifacts

Read these first after a direct refresh, live inference run, or cadence refresh:

- `artifacts/predictions/latest.json`
- `artifacts/predictions/history.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/data_quality_latest.json`
- `artifacts/runtime_runs/latest.json`
- `artifacts/runtime_runs/latest_by_mode/<mode>.json`
- `artifacts/runtime_runs/<run-id>/request.json`
- `artifacts/runtime_runs/<run-id>/events.jsonl`
- `artifacts/runtime_runs/<run-id>/summary.json`
- `artifacts/runtime_runs/<run-id>/predictions.json`
- `artifacts/runtime_runs/<run-id>/monitoring.json`
- `artifacts/runtime_runs/<run-id>/trade_ready.json`

Read these first after a reliability run:

- `artifacts/reliability/registry/latest.json`
- `artifacts/reliability/registry/latest_trustworthy.json`
- `artifacts/reliability/<run-id>/summary/workflow_manifest.json`
- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/directional_objectives.json`
- `artifacts/reliability/<run-id>/summary/walkforward_labeled_reconciliation.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

## 5. Runtime Profiles And Their Roles

Use the profile that matches the task.

- `configs/run_refresh_and_predict.default.yaml`: trusted research/comparison baseline
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`: approved live-style profile
- `configs/run_refresh_and_predict.live_conservative.yaml`: backward-compatible alias
- `configs/run_refresh_and_predict.research_safe.yaml`: research fallback that keeps the default stack but downgrades feature-coverage violations from hard-fail to warning
- `configs/run_refresh_and_predict.shadow_simplified.yaml`: cadence daily profile
- `configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml`: feature-lift shadow or paper package that swaps only the 4h candidate direction and regression artifacts and the hardened full-history trade-decision model
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`: shadow comparison left-hand profile
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`: shadow comparison right-hand profile
- `configs/run_refresh_and_predict.shadow_strict_abstention.yaml`: shadow-only stricter abstention profile

Feature-lift packaging artifacts:

- `artifacts/analysis/featurelift_20260331_rerun/shadow_rollout_4h_package.md`: operator-facing summary of the 4h shadow package
- `artifacts/models/featurelift_20260331_rerun/trade_decision_model_full_history.json`: hardened decision-gate artifact rebuilt on the long-window labeled slice

Current wrapper-emitted live-style size caps in the approved Binance-only profile:

- `15m = 0.0`
- `1h = 0.15`
- `4h = 0.35`
- `12h = 0.35`

Current config note:

- the live conservative config still contains some dormant `8h` controls, but the direct live wrapper does not emit `8h` because its default targets are `0.25,1,4,12`

## 6. Minimum Checks Before Trusting A Runtime Snapshot

Confirm all of these before acting on a fresh run:

1. `generated_at` is current enough for the decision window
2. `request.local_feature_overrides.feature_coverage.ok` is true when local overrides are used
3. source freshness in `request.local_feature_overrides.source_freshness` is acceptable when local overrides are used
4. `prompt_ready_summary.operator_summary_compact` matches the interpretation you are about to use
5. per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, `confidence_min_source`, and `abstention.reason` support the same conclusion

Operator-facing fields worth checking first:

- `prompt_ready_summary.market_outlook_strategy`
- `prompt_ready_summary.operator_summary_compact`
- `blocked_trade_analytics`
- `degradation_monitoring`

Per-horizon fields worth checking first:

- `trade_action`
- `execution_plan.status`
- `execution_plan.reason`
- `trade_decision`
- `forecast_coherence`
- `confluence`
- `abstention`
- `uncertainty`

## 7. Feature-Lift 4h Shadow Rollout

Refresh the packaged 4h candidate profile and its operator summary:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_featurelift_4h_shadow_rollout
```

Validate the packaged config without emitting a live run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml \
  --dry-run
```

Rebuild the hardened decision gate artifact if the package summary shows an outdated model path or deploy-readiness change:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.build_labeled_backtest_from_history \
  --include-reliability-snapshots \
  --lookback-rows 50000 \
  --min-rows 300 \
  --output artifacts/monitoring/labeled_backtest_1h_full_history.csv \
  --meta-output artifacts/monitoring/labeled_backtest_1h_full_history_meta.json

/workspaces/btc/.venv/bin/python -m src.scripts.enrich_backtest_with_decision_features \
  --input artifacts/monitoring/labeled_backtest_1h_full_history.csv \
  --output artifacts/monitoring/labeled_backtest_1h_full_history_enriched.csv \
  --meta-output artifacts/monitoring/labeled_backtest_1h_full_history_enriched_meta.json \
  --auto-discover-sources

/workspaces/btc/.venv/bin/python -m src.scripts.train_trade_decision_model \
  --input artifacts/monitoring/labeled_backtest_1h_full_history_enriched.csv \
  --output artifacts/models/featurelift_20260331_rerun/trade_decision_model_full_history.json \
  --candidate-only
```

### 12h Interaction Shadow Rollout

Write the dedicated 12h shadow rollout package:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_featurelift_12h_shadow_rollout
```

Validate the 12h candidate config without emitting a live run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_featurelift_12h_candidate.yaml \
  --dry-run
```

Current guidance:

- keep the validated 4h package active for the current feature-lift lane
- use the 12h candidate only for horizon-specific interaction experiments
- do not replace the main 12h artifacts with broad pruning variants unless walkforward improves versus the current untrimmed stack

## 8. Derivatives Shadow Rollout

Generate the derivatives-first shadow candidate config and readiness package:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_derivatives_shadow_rollout
```

Validate the candidate config without emitting a live run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_derivatives_candidate.yaml \
  --dry-run
```

Run the candidate through the regular shadow-comparison surface:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_derivatives_shadow_comparison
```

## 9. State-Engineering Guarded Rollout Package

Write the dedicated guarded state-engineering rollout package:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_state_engineering_guarded_rollout
```

Refresh the underlying guarded 4h-only shadow artifact when needed:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_state_engineering_guarded_shadow
```

## 10. Unified Signal-Expansion Rollout Summary

Write the combined rollout summary for the active expansion lanes:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_signal_expansion_rollout
```

Current rollout posture:

- derivatives: next-priority shadow lane with a dedicated candidate config
- 4h feature-lift: existing retrain-driven shadow package remains active
- 12h interaction experiments: use the dedicated shadow package rather than changing the validated shared stack in place
- state-engineering: keep the guarded `4h`-only runner rather than widening scope
- macro: keep deprioritized unless materially new evidence appears

### Trust Hardening Rollout Checks

For profiles with trust hardening enabled, verify these fields in `artifacts/runtime_runs/<run-id>/predictions.json` or `artifacts/runtime_runs/<run-id>/monitoring.json` before promotion to live use.

Current live-wrapper path:

- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`
- `configs/run_refresh_and_predict.live_conservative.yaml`

Current research/default comparison profile:

- `configs/run_refresh_and_predict.default.yaml`

- `trust_status`
- `trust_reasons`
- `excluded_from_voting`
- `voting_weight_after_trust`
- `trust_hardening_action`
- `trust_hardening_changed_outcome`

Expected current policy behavior:

- `4h` low trust is allowed to remain in voting but deweighted (`trust_hardening_action=deweight`, `voting_weight_after_trust=0.5`)
- `8h` is removed from live decision logic and should not appear in current live-wrapper decision artifacts
- fail-closed is enabled (`fail_closed=true`), so missing metadata should produce `missing_required_trust_metadata`

Operational blocker interpretation:

1. if any covered horizon reports `missing_required_trust_metadata`, treat the snapshot as unsafe for live promotion
2. if `trust_hardening_changed_outcome=true`, require manual review of `blocked_trade_analytics` and `prompt_ready_summary` before action
3. if trust fields are absent, treat trust hardening as not active for that profile

## 7. Reliability Review Order

Read these in order:

1. `summary/champion_gate_alignment_check.json`
2. `summary/promotion_gate.json`
3. `summary/directional_objectives.json`
4. `summary/trade_decision_model_shift_guard.json`
5. `summary/edge_trustworthiness.json`
6. `summary/walkforward_labeled_reconciliation.json`
7. `summary/calibration_robustness.json` when present
8. `summary/rolling_ab_report.json` when present

Operational interpretation rules:

1. if alignment fails, treat the run as workflow-invalid
2. if promotion is blocked, treat the run as diagnostic only
3. if trade-decision model shift guard fails, do not deploy from that run
4. if edge trustworthiness is false, do not use that run as the cadence source bundle

## 8. Current Shell Cadence Behavior

### Daily

1. resolve latest trustworthy reliability run
2. run `run_refresh_and_predict` with `configs/run_refresh_and_predict.shadow_simplified.yaml`
3. write prediction and monitoring artifacts

### Weekly

1. run reliability workflow with `configs/reliability_workflow.runtime.yaml`
2. re-resolve latest trustworthy run
3. execute the daily refresh path

### Monthly

1. run reliability workflow with `configs/reliability_workflow.default.yaml`
2. re-resolve latest trustworthy run
3. execute the daily refresh path

### Shadow

1. resolve latest trustworthy reliability run
2. run `src.scripts.run_shadow_profile_comparison`
3. compare `shadow_direction_enhanced_relaxed_chop` vs `shadow_chop_suppression`

## 9. Service API Surface

FastAPI service lives in `src/service/main.py`.

Current endpoints:

- `GET /health`
- `GET /jobs`
- `POST /jobs/{job_name}`
- `POST /run-signal`
- `POST /run-dataset-refresh`
- `POST /run-reliability-workflow`
- `POST /run-walkforward`

Current registered job names:

- `live-inference`
- `research-refresh`
- `reliability-workflow`
- `walkforward-validation`

Current non-feature:

- `POST /run-papertrade` returns `501`

## 10. Local Data And Feature Refresh Notes

Fresh local rebuild paths currently attempt:

- spot ingestion
- intrabar aggregation when enabled
- best-effort derivatives refresh
- best-effort macro refresh
- best-effort on-chain refresh
- local feature-bundle preparation for inference override

Useful local support outputs:

- `data/processed/funding/hourly_features.parquet`
- `data/processed/funding/source_manifest.json`
- `data/processed/macro/daily_features.parquet`
- `data/processed/macro/source_manifest.json`
- `data/processed/onchain/hourly_features.parquet`
- `data/processed/onchain/source_manifest.json`
- `artifacts/monitoring/meta_baseline.json`
- `artifacts/monitoring/meta_baseline.parquet`

These are working-state files, not operator authorization artifacts.

Current feature-coverage gate scope in the approved live profile:

- stale-source blocking applies to core live inputs (spot/features/technical)
- `funding`, `macro`, and `onchain` are treated as best-effort support inputs and are excluded from stale-source blocking

## 11. Agent Audit Workflow

When an agent needs to inspect the system the way an operator would, use this sequence before inventing new entry points:

1. `python -m src.scripts.run_research_refresh --config configs/run_refresh_and_predict.default.yaml` for the standard research path
2. `python -m src.scripts.run_research_refresh --config configs/run_refresh_and_predict.research_safe.yaml` only when remaining feature-coverage violations should warn instead of aborting
3. `python -m src.scripts.run_live_inference` for the constrained live path
4. `python -m src.scripts.audit_train_live_feature_parity` to compare checked training metadata with live-enforced feature families
5. `python -m src.scripts.simulate_macro_shadow_enforcement` and `python -m src.scripts.simulate_state_orderflow_shadow_enforcement` for shadow-policy sweeps
6. `python -m src.scripts.confirm_state_orderflow_outcomes`, `python -m src.scripts.confirm_orderflow_two_window_stability`, `python -m src.scripts.confirm_orderflow_rolling_stability`, and `python -m src.scripts.confirm_state_engineering_narrow_scope` for follow-up confirmation
7. `python -m src.scripts.run_state_engineering_guarded_shadow` and `python -m src.scripts.summarize_signal_program_status` for guarded-shadow and program-status summaries

Current discipline:

- treat those audit and confirmation scripts as analysis-only unless a separate promotion decision changes a runtime config or wrapper

## 12. Troubleshooting

1. If `scripts/run_cadence.sh` cannot find a trustworthy run, restore cadence artifacts first. The shell wrapper depends on local `artifacts/reliability/*` state.
2. If `.venv` is unavailable, set `PYTHON_BIN=python` when invoking the cadence shell wrapper.
3. Treat fail-fast config validation errors as correctness failures, not warnings.
4. `--replay-offset-bars` is hourly-only and incompatible with `--use-local-features`.
5. Keep scratch validation and replay work under `artifacts/tmp_validation/`.

## 13. Trust Hardening Rollback Criteria

Use rollback only when trust metadata plumbing is unstable or causes repeated operator-unsafe ambiguity.

Rollback triggers:

1. repeated `missing_required_trust_metadata` in consecutive live-style runs
2. trust fields missing unexpectedly from profile expected to be trust-enabled
3. persistent increase in operator ambiguity from repeated `trust_hardening_changed_outcome=true` without stable blocker diagnostics

Rollback levers (least to most aggressive):

1. set `fail_closed: false` to fail-open while metadata pathing is repaired
2. keep trust enabled but reduce strictness via `action_by_horizon` / `deweight_factor_by_horizon` for `4h`
3. set `trust_hardening_policy.enabled: false` as emergency disable

Post-rollback requirement:

1. capture one clean run where expected trust telemetry appears and no metadata-missing reason is present before restoring fail-closed

## 14. Minimal Safe Handoff Path

Read in this order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/agent_system_handoff_20260320.md`
4. `artifacts/monitoring/reliability_promotion_deploy_manifest.json` when present
5. `artifacts/predictions/latest.json` when present
6. `artifacts/monitoring/latest.json` when present

That gives the operating map first, then the current deployed state, then the latest runtime state.