# Operations Runbook

This document is the operating reference for running the repository safely from the current workspace state.

It is written to answer four questions clearly:

1. Which command path should be used for each operating mode?
2. Which artifacts are the source of truth after each run?
3. Which checks block promotion, deployment, or trade action?
4. How should an operator or agent respond when cadence or runtime behavior drifts?

## 1. Operating Modes

The repository currently supports four cadence modes:

- `daily`: refresh live-style predictions from the latest trustworthy reliability run
- `weekly`: run the runtime reliability workflow, then refresh predictions
- `monthly`: run the full default reliability workflow, then refresh predictions
- `shadow`: run the shadow profile comparison workflow and archive comparison artifacts

Supported wrapper commands:

```bash
bash ./scripts/run_cadence.sh daily
bash ./scripts/run_cadence.sh weekly
bash ./scripts/run_cadence.sh monthly
bash ./scripts/run_cadence.sh shadow
```

## 2. Golden Rules

1. Use runtime artifacts, not static documentation, as the source of truth for the current market state.
2. Do not treat `default`, `live_conservative`, `shadow_simplified`, and shadow-comparison profiles as interchangeable.
3. Do not force entries when `execution_plan.status` is `waiting_pullback`, `bias_only_ready`, or `rejected`.
4. Do not deploy artifacts from a reliability run that fails promotion or alignment checks.
5. Treat feature coverage, source freshness, and trustworthy-run resolution as preconditions, not optional diagnostics.

## 3. Current Source Of Truth Artifacts

Read these first after any refresh, cadence run, or reliability workflow:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

For reliability review, also read:

- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/directional_objectives.json`
- `artifacts/reliability/<run-id>/summary/walkforward_labeled_reconciliation.json`

Minimum runtime checks before trusting a fresh prediction snapshot:

- `generated_at` is current
- `request.local_feature_overrides.feature_coverage.ok = true`
- `request.local_feature_overrides.source_freshness` is acceptable
- `prompt_ready_summary.operator_summary_compact` matches the interpretation being used
- per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, `confidence_min_source`, and `abstention.reason` support the same conclusion

## 4. Runtime Profiles And Their Roles

Use the profile that matches the task. The intended roles are:

- `configs/run_refresh_and_predict.default.yaml`: trusted research and comparison baseline
- `configs/run_refresh_and_predict.live_conservative.yaml`: approved conservative live profile
- `configs/run_refresh_and_predict.shadow_simplified.yaml`: artifact-writing cadence refresh profile used by `daily`
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`: active left-hand shadow comparison profile used by `shadow`
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`: active right-hand shadow comparison profile used by `shadow`

Current live-conservative size caps from config:

- `15m = 0.0`
- `1h = 0.15`
- `4h = 0.35`
- `8h = 0.20`
- `12h = 0.35`

## 5. Exact Cadence Behavior

The current wrapper behavior in `scripts/run_cadence.sh` is:

### Daily

1. Resolve the latest trustworthy reliability run.
2. Run `src.scripts.run_refresh_and_predict` with `configs/run_refresh_and_predict.shadow_simplified.yaml`.
3. Write the latest prediction and monitoring artifacts.

Equivalent command path:

```bash
RUN_ID=$(
  /workspaces/btc/.venv/bin/python - <<'PY'
import json
from pathlib import Path

run_root = Path('artifacts/reliability')
for run_dir in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
    edge_path = run_dir / 'summary' / 'edge_trustworthiness.json'
    thresholds_path = run_dir / 'summary' / 'calibrated_thresholds.json'
    platt_path = run_dir / 'summary' / 'platt_calibration.json'
    if not edge_path.exists() or not thresholds_path.exists() or not platt_path.exists():
        continue
    payload = json.loads(edge_path.read_text(encoding='utf-8'))
    if bool(payload.get('edge_trustworthy', False)):
        print(run_dir.name)
        break
PY
)

/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_simplified.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

### Weekly

1. Run `configs/reliability_workflow.runtime.yaml`.
2. Re-resolve the latest trustworthy run.
3. Run the daily refresh path.

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail

bash ./scripts/run_cadence.sh daily
```

### Monthly

1. Run `configs/reliability_workflow.default.yaml`.
2. Re-resolve the latest trustworthy run.
3. Run the daily refresh path.

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail

bash ./scripts/run_cadence.sh daily
```

### Shadow

1. Resolve the latest trustworthy reliability run.
2. Run `src.scripts.run_shadow_profile_comparison`.
3. Compare:
   - `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`
   - `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`
4. Restore the latest artifact to the right-hand profile snapshot after comparison.

```bash
bash ./scripts/run_cadence.sh shadow
```

## 6. Reliability Workflow Expectations

The reliability workflow is responsible for generating and validating the artifacts used by runtime refreshes.

Runtime profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Default profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail
```

Runtime reliability currently includes directional-objective gating through `src.scripts.evaluate_directional_objectives`.

Current runtime directional-objective thresholds:

- `group_min_rows: 40`
- `max_brier: 0.255`
- `max_ece_by_regime.chop: 0.18`

Evaluator behavior that matters operationally:

- label resolution accepts `y`, `y_true`, `target_up`, `label`, or `direction_target`
- missing or invalid regime labels are normalized to `unknown`

Treat any non-empty `failed_checks` in `summary/directional_objectives.json` as a reliability gate failure.

## 7. Review Order After A Reliability Run

Read these in order:

1. `summary/champion_gate_alignment_check.json`
2. `summary/promotion_gate.json`
3. `summary/directional_objectives.json`
4. `summary/trade_decision_model_shift_guard.json`
5. `summary/edge_trustworthiness.json`
6. `summary/walkforward_labeled_reconciliation.json`
7. `summary/overlap_feature_drift_guard.json` when present
8. `summary/overlap_triggered_trade_diagnostics.json` when present
9. `summary/calibration_robustness.json`
10. `summary/rolling_ab_report.json`

Interpretation rules:

- if alignment fails, treat the run as workflow-invalid regardless of model metrics
- if promotion is blocked, treat the run as diagnostic only
- if trade-decision model shift guard fails, do not deploy from that run
- if edge trustworthiness is false, do not use that run as the cadence source bundle

## 8. Reading Runtime Execution State

Use these runtime fields first:

- `prompt_ready_summary.market_outlook_strategy`
- `prompt_ready_summary.operator_summary_compact`
- `blocked_trade_analytics`
- `degradation_monitoring`

Per-horizon, read:

- `trade_action`
- `execution_plan.status`
- `execution_plan.reason`
- `trade_decision`
- `forecast_coherence`
- `confluence`
- `abstention`
- `uncertainty`

Execution state meaning:

- `ready`: setup is actionable within the configured profile
- `waiting_pullback`: bias is acceptable, but entry quality is not yet acceptable at the current price
- `bias_only_ready`: execution structure is acceptable, but the predictive model still abstained
- `rejected`: a hard execution or policy guard blocked the setup

Current runtime diagnostics worth checking explicitly:

- `execution_plan.stop_management.stop_scaling`
- `execution_plan.target_management.dynamic_rr_floor_applied`
- `execution_plan.target_management.dynamic_realized_rr_ratio`
- `direction_output.probability_shrinkage`
- `trade_decision.threshold_source`
- `confidence_min_source`
- `abstention.reason`
- `uncertainty.effective_policy`

Common blocking reasons that require operator discipline rather than manual override:

- `short_term_disagreement`
- `forecast_coherence_gate`
- `pullback_quality_insufficient`
- `stop_too_tight_near_invalidation`
- `stop_too_wide`
- `risk_reward_below_floor`
- `low_execution_confluence`
- `upstream_model_hold`

## 9. Shadow Comparison Review

After `bash ./scripts/run_cadence.sh shadow`, inspect:

1. `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
2. `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
3. `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`
4. `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`

Operational guidance:

- use the Markdown summary for the fastest operator review
- use the CSV to compare runs over time
- use the longitudinal JSON as the automation source of truth
- keep `source_reliability_run_id` separate from the shadow comparison run id when reasoning about lineage

## 10. Promotion And Deployment Discipline

Promotion and deploy behavior is controlled by the reliability workflow and the `quality.promotion_deploy` targets defined in config.

Use these rules:

1. Deploy only from runs that pass promotion and alignment checks.
2. If promotion is blocked, keep the existing deployed artifacts in place.
3. Treat `summary/trade_decision_model_shift_guard.json` as a hard blocker.
4. Treat `artifacts/monitoring/reliability_promotion_deploy_manifest.json` as the current deployed-state record.

Shared deployment targets currently include:

- thresholds
- Platt calibration
- trade-decision model
- promoted and incumbent labeled profiles
- promotion and calibration summaries
- the deployment manifest

## 11. Rollback Guidance

Rollback source of truth:

- `artifacts/reliability/<run-id>/summary/promotion_deploy_manifest.json`

Current deployment record:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

Use rollback when:

- a newly selected run fails alignment checks
- a newly selected run fails model-shift guard checks
- the active variant loses trustworthy-run status on the next qualified review
- live refresh behavior degrades for policy reasons rather than normal market-state variation

Restore from a prior known-good deployment manifest instead of copying files manually from memory or old notes.

## 12. GitHub Actions And Runner Model

The cadence workflow in `.github/workflows/cadence.yml` currently:

- runs on `self-hosted`
- supports manual dispatch for `daily`, `weekly`, and `monthly`
- schedules daily, weekly, and monthly cadences in UTC
- validates the artifact root before running cadence
- rejects remote artifact URIs
- runs `python -m src.scripts.bootstrap_cadence_artifacts` before `scripts/run_cadence.sh`

This workflow expects a local filesystem artifact root visible to the runner.

For durable unattended operation:

- use a non-ephemeral self-hosted machine
- do not rely on a Codespaces shell session started with `./run.sh`
- use `scripts/setup_self_hosted_runner.sh` to install and configure the Linux x64 runner when needed

## 13. Manual Live-Style Refresh

Use this path when you need an immediate direct runtime read outside cadence.

Trusted default refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml
```

Approved conservative live refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative.yaml
```

After either command, inspect:

1. `artifacts/monitoring/latest.json`
2. `artifacts/predictions/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` if that path refreshed it

## 14. Troubleshooting

- If `scripts/run_cadence.sh` cannot locate a trustworthy run, restore the artifact bundle first; cadence depends on prebuilt reliability artifacts that are not tracked in git.
- If `.venv` is unavailable, set `PYTHON_BIN=python` when calling cadence.
- Treat fail-fast runtime config validation errors as correctness issues, not warnings.
- Use `src.scripts.audit_feature_parity` before trusting local-feature override changes.
- Keep scratch replay and validation outputs under `artifacts/tmp_validation/`.

## 15. Minimal Safe Handoff Path

For a new agent or operator taking over the workspace, use this read order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/agent_system_handoff_20260320.md`
4. `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
5. `artifacts/predictions/latest.json`
6. `artifacts/monitoring/latest.json`

That sequence gives the operating map first, then the current deployment state, then the latest runtime state.
