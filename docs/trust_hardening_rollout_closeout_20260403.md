# Trust Hardening Rollout Closeout (2026-04-03)

Status: Historical closeout note with still-useful validation context.

Use this for rollout rationale and audit background only. For current commands and live operating behavior, use `README.md`, `docs/operations_runbook.md`, `docs/live_operator_checklist_20260320.md`, and `docs/agent_system_handoff_20260320.md` first.

## 1. Final Decision

- Decision: GO for default live wrapper path.
- Default wrapper remains unchanged and points to Binance-only live profile:
  - `src/scripts/run_live_inference.py` (`DEFAULT_LIVE_CONFIG`)
- Trust hardening is now active on that default path via:
  - `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`

## 2. Active Deployed Configuration

Default live profile trust policy (active on wrapper default):

- `enabled: true`
- `horizons: [4]`
- `high_impact_horizons: [4]`
- `fail_closed: true`
- `default_action: exclude`
- `action_by_horizon: 4h=deweight`
- `deweight_factor_by_horizon: 4h=0.5`
- `divergence_abs_gap_min: 0.12`
- `divergence_flip_required: true`
- `metadata_checks.require_metadata: true`
- model summaries:
  - `artifacts/models/lgbm_dir4h_v1/summary.json`

Default live emitted target set:

- `0.25,1,4,12`

Live decision-path note:

- `8h` is removed from live decision logic and is not emitted on the default live wrapper path.

Profile parity note:

- `configs/run_refresh_and_predict.live_conservative.yaml` matches the same simplified live-policy scope.

## 3. Validation Evidence (Audit Paths)

### 3.1 Replay Validation

Artifact:

- `artifacts/analysis/trust_hardening_recent_window_validation_20260403.json`

Key results used for sign-off:

- `deweight4_exclude8` over 24 windows:
  - `predictions_with_changed_voting_set: 24`
  - `final_outcome_change_count: 0`
- fail-closed missing-metadata simulation (`missing_meta_fail_closed`):
  - `trust_reason_counts.missing_required_trust_metadata: 48`

Interpretation:

- trust hardening changes voting participation as designed, while replay did not show unstable final-outcome flips in the sampled window.
- fail-closed metadata protection is confirmed to trigger when metadata paths are unavailable.

Post-simplification replay sign-off artifacts:

- `artifacts/analysis/live_policy_simplification_4h8h_postimpl_20260403.json`
- `artifacts/analysis/live_policy_simplification_signoff_replay_20260403.json`

### 3.2 Wrapper-Path Dry Run

Signed-off run:

- `artifacts/runtime_runs/live-20260403T023108-40a3e686`

Run status artifacts:

- `summary.json` shows `mode: live`, `status: succeeded`.
- `request.json` shows `dry_run: true`.

### 3.3 Observed 4h Deweight Behavior

Artifact:

- `artifacts/runtime_runs/live-20260403T023108-40a3e686/predictions.json`

Observed fields (4h horizon):

- `trust_status: low_trust`
- `excluded_from_voting: false`
- `voting_weight_after_trust: 0.5`
- `trust_hardening_action: deweight`

### 3.4 Observed 8h Removal Behavior

Artifact:

- `artifacts/analysis/live_policy_simplification_signoff_runs_20260403.json`

Observed result:

- `8h` is absent from emitted live decision horizons in the audited post-simplification wrapper runs.
- `8h` is absent from support-horizon decision summaries on the default live wrapper path.

### 3.5 Fail-Closed Real-Path Check

Artifacts checked:

- `artifacts/runtime_runs/live-20260403T023108-40a3e686/predictions.json`
- `artifacts/runtime_runs/live-20260403T023108-40a3e686/monitoring.json`

Result:

- no `missing_required_trust_metadata` observed in this real wrapper-path run.
- run completed successfully, so fail-closed did not break runtime under current metadata pathing.

### 3.6 Operator-Visible Telemetry Presence

Telemetry confirmed in both artifacts:

- `artifacts/runtime_runs/live-20260403T023108-40a3e686/predictions.json`
- `artifacts/runtime_runs/live-20260403T023108-40a3e686/monitoring.json`

Fields confirmed:

- `trust_status`
- `trust_reasons`
- `excluded_from_voting`
- `voting_weight_after_trust`
- `trust_hardening_changed_outcome`

## 4. Rollback Procedure (Operational)

Apply rollback in this order on active live profile (`configs/run_refresh_and_predict.live_conservative_binance_only.yaml`):

1. set `trust_hardening_policy.fail_closed: false` (temporary fail-open while repairing metadata availability)
2. relax strictness by horizon:
  - adjust `action_by_horizon.4`, or
  - increase `deweight_factor_by_horizon.4`
3. emergency fallback: set `trust_hardening_policy.enabled: false`

Post-rollback exit criteria:

1. run one fresh live-style dry run
2. verify trust fields are present in both predictions and monitoring artifacts
3. verify `missing_required_trust_metadata` is absent before re-enabling fail-closed

## 5. Operator Watch Items (First Live Runs)

For the first live sessions after rollout:

1. verify 4h low-trust maps to deweight (`voting_weight_after_trust=0.5`)
2. verify 8h remains absent from live decision artifacts
3. watch for any `missing_required_trust_metadata`
4. if `trust_hardening_changed_outcome=true`, require manual review of:
   - `blocked_trade_analytics`
   - `prompt_ready_summary`
5. confirm trust fields continue to appear in both `predictions.json` and `monitoring.json`

## 6. Known Limitations And Follow-Up Candidates

Known limitations:

1. replay evidence is 24-window recent sample, not a full walkforward study for this rollout step
2. current live-wrapper trust policy targets only `4h`; `8h` is retired from live decision logic

Low-priority follow-up candidates:

1. add automated CI assertion for trust telemetry presence on live wrapper dry-run path
2. add automated alerting on `missing_required_trust_metadata`
3. evaluate whether any regime-specific trust action tuning is needed after additional live-paper samples

## 7. Automated Wrapper Smoke Check

Run command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_wrapper_smoke_check
```

What it verifies:

1. default live wrapper path executes successfully with a temporary dry-run config copy
2. runtime reaches `prediction` stage and writes `predictions.json` / `monitoring.json`
3. trust telemetry fields are present per horizon
4. `4h` low-trust maps to `deweight` with `voting_weight_after_trust=0.5`
5. `8h` is not emitted on the current live wrapper path
6. no `missing_required_trust_metadata` appears

Failure interpretation:

1. process/summary/prediction-stage failure indicates wrapper-path operational regression
2. missing trust fields indicates telemetry contract regression
3. `4h` action mismatch or unexpected `8h` emission indicates live-policy regression

## 8. Final Post-Simplification Sign-Off (Canonical)

This section is the final source for the simplified live-policy rollout state (8h removed from decision logic, 4h trust-guarded).

Canonical evidence artifacts:

- replay sign-off: `artifacts/analysis/live_policy_simplification_signoff_replay_20260403.json`
- first-live-run audit: `artifacts/analysis/live_policy_simplification_signoff_runs_20260403.json`

Audited live wrapper run IDs:

- `live-20260403T183722-67b61962`
- `live-20260403T183914-2a35866d`
- `live-20260403T184943-a160165f`

Final operational confirmations:

1. 8h is absent from decision logic in audited live runs (`has_8h_in_emitted=false`, `has_8h_in_support_horizons=false`).
2. 4h remains trust-guarded and deweights correctly when low trust (`trust_hardening_action=deweight`, `voting_weight_after_trust=0.5`).
3. all three audited live runs succeeded and completed prediction stage.

Replay/live telemetry caveat:

1. replay-offset sign-off is hourly-only by runtime contract, so replay evidence is evaluated on `1h,4h,12h`.
2. live wrapper outputs still include `15m` telemetry; this is expected and not a regression of the simplified decision policy.
