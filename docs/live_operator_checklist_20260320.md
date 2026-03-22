# Live Operator Checklist (2026-03-20)

This checklist is the shortest operator path for the approved conservative live rollout profile:

- `configs/run_refresh_and_predict.live_conservative.yaml`

Use it together with:

- `docs/live_trading_rollout_20260320.md`
- `docs/trade_decision_8h_hardening_memo_20260320.md`

## Scope Boundary

This checklist applies only to the approved live conservative path:

- `configs/run_refresh_and_predict.live_conservative.yaml`

Do not mix in observational shadow comparison outputs when making a live decision from this checklist.

The following artifacts are shadow-only diagnostics and are not live trade-authorization inputs:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Use them to judge whether the chop-suppression candidate is drifting operationally from `shadow_simplified`, not to override the current live conservative execution state.

## Current Snapshot Discipline

Do not trust a hardcoded snapshot summary in this document.

After each fresh conservative run, read the current state directly from:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` when that run path refreshed artifact-writing summaries

Minimum fields to confirm before acting:

- top-level `generated_at` is recent enough for the decision you are about to make
- `request.local_feature_overrides.feature_coverage.ok = true`
- source freshness remains acceptable in `request.local_feature_overrides.source_freshness`
- `prompt_ready_summary.operator_summary_compact` matches the preferred horizon and recommended action you intend to follow
- per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, `position_size_cap`, `confidence_min_source`, and `abstention.reason` support the same interpretation

## Pre-Run Checks

Before each live refresh:

1. Confirm the active profile is `configs/run_refresh_and_predict.live_conservative.yaml`.
2. Confirm the latest trustworthy reliability deployment is still the intended incumbent.
3. Confirm no manual local edits have changed live sizing or threshold controls.
4. Confirm data inputs are current enough for a live read.

## Run Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative.yaml
```

## Post-Run Hard Gates

Inspect these first:

1. `artifacts/monitoring/latest.json`
2. `artifacts/predictions/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` only if the run path refreshed artifact-writing outputs

Live trading stays enabled only if all of these remain true:

1. feature coverage remains `ok = true`
2. source freshness lag remains clean
3. request-level caps still show:
   - `15m = 0.0`
   - `1h = 0.15`
   - `4h = 0.35`
   - `8h = 0.20`
   - `12h = 0.35`
4. preferred live-ready horizon remains inside `4h`, `8h`, or `12h`
5. no new cluster of `forecast_coherence_gate` or `bias_direction_conflict` appears in the mid-term stack

Current workspace note:

- direct conservative refreshes should be read from `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` may lag if the selected run path does not rewrite the artifact-writing summary outputs
- shadow comparison summaries are observational only and should not be used as a substitute for the direct conservative refresh artifacts above

## Execution Decision Tree

1. If feature coverage fails, pause live trading.
2. If the preferred horizon is `ready` and `prompt_ready_summary.operator_summary_compact.recommended_operator_action = enter_now`, use that horizon's execution plan and keep size inside that horizon's configured cap.
3. If lower horizons are rejected for execution-quality reasons such as `insufficient_mfe_headroom`, treat them as non-confirming rather than as automatic overrides of a ready mid-term setup.
4. If a horizon is `bias_only_ready`, keep the directional bias but do not substitute that horizon for a separate horizon that is actually `ready`.
5. If a later refresh changes the preferred ready horizon, re-evaluate from the fresh snapshot instead of carrying forward a stale earlier bias.

## Escalate Or Pause

Reduce risk or pause if any of these appears:

1. repeated `8h`-preferred setups without confirmation from `4h` or `12h`
2. repeated preferred-horizon setups with deteriorating `edge_over_fee` or `insufficient_mfe_headroom` across the confirming stack
3. new feature freshness or coverage failure
4. next qualified reliability run fails promotion or model-shift guards

## Logging Discipline

For every live session, record:

1. run timestamp
2. chosen profile
3. preferred horizon
4. execution status and reason for `4h`, `8h`, and `12h`
5. whether the operator waited, entered, reduced risk, or paused

If a discretionary override is made, log the exact reason and why it was stronger than the default checklist.
