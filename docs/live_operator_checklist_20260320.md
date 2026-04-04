# Live Operator Checklist

Status: Current operator checklist for direct live-style refreshes.

Use this with `README.md` and `docs/operations_runbook.md` for the current operating surface.

This is the shortest operator path for the approved conservative live-style profile in the current workspace.

Approved profile:

- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`

Trust-hardening scope note:

- trust hardening is configured for 4h in `configs/run_refresh_and_predict.live_conservative_binance_only.yaml` and `configs/run_refresh_and_predict.live_conservative.yaml`; default profile may still include broader trust scope for research comparisons
- for the default live wrapper path, expect trust telemetry in runtime artifacts
- 8h is removed from live decision logic and live emitted target set

Preferred command path:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference
```

Use this checklist only for direct live-style refreshes. Do not substitute cadence or shadow outputs for a live operator decision.

## 1. Scope Boundary

This checklist applies only to direct live-style refreshes.

Do not use these as live authorization inputs:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Those files are observational only.

## 2. Pre-Run Checks

Before each live-style refresh:

1. Confirm the intended profile is `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`.
2. Confirm no local edits have changed caps or threshold controls unexpectedly.
3. Confirm you are not accidentally using cadence outputs in place of a direct run.
4. Confirm spot data recency is acceptable for the decision window.
5. Treat Binance spot as the hard live source unless the selected profile explicitly changes that contract.

Current runtime note:

- market preparation may attempt best-effort derivatives, macro, and on-chain local refreshes during a fresh rebuild
- the approved live profile ignores stale `funding`, `macro`, and `onchain` sources in its feature-coverage gate, so auxiliary lag alone should not block the live wrapper path
- those support local feature assembly, but they are not the operator-facing source of truth for a live decision

## 3. Run Command

Default:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference
```

Explicit profile form:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Current wrapper behavior:

- defaults `--targets` to `0.25,1,4,12`
- defaults `--hours` to `360`
- automatically forwards `--intrabar-enabled` when `0.25` is present in the target set
- writes trade-ready artifacts unless `--no-write-artifacts` is supplied
- replay-offset validation is hourly-only by runtime contract, so replay sign-off uses `1h,4h,12h` even though wrapper output still includes `15m` telemetry

## 4. Post-Run Read Order

Inspect these first:

1. `artifacts/monitoring/latest.json`
2. `artifacts/predictions/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` when the run wrote trade-ready artifacts
4. `artifacts/runtime_runs/<run-id>/summary.json` when you need exact runtime lineage or event context

## 5. Hard Gates Before Acting

All of these should still hold before acting on the output:

1. feature coverage remains acceptable when local overrides are used
2. source freshness remains acceptable when local overrides are used
3. the request-level caps still show:
   - `15m = 0.0`
   - `1h = 0.15`
   - `4h = 0.35`
   - `12h = 0.35`
  - ignore dormant `8h` cap settings in the config when the direct live wrapper path is used, because `8h` is not emitted by that wrapper
4. the preferred horizon and recommended action from `prompt_ready_summary.operator_summary_compact` match the conclusion you intend to act on
5. the chosen horizon is not blocked by `forecast_coherence`, `abstention`, or execution-plan rejection reasons that contradict the trade thesis

Trust-specific gates (when trust policy is enabled in the selected profile):

1. `4h` must expose `trust_status`, `trust_reasons`, `excluded_from_voting`, and `voting_weight_after_trust`
2. `missing_required_trust_metadata` is a hard pause condition under fail-closed
3. `4h` low trust should map to deweight (`excluded_from_voting=false`, `voting_weight_after_trust=0.5`)
4. if `trust_hardening_changed_outcome=true`, require manual confirmation against `blocked_trade_analytics` before entry

## 6. Execution Decision Tree

1. If feature coverage fails, pause live action.
2. If the preferred horizon is `ready` and the recommended operator action is immediate entry, use that horizon's execution plan inside its cap.
3. If the preferred horizon is `waiting_pullback`, wait rather than forcing the entry.
4. If the preferred horizon is `bias_only_ready`, keep the directional read but do not treat it as an executable entry.
5. If a later refresh changes the preferred horizon or execution state, re-evaluate from the fresh snapshot.

## 7. Escalate Or Pause Conditions

Reduce risk or pause if any of these appears:

1. new feature freshness or coverage failure
2. repeated `forecast_coherence_gate` or similar mid-term blocking reasons across the preferred stack
3. repeated preferred-horizon setups with deteriorating execution quality
4. the next qualified reliability run fails promotion or model-shift guards
5. trust telemetry is missing for a profile expected to be trust-enabled
6. repeated `missing_required_trust_metadata` appears in consecutive live-style runs

## 8. Rollback Controls

Use these controls in order:

1. set `trust_hardening_policy.fail_closed: false` to prevent metadata outages from forcing low-trust exclusions
2. relax per-horizon action (`exclude` to `deweight`) or increase `deweight_factor_by_horizon`
3. set `trust_hardening_policy.enabled: false` only as emergency fallback

After rollback, require at least one clean run with expected trust fields and no `missing_required_trust_metadata` before re-enabling fail-closed.

## 9. Session Logging Discipline

For each live session, log:

1. run timestamp
2. chosen profile
3. preferred horizon
4. execution status and reason for `15m`, `1h`, `4h`, and `12h`
  - log `15m` as informational-only and `1h` as context-sized unless one of them becomes the active operator focus in the emitted snapshot
  - log `8h` only when reviewing non-wrapper research or comparison profiles
5. whether the operator waited, entered, reduced risk, or paused

If a discretionary override is made, log the exact reason and why it overrode the default checklist.