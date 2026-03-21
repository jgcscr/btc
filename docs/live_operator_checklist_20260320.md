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

## Latest Fresh Conservative Snapshot

Fresh run used for this checklist:

- generated at `2026-03-21T05:10:33.769700+00:00`

Current operator-facing state:

- feature coverage: `ok = true`
- directional bias: bullish across the mid-term stack
- `1h`: rejected by `insufficient_mfe_headroom`
- `4h`: rejected by `insufficient_mfe_headroom`
- `8h`: `ready`, side `long`, `position_size = 0.0025482643209240163`, `position_size_cap = 0.20`, `confidence_min_source = 8h@trend_ignition`
- `12h`: `bias_only_ready`, `trade_action = hold`, `abstention.reason = edge_over_fee_below_min`

Practical reading:

- keep the bullish bias,
- the current model-approved live path is the `8h` setup,
- do not upsize the `8h` trade beyond the configured `0.20` cap,
- do not substitute the `12h` bias-only setup for the ready `8h` setup while `12h` remains blocked by `edge_over_fee_below_min`.

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
2. If `8h` is `ready` and `prompt_ready_summary.operator_summary_compact.recommended_operator_action = enter_now`, use the `8h` execution plan and keep size inside the configured `0.20` cap.
3. If `1h` or `4h` is rejected for `insufficient_mfe_headroom`, treat those horizons as non-confirming rather than as overrides of the ready `8h` setup.
4. If `12h` is `bias_only_ready` because `edge_over_fee_below_min`, keep the directional bias but do not substitute `12h` for the ready `8h` plan.
5. If a later refresh promotes `4h` or `12h` to `ready` with stronger follow-through, re-evaluate the preferred horizon instead of carrying forward a stale `8h` bias.

## Escalate Or Pause

Reduce risk or pause if any of these appears:

1. repeated `8h`-preferred long setups without confirmation from `4h` or `12h`
2. repeated `8h`-preferred long setups with deteriorating `edge_over_fee` or `insufficient_mfe_headroom` across the confirming stack
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
