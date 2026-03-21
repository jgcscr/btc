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

- generated at `2026-03-20T05:58:27.037243+00:00`

Current operator-facing state:

- feature coverage: `ok = true`
- directional bias: bullish across the mid-term stack
- `1h`: rejected by `forecast_coherence_gate`
- `4h`: `waiting_pullback`, side `long`, `pending_trade_action = long`, `position_size = 0.2400201025766676`, `position_size_cap = 0.35`
- `8h`: rejected by `insufficient_mfe_headroom`, side `long`, `position_size = 0.07313213844161717`, `position_size_cap = 0.20`
- `12h`: rejected by `insufficient_mfe_headroom`, side `long`, `position_size = 0.25781212460742925`, `position_size_cap = 0.35`

Practical reading:

- keep the bullish bias,
- do not chase a market entry,
- the only acceptable live path in this snapshot is to wait for the preferred `4h` pullback structure,
- do not override the model into a standalone `8h` long while `8h` is rejected for headroom.

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
2. If `4h` is `waiting_pullback` and `8h` or `12h` is rejected for `insufficient_mfe_headroom`, keep the directional bias but do not force a market entry.
3. If `1h` is rejected by `forecast_coherence_gate`, treat it as a context warning and do not use it to justify overriding the `4h` or `12h` execution state.
4. If `8h` is the only horizon that looks tradeable, do not promote it manually unless `4h` and `12h` are at least non-conflicting and `8h` is not rejected for `insufficient_mfe_headroom`.
5. If `4h` or `12h` becomes `ready`, prefer that horizon over a standalone `8h` continuation setup.

## Escalate Or Pause

Reduce risk or pause if any of these appears:

1. repeated `8h`-preferred long setups without confirmation from `4h` or `12h`
2. repeated `insufficient_mfe_headroom` rejections on the same side while directional bias stays one-sided
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
