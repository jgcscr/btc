# Live Operator Checklist

This checklist is the shortest operator path for the approved conservative Binance-only live profile:

- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`

Use it together with:

- `docs/live_trading_rollout_20260320.md`
- `docs/trade_decision_8h_hardening_memo_20260320.md`

## 1. Scope Boundary

This checklist applies only to direct conservative live-style refreshes.

Do not use cadence outputs or shadow comparison outputs as a substitute for a direct live-style refresh when making a live decision.

Shadow-only diagnostics are:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Those files are observational only. They are used to compare `shadow_direction_enhanced_relaxed_chop` against `shadow_chop_suppression`, not to authorize a live trade.

## 2. Snapshot Discipline

After each fresh conservative run, read the current state directly from:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` only if that run path refreshed it

Minimum fields to confirm before acting:

- `generated_at` is recent enough for the decision window
- `request.local_feature_overrides.feature_coverage.ok = true`
- `request.local_feature_overrides.source_freshness` is acceptable
- `prompt_ready_summary.operator_summary_compact` matches the interpretation you intend to use
- per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, `position_size_cap`, `confidence_min_source`, and `abstention.reason` support the same interpretation

## 3. Pre-Run Checks

Before each direct live-style refresh:

1. Confirm the active profile is `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`.
2. Confirm the latest trustworthy deployment is still the intended incumbent.
3. Confirm no manual local edits have changed live sizing or threshold controls.
4. Confirm data inputs are current enough for a live read.
5. Treat Binance spot as the only hard live source unless the selected profile explicitly wires macro or other external context into the refresh path.

Current local-runtime note:

- the refresh path now attempts best-effort macro and on-chain local feature refreshes
- those enrich the local feature bundle when present, but they are not a substitute for the approved hard live-source contract

## 4. Run Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Compatibility note:

- `configs/run_refresh_and_predict.live_conservative.yaml` remains available as a backward-compatible equivalent profile.
- Use the Binance-only named profile for new operator runs so the live data contract is explicit.

## 5. Post-Run Hard Gates

Inspect these first:

1. `artifacts/monitoring/latest.json`
2. `artifacts/predictions/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` only if the run path refreshed it

Live trading stays enabled only if all of these remain true:

1. feature coverage remains `ok = true`
2. source freshness remains clean
3. request-level caps still show:
   - `15m = 0.0`
   - `1h = 0.15`
   - `4h = 0.35`
   - `8h = 0.20`
   - `12h = 0.35`
4. the preferred actionable horizon remains within the mid-term live stack
5. no new cluster of `forecast_coherence_gate`, `bias_direction_conflict`, or similar mid-term execution blockers appears

Operational note:

- direct conservative refreshes should be read from `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` can lag if the selected path does not rewrite that summary
- `data/processed/macro/` and `data/processed/onchain/` are local support bundles, not operator-facing decision artifacts

## 6. Execution Decision Tree

1. If feature coverage fails, pause live trading.
2. If the preferred horizon is `ready` and `recommended_operator_action = enter_now`, use that horizon's execution plan and keep size inside that horizon's cap.
3. If lower horizons are rejected for execution-quality reasons, treat them as non-confirming rather than as automatic overrides of a ready mid-term setup.
4. If a horizon is `bias_only_ready`, keep the bias but do not substitute it for a different horizon that is actually `ready`.
5. If a later refresh changes the preferred horizon or execution state, re-evaluate from the fresh snapshot rather than carrying forward a stale decision.

## 7. Pause Or Escalate Conditions

Reduce risk or pause if any of these appears:

1. repeated `8h`-preferred setups without meaningful confirmation from `4h` or `12h`
2. repeated preferred-horizon setups with deteriorating execution quality across the confirming stack
3. new feature freshness or coverage failure
4. the next qualified reliability run fails promotion or model-shift guards

## 8. Logging Discipline

For every live session, record:

1. run timestamp
2. chosen profile
3. preferred horizon
4. execution status and reason for `4h`, `8h`, and `12h`
5. whether the operator waited, entered, reduced risk, or paused

If a discretionary override is made, log the exact reason and why it was stronger than the default checklist.
