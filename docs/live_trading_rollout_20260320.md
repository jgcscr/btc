# Live Trading Rollout Policy (2026-03-20)

## Approved Runtime Profiles

- `configs/run_refresh_and_predict.default.yaml` remains the trusted research and comparison baseline.
- `configs/run_refresh_and_predict.live_conservative.yaml` is the approved initial live rollout profile.

## Why The Conservative Profile Exists

The combined dataset-covered post-fix validation is positive overall, but horizon quality is not uniform.

Covered aggregate used for the rollout decision:

- `45` added trades
- average signed return proxy `+0.004573014671526228`
- `12h`: `+0.015200327994534746` over `12` trades
- `4h`: `+0.0018605283310700377` over `22` trades
- `8h`: `-0.001595445363570682` over `11` trades

Practical implication:

- the system is trusted for live trading,
- but `8h` should carry less capital than `4h` or `12h` until additional covered evidence improves it.

Supporting operator references:

- `docs/live_operator_checklist_20260320.md`
- `docs/trade_decision_8h_hardening_memo_20260320.md`

Shadow comparison outputs remain observational only:

- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

They are useful for candidate monitoring and drift review, but they are not part of the live conservative authorization path.

## Conservative Risk Limits

The live rollout profile enforces these limits:

- `confidence_min = 0.33`
- global `position_size_cap = 0.35`
- per-horizon caps:
  - `15m`: `0.00`
  - `1h`: `0.15`
  - `4h`: `0.35`
  - `8h`: `0.20`
  - `12h`: `0.35`

Interpretation:

- `15m` remains informational only,
- `1h` can contribute to context but should not dominate live size,
- `8h` remains enabled but capped below `4h` and `12h`,
- `4h` and `12h` are the primary live-carry horizons.

Scoped runtime overrides now active in the checked-in live conservative profile:

- `confidence_min_by_horizon_regime.8h.trend_ignition = 0.23`
- `trade_decision_policy.thresholds_by_horizon_regime.8h.trend_ignition = 0.4175`
- `trade_decision_policy.thresholds_by_horizon_regime.12h.trend_ignition = 0.4175`
- `abstention_policy.thresholds_by_horizon_regime.8h.trend_ignition.hold_prob_band = 0.0`
- `confluence_policy.min_support_ratio_by_horizon.4h = 0.8`
- `confluence_policy.min_support_ratio_by_horizon.8h = 0.8`
- `execution_policy.adaptive_take_profit.min_rr_fraction_of_floor = 0.75`
- `execution_policy.regime_templates.trend_ignition.tp_multiplier = 1.1`

## Exact Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative.yaml
```

## Required Monitoring Artifacts

Inspect these after each live refresh:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` when the selected run path rewrites artifact-writing summaries

Primary fields:

- request-level `local_feature_overrides.feature_coverage.ok`
- request-level `local_feature_overrides.source_freshness`
- per-horizon `trade_decision.expected_net`
- per-horizon `execution_plan.reason`
- per-horizon `execution_plan.status`
- per-horizon `position_size`
- request-level `position_size_cap_by_horizon`

Current operating note:

- direct conservative refreshes should be interpreted from `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json`
- the `prompt_ready_summary.market_outlook_strategy` fields are useful context but are not the primary live authorization read path
- shadow comparison artifacts under `artifacts/predictions/comparisons/` should not be used to override the live conservative read path

Current snapshot discipline:

- do not rely on a hardcoded snapshot summary in this rollout note
- confirm recency from top-level `generated_at` in `artifacts/predictions/latest.json`
- confirm feature coverage and source freshness from `artifacts/monitoring/latest.json`
- confirm the preferred horizon and operator action from `prompt_ready_summary.operator_summary_compact`
- confirm per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, and position-size limits before acting

## Live Escalation Rules

Keep live trading enabled only while these conditions remain true:

- feature coverage remains `ok = true`
- the preferred live-ready horizon remains within `{4h, 8h, 12h}`
- `8h` does not become the dominant source of consecutive live-ready setups with weak follow-through
- no new wave of `forecast_coherence_gate` or `bias_direction_conflict` failures appears across the mid-term stack

Reduce risk or pause live trading if any of these happens:

- `8h` becomes the preferred horizon repeatedly while `4h` and `12h` weaken materially
- live-ready states repeat but execution reasons deteriorate into `insufficient_mfe_headroom` or coherence failures on the same side
- feature freshness or coverage fails
- the next qualified reliability run fails promotion or model-shift guards

## 8h Hardening Status

What was tried:

- a direct `8h` execution suppression candidate was tested,
- it removed some weak `8h` trades,
- but it rerouted too much flow into weaker `12h` replacements and degraded the covered aggregate,
- so it was rejected and not promoted.

What is active now:

- keep the validated default signal logic,
- harden live deployment by reducing `8h` capital exposure and using scoped horizon/regime overrides instead of forcing a blanket routing rule,
- continue collecting covered post-fix evidence before making another structural `8h` policy change.

Latest direct operator-caution extraction from the covered `8h` added-trade set:

- `11` trades
- average signed return proxy `-0.004160029236862267`
- `8h` longs: `7` trades, average `-0.005096512073318341`
- `8h` shorts: `4` trades, average `-0.0025211842730641365`

Operational consequence:

- keep `8h` enabled but underweighted,
- respect the `0.20` `8h` size cap even when `8h` is the current ready horizon,
- do not manually upsize or substitute `12h` when it remains `bias_only_ready` because `edge_over_fee_below_min`.
