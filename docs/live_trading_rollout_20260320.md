# Live Trading Rollout Policy

This document defines the current live trading posture for the repository.

It is intentionally narrower than the README and runbook. Its purpose is to explain which runtime profile is approved for live-style operation, why it is conservative, and how an operator should interpret the live risk controls.

## 1. Approved Runtime Profiles

The approved runtime profiles are:

- `configs/run_refresh_and_predict.default.yaml`: trusted research and comparison baseline
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`: approved live profile for current operations

These profiles serve different purposes. The default profile is the research and validation anchor. The Binance-only live conservative profile is the operational profile for direct live-style refreshes.

Backward compatibility note:

- `configs/run_refresh_and_predict.live_conservative.yaml` remains available as a legacy-equivalent alias of the same policy set.
- Operators should prefer the Binance-only named profile because it matches the current live source contract.

## 2. Why The Conservative Live Profile Exists

The post-fix validation basis is positive overall, but horizon quality is not uniform across the covered aggregate.

Covered aggregate used for the live rollout decision:

- `45` added trades
- average signed return proxy `+0.004573014671526228`
- `12h`: `+0.015200327994534746` over `12` trades
- `4h`: `+0.0018605283310700377` over `22` trades
- `8h`: `-0.001595445363570682` over `11` trades

Operational interpretation:

- the system is trusted for live-style use with conservative discipline
- `4h` and `12h` are the strongest live-carry horizons
- `8h` remains enabled, but should carry less capital until additional covered evidence improves confidence

Supporting operator references:

- `docs/live_operator_checklist_20260320.md`
- `docs/trade_decision_8h_hardening_memo_20260320.md`

Shadow comparison outputs remain observational only and are not part of the live authorization path.

## 3. Conservative Risk Limits

The checked-in live conservative Binance-only profile enforces:

- `confidence_min = 0.33`
- global `position_size_cap = 0.35`
- per-horizon caps:
  - `15m = 0.00`
  - `1h = 0.15`
  - `4h = 0.35`
  - `8h = 0.20`
  - `12h = 0.35`

Interpretation:

- `15m` remains informational only
- `1h` can contribute context but should not dominate live size
- `8h` remains enabled but underweighted relative to `4h` and `12h`
- `4h` and `12h` remain the main live-carry horizons

## 4. Active Live Conservative Overrides

The checked-in live conservative Binance-only profile currently includes these notable scoped overrides:

- `confidence_min_by_horizon_regime.8h.trend_ignition = 0.23`
- `trade_decision_policy.thresholds_by_horizon_regime.8h.trend_ignition = 0.4175`
- `trade_decision_policy.thresholds_by_horizon_regime.12h.trend_ignition = 0.4175`
- `abstention_policy.thresholds_by_horizon_regime.8h.trend_ignition.hold_prob_band = 0.0`
- `confluence_policy.min_support_ratio_by_horizon.4h = 0.8`
- `confluence_policy.min_support_ratio_by_horizon.8h = 0.8`
- `execution_policy.adaptive_take_profit.min_rr_fraction_of_floor = 0.75`
- `execution_policy.regime_templates.trend_ignition.tp_multiplier = 1.1`

These are intended to harden live operation without introducing a blanket routing or horizon-suppression rule.

## 5. Direct Live Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

This is the direct live-style refresh path. Do not substitute cadence or shadow commands when the task is a live decision.

Live data contract:

- Binance spot is the only hard live source assumed by the approved direct live profile.
- Macro context remains research-contextual and runtime-optional until a future live profile explicitly wires it in on every refresh.

## 6. Required Monitoring Artifacts

Inspect these after each direct live-style refresh:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` only if that path refreshed it

Primary fields:

- request-level `local_feature_overrides.feature_coverage.ok`
- request-level `local_feature_overrides.source_freshness`
- per-horizon `trade_decision.expected_net`
- per-horizon `execution_plan.reason`
- per-horizon `execution_plan.status`
- per-horizon `position_size`
- request-level `position_size_cap_by_horizon`

Reading discipline:

- use `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json` as the primary live authorization artifacts
- treat `prompt_ready_summary` as an operator aid, not as the only authorization read path
- do not use shadow comparison artifacts to override the direct conservative read path

## 7. Snapshot Discipline

Before acting on a fresh live snapshot:

- confirm recency from top-level `generated_at`
- confirm feature coverage and source freshness
- confirm the preferred horizon and operator action from `prompt_ready_summary.operator_summary_compact`
- confirm per-horizon `trade_action`, `execution_plan.status`, `execution_plan.reason`, and position-size limits

Do not rely on a hardcoded snapshot summary in this policy document.

## 8. Live Escalation Rules

Keep live trading enabled only while these conditions remain true:

- feature coverage remains `ok = true`
- the preferred actionable horizon remains within `{4h, 8h, 12h}`
- `8h` does not become the dominant source of repeated live-ready setups with weak follow-through
- no new cluster of `forecast_coherence_gate` or `bias_direction_conflict` failures appears across the mid-term stack

Reduce risk or pause live trading if any of these occurs:

- `8h` becomes the preferred horizon repeatedly while `4h` and `12h` weaken materially
- live-ready states repeat but execution reasons deteriorate into coherence or execution-quality failures on the same side
- feature freshness or coverage fails
- the next qualified reliability run fails promotion or model-shift guards

## 9. 8h Operating Stance

The repository does not currently use a blanket `8h` suppression rule.

What was learned from prior hardening attempts:

- direct `8h` suppression can remove weak `8h` trades
- but it can also reroute flow into weaker replacements and degrade the covered aggregate

What is active now:

- keep `8h` enabled
- underweight it through profile-level risk controls
- use scoped horizon and regime overrides rather than blanket routing changes
- continue collecting covered evidence before making another structural `8h` policy change

Supporting covered operator-caution slice:

- `11` trades
- average signed return proxy `-0.004160029236862267`
- `8h` longs: `7` trades, average `-0.005096512073318341`
- `8h` shorts: `4` trades, average `-0.0025211842730641365`

Operational consequence:

- keep `8h` enabled but underweighted
- respect the `0.20` `8h` cap even when `8h` is the current ready horizon
- do not manually upsize or replace a blocked `12h` setup just because `8h` is active
