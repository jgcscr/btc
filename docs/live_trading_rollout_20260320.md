# Live Trading Rollout Policy

Status: Current live policy context, but not the primary operator runbook.

For exact commands and day-to-day decision flow, use `docs/live_operator_checklist_20260320.md`, `docs/operations_runbook.md`, and `README.md` first.

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
- `4h` and `12h` are the strongest direct live-carry horizons
- the current live wrapper emits `15m`, `1h`, `4h`, and `12h`; it does not emit `8h`
- legacy `8h` controls still exist in the live conservative config, but they are not part of the direct wrapper target set

Supporting operator references:

- `docs/live_operator_checklist_20260320.md`
- `docs/trade_decision_8h_hardening_memo_20260320.md`

Shadow comparison outputs remain observational only and are not part of the live authorization path.

## 3. Conservative Risk Limits

The checked-in live conservative Binance-only profile enforces:

- `confidence_min = 0.33`
- global `position_size_cap = 0.35`
- wrapper-emitted per-horizon caps:
  - `15m = 0.00`
  - `1h = 0.15`
  - `4h = 0.35`
  - `12h = 0.35`

Interpretation:

- `15m` remains informational only
- `1h` can contribute context but should not dominate live size
- `4h` and `12h` remain the main live-carry horizons

Current config note:

- the config file still contains an `8h = 0.20` cap and some `8h`-specific overrides, but those are dormant under the direct live wrapper because its default targets are `0.25,1,4,12`

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

Current wrapper interpretation:

- the `8h` entries above remain checked-in config state, but they are inactive for the direct live wrapper path unless an operator explicitly changes the target set outside the wrapper defaults

## 5. Direct Live Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

This is the direct live-style refresh path. Do not substitute cadence or shadow commands when the task is a live decision.

Live data contract:

- Binance spot is the only hard live source assumed by the approved direct live profile.
- `funding`, `macro`, and `onchain` are runtime-optional support inputs for this profile and are excluded from stale-source blocking.

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
- the preferred actionable horizon remains within `{4h, 12h}` for the direct wrapper path, with `1h` treated as context-sized when emitted
- no new cluster of `forecast_coherence_gate` or `bias_direction_conflict` failures appears across the mid-term stack

Reduce risk or pause live trading if any of these occurs:

- live-ready states repeat but execution reasons deteriorate into coherence or execution-quality failures on the same side
- feature freshness or coverage fails
- the next qualified reliability run fails promotion or model-shift guards

## 9. Dormant 8h Config State

The checked-in live conservative config still carries some `8h` sizing and regime overrides, but the default direct live wrapper does not emit `8h` because its default target set is `0.25,1,4,12`.

Operational consequence:

- do not treat `8h` as part of the approved direct live wrapper decision surface unless you intentionally bypass the wrapper defaults
- if a separate research or comparison path emits `8h`, treat that as outside the narrow live checklist unless a new rollout decision explicitly re-approves it
