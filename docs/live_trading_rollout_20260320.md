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

## Exact Command

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative.yaml
```

## Required Monitoring Artifacts

Inspect these after each live refresh:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`

Primary fields:

- `prompt_ready_summary.market_outlook_strategy.selected_direction`
- `prompt_ready_summary.market_outlook_strategy.preferred_horizon`
- `prompt_ready_summary.market_outlook_strategy.execution_state`
- `prompt_ready_summary.market_outlook_strategy.tradeable`
- per-horizon `trade_decision.expected_net`
- per-horizon `execution_plan.reason`
- per-horizon `position_size`
- request-level `position_size_cap_by_horizon`

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
- harden live deployment by reducing `8h` capital exposure instead of forcing a weaker routing rule,
- continue collecting covered post-fix evidence before making another structural `8h` policy change.
