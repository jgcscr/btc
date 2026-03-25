# 8h Hardening Memo

Historical caution reference.

This note records the operator-safe interpretation of `8h` weakness after the post-fix validation pass.

It does not replace the current live profile, the current trust basis, or the current runtime artifacts.

Use these current sources instead:

- `docs/live_trading_rollout_20260320.md`
- `docs/trade_decision_post_fix_trust_basis_20260319.md`
- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`

## Scope

It does not replace the trusted default or the conservative live profile. It explains why the current live stance remains:

- keep the validated signal logic,
- underweight `8h` capital,
- avoid forcing standalone `8h` longs without stronger stack confirmation.

## 1. Direct Covered 8h Extraction

The latest check used the dataset-covered pairwise summaries and filtered only rows where:

- timestamps were aligned,
- the base profile stayed `hold`,
- the candidate profile added a trade,
- the candidate `prompt_preferred_horizon` was `8h`.

Those rows were then matched directly to `artifacts/datasets/btc_features_multi_horizon_splits.npz` using the `8h` label timestamps.

Resulting covered `8h` added-trade set:

- total trades: `11`
- average signed return proxy: `-0.004160029236862267`
- positive vs non-positive: `3` vs `8`

By side:

- `8h` longs: `7` trades, average `-0.005096512073318341`, positive vs non-positive `2` vs `5`
- `8h` shorts: `4` trades, average `-0.0025211842730641365`, positive vs non-positive `1` vs `3`

Largest losses in the covered `8h` set were concentrated in added longs, including:

- window `default_profile_pairwise_extension960_1152_post_fix_20260319`, offset `1116`, signed return `-0.020912032574415207`
- window `default_profile_pairwise_extension6_954_post_fix_20260320`, offset `456`, signed return `-0.01903829723596573`
- window `default_profile_pairwise_extension960_1152_post_fix_20260319`, offset `1146`, signed return `-0.015797356143593788`

## 2. Interpretation

This is weaker than the broad horizon-level aggregate and should be treated as an operator caution lens, not as a replacement for the official trust-basis aggregate.

What it means operationally:

- `8h` remains the weakest live-carry horizon,
- weakness is more concentrated in `8h` longs than in `8h` shorts,
- the current risk control should continue to be capital underweighting rather than structural rerouting,
- a blanket `8h` suppression rule is still not justified because the earlier direct suppression experiment degraded the wider covered aggregate.

## 3. Safe Action Now

Keep these rules active:

1. leave `configs/run_refresh_and_predict.default.yaml` unchanged
2. keep `configs/run_refresh_and_predict.live_conservative.yaml` with `8h` capped below `4h` and `12h`
3. do not manually force a standalone `8h` long when `8h` is rejected for `insufficient_mfe_headroom`
4. do not manually promote an `8h` long if `4h` is not ready or if `12h` is simultaneously failing headroom on the same side

## 4. Safest Next Structural Candidate

If another code-level hardening pass is attempted, the safest candidate is:

- a conditional `8h`-long tightening rule only,
- validated first on covered replay,
- without suppressing all `8h` flow,
- and without rerouting marginal cases blindly into `12h`.

Until that validation exists, the approved production stance remains operator discipline plus horizon-specific size caps.
