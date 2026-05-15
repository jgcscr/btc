# Direction Ensemble Family Retune Closeout 2026-05-15

Status: Current closeout note for the May 15, 2026 ensemble-family retune.

Use this together with `docs/operations_runbook.md` when checking why the active default/live/research profiles no longer prefer the older recurrent-heavy 1h weighting.

## What Changed

The runtime direction stack was already diversity-aware, but the checked-in profile weights and priorities still overemphasized recurrent models and did not cleanly encode the optional `regime_logit` family.

The landed changes do three things:

- expand the default direction-model registry so runtime defaults match the broader checked-in profile universe
- scope direction-model weights and diversity policy to the per-horizon active model set at inference time, so stale config names do not break live execution
- retune the active runtime profiles to prefer `tree` and `attention`, keep `volatility` as support, demote recurrent-heavy reranking, and allow one optional `regime_logit` member when a matching artifact exists

## Why The Retune Changed

The retune was based on a new family-value audit over the labeled monitoring slice.

Audit artifact:

- `artifacts/analysis/direction_family_value_latest.json`

Audit entrypoint:

- `src/scripts/analyze_direction_family_value.py`

The audit uses the component probability columns already persisted in labeled backtests, groups them by family via `resolve_component_group_map()`, and reports:

- family-level calibration and decision-proxy metrics
- regime and fold stability
- leave-one-out deltas to estimate incremental value by family
- pairwise correlation ranking to spot redundant components

The operating conclusion from the checked-in audit run was:

- `tree` was the strongest incremental family on the labeled slice
- `attention` remained useful enough to keep near the front of the policy
- recurrent families were still available but showed weaker marginal value on that slice
- `regime_logit` was worth keeping as an optional orthogonal family rather than a base-registry requirement

## Config And Runtime Consequences

Active profiles updated:

- `configs/run_refresh_and_predict.default.yaml`
- `configs/run_refresh_and_predict.research_safe.yaml`
- `configs/run_refresh_and_predict.live_conservative.yaml`
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`

Runtime support updated:

- `src/config_trading.py`
- `src/runtime/model_resolution_support.py`
- `src/runtime/prediction_pipeline.py`
- `src/runtime/prediction_result_support.py`
- `src/trading/signals.py`

Important compatibility rule:

- top-level `dir_model_weights` stays limited to the base direction-model registry because CLI config validation runs before optional horizon-specific `regime_logit` discovery
- `regime_logit` is instead introduced through `direction_ensemble_policy` and `regime_model_weights`, after per-horizon model resolution confirms the artifact exists

## Validation

Focused checks passed after the retune:

- `pytest tests/test_model_resolution_support.py`
- `pytest tests/test_prediction_result_support.py`
- `pytest tests/test_analyze_direction_family_value.py`
- `pytest tests/test_runtime_profile_configs.py`
- `python -m src.scripts.run_research_refresh --config configs/run_refresh_and_predict.default.yaml --targets 0.25,1,4,8,12 --write-artifacts`

The fresh default refresh succeeded after the retune. On the latest validated run, the stack still produced a mid-term long bias but rejected execution because trade filters blocked the setup, which is the expected behavior when the decision gate sees insufficient execution quality despite directional bias.