# Trade Decision Post-Fix Trust Basis (2026-03-19)

This document remains the rationale for why `configs/run_refresh_and_predict.default.yaml` is the trusted operating default.

For the current deployed bundle and current live-style runtime state, use:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`

## Recommended Operating Default

The recommended operating profile is the current default runtime config:

- `configs/run_refresh_and_predict.default.yaml`

This recommendation is based on post-fix replay validation after correcting a confluence-policy enforcement bug in `src/scripts/run_refresh_and_predict.py`.

## Why This Default Is Frozen

The strongest negative evidence against the promoted default was traced to a runtime bug rather than a stable model defect.

Root cause:

- horizon-specific confluence overrides (`min_support_ratio_by_horizon`, `min_aligned_horizons_by_horizon`) were present in config,
- but `_resolve_confluence_policy(...)` dropped those mappings during normalization,
- so 4h and 8h unanimity overrides were not actually enforced at runtime.

The fix is now present in:

- `src/scripts/run_refresh_and_predict.py`
- `tests/test_prediction_coherence_controls.py`

## Post-Fix Validation Basis

### Targeted leak probe

Artifacts:

- `artifacts/tmp_validation/default_profile_pairwise_targeted_4h_leak_fix_probe_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_targeted_4h_leak_fix_probe_20260319/return_proxy_summary.json`

Observed result:

- completed portion of the previously bad 4h leak window produced `0` added trades,
- which is consistent with the fixed runtime blocking the leaked 4h chop additions.

### Dataset-covered post-fix extension slices

Artifacts:

- `artifacts/tmp_validation/default_profile_pairwise_extension6_954_post_fix_20260320/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension6_954_post_fix_20260320/return_proxy_summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension960_1152_post_fix_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension960_1152_post_fix_20260319/return_proxy_summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/return_proxy_summary.json`

Current scored checkpoints:

- `6`-`954`: `159` offsets, `123` aligned rows, `26` added trades, average signed return proxy `+0.006205112119250071`
- `960`-`1152`: `33` offsets, `33` aligned rows, `12` added trades, average signed return proxy `+0.001547706492904884`
- `1158`-`1410`: `43` offsets, `28` aligned rows, `7` added trades, average signed return proxy `+0.003697181029045688`

Combined dataset-covered post-fix aggregate:

- added trades: `45`
- added-trade average signed return proxy: `+0.004573014671526228`
- positive vs nonpositive added trades: `29` vs `16`

By horizon across the combined covered aggregate:

- `12h`: `12` added trades, average `+0.015200327994534746`
- `4h`: `22` added trades, average `+0.0018605283310700377`
- `8h`: `11` added trades, average `-0.001595445363570682`

Coverage note:

- `artifacts/tmp_validation/default_profile_pairwise_extension1416_1608_post_fix_20260319/summary.json` is a clean replay-only extension with `33` offsets and `32` aligned rows,
- but its timestamps fall earlier than the local return-label dataset coverage, so it should not be used as scored trust evidence.

Interpretation:

- post-fix evidence is positive overall,
- `12h` remains the strongest contributor,
- `4h` remains positive in aggregate instead of being the drag that invalidated the earlier pre-fix fresh slice,
- `8h` is now much closer to neutral than it appeared in the early small slices, but it still remains the weakest horizon in the combined covered aggregate.

## Operating Conclusion

As of 2026-03-19, the current default is the best validated runtime profile in the repository and should remain the active operating default.

Trust statement:

- predictions are trusted for operation relative to the prepromotion baseline,
- the profile is considered post-fix trusted,
- broader replay validation should continue, but no broad policy rollback is justified by the current evidence,
- the current evidence is sufficient to justify live trading with conservative risk discipline,
- horizon-aware monitoring should remain in place because `8h` continues to lag `4h` and `12h` in the covered post-fix aggregate.

## Validation Workflow Going Forward

Use these utilities for continued validation:

- `artifacts/tmp_validation/run_pairwise_replay_matrix.py`
- `artifacts/tmp_validation/rebuild_pairwise_summary_from_snapshots.py`
- `artifacts/tmp_validation/score_pairwise_return_proxy.py`

Recommended practice:

- continue appending fresh post-fix snapshot pairs into a dedicated output directory,
- rebuild `summary.json` from completed snapshot pairs,
- score with the return-proxy script,
- treat recovered results as trustworthy only when the directory manifest matches the intended configs.