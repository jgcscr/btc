# Trade Decision Post-Fix Trust Basis (2026-03-19)

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

### Broader post-fix extension slice

Artifacts:

- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/return_proxy_summary.json`

Current scored checkpoint from that directory:

- offset count: `38`
- aligned rows: `28`
- added trades: `7`
- added-trade average signed return proxy: `+0.003697181029045688`
- positive vs nonpositive added trades: `4` vs `3`

By horizon:

- `12h`: `3` added trades, average `+0.006069614241520564`
- `4h`: `4` added trades, average `+0.0019178561196895316`

Interpretation:

- post-fix evidence is positive overall,
- `12h` remains the strongest contributor,
- `4h` remains positive in aggregate instead of being the drag that invalidated the earlier pre-fix fresh slice.

## Operating Conclusion

As of 2026-03-19, the current default is the best validated runtime profile in the repository and should remain the active operating default.

Trust statement:

- predictions are trusted for operation relative to the prepromotion baseline,
- the profile is considered post-fix trusted,
- broader replay validation should continue, but no broad policy rollback is justified by the current evidence.

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