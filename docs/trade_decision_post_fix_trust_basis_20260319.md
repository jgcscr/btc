# Trade Decision Post-Fix Trust Basis

Status: Historical trust-basis memo with still-useful rationale.

Do not use this as the sole operator reference. For current commands and runtime behavior, use `README.md`, `docs/operations_runbook.md`, `docs/live_operator_checklist_20260320.md`, and `docs/agent_system_handoff_20260320.md` first.

This document explains why `configs/run_refresh_and_predict.default.yaml` remains the trusted operating default.

For the current deployed bundle and current live-style runtime state, read:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`

## 1. Recommended Operating Default

The recommended operating profile remains:

- `configs/run_refresh_and_predict.default.yaml`

This recommendation is based on replay validation after fixing a runtime confluence-policy enforcement bug in `src/scripts/run_refresh_and_predict.py`.

## 2. Why The Default Is Trusted

The strongest negative evidence against the promoted default traced back to a runtime bug rather than to a stable model defect.

Root cause:

- horizon-specific confluence overrides such as `min_support_ratio_by_horizon` and `min_aligned_horizons_by_horizon` were present in config
- `_resolve_confluence_policy(...)` did not preserve those mappings correctly during normalization
- as a result, 4h and 8h unanimity overrides were not fully enforced at runtime in the affected path

The fix is reflected in:

- `src/scripts/run_refresh_and_predict.py`
- `tests/test_prediction_coherence_controls.py`

The key trust argument is that the earlier negative evidence was materially contaminated by incorrect runtime behavior. Once the confluence normalization path was corrected, the default profile recovered to a positive covered aggregate.

## 3. Post-Fix Validation Basis

### Targeted leak probe

Supporting artifacts:

- `artifacts/tmp_validation/default_profile_pairwise_targeted_4h_leak_fix_probe_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_targeted_4h_leak_fix_probe_20260319/return_proxy_summary.json`

Observed result:

- the completed portion of the previously problematic 4h leak window produced `0` added trades
- that is consistent with the corrected runtime blocking the leaked 4h chop additions

### Dataset-covered post-fix extension slices

Supporting artifacts:

- `artifacts/tmp_validation/default_profile_pairwise_extension6_954_post_fix_20260320/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension6_954_post_fix_20260320/return_proxy_summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension960_1152_post_fix_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension960_1152_post_fix_20260319/return_proxy_summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/summary.json`
- `artifacts/tmp_validation/default_profile_pairwise_extension1158_1410_post_fix_20260319/return_proxy_summary.json`

Scored checkpoints:

- `6`-`954`: `159` offsets, `123` aligned rows, `26` added trades, average signed return proxy `+0.006205112119250071`
- `960`-`1152`: `33` offsets, `33` aligned rows, `12` added trades, average signed return proxy `+0.001547706492904884`
- `1158`-`1410`: `43` offsets, `28` aligned rows, `7` added trades, average signed return proxy `+0.003697181029045688`

Combined covered post-fix aggregate:

- added trades: `45`
- added-trade average signed return proxy: `+0.004573014671526228`
- positive vs nonpositive added trades: `29` vs `16`

By horizon across the combined covered aggregate:

- `12h`: `12` added trades, average `+0.015200327994534746`
- `4h`: `22` added trades, average `+0.0018605283310700377`
- `8h`: `11` added trades, average `-0.001595445363570682`

Coverage boundary:

- `artifacts/tmp_validation/default_profile_pairwise_extension1416_1608_post_fix_20260319/summary.json` is a clean replay-only extension
- but it falls earlier than local return-label dataset coverage and should not be treated as scored trust evidence

## 4. Interpretation

The post-fix evidence supports the following conclusions:

- the default profile is positive overall after the confluence-policy fix
- `12h` remains the strongest contributor in the covered aggregate
- `4h` remains positive instead of acting as the drag that invalidated earlier pre-fix interpretation
- `8h` remains the weakest horizon, but it is materially closer to neutral than the pre-fix evidence suggested

This is why the default profile remains trusted as the operating baseline, while live deployment still uses conservative risk discipline and horizon-aware monitoring.

## 5. Operating Conclusion

The current default remains the best validated runtime profile in the repository and should remain the operating default for research, comparison, and cadence-linked runtime interpretation.

Trust statement:

- predictions are trusted for operation relative to the earlier pre-promotion baseline
- the profile is treated as post-fix trusted
- broader replay validation should continue, but the current evidence does not justify a broad rollback
- the evidence is strong enough to support live-style use under conservative controls
- horizon-aware monitoring should remain in place because `8h` still lags `4h` and `12h`

## 6. Validation Workflow Going Forward

Use these utilities for continued validation:

- `artifacts/tmp_validation/run_pairwise_replay_matrix.py`
- `artifacts/tmp_validation/rebuild_pairwise_summary_from_snapshots.py`
- `artifacts/tmp_validation/score_pairwise_return_proxy.py`

Recommended practice:

- append fresh post-fix snapshot pairs into a dedicated validation output directory
- rebuild `summary.json` from completed snapshot pairs
- score the completed set with the return-proxy script
- treat replay results as trustworthy only when the directory manifest matches the intended configs and snapshot lineage