## Trade-Decision Operator Handoff

Date: 2026-03-16
Final reference runs: `20260316T011019Z`, `20260316T013439Z`, `20260316T030147Z`

Current operating decision:

- Keep the current promoted shadow deployment from run `20260316T030147Z` active.
- Treat the neutral `p_up` cap branch as both the selected shadow winner and the current deployed profile.
- Do not relax promotion gates.

What changed materially:

- Raw reference-feature ablation is no longer the main candidate of interest.
- Threshold tuning at `0.555` produced the first ablation-derived branch with positive economics and significant companion evidence, but it still failed calibration.
- A neutral-only calibration guard fixed calibration error, but cut support too far and weakened ranking quality.
- A neutral `abs_ret_pred >= 0.00212` branch improved recent ranking quality and calibration together and became the workflow's `best_ineligible_variant`, but it still did not retain enough recent rows to pass the selection policy.
- A neutral `p_up < 0.499` cap on top of ablation `0.555` then recovered enough support to pass the official shadow selector in run `20260316T013439Z`.

How to read the final state:

- Best economic base branch: `reference_feature_ablation_threshold_0p555`
- Best calibration-fix diagnostic branch: `reference_feature_ablation_threshold_0p555_selection_calibration_guard`
- Best earlier diagnostic ineligible branch: `reference_feature_ablation_threshold_0p555_neutral_abs_ret_pred_floor_0p00212`
- Current selected eligible and deployed shadow branch: `reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499`
- Deployment completed on run `20260316T030147Z` after four workflow fixes: selected-shadow companion routing, selected-shadow overlap-triggered diagnostics routing, deploy-after-final-gate ordering, and model-shift normalization against the effective promoted model with current incumbent reference provenance.
- Final deployed promotion evidence on `20260316T030147Z`: companion gate `promote = true`, overlap-triggered slice `10` trades with `+0.0162` net return and `0.60` hit rate, rolling windows `5/5` candidate wins, and trade-decision model-shift guard passed with `max_reference_coef_delta = 0.0065`, `source_not_current_count = 0`, and stable reference source `artifacts/monitoring/labeled_backtest_1h_incumbent.csv`.

Recommended next work:

- Monitor the new deployed branch for reference-feature drift and overlap-triggered degradation under live cadence, rather than reopening threshold tuning immediately.
- Keep the older `selection_calibration_guard` deployment only as rollback context; the active deployment is now the ablation-derived neutral `p_up` cap branch.
- Do not weaken row-count, AUC, or significance gates.