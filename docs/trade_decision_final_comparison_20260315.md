# Trade-Decision Final Comparison

Historical reference only.

This file records a finalized comparison from 2026-03-16. It is useful for understanding why a branch was selected at that time, but it is not the current operating source of truth.

Use these current sources instead:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `docs/trade_decision_post_fix_trust_basis_20260319.md`
- `docs/operations_runbook.md`

## Reference Runs

- `20260316T011019Z`
- `20260316T013439Z`
- `20260316T030147Z`

## Compared Candidates

- Base candidate: `none`
- Raw ablation: `reference_feature_ablation`
- Best tuned ablation threshold: `reference_feature_ablation_threshold_0p555`
- Diagnostic calibration guard on tuned ablation: `reference_feature_ablation_threshold_0p555_selection_calibration_guard`
- Diagnostic ranking-quality branch on tuned ablation: `reference_feature_ablation_threshold_0p555_neutral_abs_ret_pred_floor_0p00212`
- Neutral `p_up` cap branch on tuned ablation: `reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499`

## Summary

| Variant | Trade count | Net return total | Companion | Rolling delta | Recent rows | Recent AUC | Recent ECE | ECE drift | Status |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base `none` | 13 | -0.0151 | fail | +0.0061 | 3 | 0.0000 | 0.1423 | 0.0067 | too sparse, not promotable |
| Raw ablation | 101 | -0.0923 | fail | -0.0711 | 61 | 0.4513 | 0.0663 | 0.0441 | overtrades, not promotable |
| Ablation `0.555` | 42 | +0.0830 | pass | +0.1042 | 16 | 0.5667 | 0.0971 | 0.0770 | economics good, calibration fails |
| Ablation `0.555` + neutral guard `0.46` | 34 | +0.0637 | fail at `p=0.07` | +0.0849 | 11 | 0.4000 | 0.0258 | -0.0251 | calibration improved, but ranking weakened |
| Ablation `0.555` + neutral `abs_ret_pred >= 0.00212` | 28 | +0.0396 | fail at `p=0.118` | +0.0607 | 11 | 0.6000 | 0.0171 | -0.0114 | ranking and calibration improve, support still fails |
| Ablation `0.555` + neutral `p_up < 0.499` | 25 | +0.0825 | pass at `p=0.039` | +0.1037 | 15 | 0.6296 | 0.0703 | -0.0072 | selected eligible shadow and deployed on `20260316T030147Z` |

## Historical Decision

- Keep the current promoted deployment from `20260316T030147Z` active.
- Keep raw ablation out of consideration for promotion.
- Keep ablation `0.555` in the official shadow pool because it is the first ablation-derived branch with clearly positive economics and statistically positive companion evidence.
- Keep the neutral `min_p_up = 0.46` guard as a useful calibration diagnostic, but not the lead follow-up branch.
- `reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499` remains the strongest workflow-integrated follow-up branch. It first became `selected_variant` and `best_eligible_variant` in run `20260316T013439Z`, then completed promotion and deployment in run `20260316T030147Z`.
- The successful deployment required four workflow corrections: selected-shadow companion routing, selected-shadow overlap-triggered diagnostics routing, deploy-after-final-gate ordering, and evaluating the trade-decision model-shift guard against the effective promoted model with normalized incumbent reference provenance.

## Interpretation

- The repo now has evidence that threshold tuning plus a neutral `p_up` cap can recover enough support to clear the current selection bar without weakening gates.
- The former workflow blockers are now resolved, and the branch has been deployed without weakening promotion gates.
- The next best-practice step is live monitoring and rollback readiness, not additional gate relaxation.