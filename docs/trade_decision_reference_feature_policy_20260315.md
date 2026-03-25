# Trade-Decision Reference Feature Policy

Historical policy record.

This file captures the 2026-03-15 policy decision context around reference features. It remains useful background, but current deployment state and current live operation should be read from the active manifest and current runbook artifacts.

Use these current sources instead:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `docs/operations_runbook.md`
- `docs/live_trading_rollout_20260320.md`

## Historical Decision

- Keep source-aware reference-feature controls enabled by default.
- Keep `disable_on_source_mismatch` as the training default for live workflow runs.
- Keep reference-feature clipping enabled at `max_abs_value: 0.25`.
- Keep the trade-decision model-shift guard enabled for promotion.
- Keep `reference_feature_ablation` in the official shadow candidate pool as a diagnostic candidate, but do not promote it unless it clears the same companion, calibration, and rolling checks as any other shadow variant.

## Validated Evidence

Primary run:

- `20260315T200516Z`

Observed evidence:

- Base candidate remained selected as `official_shadow_variant: none`.
- Base policy-aligned candidate: `13` trades, `net_return_total = -0.0151`, rolling delta versus incumbent `+0.0061`, companion gate `promote = false`.
- Reference-feature ablation candidate: `101` trades, `net_return_total = -0.0923`, rolling delta versus incumbent `-0.0711`, companion gate `promote = false`.
- Ablation improved trade coverage materially, but it failed the quality bar on economics and calibration. Recent active-trade calibration for the ablation variant showed `auc = 0.4513` and `ece_drift = 0.0441`, both outside current promotion thresholds.
- The trade-decision model-shift guard still correctly blocked promotion on reference coefficient drift, `source_not_current_count`, and reference-source instability.

## Interpretation

- Coverage recovery alone is not enough to justify deployment.
- The current best-practice posture is conservative: keep the stricter reference-feature controls, keep the guard, and treat the ablation variant as an evaluation branch until its economics and calibration improve.

## Recommended Next Tuning Target

- A strict ablation threshold sweep above `0.555` is too blunt. Follow-up evaluation showed thresholds `0.57+` collapsed to zero trades, while the finalized replay on run `20260315T210636Z` confirmed that `0.555` is the only meaningful stricter boundary: it passed companion significance and produced positive rolling economics, but still failed recent active-trade calibration (`ece_10 = 0.0971`, `ece_drift = 0.0770`).
- If ablation tuning continues, focus on improving recent active-trade calibration for the retained neutral slice rather than expanding coverage further or raising thresholds again.