# Meta Stack Shadow Rollout Operator Guide

This guide documents the shadow-only promotion workflow for evaluating the existing `meta_stack` candidate against the incumbent live ensemble path.

## Intent

- Keep the current live ensemble unchanged while evaluating `meta_stack` offline.
- Focus evaluation on `4h` and `12h`, where directional bias matters most and the live ensemble already prunes to a smaller active set.
- Block any move toward a more complex boosting architecture unless the current `meta_stack` clears the guarded promotion bar.
- Prioritize the `4h` trust and calibration defects before considering live promotion.

## Command

Run the default shadow rollout:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_meta_stack_shadow_rollout
```

Outputs:

- `artifacts/analysis/meta_stack_shadow_rollout_latest.json`
- `artifacts/analysis/meta_stack_shadow_rollout_latest.md`
- per-horizon comparison artifacts under `artifacts/analysis/`

## What The Script Checks

For each focus horizon (`4h`, `12h`), the workflow compares `xgb`, `meta_stack`, and `selector_simple` using the existing guarded walk-forward tooling.

Promotion checks require all of the following:

- `meta_stack` selected by the guarded policy
- better or equal expanding-window net return versus `xgb`
- better or equal expanding-window AUC versus `xgb`
- no material ECE degradation beyond the configured tolerance
- better or equal rolling-window net return versus `xgb`
- better or equal rolling-window AUC versus `xgb`
- minimum rolling trade count satisfied

## Operating Rule

- If `4h` remains `low_trust`, treat that as a blocker for live promotion even if `meta_stack` performs well in shadow.
- Keep the current live ensemble in place until both `4h` trust issues and guarded walk-forward promotion checks are cleared.
- Do not add a deeper stacking or boosting layer until the existing `meta_stack` proves that added complexity is earning its keep.

## Current Default Posture

- live ensemble: keep
- meta stack: shadow only
- more complex boosting: defer
- `4h` trust/calibration: fix first