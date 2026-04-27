# Intrabar Featurelift Handoff

Generated after the April 27 intrabar transition feature rollout and follow-up 4h/12h evaluation pass.

## Scope

- added six new shared intrabar transition features to the train/live pipeline
- rebuilt the dataset stack and retrained 4h and 12h models
- reran 4h and 12h feature-reliability and walkforward checks
- packaged a refreshed 4h shadow rollout candidate
- tested a trimmed 12h ablation and rejected it after validation

## Current Outcome

- `4h` remains the active feature-lift shadow lane
- `12h` improved on the untrimmed stack versus the prior leakage-safe rerun, but a follow-up trim of the two weakest new intrabar columns degraded walkforward and was not retained
- current main 12h artifacts were restored to the stronger untrimmed state

## Comparison Snapshot

Primary comparison artifact:

- `artifacts/analysis/intrabar_featurelift_apr2026/comparison_vs_rerun.json`
- `artifacts/analysis/intrabar_featurelift_apr2026/comparison_vs_rerun.md`

Key deltas versus the prior leakage-safe rerun:

- `4h` direction test F1: `-0.0104`
- `4h` regression test RMSE: `-0.00036` improvement
- `4h` walkforward AUC mean: `+0.0140`
- `12h` direction test F1: `+0.0067`
- `12h` regression test RMSE: `-0.00245` improvement
- `12h` walkforward AUC mean: `+0.0143`

Interpretation:

- the `4h` lane improved on robustness and regression fit more than on direct test F1, which is still sufficient to keep it as the active shadow lane
- the `12h` lane benefited from the added intrabar features overall, but not from post-hoc broad pruning

## 4h Operator Path

Packaging command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_featurelift_4h_shadow_rollout \
  --direction-dir artifacts/models/xgb_dir4h_v1 \
  --regression-dir artifacts/models/xgb_ret4h_v1 \
  --walkforward-path artifacts/analysis/intrabar_featurelift_apr2026/walkforward_4h.json \
  --output-json artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_4h_package.json \
  --output-markdown artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_4h_package.md
```

Validation command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml \
  --dry-run
```

Current package artifacts:

- `configs/run_refresh_and_predict.shadow_featurelift_4h_candidate.yaml`
- `artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_4h_package.json`
- `artifacts/analysis/intrabar_featurelift_apr2026/shadow_rollout_4h_package.md`

## 12h Shadow Path

Packaging command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.package_featurelift_12h_shadow_rollout
```

Validation command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_featurelift_12h_candidate.yaml \
  --dry-run
```

Current intent:

- keep `12h` experiments isolated in a dedicated shadow package
- prefer new horizon-specific interaction features over removing shared intrabar features that still help `4h`
- do not overwrite the validated main 12h stack with pruning experiments unless walkforward improves

## Rejected 12h Trim

Rejected experiment artifacts:

- `artifacts/datasets/btc_features_multi_horizon_12h_trimmed_intrabar_splits.npz`
- `artifacts/analysis/intrabar_featurelift_apr2026/trimmed_12h_feature_selection.json`
- `artifacts/analysis/intrabar_featurelift_apr2026/walkforward_12h_trimmed.json`

Rejected because:

- trimmed `12h` walkforward AUC fell from `0.5422` to `0.5112`
- trimmed `12h` cumulative net return fell from `98.73` to `90.70`

## Runtime Check

Post-change live-style validation was rerun successfully with the approved conservative live wrapper:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Use `artifacts/monitoring/latest.json` and `artifacts/predictions/latest.json` as the source-of-truth runtime read after any further shadow-package refresh.