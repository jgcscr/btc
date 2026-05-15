# Model Suite Training Guide

This document describes how to reproduce the multi-horizon model suite (regression + direction) across 15m/1h/4h/8h/12h.

## Scope

The suite trains:
- Regression models (XGBoost) for returns at each horizon.
- Direction classifiers (XGBoost, LightGBM).
- Sequence classifiers (LSTM, GRU, BiLSTM, CNN-LSTM, CNN-BiLSTM, GARCH-LSTM).
- Transformer classifiers.
- A sparse regime-focused logistic classifier for orthogonal macro/volatility/regime features.

Current runtime note:

- the active default/live/research runtime profiles do not treat every trained family equally
- current checked-in ensemble policy prioritizes `tree` and `attention`, keeps `volatility` as support, and lets `regime_logit` join as an optional orthogonal family when the matching artifact exists for that horizon
- recurrent families remain available, but their rerank and regime-specific weights are intentionally lower than the tree/attention stack after the May 15, 2026 family-value audit

To refresh the family-value audit from the canonical labeled monitoring slice:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.analyze_direction_family_value \
  --input artifacts/monitoring/labeled_backtest_1h.csv \
  --output artifacts/analysis/direction_family_value_latest.json
```

## One-command suite runner

Use the orchestration script to rebuild datasets and train the full stack in a single run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 0.25,1,4,8,12 \
  --rebuild-datasets \
  --train-regression \
  --train-direction \
  --train-lgbm \
  --train-sequence \
  --train-transformer \
  --train-regime-logit
```

Recommended for CI smoke tests:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 1 \
  --rebuild-datasets \
  --train-direction \
  --train-regression
```

## Fine-grained runs

Train only regression models:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 0.25,1,4,8,12 \
  --train-regression
```

Train only direction models (tree + LightGBM):

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 0.25,1,4,8,12 \
  --train-direction \
  --train-lgbm
```

Train only sequence and transformer models:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 0.25,1,4,8,12 \
  --train-sequence \
  --compact-sequence-set \
  --train-transformer
```

Train only the regime specialist:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 0.25,1,4,8,12 \
  --train-regime-logit
```

Train a selected subset of sequence models explicitly:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.train_model_suite \
  --targets 1,4,12 \
  --train-sequence \
  --sequence-models gru,garch_lstm
```

In this codespace, prefer the explicit interpreter path above unless you have already activated `.venv` in your shell.

## Current March 31, 2026 Notes

The current codespace includes additional feature-engineering and validation paths that are not captured by the older one-command summary alone.

Current important facts:

- 15m and runtime 1h intrabar aggregation now share `src/trading/intrabar_features.py`
- 1h regression dataset rebuilds can use slice-aware reliability filtering via `--feature-reliability-json` and `--feature-reliability-min-score`
- the tuned 1h regression score threshold used in this codespace is `0.80`
- the corrected March 31 comparison report is `artifacts/analysis/featurelift_20260331_rerun/comparison_report.md`
- the older pre-rerun March 31 comparison summary is superseded and should not be used as a model-quality summary

When reproducing the corrected rerun, refresh local external features first:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_macro_features --full-refresh
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_onchain_features --full-refresh
```

Then rebuild the dataset stack with the current thresholds:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_15m --output-dir artifacts/datasets
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_direction_15m \
  --output-dir artifacts/datasets \
  --feature-reliability-json artifacts/analysis/feature_reliability_15m_1h_slice_20260331.json \
  --feature-reliability-min-score 0.55
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset \
  --output-dir artifacts/datasets \
  --feature-reliability-json artifacts/analysis/feature_reliability_15m_1h_slice_20260331.json \
  --feature-reliability-min-score 0.80
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_direction \
  --output-dir artifacts/datasets \
  --feature-reliability-json artifacts/analysis/feature_reliability_15m_1h_slice_20260331.json \
  --feature-reliability-min-score 0.55
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_multi_horizon \
  --output-dir artifacts/datasets --horizons 1 4 8 12
```

## Datasets used per horizon

- 15m: `btc_features_15m_splits.npz`, `btc_features_15m_direction_splits.npz`
- 1h: `btc_features_1h_splits.npz`, `btc_features_1h_direction_splits.npz`
- 4h/8h/12h: `btc_features_multi_horizon_splits.npz`

## Model outputs

The suite writes model artifacts to:

- Regression: `artifacts/models/xgb_ret{horizon}_v1/`
- Direction (XGBoost): `artifacts/models/xgb_dir{horizon}_v1/`
- Direction (LightGBM): `artifacts/models/lgbm_dir{horizon}_v1/`
- Direction (Regime logistic): `artifacts/models/regime_logit_dir{horizon}_v1/`
- Sequence models: `artifacts/models/{lstm,gru,bilstm,cnn_lstm,cnn_bilstm,garch_lstm}_dir{horizon}_v1/`
- Transformer: `artifacts/models/transformer_dir{horizon}_v1/`

Where `{horizon}` is one of `15m`, `1h`, `4h`, `8h`, `12h`.

## Notes

- 15m/1h targets use flat label keys in the dataset; multi-horizon targets use `y_dir{horizon}h_*` and `y_ret{horizon}h_*` fields.
- The suite script shells out to the existing CLIs so model hyperparameters and logging remain consistent with prior runs.
- `--compact-sequence-set` is a shortcut for `--sequence-models gru,garch_lstm`.
- Multi-horizon feature builders now explicitly exclude forward-return leakage columns.
- Use `src.scripts.generate_featurelift_comparison_report` after retraining if the checked-in March 31 comparison artifacts need to be refreshed.
