# Model Suite Training Guide

This document describes how to reproduce the multi-horizon model suite (regression + direction) across 15m/1h/4h/8h/12h.

## Scope

The suite trains:
- Regression models (XGBoost) for returns at each horizon.
- Direction classifiers (XGBoost, LightGBM).
- Sequence classifiers (LSTM, GRU, BiLSTM, CNN-LSTM, CNN-BiLSTM, GARCH-LSTM).
- Transformer classifiers.

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
  --train-transformer
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
  --train-transformer
```

In this codespace, prefer the explicit interpreter path above unless you have already activated `.venv` in your shell.

## Datasets used per horizon

- 15m: `btc_features_15m_splits.npz`, `btc_features_15m_direction_splits.npz`
- 1h: `btc_features_1h_splits.npz`, `btc_features_1h_direction_splits.npz`
- 4h/8h/12h: `btc_features_multi_horizon_splits.npz`

## Model outputs

The suite writes model artifacts to:

- Regression: `artifacts/models/xgb_ret{horizon}_v1/`
- Direction (XGBoost): `artifacts/models/xgb_dir{horizon}_v1/`
- Direction (LightGBM): `artifacts/models/lgbm_dir{horizon}_v1/`
- Sequence models: `artifacts/models/{lstm,gru,bilstm,cnn_lstm,cnn_bilstm,garch_lstm}_dir{horizon}_v1/`
- Transformer: `artifacts/models/transformer_dir{horizon}_v1/`

Where `{horizon}` is one of `15m`, `1h`, `4h`, `8h`, `12h`.

## Notes

- 15m/1h targets use flat label keys in the dataset; multi-horizon targets use `y_dir{horizon}h_*` and `y_ret{horizon}h_*` fields.
- The suite script shells out to the existing CLIs so model hyperparameters and logging remain consistent with prior runs.
