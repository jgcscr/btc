# BTCUSDT Forecasting Pipeline (Binance-Only)

This repository contains a Binance-only BTCUSDT forecasting pipeline with:

- Binance spot kline ingestion via `data.ingestors.binance_us_spot` (primary refresh path)
- Binance spot kline ingestion via `data.ingestors.binance_spot_klines` (raw/day-to-GCS utility)
- Technical feature computation via `data.processed.compute_technical_features`
- Dataset builders, model trainers, and reliability tooling under `src/scripts/`
- Prediction and monitoring artifacts under `artifacts/`


## 1. Environment Setup

```bash
cd /workspaces/btc
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tests.txt
```

## 2. Quick Prediction Run

Primary entrypoint:

```bash
python -m src.scripts.run_refresh_and_predict --targets 1,4,8,12
```

Config-driven run:

```bash
python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml
```

Artifact-driven run (recommended after reliability workflow):

```bash
python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_simplified.yaml \
  --targets 1,4,8,12 \
  --thresholds-json artifacts/reliability/<run-id>/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/<run-id>/summary/platt_calibration.json \
  --write-artifacts
```

Dry run:

```bash
python -m src.scripts.run_refresh_and_predict --dry-run --targets 1,4,8,12
```

Main outputs:

- `artifacts/predictions/latest.json`
- `artifacts/predictions/history.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`

`artifacts/monitoring/latest.json` also includes local feature alignment diagnostics under `request.local_feature_overrides.feature_alignment`.

## 3. Ingestion Scripts Available

Current active data source is Binance spot klines.

Active command used by refresh/prediction flow:

```bash
python -m data.ingestors.binance_us_spot
```

```bash
python -m data.ingestors.binance_spot_klines
```

## 4. Feature Processing Scripts Available

The following processors exist in `data/processed/`:

- `data.processed.compute_technical_features`

Example run:

```bash
python -m data.processed.compute_technical_features
```

## 5. Dataset Build and Training Entrypoints

Dataset builders currently present:

- `src.scripts.build_training_dataset`
- `src.scripts.build_training_dataset_15m`
- `src.scripts.build_training_dataset_direction`
- `src.scripts.build_training_dataset_direction_15m`
- `src.scripts.build_training_dataset_multi_horizon`
- `src.scripts.build_sequence_direction_dataset`

Model training/search scripts currently present include:

- Baselines and suites:
  - `src.scripts.train_baseline_model`
  - `src.scripts.train_model_suite`
  - `src.scripts.train_direction_model`
  - `src.scripts.train_lgbm_dir`
  - `src.scripts.train_target_ranges`
- Sequence models:
  - `src.scripts.train_lstm_dir1h`
  - `src.scripts.train_bilstm_dir1h`
  - `src.scripts.train_gru_dir1h`
  - `src.scripts.train_cnn_lstm_dir1h`
  - `src.scripts.train_cnn_bilstm_dir1h`
  - `src.scripts.train_garch_lstm_dir1h`
  - `src.scripts.train_transformer_dir1h`
  - `src.scripts.train_transformer_dir1h_large`
- Hyperparameter search:
  - `src.scripts.search_xgb_optuna`
  - `src.scripts.search_lstm_optuna`
  - `src.scripts.search_transformer_optuna`
  - `src.scripts.search_ensemble_thresholds`
  - `src.scripts.search_ensemble_weights`

## 6. Reliability and Evaluation

Reliability workflow:

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml
```

Alternative config:

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml
```

Keep workflow running when promotion gate blocks promotion:

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Standalone trigger check:

```bash
python -m src.scripts.check_reliability_triggers \
  --config-path configs/reliability_workflow.default.yaml \
  --history-path artifacts/predictions/history.json \
  --horizons 1h,4h,8h,12h
```

Backtest/evaluation scripts present:

- `src.scripts.evaluate_ensemble_signals`
- `src.scripts.backtest_signals`
- `src.scripts.backtest_signals_4h`
- `src.scripts.backtest_signals_1h4h_confirm`
- `src.scripts.eval_equity_curves`

Reliability quality/gating helpers present:

- `src.scripts.build_labeled_backtest_from_history`
- `src.scripts.evaluate_model_quality`
- `src.scripts.evaluate_shadow_promotion`
- `src.scripts.evaluate_rolling_ab`
- `src.scripts.tune_joint_signal_thresholds`
- `src.scripts.evaluate_calibration_robustness`

## 7. Trade-Ready Reporting

Build trade-ready report artifact:

```bash
REPORT_BUCKET=gs://<your-bucket> \
WORKSPACE=/workspace \
python -m src.scripts.build_trade_ready_report
```

## 8. Fresh Prediction Checklist

Use this sequence for a fresh, reliability-calibrated prediction snapshot.

1. Run reliability workflow (runtime profile)

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

2. Resolve latest reliability run id

```bash
RUN_ID=$(ls -1 artifacts/reliability | sort | tail -n 1)
echo "$RUN_ID"
```

3. Run refresh + predict with latest calibrated thresholds and Platt calibration

```bash
python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_simplified.yaml \
  --targets 1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

4. Read outputs

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`

Quick validation checks:

- `generated_at` is recent.
- Horizon `timestamp` values match latest candle time.
- `request.local_feature_overrides.feature_alignment` in `artifacts/monitoring/latest.json` lists any unresolved imputed columns.


