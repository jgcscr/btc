# BTCUSDT Forecasting Pipeline (Binance-Only)

This repository contains a Binance-only BTCUSDT forecasting and reliability pipeline with:

- Binance US spot kline ingestion for live refresh and inference
- Hourly and intrabar feature generation for 15m, 1h, 4h, 8h, and 12h forecasting
- Dataset builders, model trainers, and evaluation tooling under `src/scripts/`
- Reliability workflows with calibration, walk-forward validation, overlap trust checks, and promotion gating
- Default-vs-midband matched-cycle paper tracking and longitudinal watchlist artifacts
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

In this workspace, commands are typically run with:

```bash
/workspaces/btc/.venv/bin/python -m <module>
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
- `artifacts/monitoring/meta_baseline.json`
- `artifacts/monitoring/meta_baseline.parquet`
- `artifacts/monitoring/trend_ignition_state.json`
- `artifacts/monitoring/direction_fallback_state.json`
- `artifacts/monitoring/data_quality_latest.json`

`artifacts/predictions/latest.json` includes per-horizon:

- `entry_price`
- `direction_next`
- `trade_action`
- `stop_loss`
- `take_profit`
- `expected_value`
- `regime_state`

`artifacts/monitoring/latest.json` also includes local feature alignment diagnostics under `request.local_feature_overrides.feature_alignment`.

Notes:

- If `artifacts/models/target_ranges/metadata.json` exists, `run_refresh_and_predict` auto-enables target-range inference for supported horizons.
- The default config currently requests targets `0.25,1,4,8,12`, so live output includes a 15m horizon in addition to hourly horizons.

## 3. Config Files

Prediction configs:

- `configs/run_refresh_and_predict.default.yaml`
- `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `configs/run_refresh_and_predict.shadow_strict_abstention.yaml`

Reliability configs:

- `configs/reliability_workflow.default.yaml`
- `configs/reliability_workflow.runtime.yaml`
- `configs/reliability_workflow.midband_paper.yaml`

Other configs currently present:

- `configs/conservative_trading.yaml`
- `configs/monitoring_sla_overrides.yaml`
- `configs/walkforward/`

## 4. Ingestion Scripts Available

Current active data source is Binance spot klines.

Active command used by refresh/prediction flow:

```bash
python -m data.ingestors.binance_us_spot
```

```bash
python -m data.ingestors.binance_spot_klines
```

## 5. Feature Processing Scripts Available

The following processors exist in `data/processed/`:

- `data.processed.compute_technical_features`

Example run:

```bash
python -m data.processed.compute_technical_features
```

## 6. Dataset Build and Training Entrypoints

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
  - `src.scripts.train_meta_ensemble`
  - `src.scripts.train_platt_calibration`
  - `src.scripts.train_target_ranges`
  - `src.scripts.train_trade_decision_model`
  - `src.scripts.train_trend_ignition_xgb`
- Sequence models:
  - `src.scripts.train_lstm_dir1h`
  - `src.scripts.train_lstm_direction_model`
  - `src.scripts.train_bilstm_dir1h`
  - `src.scripts.train_gru_dir1h`
  - `src.scripts.train_cnn_lstm_dir1h`
  - `src.scripts.train_cnn_bilstm_dir1h`
  - `src.scripts.train_garch_lstm_dir1h`
  - `src.scripts.train_transformer_dir1h`
  - `src.scripts.train_transformer_dir1h_large`
- Additional point trainers:
  - `src.scripts.train_xgb_dir4h_v1`
  - `src.scripts.train_xgb_ret4h_v1`
- Hyperparameter search:
  - `src.scripts.search_xgb_optuna`
  - `src.scripts.search_lstm_optuna`
  - `src.scripts.search_transformer_optuna`
  - `src.scripts.search_ensemble_thresholds`
  - `src.scripts.search_ensemble_weights`

## 7. Reliability and Evaluation

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

Midband paper-evaluation profile:

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.midband_paper.yaml
```

Keep workflow running when promotion gate blocks promotion:

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

With `--continue-on-promotion-fail`, the workflow can keep running and still write downstream summary artifacts even when the promotion gate blocks promotion.

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
- `src.scripts.run_walkforward_validation`
- `src.scripts.compare_walkforward_models`
- `src.scripts.run_cv_stress_sweep`
- `src.scripts.run_label_ablation`
- `src.scripts.evaluate_champion_challenger`
- `src.scripts.evaluate_rolling_ab`
- `src.scripts.evaluate_shadow_promotion`
- `src.scripts.evaluate_regime_weakness`

Reliability quality/gating helpers present:

- `src.scripts.build_labeled_backtest_from_history`
- `src.scripts.evaluate_model_quality`
- `src.scripts.evaluate_shadow_promotion`
- `src.scripts.evaluate_rolling_ab`
- `src.scripts.tune_joint_signal_thresholds`
- `src.scripts.evaluate_calibration_robustness`
- `src.scripts.evaluate_feature_reliability`
- `src.scripts.audit_point_in_time_integrity`
- `src.scripts.enrich_backtest_with_decision_features`
- `src.scripts.train_trade_decision_model`
- `src.scripts.apply_trade_decision_policy_to_backtest`
- `src.scripts.analyze_ensemble_hygiene`
- `src.scripts.slice_direction_dataset_by_timestamps`
- `src.scripts.analyze_overlap_trust_stability`
- `src.scripts.analyze_overlap_triggered_trade_diagnostics`

Reliability run outputs are written under:

- `artifacts/reliability/<run-id>/summary/`

Common artifacts include:

- `workflow_manifest.json`
- `calibrated_thresholds.json`
- `platt_calibration.json`
- `walkforward_labeled_reconciliation.json`
- `edge_trustworthiness.json`
- `overlap_trust_stability.json`

## 8. Matched-Cycle Default vs Midband Tracking

The repository includes matched-cycle tooling to compare the default runtime against the midband paper profile on aligned data lineage.

Run one matched cycle:

```bash
python -m src.scripts.run_default_midband_matched_cycle \
  --default-config configs/reliability_workflow.default.yaml \
  --midband-config configs/reliability_workflow.midband_paper.yaml \
  --run-root artifacts/reliability \
  --continue-on-promotion-fail
```

Replay a prior trusted default snapshot deterministically:

```bash
python -m src.scripts.run_default_midband_matched_cycle \
  --default-config configs/reliability_workflow.default.yaml \
  --midband-config configs/reliability_workflow.midband_paper.yaml \
  --run-root artifacts/reliability \
  --default-pinned-snapshot artifacts/reliability/<trusted-run>/summary/btc_features_1h_direction_splits.snapshot.npz \
  --default-pinned-snapshot-meta artifacts/reliability/<trusted-run>/summary/btc_features_1h_direction_meta.snapshot.json \
  --default-pinned-labeled-csv artifacts/reliability/<trusted-run>/summary/labeled_backtest.snapshot.csv \
  --continue-on-promotion-fail
```

Replay directly from a prior self-contained run id:

```bash
python -m src.scripts.run_default_midband_matched_cycle \
  --default-pinned-run-id <trusted-run-id> \
  --run-root artifacts/reliability \
  --continue-on-promotion-fail
```

Replay directly from a prior matched-cycle id:

```bash
python -m src.scripts.run_default_midband_matched_cycle \
  --default-pinned-cycle-id <trusted-cycle-id> \
  --run-root artifacts/reliability \
  --continue-on-promotion-fail
```

Runs created by the newer replay-support workflow also preserve a run-local labeled backtest snapshot at `summary/labeled_backtest.snapshot.csv`, so future replays do not depend on the mutable monitoring CSV.

Compare a trusted replay run against a drifting latest run:

```bash
python -m src.scripts.analyze_overlap_trust_flip \
  --trusted-run-id <trusted-default-run-id> \
  --drift-run-id <drifting-default-run-id> \
  --run-root artifacts/reliability \
  --output artifacts/analysis/overlap_trust_flip_latest.json
```

For exact fold-level bar attribution, rerun overlap compare with the standard workflow settings and inspect the emitted `*_rows.csv` files for the selected model. `src.scripts.compare_walkforward_models` now writes per-bar fold exports alongside each model summary JSON.

Supporting comparison/watchlist scripts currently present:

- `src.scripts.build_default_vs_midband_paper_live_longitudinal`
- `src.scripts.build_default_vs_midband_paper_live_watchlist`
- `src.scripts.compare_default_vs_midband_paper_live_snapshots`
- `src.scripts.compare_default_vs_midband_profile_metrics`
- `src.scripts.analyze_paired_trigger_overlap`
- `src.scripts.evaluate_midband_shadow_retrospective`
- `src.scripts.update_midband_shadow_longitudinal`

Canonical matched-cycle/watchlist artifacts:

- `artifacts/reliability/default_midband_matched_cycle_latest.json`
- `artifacts/reliability/default_vs_midband_paper_live_watchlist.json`

## 9. Trade-Ready Reporting

`src.scripts.build_trade_ready_report` is a specialized CI/GCS reporting helper. It expects captured workflow outputs such as `run_dataset_refresh.json` and `run_signal.json` in a workspace directory, plus a `REPORT_BUCKET` destination.

Example:

```bash
REPORT_BUCKET=gs://<your-bucket> \
WORKSPACE=/workspace \
python -m src.scripts.build_trade_ready_report
```

This is not the main local prediction entrypoint; for local fresh predictions use `src.scripts.run_refresh_and_predict`.

## 10. Fresh Prediction Checklist

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
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

4. Read outputs

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/reliability/${RUN_ID}/summary/edge_trustworthiness.json`
- `artifacts/reliability/${RUN_ID}/summary/walkforward_labeled_reconciliation.json`

Quick validation checks:

- `generated_at` is recent.
- Horizon `timestamp` values match latest candle time.
- `entry_price`, `stop_loss`, and `take_profit` are present for each horizon in `artifacts/predictions/latest.json`.
- `request.local_feature_overrides.feature_alignment` in `artifacts/monitoring/latest.json` lists any unresolved imputed columns.
- Reliability trust artifacts show whether the run is overlap-trustworthy before treating new thresholds as deployable.


