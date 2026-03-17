# BTCUSDT Forecasting Pipeline (Binance-Only)

This repository contains a Binance-only BTCUSDT forecasting and reliability pipeline with:

- Binance US spot kline ingestion for live refresh and inference
- Hourly and intrabar feature generation for 15m, 1h, 4h, 8h, and 12h forecasting
- Runtime trade-decision gating with confluence, feature-coverage, and target-range overlays
- Automatic best-version model resolution for versioned model families during live inference
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
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict --targets 0.25,1,4,8,12
```

Config-driven run with the current promoted default profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml
```

Run explicitly against the currently deployed shared bundle:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --thresholds-json artifacts/models/calibrated_thresholds_merged.json \
  --platt-calibration artifacts/models/platt_calibration.json \
  --trade-decision-model artifacts/models/trade_decision_model.json
```

Artifact-driven run with the latest trustworthy reliability bundle:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/<run-id>/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/<run-id>/summary/platt_calibration.json \
  --write-artifacts
```

Shadow/cadence-compatible run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_simplified.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/<run-id>/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/<run-id>/summary/platt_calibration.json \
  --write-artifacts
```

Dry run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 0.25,1,4,8,12
```

Main outputs:

- `artifacts/predictions/latest.json`
- `artifacts/predictions/history.json`
- `artifacts/analysis/prediction_coherence_latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `artifacts/monitoring/meta_baseline.json`
- `artifacts/monitoring/meta_baseline.parquet`
- `artifacts/monitoring/trend_ignition_state.json`
- `artifacts/monitoring/direction_fallback_state.json`
- `artifacts/monitoring/data_quality_latest.json`

`artifacts/predictions/latest.json` includes per-horizon:

- `entry_price`
- `direction_next`
- `direction_next_display`
- `trade_action`
- `stop_loss`
- `take_profit`
- `expected_value`
- `regime_state`
- `probability_calibration`
- `direction_output`
- `trade_decision`
- `confluence`
- `execution_plan`
- `execution_prior_provenance`
- `forecast_coherence`
- `projected_high` / `projected_low` when target-range models are enabled

The top-level payload also includes `execution_prior_summary`, which aggregates whether execution priors came from global history, regime/volatility buckets, and the stop/target source mix used by the current snapshot.

`execution_plan.stop_management` records whether stop guardrails widened, capped, or swapped the selected stop candidate to keep the stop width inside the configured ATR band.

`probability_calibration` records the requested horizon/regime calibration key, the key actually applied, and whether runtime fell back to the base horizon calibration.

`direction_output` is a user-facing direction payload. It can apply a separate direction-only calibration/remap, expose the probability used for display, and emit `neutral` inside the configured band without changing `trade_action` or the internal `direction_next` used by policy gates.

`artifacts/monitoring/latest.json` also includes local feature alignment diagnostics under `request.local_feature_overrides.feature_alignment`, source freshness under `request.local_feature_overrides.source_freshness`, and feature-coverage enforcement results under `request.local_feature_overrides.feature_coverage`.

Notes:

- If `artifacts/models/target_ranges/metadata.json` exists, `run_refresh_and_predict` auto-enables target-range inference for supported horizons.
- The default config currently requests targets `0.25,1,4,8,12`, so live output includes a 15m horizon in addition to hourly horizons.
- `configs/run_refresh_and_predict.default.yaml` is the promoted default runtime profile.
- `configs/run_refresh_and_predict.shadow_simplified.yaml` remains the cadence/shadow wrapper profile and is still used by `scripts/run_cadence.sh` because it writes artifacts directly.
- Live inference now resolves the best available versioned model artifact within a model family before loading it.
- The current deployed bundle is recorded in `artifacts/monitoring/reliability_promotion_deploy_manifest.json`. At the time of this README update it points to run `20260316T030147Z` and variant `reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499`.

Execution plan quick guide:

- `bias_only_ready`: the execution layer found an acceptable setup, but the upstream model still abstained so the final `trade_action` remains `hold`.
- `waiting_pullback`: bias and confluence are acceptable, but price is outside the preferred entry zone and the plan is waiting for a retest.
- `rejected`: a hard guard blocked the setup; inspect `execution_plan.reason` before overriding it.
- Common reasons are `bias_direction_conflict`, `stop_too_tight_near_invalidation`, `stop_too_wide`, `risk_reward_below_floor`, `low_execution_confluence`, and `upstream_model_hold`.

Cadence entrypoint:

```bash
batch ./scripts/run_cadence.sh daily
batch ./scripts/run_cadence.sh weekly
batch ./scripts/run_cadence.sh monthly
```

That wrapper resolves the latest trustworthy reliability run for daily predictions, runs the runtime reliability profile for weekly refreshes, and runs the full default reliability profile for monthly retraining before refreshing predictions.

## 3. Config Files

Prediction configs:

- `configs/run_refresh_and_predict.default.yaml` - promoted runtime policy with trade-decision, confluence, feature-coverage, adaptive-threshold, regime-model, target-range, and execution-policy blocks.
- `configs/run_refresh_and_predict.default.yaml` now also carries a `forecast_coherence_policy` block that can force `hold` and exclude incoherent higher-horizon forecasts from confluence/bias voting.
- `configs/run_refresh_and_predict.shadow_simplified.yaml` - shadow/cadence profile that mirrors the promoted policy while writing artifacts by default.
- `configs/run_refresh_and_predict.shadow_strict_abstention.yaml`

Reliability configs:

- `configs/reliability_workflow.default.yaml` - full monthly/default workflow with labeled-dataset rebuilds, overlap drift guard, trusted baseline pack generation, raw direction snapshots, and deployable-threshold fallback.
- `configs/reliability_workflow.runtime.yaml` - lighter runtime workflow pinned to the current trusted snapshot lineage and shadow paper-live config.
- `configs/reliability_workflow.midband_paper.yaml`

Other configs currently present:

- `configs/conservative_trading.yaml`
- `configs/monitoring_sla_overrides.yaml`
- `configs/walkforward/`

## 4. Ingestion Scripts Available

Current active data source is Binance spot klines.

Active command used by refresh/prediction flow:

```bash
/workspaces/btc/.venv/bin/python -m data.ingestors.binance_us_spot
```

```bash
/workspaces/btc/.venv/bin/python -m data.ingestors.binance_spot_klines
```

Chunked historical backfill helper added for local history refreshes:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.backfill_binance_us_spot \
  --interval 1h --days 365
```

## 5. Feature Processing Scripts Available

The following processors exist in `data/processed/`:

- `data.processed.compute_technical_features`

Example run:

```bash
/workspaces/btc/.venv/bin/python -m data.processed.compute_technical_features
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

Direction-model audit helper:

- `src.scripts.audit_direction_models` writes `artifacts/analysis/direction_model_audit_latest.json` and can be used to derive component-weight recommendations for `train_meta_ensemble` / runtime regime weighting.

Example:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.audit_direction_models
```

Marginal 1h direction audit helper:

- `src.scripts.analyze_direction_marginal_calibration` writes `artifacts/analysis/direction_marginal_1h_latest.json` plus a CSV export of the marginal rows so you can inspect the 0.50-0.60 `p_up` slice, its realized accuracy, regime mix, and the largest feature shifts versus the non-marginal baseline.
- Scratch validation runs should stay under `artifacts/tmp_validation/`; the canonical retained outputs are the copies under `artifacts/analysis/` and the workflow run summary directory.

Example:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.analyze_direction_marginal_calibration \
  --include-reliability-snapshots
```

## 7. Reliability and Evaluation

Reliability workflow:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml
```

Alternative config:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml
```

Midband paper-evaluation profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.midband_paper.yaml
```

Keep workflow running when promotion gate blocks promotion:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

With `--continue-on-promotion-fail`, the workflow can keep running and still write downstream summary artifacts even when the promotion gate blocks promotion.

Promotion and deploy behavior:

- A successful promotion can now deploy the promoted thresholds, Platt calibration, trade-decision model, incumbent labeled backtest, and monitoring summaries into the shared `artifacts/models` and `artifacts/monitoring` targets configured under `quality.promotion_deploy`.
- If promotion is blocked, the workflow still writes summary artifacts and does not overwrite the active shared baseline.
- `summary/champion_gate_alignment_check.json` is the fail-fast guard that verifies the enforced champion gate source matches the expected source. For `official_shadow_variant: none`, the check only enforces labeled source and path consistency. For active official shadows, it also enforces metric equality against the policy-aligned companion gate.
- `summary/trade_decision_model_shift_guard.json` is a hard promotion gate for trade-decision model drift. Treat failures as deployment blockers, not advisory diagnostics.
- The current trade-decision default is conservative: reference features are source-aware, can be disabled on source mismatch, and may be clipped before training; the `reference_feature_ablation` shadow variant is evaluated but should only be deployed if it passes the same promotion checks as any other candidate.

Regression check:

```bash
/workspaces/btc/.venv/bin/python -m unittest discover -s tests
```

Current workflow behavior to be aware of:

- The default workflow now builds the canonical labeled dataset, overlap feature-drift guard, raw direction-feature snapshots, and trusted baseline pack manifests.
- Joint threshold tuning can fall back to `artifacts/monitoring/calibrated_thresholds_last_deployable.json` when the latest candidate is rejected.
- Runtime paper-live resolution is config-driven via `search.paper_live_config` and no longer assumes the same prediction config for every workflow profile.

Cadence automation:

- Shell entrypoint: `scripts/run_cadence.sh`
- GitHub Actions schedule: `.github/workflows/cadence.yml`
- Operations runbook: `docs/operations_runbook.md`
- Trade-decision operator handoff: `docs/trade_decision_operator_handoff_20260316.md`
- Trade-decision final comparison: `docs/trade_decision_final_comparison_20260315.md`

The GitHub Actions workflow supports manual dispatch plus three UTC cadences:

- daily at `01:15`
- weekly on Monday at `02:30`
- monthly on day 1 at `03:45`

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
- `platt_calibration_coverage.json`
- `platt_calibration_policy_aligned_labeled.csv`
- `direction_output_labeled_1h.csv`
- `direction_output_isotonic_1h.json`
- `paper_live_direction_output_shadow_config.yaml`
- `paper_live_upstream_direction_candidate.yaml`
- `direction_marginal_1h.json`
- `promotion_gate.json`
- `champion_gate_alignment_check.json`
- `trade_decision_model_shift_guard.json`
- `walkforward_labeled_reconciliation.json`
- `edge_trustworthiness.json`
- `overlap_trust_stability.json`

`platt_calibration.json` can contain both base horizon keys such as `1h` and regime-aware keys such as `1h@neutral` or `1h@trend_ignition`.

`platt_calibration_coverage.json` records whether the labeled input actually supported horizon-regime calibration, including when the workflow had to default missing horizon labels to `1h`.

`platt_calibration_policy_aligned_labeled.csv` is the dedicated history-plus-OHLCV calibration source used when the workflow regenerates multi-horizon regime-aware calibration entries such as `4h@neutral` or `8h@chop`.

`direction_output_labeled_1h.csv` and `direction_output_isotonic_1h.json` are the separate direction-only calibration inputs emitted for the shadow display policy. `paper_live_direction_output_shadow_config.yaml` points runtime direction display at that artifact and can optionally add a marginal 1h component rerank inside the audited `0.50-0.60` band without changing internal `direction_next` or `trade_action`.

The runtime and midband paper workflow profiles now opt in to applying `paper_live_direction_output_shadow_config.yaml` during stage-7 paper-live refresh. The default reliability profile still emits the shadow bundle without applying it automatically.

`paper_live_upstream_direction_candidate.yaml` is a candidate-only internal 1h weight update derived from the same marginal audit. The workflow emits it for replay validation, but the checked-in profiles keep `apply_to_paper_live: false` so it does not change paper-live execution by default.

`direction_marginal_1h.json` is the workflow-emitted marginal-slice audit used to derive the optional 1h rerank weights for the shadow profile.

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

Trustworthy runs now also write `summary/trusted_baseline_pack.json`. The baseline pack is a run-local manifest that bundles the snapshot NPZ, labeled backtest snapshot, overlap dataset snapshot, compare summaries, and trust artifacts under that trusted run id so later replay and drift checks can resolve a single manifest instead of manual paths.

Newer runs also snapshot the raw pre-normalization direction feature frame under `summary/direction_features_raw.snapshot.csv` plus a labeled-overlap slice under `summary/direction_features_raw.labeled_overlap.csv`. Those files are intended to preserve the exact raw feature values needed for future overlap trust-flip analysis, instead of relying on reconstructed values from the mutable canonical source.

Compare a trusted replay run against a drifting latest run:

```bash
python -m src.scripts.analyze_overlap_trust_flip \
  --trusted-run-id <trusted-default-run-id> \
  --drift-run-id <drifting-default-run-id> \
  --run-root artifacts/reliability \
  --output artifacts/analysis/overlap_trust_flip_latest.json
```

For exact fold-level bar attribution, rerun overlap compare with the standard workflow settings and inspect the emitted `*_rows.csv` files for the selected model. `src.scripts.compare_walkforward_models` now writes per-bar fold exports alongside each model summary JSON.

Export raw overlap feature-row deltas for the bars that actually flipped trust:

```bash
python -m src.scripts.analyze_overlap_feature_drift \
  --trusted-run-id <trusted-default-run-id> \
  --drift-run-id <drifting-default-run-id> \
  --run-root artifacts/reliability \
  --detail-analysis artifacts/analysis/overlap_trust_flip_detailed_latest.json \
  --output artifacts/analysis/overlap_feature_drift_latest.json
```

The feature-drift artifact includes the worst-fold train-window boundaries plus the matched per-bar `p_up`, signal, and `ret_net` values for each changed row when the detailed overlap row exports are available. For datasets created after the scaler-stats update, the same artifact now also exports raw pre-normalization feature deltas and raw row snapshots for the exact changed fold rows.

Create or refresh a baseline pack manifest manually for a trusted run when needed:

```bash
python -m src.scripts.create_trusted_baseline_pack \
  --run-id <trusted-run-id> \
  --run-root artifacts/reliability
```

The default reliability workflow also enables an overlap feature-drift guard. Once a prior trusted baseline pack exists, the workflow auto-discovers the latest trusted pack, compares the current overlap tail against that baseline, and writes `summary/overlap_feature_drift_guard.json`. If the monitored intrabar or order-flow features move too far in trusted-train standard-deviation units, paper-live is forced onto conservative hold thresholds even before the overlap trust check would have silently degraded.

If you need to backfill raw snapshots for an older run that predates this workflow change, you can export them directly from that run's saved direction datasets:

```bash
python -m src.scripts.export_direction_feature_snapshot \
  --dataset artifacts/reliability/<run-id>/summary/btc_features_1h_direction_splits.snapshot.npz \
  --output artifacts/reliability/<run-id>/summary/direction_features_raw.snapshot.csv \
  --meta-output artifacts/reliability/<run-id>/summary/direction_features_raw.snapshot_meta.json
```

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
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

2. Resolve the latest trustworthy reliability run id

```bash
RUN_ID=$(
  /workspaces/btc/.venv/bin/python - <<'PY'
import json
from pathlib import Path

for run_dir in sorted((p for p in Path('artifacts/reliability').iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
    edge_path = run_dir / 'summary' / 'edge_trustworthiness.json'
    thresholds_path = run_dir / 'summary' / 'calibrated_thresholds.json'
    platt_path = run_dir / 'summary' / 'platt_calibration.json'
    if not edge_path.exists() or not thresholds_path.exists() or not platt_path.exists():
        continue
    payload = json.loads(edge_path.read_text())
    if payload.get('edge_trustworthy'):
        print(run_dir.name)
        break
PY
)
echo "$RUN_ID"
```

3. Run refresh + predict with latest calibrated thresholds and Platt calibration

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

4. Read outputs

- `artifacts/predictions/latest.json`
- `artifacts/analysis/prediction_coherence_latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/reliability/${RUN_ID}/summary/edge_trustworthiness.json`
- `artifacts/reliability/${RUN_ID}/summary/walkforward_labeled_reconciliation.json`

Quick validation checks:

- `generated_at` is recent.
- Horizon `timestamp` values match latest candle time.
- `entry_price`, `stop_loss`, and `take_profit` are present for each horizon in `artifacts/predictions/latest.json`.
- `request.local_feature_overrides.feature_alignment` in `artifacts/monitoring/latest.json` lists any unresolved imputed columns.
- `request.local_feature_overrides.feature_coverage.ok` remains true before treating the output as trade-ready.
- Reliability trust artifacts show whether the run is overlap-trustworthy before treating new thresholds as deployable.

For unattended cadence execution and the exact daily/weekly/monthly command breakdown, see `docs/operations_runbook.md`.

## 11. Troubleshooting / Usage Notes

- `scripts/run_cadence.sh` must be invoked with the batch prefix in this workspace: `batch ./scripts/run_cadence.sh daily`.
- Do not run `./scripts/run_cadence.sh daily` directly. Direct execution fails and is not the supported invocation for future agents or users.
- Historical replay is available for hourly horizons with cached artifacts:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

- Replay mode currently supports hourly horizons only and will overwrite `artifacts/predictions/latest.json` with the replayed snapshot, so run a fresh live prediction afterward if you want to restore the latest live artifact.


