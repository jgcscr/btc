# BTCUSDT Forecasting Pipeline

This repository contains the BTCUSDT forecasting, reliability, and execution-decision stack used in this workspace.

It supports:

- Binance US spot kline ingestion for live-style refreshes
- Free macro context ingestion for dollar strength, US10Y, and EUR/USD
- Feature generation for 15m, 1h, 4h, 8h, and 12h horizons
- Multi-horizon direction forecasting with calibrated runtime policies
- Trade-decision, confluence, coherence, uncertainty, and execution-plan gating
- Reliability workflows for calibration, promotion, overlap checks, and deployment handoff
- Shadow-profile comparison and cadence automation
- Shared 15m-to-1h intrabar feature generation for both training and runtime inference
- Slice-aware feature reliability filtering for 15m and 1h training paths
- Local macro and on-chain feature refresh paths that can run without private paid data sources

## Current Codespace State

This codespace includes the March 31, 2026 feature-lift and leakage-fix work.

That state includes:

- a shared intrabar feature builder at `src/trading/intrabar_features.py`
- expanded hourly price-state features in `src/trading/feature_engineering.py`
- processed on-chain features refreshed to `data/processed/onchain/hourly_features.parquet`
- a corrected feature-lift rerun report at `artifacts/analysis/featurelift_20260331_rerun/comparison_report.md`
- a validation workflow at `.github/workflows/validation-guards.yml`

Important workspace caveat:

- `data/` and most of `artifacts/` are gitignored local state
- another agent or a fresh clone should regenerate local processed features and runtime artifacts rather than assuming they already exist
- the corrected feature-lift report is authoritative; the older `featurelift_20260331` report is intentionally marked superseded

## 1. What This Repository Does

The runtime flow is organized around two related but distinct jobs:

1. Generate fresh prediction snapshots from the current market state.
2. Rebuild, validate, and promote reliability artifacts that the runtime uses for calibrated inference.

The core runtime entrypoint is:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict
```

The core reliability workflow entrypoint is:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow
```

The cadence wrapper is:

```bash
bash ./scripts/run_cadence.sh <daily|weekly|monthly|shadow>
```

For agent handoff and operating discipline, also read:

- `docs/operations_runbook.md`
- `docs/agent_system_handoff_20260320.md`
- `docs/live_operator_checklist_20260320.md`

## 2. Source Of Truth For Current State

Use runtime artifacts for the latest state. Do not rely on static notes in this README for the current market snapshot.

Primary runtime state sources:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `artifacts/analysis/featurelift_20260331_rerun/comparison_report.json` for the corrected March 31 feature-lift evaluation

Before acting on any fresh run, confirm:

- top-level `generated_at` is current
- `request.local_feature_overrides.feature_coverage.ok` is `true`
- source freshness in `request.local_feature_overrides.source_freshness` is acceptable
- the preferred horizon and recommended action in `prompt_ready_summary.operator_summary_compact` match the interpretation you are about to use

Before trusting any model-improvement narrative, confirm:

- you are reading `artifacts/analysis/featurelift_20260331_rerun/comparison_report.json` or `.md`
- you are not using the older pre-rerun March 31 feature-lift summary as a performance source

## 3. Runtime Profiles

The checked-in runtime profiles are not interchangeable.

- `configs/run_refresh_and_predict.default.yaml`: trusted research and comparison baseline; includes trade-decision, confluence, forecast-coherence, uncertainty, direction-output, and execution-policy controls.
- `configs/run_refresh_and_predict.live_conservative.yaml`: backward-compatible conservative live profile.
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`: approved conservative live profile for current Binance-only operations; keeps the same target set while enforcing tighter confidence and size discipline.
- `configs/run_refresh_and_predict.shadow_simplified.yaml`: cadence-friendly artifact-writing runtime profile used by the daily cadence refresh.
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`: active left-hand shadow comparison profile used by the `shadow` cadence path.
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`: shadow comparison candidate profile.
- `configs/run_refresh_and_predict.shadow_strict_abstention.yaml`: additional shadow-only profile for stricter abstention experiments.

Current live-source assumption:

- Approved live-style runtime refreshes are Binance-spot first.
- Macro context is available for research and local augmentation, but it is not currently treated as a blocking live coverage dependency for Binance-only runtime profiles.
- If macro is later promoted to a required live dependency, the runtime profile must merge that source on every refresh before tightening feature-coverage gates again.

Current live-conservative position caps are defined in `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`:

- `15m = 0.0`
- `1h = 0.15`
- `4h = 0.35`
- `8h = 0.20`
- `12h = 0.35`

The live conservative profile also keeps scoped `8h@trend_ignition` overrides for:

- trade-decision threshold
- confidence minimum
- abstention hold-band

## 4. Environment Setup

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

## 5. Prediction And Refresh Commands

Primary refresh command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict --targets 0.25,1,4,8,12
```

Trusted default runtime refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml
```

Approved conservative live refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

Refresh against the currently deployed shared bundle:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --thresholds-json artifacts/models/calibrated_thresholds_merged.json \
  --platt-calibration artifacts/models/platt_calibration.json \
  --trade-decision-model artifacts/models/trade_decision_model.json
```

Refresh against a specific trustworthy reliability run:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/<run-id>/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/<run-id>/summary/platt_calibration.json \
  --write-artifacts
```

Daily cadence-equivalent refresh:

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

Refresh the free macro context bundle used by local dataset builds:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_macro_features --full-refresh
```

This writes:

- `data/processed/macro/daily_features.parquet`
- `data/processed/macro/source_manifest.json`

Refresh the on-chain bundle used by local dataset builds and runtime augmentation:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_onchain_features --full-refresh
```

This writes:

- `data/processed/onchain/hourly_features.parquet`
- `data/processed/onchain/source_manifest.json`

On-chain refresh behavior in this codespace:

- first uses the configured on-chain API when available
- otherwise falls back to public Blockchain chart series for the supported BTC metrics
- derives hourly change, 24h z-score, and 6h trend columns from the raw metric frame

`run_refresh_and_predict` now attempts to refresh both macro and on-chain local feature bundles automatically on each local rebuild path. Failures in those refreshes are warning-level unless the selected runtime profile later makes them hard dependencies.

The checked-in macro implementation uses:

- `DTWEXBGS` from FRED as the operational free dollar-strength proxy instead of exact DXY
- `DGS10` from FRED for the 10-year Treasury yield
- `EUR/USD` from Frankfurter/ECB for FX context

Macro observations are timestamped at the next UTC midnight before merge to avoid same-day publication leakage.

Replay mode for hourly horizons:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

Replay mode overwrites the standard prediction artifact paths, so run a fresh live-style refresh afterward if you want to restore the latest live snapshot.

## 6A. Training And Feature-Lift Rebuilds

For the current March 31, 2026 feature-engineering stack, the exact rebuild sequence used in this codespace was:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_macro_features --full-refresh
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_onchain_features --full-refresh
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

Current reliability expectations for those dataset builders:

- 15m direction uses the slice-aware reliability payload at horizon `0.25`
- 1h regression uses the same payload with a tuned minimum score of `0.80`
- 1h direction uses the same payload with a minimum score of `0.55`
- multi-horizon builders now exclude forward-return leakage columns instead of allowing them into the feature matrix

Corrected feature-lift outputs now live under:

- `artifacts/models/featurelift_20260331_rerun/`
- `artifacts/analysis/featurelift_20260331_rerun/`

## 6. Runtime Outputs

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

Each horizon payload in `artifacts/predictions/latest.json` includes, among other fields:

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
- `projected_high` and `projected_low` when target-range models are active

Top-level operator-facing fields now include:

- `blocked_trade_analytics`
- `degradation_monitoring`
- `prompt_ready_summary`
- `prompt_ready_summary.operator_summary_compact`

Important execution diagnostics recorded in runtime output:

- `execution_plan.stop_management.stop_scaling`
- `execution_plan.target_management.dynamic_rr_floor_applied`
- `execution_plan.target_management.dynamic_realized_rr_ratio`
- `direction_output.probability_shrinkage`
- `trade_decision.threshold_source`
- `confidence_min_source`
- `abstention.reason`
- `uncertainty.effective_policy`

Additional local feature-source outputs relevant to agents:

- `data/processed/macro/daily_features.parquet`
- `data/processed/macro/source_manifest.json`
- `data/processed/onchain/hourly_features.parquet`
- `data/processed/onchain/source_manifest.json`

These are local working-state files. Regenerate them if they are absent or stale.

## 6B. Validation Guards

The repository now includes a dedicated validation workflow:

- `.github/workflows/validation-guards.yml`

Its local equivalent command path is:

```bash
/workspaces/btc/.venv/bin/python -m pytest \
  tests/test_runtime_feature_parity_and_validation.py \
  tests/test_intrabar_feature_parity.py \
  tests/test_macro_loader_and_integration.py \
  tests/test_onchain_loader_and_integration.py \
  tests/test_direction_feature_reliability_filters.py \
  tests/test_feature_leakage_guards.py \
  tests/test_featurelift_report_reference_check.py

/workspaces/btc/.venv/bin/python -m src.scripts.generate_featurelift_comparison_report
/workspaces/btc/.venv/bin/python -m src.scripts.check_featurelift_report_references
```

Those guards are intended to catch:

- train/runtime intrabar drift
- macro and on-chain integration regressions
- leakage reintroduction in hourly or multi-horizon datasets
- stale references to the superseded March 31 feature-lift report

## 7. Cadence Operations

The cadence wrapper supports four operating paths:

- `daily`: refresh predictions from the latest trustworthy reliability run
- `weekly`: run the runtime reliability workflow, then refresh predictions
- `monthly`: run the full default reliability workflow, then refresh predictions
- `shadow`: run the shadow profile comparison workflow

Commands:

```bash
bash ./scripts/run_cadence.sh daily
bash ./scripts/run_cadence.sh weekly
bash ./scripts/run_cadence.sh monthly
bash ./scripts/run_cadence.sh shadow
```

Important current behavior from `scripts/run_cadence.sh`:

- `daily` refreshes with `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `weekly` runs `configs/reliability_workflow.runtime.yaml` before the same daily refresh path
- `monthly` runs `configs/reliability_workflow.default.yaml` before the same daily refresh path
- `shadow` compares `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml` against `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`

The cadence wrapper resolves the latest trustworthy reliability run by scanning:

- `artifacts/reliability/*/summary/edge_trustworthiness.json`
- `artifacts/reliability/*/summary/calibrated_thresholds.json`
- `artifacts/reliability/*/summary/platt_calibration.json`

GitHub Actions schedule in `.github/workflows/cadence.yml`:

- daily at `01:15 UTC`
- weekly on Monday at `02:30 UTC`
- monthly on day `1` at `03:45 UTC`

The workflow supports manual dispatch for `daily`, `weekly`, and `monthly`. The `shadow` cadence remains a local wrapper path unless the workflow is extended.

## 8. Shadow Comparison Outputs

Shadow comparison outputs are written under:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Use `bash ./scripts/run_cadence.sh shadow` for the supported end-to-end path.

Interpretation rules:

- use the Markdown summary for the fastest operator-facing comparison read
- use the CSV for quick per-run history review
- use the longitudinal JSON as the automation source of truth
- treat `source_reliability_run_id` as distinct from the shadow comparison run id

## 9. Reliability Workflow

Runtime reliability workflow:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml
```

Full default reliability workflow:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml
```

Midband paper profile:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.midband_paper.yaml
```

Continue even when promotion is blocked:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

The runtime workflow includes directional-objective evaluation through `src.scripts.evaluate_directional_objectives`.

Current runtime directional-objective thresholds from `configs/reliability_workflow.runtime.yaml`:

- `group_min_rows: 40`
- `max_brier: 0.255`
- `max_ece_by_regime.chop: 0.18`

Current default workflow directional-objective thresholds from `configs/reliability_workflow.default.yaml`:

- `group_min_rows: 80`
- `max_brier: 0.25`

Evaluator behavior from `src/scripts/evaluate_directional_objectives.py`:

- auto-resolves the label column from `y`, `y_true`, `target_up`, `label`, or `direction_target`
- normalizes missing or invalid regime labels to `unknown`

Reliability outputs are written under:

- `artifacts/reliability/<run-id>/summary/`

Common artifacts include:

- `workflow_manifest.json`
- `calibrated_thresholds.json`
- `platt_calibration.json`
- `platt_calibration_coverage.json`
- `promotion_gate.json`
- `champion_gate_alignment_check.json`
- `trade_decision_model_shift_guard.json`
- `edge_trustworthiness.json`
- `walkforward_labeled_reconciliation.json`
- `directional_objectives.json`
- `trusted_baseline_pack.json`
- `overlap_feature_drift_guard.json`

## 10. Reliable Fresh Prediction Flow

Use this when you want a fresh prediction snapshot tied to the latest trustworthy reliability artifacts.

1. Run the runtime reliability workflow.
2. Resolve the latest trustworthy run id.
3. Run refresh and predict with that run's thresholds and Platt calibration.
4. Read the generated runtime artifacts.

Resolve the latest trustworthy run id:

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
    payload = json.loads(edge_path.read_text(encoding='utf-8'))
    if payload.get('edge_trustworthy'):
        print(run_dir.name)
        break
PY
)
echo "$RUN_ID"
```

Then refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

Read these outputs immediately after the run:

- `artifacts/predictions/latest.json`
- `artifacts/analysis/prediction_coherence_latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/reliability/${RUN_ID}/summary/edge_trustworthiness.json`
- `artifacts/reliability/${RUN_ID}/summary/walkforward_labeled_reconciliation.json`

## 11. Data Ingestion And Feature Processing

Current active market data source is Binance US spot klines.

Refresh/prediction flow uses:

```bash
/workspaces/btc/.venv/bin/python -m src.ingest_spot_klines --interval 1h --hours 360
```

Available ingestors in this workspace:

- `data.ingestors.binance_spot_klines`
- `data.ingestors.binance_us_spot`

Direct ingestor entrypoint:

```bash
/workspaces/btc/.venv/bin/python -m data.ingestors.binance_spot_klines
```

Historical backfill helper:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.backfill_binance_us_spot \
  --interval 1h --days 365
```

Feature processing entrypoint:

```bash
/workspaces/btc/.venv/bin/python -m data.processed.compute_technical_features
```

Shared runtime and training feature logic lives in:

- `src/trading/feature_engineering.py`

## 12. Training, Search, And Evaluation Entrypoints

Dataset builders currently present:

- `src.scripts.build_training_dataset`
- `src.scripts.build_training_dataset_15m`
- `src.scripts.build_training_dataset_direction`
- `src.scripts.build_training_dataset_direction_15m`
- `src.scripts.build_training_dataset_multi_horizon`
- `src.scripts.build_sequence_direction_dataset`

Core training and model assembly scripts:

- `src.scripts.train_baseline_model`
- `src.scripts.train_model_suite`
- `src.scripts.train_direction_model`
- `src.scripts.train_lgbm_dir`
- `src.scripts.train_meta_ensemble`
- `src.scripts.train_platt_calibration`
- `src.scripts.train_target_ranges`
- `src.scripts.train_trade_decision_model`
- `src.scripts.train_trend_ignition_xgb`

Sequence-model training scripts:

- `src.scripts.train_lstm_dir1h`
- `src.scripts.train_lstm_direction_model`
- `src.scripts.train_bilstm_dir1h`
- `src.scripts.train_gru_dir1h`
- `src.scripts.train_cnn_lstm_dir1h`
- `src.scripts.train_cnn_bilstm_dir1h`
- `src.scripts.train_garch_lstm_dir1h`
- `src.scripts.train_transformer_dir1h`
- `src.scripts.train_transformer_dir1h_large`

Additional point-model trainers:

- `src.scripts.train_xgb_dir4h_v1`
- `src.scripts.train_xgb_ret4h_v1`

Hyperparameter and threshold search scripts:

- `src.scripts.search_xgb_optuna`
- `src.scripts.search_lstm_optuna`
- `src.scripts.search_transformer_optuna`
- `src.scripts.search_ensemble_thresholds`
- `src.scripts.search_ensemble_weights`

## 13. Diagnostics, Audits, And Comparison Tools

Key analysis helpers currently present:

- `src.scripts.audit_direction_models`
- `src.scripts.audit_feature_parity`
- `src.scripts.analyze_direction_marginal_calibration`
- `src.scripts.analyze_probability_branch_alignment`
- `src.scripts.analyze_probability_calibration_alignment`
- `src.scripts.analyze_overlap_trust_stability`
- `src.scripts.analyze_overlap_triggered_trade_diagnostics`
- `src.scripts.compare_live_profile_snapshots`
- `src.scripts.run_shadow_profile_comparison`
- `src.scripts.update_shadow_profile_comparison_longitudinal`
- `src.scripts.summarize_shadow_profile_comparison_longitudinal`

Train/serve parity check:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.audit_feature_parity \
  --dataset-path artifacts/datasets/btc_features_multi_horizon_splits.npz \
  --features-path artifacts/tmp/parity_features_1h.parquet \
  --target-column ret_1h
```

Regression check:

```bash
/workspaces/btc/.venv/bin/python -m unittest discover -s tests
```

## 14. GitHub Actions And Self-Hosted Runner Notes

Cadence automation is defined in `.github/workflows/cadence.yml` and runs on `self-hosted` runners.

Current workflow behavior:

- validates the local artifact root first
- rejects remote artifact URIs
- runs `python -m src.scripts.bootstrap_cadence_artifacts` before cadence execution
- restores the deployed manifest, selected trustworthy summary, manifest-listed deployed files, and `artifacts/models` into the checkout

The helper script for Linux x64 runner setup is:

- `scripts/setup_self_hosted_runner.sh`

For unattended operation, prefer a durable non-ephemeral host over a session-bound Codespaces container.

## 15. Troubleshooting Notes

- invoke cadence through `bash ./scripts/run_cadence.sh <cadence>` when needed from the shell
- if `.venv` is unavailable, set `PYTHON_BIN=python` before running the cadence wrapper
- runtime config validation is fail-fast for unknown top-level keys, unknown direction-model weight override keys, and malformed or duplicate normalized threshold entries
- `disabled_horizons` is supported explicitly in runtime configs for suppressing horizons without editing target lists
- scratch validation work should stay under `artifacts/tmp_validation/`

## 16. Read Order For A New Operator Or Agent

Recommended read order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/agent_system_handoff_20260320.md`
4. `docs/trade_decision_post_fix_trust_basis_20260319.md`
5. `docs/live_trading_rollout_20260320.md`
6. `docs/live_operator_checklist_20260320.md`
7. `docs/trade_decision_8h_hardening_memo_20260320.md`

That sequence gives the highest-level system map first, then the operating procedure, then the more detailed policy and deployment context.
