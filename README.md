# BTCUSDT Forecasting Workspace

This repository contains the current BTCUSDT research refresh, live inference, reliability, cadence, and service stack used in this workspace.

The codebase is no longer organized around a single monolithic script path for day-to-day operations. The current split is:

- wrapper CLIs in `src/scripts/` for research refresh, live inference, reliability, shadow comparison, and cadence helpers
- runtime orchestration in `src/runtime/` for refresh, market preparation, summary writing, monitoring artifacts, and runtime telemetry
- service endpoints in `src/service/` for in-process job execution
- legacy execution logic still retained in `src/scripts/run_refresh_and_predict.py` for the full argparse surface and the dense `run_predictions(...)` executor
- legacy workflow logic still retained in `src/scripts/run_reliability_workflow.py`, wrapped by `src/runtime/reliability_pipeline.py`

The most important practical consequence is this: for normal operation, use the wrapper commands documented below. Read `run_refresh_and_predict.py` directly only when you need the full legacy CLI surface or the remaining prediction-stage internals.

## What Exists In This Codespace

Current checked-in operating surface:

- direct research refresh wrapper: `src.scripts.run_research_refresh`
- direct live inference wrapper: `src.scripts.run_live_inference`
- runtime reliability wrapper: `src.scripts.run_reliability_pipeline`
- full reliability workflow: `src.scripts.run_reliability_workflow`
- cadence shell wrapper: `scripts/run_cadence.sh`
- shadow comparison runner: `src.scripts.run_shadow_profile_comparison`
- FastAPI service: `src/service/main.py`
- service registry/orchestration: `src/service/job_runner.py` and `src/service/orchestration.py`
- self-hosted cadence workflow: `.github/workflows/cadence.yml`
- validation workflow: `.github/workflows/validation-guards.yml`

Local state caveat:

- `data/` and most of `artifacts/` are local working state and are not a safe assumption on a fresh clone.
- Agents should regenerate local processed features and runtime artifacts when needed instead of assuming they already exist.

## Preferred Entry Points

Use these wrappers first.

### Research Refresh

Recommended baseline command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.default.yaml
```

This wrapper:

- parses the same config and CLI surface as `run_refresh_and_predict`
- executes `src.runtime.refresh_pipeline.execute_refresh_pipeline(..., mode=RuntimeMode.RESEARCH)`
- writes runtime telemetry under `artifacts/runtime_runs/<run-id>/`

### Live Inference

Recommended live-style command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference
```

The live wrapper constrains the CLI surface intentionally:

- default config: `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`
- default targets: `0.25,1,4,8,12`
- supported provider: `binanceus`
- writes trade-ready artifacts unless `--no-write-artifacts` is supplied

### Reliability Workflow

Recommended runtime reliability command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Recommended full default reliability command:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail
```

This wrapper:

- parses with `src.scripts.run_reliability_workflow.parse_args(...)`
- executes `src.runtime.reliability_pipeline.execute_reliability_pipeline(...)`
- records pipeline events under `artifacts/runtime_runs/<run-id>/`
- still delegates actual workflow steps to `src.scripts.run_reliability_workflow`

### Legacy Full-Surface Refresh Script

Use this only when you need options not exposed by the narrower wrappers:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict
```

Current note for agents:

- this file still owns the full CLI parse surface and the remaining dense `run_predictions(...)` body
- the runtime support modules now own much of the config normalization, prepared-data loading, summary writing, monitoring writing, horizon helpers, and policy support around it

## Current Runtime Architecture

The main runtime flow is:

1. `src.scripts.run_research_refresh` or `src.scripts.run_live_inference`
2. `src.runtime.refresh_pipeline.execute_refresh_pipeline`
3. `src.runtime.market_preparation.prepare_market_data`
4. `src.runtime.refresh_support.resolve_prediction_inputs`
5. `src.scripts.run_refresh_and_predict.run_predictions`
6. `src.runtime.summary_support.write_prediction_summary`
7. `src.runtime.output_support.write_monitoring_artifact`

Important runtime modules for agents:

- `src/runtime/refresh_pipeline.py`: top-level research/live orchestration and runtime telemetry
- `src/runtime/market_preparation.py`: ingestion, local feature bundle assembly, replay override, and quality/coverage gating
- `src/runtime/refresh_support.py`: config loading, threshold/platt resolution, dataset selection, prepared-data loading
- `src/runtime/horizon_support.py`: horizon normalization, label formatting, and sort keys
- `src/runtime/summary_support.py`: prompt summary, degradation monitoring, blocked-trade analytics, prediction payload writing
- `src/runtime/output_support.py`: monitoring artifacts and meta-baseline refresh
- `src/runtime/reliability_pipeline.py`: runtime telemetry wrapper around the reliability workflow

Current deliberate legacy boundaries:

- `src/scripts/run_refresh_and_predict.py` still owns `parse_args(...)` and `run_predictions(...)`
- `src/scripts/run_reliability_workflow.py` still owns reliability workflow execution
- `src/runtime/reliability_pipeline.py` still uses the reliability workflow step-event sink from the legacy workflow module

## Service API

The FastAPI service is defined in `src/service/main.py`.

Current stable endpoints:

- `GET /health`
- `GET /jobs`
- `POST /jobs/{job_name}`
- `POST /run-signal`
- `POST /run-dataset-refresh`
- `POST /run-reliability-workflow`
- `POST /run-walkforward`

Registered job names from `src/service/job_runner.py`:

- `live-inference`
- `research-refresh`
- `reliability-workflow`
- `walkforward-validation`

Compatibility note:

- `POST /run-papertrade` currently returns `501` and is not implemented in this workspace.

Service implementation note:

- jobs are executed in process via imported modules, not by shelling out to missing external wrappers

## Source Of Truth Artifacts

Read these first after any runtime refresh or cadence run:

- `artifacts/predictions/latest.json`
- `artifacts/predictions/history.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/monitoring/data_quality_latest.json`
- `artifacts/runtime_runs/<run-id>/request.json`
- `artifacts/runtime_runs/<run-id>/events.jsonl`
- `artifacts/runtime_runs/<run-id>/summary.json`
- `artifacts/runtime_runs/<run-id>/predictions.json`
- `artifacts/runtime_runs/<run-id>/monitoring.json`
- `artifacts/runtime_runs/<run-id>/trade_ready.json`

Read these first after a reliability run:

- `artifacts/reliability/<run-id>/summary/workflow_manifest.json`
- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/directional_objectives.json`
- `artifacts/reliability/<run-id>/summary/walkforward_labeled_reconciliation.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

## Runtime Profiles In Current Use

The important runtime configs are:

- `configs/run_refresh_and_predict.default.yaml`: trusted research and comparison baseline
- `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`: approved constrained live-style profile
- `configs/run_refresh_and_predict.live_conservative.yaml`: backward-compatible legacy-equivalent alias
- `configs/run_refresh_and_predict.shadow_simplified.yaml`: cadence daily refresh profile
- `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`: shadow comparison left-hand profile
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`: shadow comparison right-hand profile
- `configs/run_refresh_and_predict.shadow_strict_abstention.yaml`: additional shadow-only diagnostic profile

Current live-style sizing caps from `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`:

- `15m = 0.0`
- `1h = 0.15`
- `4h = 0.35`
- `8h = 0.20`
- `12h = 0.35`

## Cadence Operations

The shell wrapper supports four cadences:

```bash
bash ./scripts/run_cadence.sh daily
bash ./scripts/run_cadence.sh weekly
bash ./scripts/run_cadence.sh monthly
bash ./scripts/run_cadence.sh shadow
```

Current shell behavior from `scripts/run_cadence.sh`:

- `daily`: resolve the latest trustworthy reliability run and refresh predictions with `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `weekly`: run `configs/reliability_workflow.runtime.yaml`, then the same daily refresh path
- `monthly`: run `configs/reliability_workflow.default.yaml`, then the same daily refresh path
- `shadow`: compare `shadow_direction_enhanced_relaxed_chop` vs `shadow_chop_suppression`

Current GitHub Actions behavior from `.github/workflows/cadence.yml`:

- scheduled/manual workflow supports `daily`, `weekly`, and `monthly`
- `shadow` is not wired into workflow dispatch or schedule
- the workflow runs on `self-hosted`
- it preflights and bootstraps local cadence artifacts via `src.scripts.bootstrap_cadence_artifacts`
- it rejects remote artifact URIs and expects local filesystem paths visible to the runner

## Common Exact Commands

### Default Research Refresh

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12
```

### Conservative Live Refresh

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference \
  --config configs/run_refresh_and_predict.live_conservative_binance_only.yaml
```

### Refresh Against A Specific Reliability Run

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.default.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/<run-id>/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/<run-id>/summary/platt_calibration.json \
  --write-artifacts
```

### Replay Validation

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run \
  --targets 1,4,8,12 \
  --replay-offset-bars 24
```

Current replay rules enforced in code:

- hourly horizons only
- implies dry-run when offset is positive
- incompatible with `--use-local-features`

### Runtime Reliability

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

### Refresh Macro And On-Chain Bundles

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_macro_features --full-refresh
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_onchain_features --full-refresh
```

Those write local support bundles under:

- `data/processed/macro/`
- `data/processed/onchain/`

Current runtime note:

- fresh local rebuild paths attempt best-effort macro and on-chain refreshes during market preparation
- those enrich local feature bundles but are not the operator-facing source of truth for trade decisions

## Training, Validation, And Analysis Surface

The workspace still contains a broad modeling and audit surface under `src/scripts/`.

Useful categories:

- dataset builders: `build_training_dataset*`, `build_sequence_direction_dataset`
- trainers: `train_*`
- search/tuning: `search_*`, `tune_joint_signal_thresholds.py`
- reliability evaluators: `evaluate_*`, `compare_walkforward_models.py`
- audits and diagnostics: `audit_*`, `analyze_*`, `compare_*`, `summarize_*`

For the model-building surface, also read:

- `docs/model_suite.md`

## Validation Guards

The validation workflow lives at:

- `.github/workflows/validation-guards.yml`

Important local validation subset still referenced operationally:

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

## Minimal Read Order For A New Agent

Use this order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/agent_system_handoff_20260320.md`
4. `artifacts/monitoring/reliability_promotion_deploy_manifest.json` when present
5. `artifacts/predictions/latest.json` when present
6. `artifacts/monitoring/latest.json` when present

That sequence gives the system map first, then the operating procedure, then the current deployed and runtime state.