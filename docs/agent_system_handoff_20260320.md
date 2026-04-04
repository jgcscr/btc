# Agent System Handoff

Status: Current agent handoff reference.

Use this with `README.md` and `docs/operations_runbook.md` as the current operating surface.

This is the shortest safe handoff for an agent taking over the current workspace.

## 1. Start With The Real Operating Split

There are five different job paths here:

1. research refresh
2. live inference
3. reliability workflow
4. shell cadence
5. service/API execution

Do not treat them as interchangeable.

Preferred wrappers:

- research: `src.scripts.run_research_refresh`
- live: `src.scripts.run_live_inference`
- reliability: `src.scripts.run_reliability_pipeline`
- shell cadence: `scripts/run_cadence.sh`
- service: `src/service/main.py`

Important current boundary:

- `src/scripts/run_refresh_and_predict.py` still owns the remaining dense `run_predictions(...)` body and the full CLI parse surface.

Current wrapper behavior that matters in practice:

- `src.scripts.run_live_inference` defaults to `configs/run_refresh_and_predict.live_conservative_binance_only.yaml`
- it emits `0.25,1,4,12` by default and auto-forwards `--intrabar-enabled` when `0.25` is present
- the approved live profile ignores stale `funding`, `macro`, and `onchain` sources in `feature_coverage_policy`
- `configs/run_refresh_and_predict.research_safe.yaml` is the documented fallback when research refreshes should warn instead of hard-fail on remaining feature-coverage violations

## 2. Read These First

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/live_operator_checklist_20260320.md`
4. `artifacts/monitoring/reliability_promotion_deploy_manifest.json` when present
5. `artifacts/predictions/latest.json` when present
6. `artifacts/monitoring/latest.json` when present

If the task touches service execution, also read:

- `src/service/main.py`
- `src/service/orchestration.py`
- `src/service/job_runner.py`

## 3. Current Runtime Map

Direct wrappers execute runtime orchestration in `src/runtime/`.

Main files to know:

- `src/runtime/refresh_pipeline.py`: research/live runtime orchestration
- `src/runtime/market_preparation.py`: ingestion, local features, replay, quality/coverage gating
- `src/runtime/refresh_support.py`: config and model-input resolution
- `src/runtime/horizon_support.py`: horizon normalization/labels/sort keys
- `src/runtime/summary_support.py`: prompt/degradation/summary payloads
- `src/runtime/output_support.py`: monitoring artifacts and meta-baseline refresh
- `src/runtime/reliability_pipeline.py`: runtime wrapper around the reliability workflow

Current intentional legacy hook still outside the prediction executor:

- reliability step-event sink in `src/runtime/reliability_pipeline.py`

## 4. Current Source Of Truth

Read these artifacts before relying on any recent runtime state:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/runtime_runs/<run-id>/request.json`
- `artifacts/runtime_runs/<run-id>/events.jsonl`
- `artifacts/runtime_runs/<run-id>/summary.json`

Read these first for reliability state:

- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

## 5. Safe Command Defaults

Research refresh:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.default.yaml
```

Research-safe fallback:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_research_refresh \
  --config configs/run_refresh_and_predict.research_safe.yaml
```

Live inference:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_live_inference
```

Runtime reliability:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_pipeline \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Shell cadence:

```bash
bash ./scripts/run_cadence.sh daily
```

## 6. Important Current Caveats

1. `data/` and most of `artifacts/` are local working state, not guaranteed on a fresh clone.
2. The shell cadence wrapper supports `shadow`, but `.github/workflows/cadence.yml` currently schedules or dispatches only `daily`, `weekly`, and `monthly`.
3. Service execution is in process, not shelling out to missing wrappers.
4. `POST /run-papertrade` is not implemented and returns `501`.
5. Replay mode is hourly-only and incompatible with local-feature override mode.

## 7. Agent Audit Surface

When an agent needs to inspect the system rather than change live behavior, use the checked-in analysis helpers before inventing new scripts:

- `src.scripts.audit_train_live_feature_parity`
- `src.scripts.simulate_macro_shadow_enforcement`
- `src.scripts.simulate_state_orderflow_shadow_enforcement`
- `src.scripts.confirm_state_orderflow_outcomes`
- `src.scripts.confirm_orderflow_two_window_stability`
- `src.scripts.confirm_orderflow_rolling_stability`
- `src.scripts.confirm_state_engineering_narrow_scope`
- `src.scripts.run_state_engineering_guarded_shadow`
- `src.scripts.summarize_signal_program_status`
- `src.scripts.prepare_derivatives_shadow_validation`
- `src.scripts.refresh_derivatives_features`

These are analysis-only unless a separate change explicitly promotes a config or wrapper path.

## 8. Main Failure Mode To Avoid

The most common operator and agent mistake in this repository is mixing up:

- direct live-style refreshes
- cadence refreshes
- reliability workflows
- observational shadow runs

Keep those paths separate and read the artifacts for the path you actually ran.