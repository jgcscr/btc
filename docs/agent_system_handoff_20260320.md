# Agent System Handoff

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

## 7. Main Failure Mode To Avoid

The most common operator and agent mistake in this repository is mixing up:

- direct live-style refreshes
- cadence refreshes
- reliability workflows
- observational shadow runs

Keep those paths separate and read the artifacts for the path you actually ran.