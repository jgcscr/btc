# Agent System Handoff (2026-03-20)

This document is the shortest complete handoff for a new agent that needs to operate this repository safely.

It is written for the current workspace state, not for a hypothetical clean-room rebuild.

## What Is Trusted Right Now

There are two different runtime references and they serve different purposes:

- `configs/run_refresh_and_predict.default.yaml` is the trusted post-fix research and comparison baseline.
- `configs/run_refresh_and_predict.live_conservative.yaml` is the approved initial live trading profile.

Do not treat them as interchangeable.

Current live stance:

- the system is trusted for live trading with conservative risk discipline,
- `8h` remains the weakest carry horizon,
- live hardening is done with horizon-specific size caps, scoped horizon/regime overrides, and operator discipline, not with a blanket `8h` suppression rule.

Current state sources in this codespace:

- read the latest live-style runtime state from `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json`
- read the active deployed bundle from `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- treat those artifacts, not this handoff note, as the source of truth for the current run id, active variant, preferred horizon, and latest recommended action

## Read Order For A New Agent

Read these first, in this order:

1. `README.md`
2. `docs/operations_runbook.md`
3. `docs/trade_decision_post_fix_trust_basis_20260319.md`
4. `docs/live_trading_rollout_20260320.md`
5. `docs/live_operator_checklist_20260320.md`
6. `docs/trade_decision_8h_hardening_memo_20260320.md`
7. `artifacts/monitoring/reliability_promotion_deploy_manifest.json`

If you need the older deployment decision context, then read:

- `docs/trade_decision_operator_handoff_20260316.md`

## Golden Paths

### 1. Fresh live-style prediction refresh

Use this when you need the current market state under the approved live rollout policy:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.live_conservative.yaml
```

Inspect immediately after the run:

1. `artifacts/predictions/latest.json`
2. `artifacts/monitoring/latest.json`
3. `artifacts/monitoring/trade_ready_summary.json` only if that run path rewrote artifact-writing summaries

### 2. Standard daily operations

Use the cadence wrapper:

```bash
bash ./scripts/run_cadence.sh daily
```

Important behavior:

- the cadence script resolves the latest trustworthy reliability run,
- it refreshes predictions with `configs/run_refresh_and_predict.shadow_simplified.yaml`,
- it is for scheduled operating cadence, not for the conservative discretionary live rollout.
- the GitHub Actions workflow currently exposes only `daily`, `weekly`, and `monthly`; `shadow` remains a local wrapper path unless the workflow is extended.

### 2a. Shadow comparison cadence

Use this when you need repeated observational comparison between the simplified shadow profile and the chop-suppression candidate:

```bash
bash ./scripts/run_cadence.sh shadow
```

Inspect immediately after the run:

1. `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
2. `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`
3. `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`

Important behavior:

- the shadow cadence compares `shadow_simplified` vs `shadow_chop_suppression`,
- the comparison run id is timestamped independently from the trustworthy reliability source,
- the manifest records the source bundle as `source_reliability_run_id`,
- the latest shadow run is determined by `generated_at`, not by lexicographic `run_id`,
- the current decision-useful operator digest is the Markdown summary, not the raw pairwise comparison JSON.

### 3. Reliability workflow refresh

Runtime reliability pass:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail
```

Full monthly/default reliability pass:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail
```

### 4. Replay validation against historical snapshots

Use this when validating policy changes without touching the active live logic:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

For pairwise trust validation, the working utilities are:

- `artifacts/tmp_validation/run_pairwise_replay_matrix.py`
- `artifacts/tmp_validation/rebuild_pairwise_summary_from_snapshots.py`
- `artifacts/tmp_validation/score_pairwise_return_proxy.py`

## Files That Matter Most During Operations

Runtime config and policy:

- `configs/run_refresh_and_predict.default.yaml`
- `configs/run_refresh_and_predict.live_conservative.yaml`
- `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`
- `src/trading/feature_engineering.py`
- `src/scripts/audit_feature_parity.py`

Reliability workflow control:

- `configs/reliability_workflow.runtime.yaml`
- `configs/reliability_workflow.default.yaml`
- `scripts/run_cadence.sh`

Runtime outputs:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json` when the selected path refreshes artifact-writing summaries
- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Reliability outputs:

- `artifacts/reliability/<run-id>/summary/promotion_gate.json`
- `artifacts/reliability/<run-id>/summary/champion_gate_alignment_check.json`
- `artifacts/reliability/<run-id>/summary/trade_decision_model_shift_guard.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`

Scratch validation area:

- keep temporary replay work under `artifacts/tmp_validation/`
- do not treat scratch outputs as trusted until the manifest and configs match the intended run

## Safe Operating Rules

1. Do not weaken promotion gates just to keep a candidate moving.
2. Do not replace the trusted default with an unscored policy change.
3. Do not add a blanket `8h` suppression rule without covered replay validation.
4. Prefer capital underweighting and operator discipline over speculative routing changes.
5. If live output shows directional bias but `execution_plan` says `waiting_pullback` or `rejected`, do not force a market entry.
6. Treat runtime config validation failures as correctness problems, not as warnings; stale weight keys and malformed threshold maps now fail fast by design.

## What To Check Before Trusting A New Change

For reliability and deployment changes:

1. `summary/champion_gate_alignment_check.json`
2. `summary/promotion_gate.json`
3. `summary/trade_decision_model_shift_guard.json`
4. `summary/overlap_triggered_trade_diagnostics.json`
5. `summary/calibration_robustness.json`
6. `summary/rolling_ab_report.json`

For runtime policy changes:

1. pairwise replay summary integrity
2. manifest correctness for snapshot reuse
3. covered return-proxy scoring against `artifacts/datasets/btc_features_multi_horizon_splits.npz`
4. focused regression tests when code changes are involved
5. train/serve feature parity using `src.scripts.audit_feature_parity` when local-feature overrides are part of the change

## Current 8h Interpretation

The current direct covered operator-caution extraction for `8h` added trades is:

- total `8h` added trades: `11`
- average signed return proxy: `-0.004160029236862267`
- `8h` longs are the weakest slice

Operationally this means:

- keep `8h` enabled,
- keep it underweighted,
- allow the checked-in live conservative profile to act on `8h` when it is genuinely `ready`,
- do not manually upsize or substitute a blocked `12h` setup for the ready `8h` setup.

## Minimal First Session For A New Agent

If a fresh agent has to pick up the repo quickly, the minimum safe sequence is:

1. read the documents listed in the read order above
2. inspect the current deploy manifest
3. run a fresh conservative live prediction refresh
4. inspect `latest.json` and `trade_ready_summary.json`
5. only then decide whether the task is live operations, reliability, or replay validation

That sequence is enough to avoid the most common mistake in this repo: mixing up the trusted research default, the conservative live rollout profile, and the cadence/shadow operating profile.
