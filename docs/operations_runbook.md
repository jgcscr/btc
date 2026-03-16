# Operations Runbook

This repo uses three operating cadences:

- `daily`: refresh live predictions from the latest trustworthy reliability run
- `weekly`: run the runtime reliability workflow, then refresh live predictions
- `monthly`: run the full default reliability workflow, then refresh live predictions

## Single Entrypoint

Use the shell entrypoint from the repository root:

```bash
batch ./scripts/run_cadence.sh daily
batch ./scripts/run_cadence.sh weekly
batch ./scripts/run_cadence.sh monthly
```

In CI or other environments without the local `.venv`, set `PYTHON_BIN` explicitly if needed:

```bash
PYTHON_BIN=python ./scripts/run_cadence.sh daily
```

The script resolves the latest trustworthy run by reading `artifacts/reliability/*/summary/edge_trustworthiness.json` and then uses that run's `calibrated_thresholds.json` and `platt_calibration.json` for prediction.

## Exact Commands By Cadence

### Daily

Runs fresh predictions from the latest trustworthy run.

```bash
RUN_ID=$(python - <<'PY'
import json
from pathlib import Path
for run_dir in sorted((p for p in Path('artifacts/reliability').iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
    edge_path = run_dir / 'summary' / 'edge_trustworthiness.json'
    if not edge_path.exists():
        continue
    payload = json.loads(edge_path.read_text())
    if payload.get('edge_trustworthy'):
        print(run_dir.name)
        break
PY
)

python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.shadow_simplified.yaml \
  --targets 0.25,1,4,8,12 \
  --thresholds-json artifacts/reliability/${RUN_ID}/summary/calibrated_thresholds.json \
  --platt-calibration artifacts/reliability/${RUN_ID}/summary/platt_calibration.json \
  --write-artifacts
```

### Weekly

Runs the runtime reliability profile, then refreshes predictions.

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.runtime.yaml \
  --continue-on-promotion-fail

batch ./scripts/run_cadence.sh daily
```

### Monthly

Runs the full default reliability profile, then refreshes predictions.

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail

batch ./scripts/run_cadence.sh daily
```

Direct execution of `./scripts/run_cadence.sh <cadence>` is not supported in this workspace. Use the `batch` prefix for all cadence invocations.

## Promotion Handoff And Gate Alignment

- Shared live and monitoring artifacts should only move when a workflow run both passes promotion and the runtime config defines `quality.promotion_deploy` targets.
- If promotion is blocked, treat the run as diagnostic only; keep the existing incumbent deployed.
- Check `artifacts/reliability/<RUN_ID>/summary/champion_gate_alignment_check.json` before trusting a new promotion result. The expected steady states are:
  - `official_shadow_variant: none` with `expected_source: labeled` and `selected_source: labeled`
  - official shadow active with `expected_source: policy_aligned` and `selected_source: policy_aligned`
- If the alignment check fails, treat it as a workflow regression rather than a model-quality signal and do not deploy artifacts from that run.

## Trade-Decision Reference Features

- The live workflow default is conservative: keep source-aware reference-feature controls enabled, use `disable_on_source_mismatch`, and keep clipping enabled when configured.
- Treat `summary/trade_decision_model_shift_guard.json` as a hard promotion blocker, not just a diagnostic artifact.
- Treat `official_shadow_variant: reference_feature_ablation` as diagnostic until it passes the same companion, calibration, and rolling checks required of any promoted candidate.
- Coverage improvements from the ablation variant are not sufficient on their own; require acceptable recent active-trade calibration and non-negative economics before considering deployment.

## Rollback Criteria

- Current active deployment: run `20260316T030147Z`, variant `reference_feature_ablation_threshold_0p555_neutral_p_up_cap_0p499`, recorded in `artifacts/monitoring/reliability_promotion_deploy_manifest.json`.
- Immediate rollback target: run `20260315T062250Z`, variant `selection_calibration_guard`.
- Roll back if the next runtime reliability run keeps the active variant selected but records `summary/promotion_gate.json` with `promote = false` or any failed `trade_decision_model_shift_guard` checks.
- Roll back if `summary/champion_gate_alignment_check.json` fails, because that indicates workflow routing drift rather than a trustworthy model-quality comparison.
- Roll back if the active variant loses overlap-triggered support on the next qualified runtime run: `triggered_row_count < 10`, `triggered_net_return_total <= 0`, or `triggered_hit_rate < 0.45` in `summary/overlap_triggered_trade_diagnostics.json`.
- Roll back if live refresh checks show a materially worse operational posture than the prior deployment, especially if a horizon flips from trade-decision-triggered to inactive for non-price reasons tied to reference-feature drift or policy gating instability.
- Use the prior deployment artifacts from run `20260315T062250Z` as the manual restore source for `artifacts/models/trade_decision_model.json`, `artifacts/models/platt_calibration.json`, `artifacts/monitoring/labeled_backtest_1h*.json/csv`, and `artifacts/monitoring/promotion_gate_1h.json` if an emergency rollback is required.

### Manual Rollback

Use this exact restore block from the repository root if the active deployment must be reverted immediately:

```bash
set -e

cp artifacts/reliability/20260315T062250Z/summary/platt_calibration.json \
  artifacts/models/platt_calibration.json

cp artifacts/reliability/20260315T062250Z/summary/trade_decision_model.json \
  artifacts/models/trade_decision_model.json

cp artifacts/reliability/20260315T062250Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard.csv \
  artifacts/monitoring/labeled_backtest_1h.csv

cp artifacts/reliability/20260315T062250Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard_meta.json \
  artifacts/monitoring/labeled_backtest_1h_meta.json

cp artifacts/reliability/20260315T062250Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard.csv \
  artifacts/monitoring/labeled_backtest_1h_incumbent.csv

cp artifacts/reliability/20260315T062250Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard_meta.json \
  artifacts/monitoring/labeled_backtest_1h_incumbent_meta.json

cp artifacts/reliability/20260315T062250Z/summary/model_quality_candidate.json \
  artifacts/monitoring/model_quality_incumbent_1h_backtest.json

cp artifacts/reliability/20260315T062250Z/summary/model_quality_candidate.json \
  artifacts/monitoring/model_quality_incumbent_1h.json

cp artifacts/reliability/20260315T062250Z/summary/promotion_gate.json \
  artifacts/monitoring/promotion_gate_1h.json

cp artifacts/reliability/20260315T062250Z/summary/calibration_robustness.json \
  artifacts/monitoring/calibration_robustness.json

cp artifacts/reliability/20260315T062250Z/summary/rolling_ab_report.json \
  artifacts/monitoring/rolling_ab_report.json

cp artifacts/reliability/20260315T062250Z/summary/rolling_ab_report.md \
  artifacts/monitoring/rolling_ab_report.md

if [[ -f artifacts/reliability/20260315T062250Z/summary/selection_calibration_guard_rule.json ]]; then
  cp artifacts/reliability/20260315T062250Z/summary/selection_calibration_guard_rule.json \
    artifacts/monitoring/selection_calibration_guard_rule_1h.json
fi
```

After the copy completes, replace `artifacts/monitoring/reliability_promotion_deploy_manifest.json` with a manifest that points back to run `20260315T062250Z`, then run a dry-run refresh check to confirm the restored bundle loads cleanly.

## Weekly Monitoring Checklist

For the next weekly runtime run after deployment `20260316T030147Z`, inspect these items in order:

- Confirm `artifacts/monitoring/reliability_promotion_deploy_manifest.json` still points at run `20260316T030147Z` before starting, unless an approved rollback was executed.
- Run the weekly workflow and read `summary/champion_gate_alignment_check.json` first; it must stay `passed = true` with `selected_source = policy_aligned`.
- Read `summary/promotion_gate.json`; treat any failed `trade_decision_model_shift_guard` check as an immediate rollback candidate, not a soft warning.
- Read `summary/overlap_triggered_trade_diagnostics.json`; keep the deployment only if the triggered slice remains at least `10` trades, positive net return, and hit rate at or above `0.45`.
- Read `summary/rolling_ab_report.json`; confirm the promoted branch does not introduce new negative rolling windows beyond the configured cap.
- Read `summary/calibration_robustness.json`; confirm the active-trade selection slice still clears recent AUC, recent ECE, and ECE drift thresholds.
- Run a dry-run prediction refresh against the active deployment and compare the result to the prior rollback target if behavior looks unstable. The validated comparison pattern is that the new deployment may increase trade-decision triggering while final actions remain `hold`; that alone is not a rollback signal.
- If a rollback trigger fires, restore run `20260315T062250Z` first, then investigate whether the failure is a workflow regression, overlap deterioration, or reference-feature drift.

## Reading `execution_plan`

- `bias_only_ready` means the execution layer likes the structure, stop, and target layout, but the predictive model still returned `hold`.
- `waiting_pullback` means the setup is aligned and tradable only on a retest into the preferred entry zone.
- `rejected` means a hard guard failed; the most common reasons in recent replay checks were `bias_direction_conflict`, `stop_too_tight_near_invalidation`, and `stop_too_wide`.
- Treat `upstream_model_hold` as informational rather than a structural failure; treat the stop- and RR-related reasons as execution-quality failures.
- `execution_plan.stop_management` shows whether the execution layer expanded, capped, or replaced the selected stop to keep it inside the configured ATR guardrails.

Replay workflow for cached hourly bars:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

Replay mode is hourly-only for now and overwrites the usual prediction artifact paths, so restore a live run afterward when you are finished reviewing the historical snapshot.

## GitHub Actions Schedule

The workflow is defined in `.github/workflows/cadence.yml` and runs:

- daily at `01:15 UTC`
- weekly on Monday at `02:30 UTC`
- monthly on day 1 at `03:45 UTC`

It also supports manual dispatch with a `cadence` selector.

## Expected Outputs

After a successful run, inspect:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/walkforward_labeled_reconciliation.json`

## Codespaces Note

GitHub Codespaces containers are not a reliable place to depend on `cron` or `systemd` for persistent scheduling. Use the GitHub Actions workflow for unattended cadence execution.

GitHub Actions also requires the underlying model and dataset artifacts to be available on the runner. This workflow assumes the repository checkout already contains the required files or that they are restored by your environment before the cadence step runs.