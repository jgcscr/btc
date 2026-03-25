# Operations Runbook

This repo uses four operating cadences:

- `daily`: refresh live predictions from the latest trustworthy reliability run
- `weekly`: run the runtime reliability workflow, then refresh live predictions
- `monthly`: run the full default reliability workflow, then refresh live predictions
- `shadow`: run simplified vs chop-suppression shadow profiles and archive comparison artifacts

## Single Entrypoint

Use the shell entrypoint from the repository root:

```bash
bash ./scripts/run_cadence.sh daily
bash ./scripts/run_cadence.sh weekly
bash ./scripts/run_cadence.sh monthly
bash ./scripts/run_cadence.sh shadow
```

In CI or other environments without the local `.venv`, set `PYTHON_BIN` explicitly if needed:

```bash
PYTHON_BIN=python bash ./scripts/run_cadence.sh daily
```

The script resolves the latest trustworthy run by reading `artifacts/reliability/*/summary/edge_trustworthiness.json` and then uses that run's `calibrated_thresholds.json` and `platt_calibration.json` for prediction.

Recommended runtime policy reference:

- `configs/run_refresh_and_predict.default.yaml` is the trusted post-fix operating default.
- `configs/run_refresh_and_predict.live_conservative.yaml` is the conservative live rollout profile with horizon-specific size caps plus scoped `8h@trend_ignition` overrides for trade-decision threshold, confidence minimum, and abstention hold-band.
- `docs/trade_decision_post_fix_trust_basis_20260319.md` is the operator-facing record of the current trust basis and validation workflow.
- `docs/live_trading_rollout_20260320.md` is the operator-facing live rollout policy and monitoring guide.
- `docs/live_operator_checklist_20260320.md` is the shortest pre-run and post-run operator checklist for live use.
- `docs/trade_decision_8h_hardening_memo_20260320.md` records the current `8h` caution stance and the safest next hardening direction.
- `docs/agent_system_handoff_20260320.md` is the new-agent handoff for safely running the system end to end.

Current state sources:

- read the active deployed bundle from `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- read the latest live-style snapshot from `artifacts/predictions/latest.json` and `artifacts/monitoring/latest.json`
- check `generated_at`, feature coverage, and source freshness in those artifacts before acting
- do not rely on hardcoded snapshot summaries in this runbook for the latest market state

## Exact Commands By Cadence

There are two execution contexts in this repo:

- local CLI cadence: run `bash ./scripts/run_cadence.sh <cadence>` directly when the checkout already has the required local `artifacts/` tree
- GitHub Actions cadence: the self-hosted workflow first validates and bootstraps artifacts into the checkout, then calls the same shell wrapper

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

bash ./scripts/run_cadence.sh daily
```

Runtime reliability workflow quality stage now includes directional-objective gating:

- command: `python -m src.scripts.evaluate_directional_objectives`
- outputs: `summary/directional_objectives.json` and `logs/directional_objectives.log`
- checks: overall, by-horizon, and by-regime Brier/ECE/F1 with min-row guards
- runtime profile defaults currently use `prob_col: p_up`, `label_col: y` (auto-resolved to `y_true` if present in the labeled CSV), `group_min_rows: 40`, `max_brier: 0.255`, and `max_ece_by_regime.chop: 0.18`

### Monthly

Runs the full default reliability profile, then refreshes predictions.

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail

bash ./scripts/run_cadence.sh daily
```

### Shadow

Runs the current shadow-vs-shadow comparison workflow between:

- `configs/run_refresh_and_predict.shadow_simplified.yaml`
- `configs/run_refresh_and_predict.shadow_chop_suppression.yaml`

Use it through the cadence wrapper:

```bash
bash ./scripts/run_cadence.sh shadow
```

Current workflow boundary:

- `shadow` is supported by `scripts/run_cadence.sh`
- `.github/workflows/cadence.yml` currently schedules only `daily`, `weekly`, and `monthly`
- the GitHub Actions manual dispatch selector currently exposes only `daily`, `weekly`, and `monthly`
- run `shadow` locally unless you intentionally extend the workflow inputs and artifact upload paths

Important behavior:

- the shadow run uses the latest trustworthy reliability bundle for thresholds and Platt calibration,
- the shadow comparison run id is timestamped independently from the source reliability run id,
- the manifest records the source reliability run separately as `source_reliability_run_id`,
- repeated shadow cadence runs should append new comparison points instead of overwriting the prior comparison artifact.

Inspect these outputs after each shadow run:

- `artifacts/predictions/comparisons/shadow_profile_comparison_longitudinal.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.json`
- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

Interpretation rules:

- treat the Markdown summary as the fastest operator-facing snapshot,
- treat the CSV as the quickest per-run history export,
- treat the longitudinal JSON as the source of truth for automation and backfills,
- do not interpret score-only differences as operational changes unless the summary starts showing operational or decision-state deltas.

Direct execution of `./scripts/run_cadence.sh <cadence>` is not supported in this workspace because the script is not executable as checked in. Invoke it through `bash`.

## Runtime Validation Rules

Before trusting a config edit or a new local profile, expect these fail-fast checks:

- unknown runtime config keys are rejected
- unknown direction-model weight override keys are rejected
- malformed or duplicate normalized threshold entries are rejected
- stale direction model keys such as removed model families must be deleted from profile weight maps before the run will start

Train/serve parity can be checked with:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.audit_feature_parity \
  --dataset-path artifacts/datasets/btc_features_multi_horizon_splits.npz \
  --features-path artifacts/tmp/parity_features_1h.parquet \
  --target-column ret_1h
```

This is the preferred check before trusting a local-feature override path for live-style refreshes.

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

- Current active deployment is whatever run and variant are recorded in `artifacts/monitoring/reliability_promotion_deploy_manifest.json`.
- Recommended runtime policy default: `configs/run_refresh_and_predict.default.yaml`, with trust basis documented in `docs/trade_decision_post_fix_trust_basis_20260319.md`.
- Roll back if the next runtime reliability run keeps the active variant selected but records `summary/promotion_gate.json` with `promote = false` or any failed `trade_decision_model_shift_guard` checks.
- Roll back if `summary/champion_gate_alignment_check.json` fails, because that indicates workflow routing drift rather than a trustworthy model-quality comparison.
- Roll back if the active variant loses overlap-triggered support on the next qualified runtime run: `triggered_row_count < 10`, `triggered_net_return_total <= 0`, or `triggered_hit_rate < 0.45` in `summary/overlap_triggered_trade_diagnostics.json`.
- Roll back if live refresh checks show a materially worse operational posture than the prior deployment, especially if a horizon flips from trade-decision-triggered to inactive for non-price reasons tied to reference-feature drift or policy gating instability.
- Use the prior approved deployment artifacts referenced by the target run's summary manifest when an emergency rollback is required.

### Manual Rollback

Use the deploy manifest written by the target reliability run instead of a hardcoded historical copy block.

Each promoted run writes `artifacts/reliability/<RUN_ID>/summary/promotion_deploy_manifest.json`, which records the deployed targets and source files for that run. Use that file as the rollback source of truth.

Example restore flow from the repository root:

```bash
set -e

ROLLBACK_RUN_ID="<prior-known-good-run-id>"
MANIFEST="artifacts/reliability/${ROLLBACK_RUN_ID}/summary/promotion_deploy_manifest.json"
export MANIFEST

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST"])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

for spec in payload.get("deployed_files", {}).values():
    source = Path(spec["source"])
    target = Path(spec["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

Path("artifacts/monitoring/reliability_promotion_deploy_manifest.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
PY
```

After the copy completes, run a dry-run refresh check to confirm the restored bundle loads cleanly.

## Weekly Monitoring Checklist

For the next weekly runtime run after the current deployment, inspect these items in order:

- Confirm `artifacts/monitoring/reliability_promotion_deploy_manifest.json` still points at the intended incumbent before starting, unless an approved rollback was executed.
- Read `summary/directional_objectives.json` early in the review. Treat any non-empty `failed_checks` as a reliability-gate failure that must be addressed before promotion.
- Run the weekly workflow and read `summary/champion_gate_alignment_check.json` first; it must stay `passed = true` with `selected_source = policy_aligned`.
- Read `summary/promotion_gate.json`; treat any failed `trade_decision_model_shift_guard` check as an immediate rollback candidate, not a soft warning.
- Read `summary/overlap_triggered_trade_diagnostics.json`; keep the deployment only if the triggered slice remains at least `10` trades, positive net return, and hit rate at or above `0.45`.
- Read `summary/rolling_ab_report.json`; confirm the promoted branch does not introduce new negative rolling windows beyond the configured cap.
- Read `summary/calibration_robustness.json`; confirm the active-trade selection slice still clears recent AUC, recent ECE, and ECE drift thresholds.
- Run a dry-run prediction refresh against the active deployment and compare the result to the prior rollback target if behavior looks unstable. The validated comparison pattern is that the new deployment may increase trade-decision triggering while final actions remain `hold`; that alone is not a rollback signal.
- If a rollback trigger fires, restore the prior known-good deployed run first, then investigate whether the failure is a workflow regression, overlap deterioration, or reference-feature drift.

## Reading `execution_plan`

- `bias_only_ready` means the execution layer likes the structure, stop, and target layout, but the predictive model still returned `hold`.
- `waiting_pullback` means the setup is aligned and tradable only on a retest into the preferred entry zone.
- `rejected` means a hard guard failed; the most common reasons in recent replay checks were `bias_direction_conflict`, `stop_too_tight_near_invalidation`, and `stop_too_wide`.
- Treat `upstream_model_hold` as informational rather than a structural failure; treat the stop- and RR-related reasons as execution-quality failures.
- `execution_plan.stop_management` shows whether the execution layer expanded, capped, or replaced the selected stop to keep it inside the configured ATR guardrails.
- `execution_plan.pullback_quality` records the pullback score, the minimum score required for that horizon/regime, and the VWAP and candle-expansion diagnostics that can force a downgrade from immediate entry to pullback-only or a full rejection.
- `execution_plan.disagreement_severity` records the short-term versus mid-term directional conflict score. When it crosses the configured block threshold the horizon is rejected as `short_term_disagreement`; lower scores can still force pullback-only entry.
- `bias_score`, `execution_score`, `bias_support_horizons`, and `bias_support_is_8h_standalone` expose how the weighted horizon vote was built and whether the surviving bias is leaning too heavily on `8h` alone.

## Reading New Monitoring Fields

- `artifacts/predictions/latest.json` now includes `blocked_trade_analytics`, `degradation_monitoring`, and `prompt_ready_summary.operator_summary_compact` at the top level.
- `blocked_trade_analytics` is the quickest way to see whether the current snapshot is being blocked mainly by forecast coherence, short-term disagreement, pullback quality, or baseline risk/reward guards.
- `blocked_trade_analytics.gate_stage_counts` and `gate_reason_counts` show which gating stage actually blocked the stack most often in the latest snapshot.
- `degradation_monitoring` is snapshot-history based rather than realized-PnL based. Treat it as an operational posture alarm, not as a substitute for trade replay or live outcome review.
- `operator_summary_compact` is the fastest operator-facing digest: it reports market bias, preferred horizon, recommended action, primary blocker, supporting horizons, and caution flags.
- `artifacts/monitoring/latest.json` mirrors these top-level prediction summaries so operators can inspect them from one monitoring payload.
- Per-horizon `trade_decision.threshold_source`, `confidence_min_source`, and `abstention.reason` now expose which horizon/regime override was actually applied.

Replay workflow for cached hourly bars:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --dry-run --targets 1,4,8,12 \
  --replay-offset-bars 24
```

Replay mode is hourly-only for now and overwrites the usual prediction artifact paths, so restore a live run afterward when you are finished reviewing the historical snapshot.

## GitHub Actions Schedule

The workflow is defined in `.github/workflows/cadence.yml` and runs (UTC):

- daily at `01:15 UTC`
- weekly on Monday at `02:30 UTC`
- monthly on day 1 at `03:45 UTC`

It also supports manual dispatch with a `cadence` selector for `daily`, `weekly`, and `monthly`.

## Expected Outputs

After a successful run, inspect:

- `artifacts/predictions/latest.json`
- `artifacts/monitoring/latest.json`
- `artifacts/monitoring/trade_ready_summary.json`
- `artifacts/reliability/<run-id>/summary/edge_trustworthiness.json`
- `artifacts/reliability/<run-id>/summary/walkforward_labeled_reconciliation.json`

Within `artifacts/predictions/latest.json`, check these top-level sections before acting on a setup:

- `prompt_ready_summary.market_outlook_strategy`
- `prompt_ready_summary.operator_summary_compact`
- `blocked_trade_analytics`
- `degradation_monitoring`

For shadow cadence reviews, also inspect:

- `artifacts/predictions/comparisons/shadow_profile_comparison_summary.md`
- `artifacts/predictions/comparisons/shadow_profile_comparison_runs.csv`

## Codespaces Note

GitHub Codespaces containers are not a reliable place to depend on `cron` or `systemd` for persistent scheduling. Use the GitHub Actions workflow for unattended cadence execution.

GitHub Actions also requires the underlying model and dataset artifacts to be available on the runner. This workflow assumes the repository checkout already contains the required files or that they are restored by your environment before the cadence step runs.

Current GitHub Actions bootstrap path:

- the workflow is now intentionally self-hosted only
- install a self-hosted GitHub Actions runner on the same machine or server that already has the cadence artifacts
- set repository variable `CADENCE_ARTIFACTS_ROOT_URI` to the local filesystem path that mirrors the repository `artifacts/` tree on that runner
- optionally set `CADENCE_DEPLOY_MANIFEST_URI` if the deploy manifest is not at `monitoring/reliability_promotion_deploy_manifest.json` under that root
- the workflow first runs a preflight validation against that local path and fails before cadence if the manifest or required summary/deployed files are missing
- the workflow runs `python -m src.scripts.bootstrap_cadence_artifacts` before `scripts/run_cadence.sh`, which restores the deployed manifest, the selected trustworthy run summary, the manifest-listed deployed files, and the full `artifacts/models` tree into the local checkout
- remote artifact URIs are intentionally rejected by this workflow variant; use only local filesystem paths visible to the self-hosted runner

Self-hosted runner setup helper:

- use `scripts/setup_self_hosted_runner.sh` to download and configure the Linux x64 runner on the local machine once you have a repository registration token
- the script accepts `--repo`, `--token`, `--dir`, `--name`, `--labels`, `--replace`, and `--install-service`
- after the runner is online, set `CADENCE_ARTIFACTS_ROOT_URI` in GitHub repository variables to the local artifacts path visible to that runner if you need to override the default local path

## Session-Bound Runner Warning

The runner currently configured in a Codespaces or dev-container session is not durable by default:

- if you start the runner with `./run.sh`, it only lives for the lifetime of that shell or background process
- if the container restarts, is rebuilt, or goes idle long enough to be recycled, the runner stops and GitHub will show it as offline
- this is acceptable for ad hoc manual workflow runs, but it is not a reliable foundation for unattended scheduled cadence execution

## Durable Runner Recommendation

For stable unattended cadence operation, prefer a non-ephemeral self-hosted host:

- a small VM, always-on workstation, or local server with a persistent filesystem
- install the runner outside the repository checkout, for example under `$HOME/actions-runner-btc`
- point the workflow at the persistent local artifact tree, or keep the repository and `artifacts/` tree on the same host path permanently
- run the runner under the host's service manager so it comes back after reboot

If you stay in Codespaces for now, treat the runner as temporary and re-check these before trusting schedule execution:

- the runner still shows online in GitHub
- `/workspaces/btc/artifacts` still exists on the runner host
- the preflight step resolves the expected manifest path and trustworthy run id before cadence starts