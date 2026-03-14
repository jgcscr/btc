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