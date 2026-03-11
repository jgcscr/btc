# Operations Runbook

This repo uses three operating cadences:

- `daily`: refresh live predictions from the latest trustworthy reliability run
- `weekly`: run the runtime reliability workflow, then refresh live predictions
- `monthly`: run the full default reliability workflow, then refresh live predictions

## Single Entrypoint

Use the shell entrypoint from the repository root:

```bash
./scripts/run_cadence.sh daily
./scripts/run_cadence.sh weekly
./scripts/run_cadence.sh monthly
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

./scripts/run_cadence.sh daily
```

### Monthly

Runs the full default reliability profile, then refreshes predictions.

```bash
python -m src.scripts.run_reliability_workflow \
  --config configs/reliability_workflow.default.yaml \
  --continue-on-promotion-fail

./scripts/run_cadence.sh daily
```

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