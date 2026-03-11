#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_DEFAULT="$ROOT_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python executable not found at $PYTHON_BIN" >&2
    exit 1
  fi
fi

cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/run_cadence.sh <daily|weekly|monthly>

Cadences:
  daily    Run fresh predictions from the latest trustworthy reliability run.
  weekly   Run the runtime reliability workflow, then refresh predictions.
  monthly  Run the full default reliability workflow, then refresh predictions.
EOF
}

find_latest_trustworthy_run() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

run_root = Path("artifacts/reliability")
if not run_root.exists():
    raise SystemExit(1)

for run_dir in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
    edge_path = run_dir / "summary" / "edge_trustworthiness.json"
    thresholds_path = run_dir / "summary" / "calibrated_thresholds.json"
    platt_path = run_dir / "summary" / "platt_calibration.json"
    if not edge_path.exists() or not thresholds_path.exists() or not platt_path.exists():
        continue
    try:
        payload = json.loads(edge_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if bool(payload.get("edge_trustworthy", False)):
        print(run_dir.name)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

run_predictions() {
  local run_id="$1"
  echo "Using trustworthy run: $run_id"
  "$PYTHON_BIN" -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.shadow_simplified.yaml \
    --targets 0.25,1,4,8,12 \
    --thresholds-json "artifacts/reliability/${run_id}/summary/calibrated_thresholds.json" \
    --platt-calibration "artifacts/reliability/${run_id}/summary/platt_calibration.json" \
    --write-artifacts
}

run_reliability() {
  local config_path="$1"
  "$PYTHON_BIN" -m src.scripts.run_reliability_workflow \
    --config "$config_path" \
    --continue-on-promotion-fail
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

CADENCE="$1"

case "$CADENCE" in
  daily)
    RUN_ID="$(find_latest_trustworthy_run)"
    if [[ -z "$RUN_ID" ]]; then
      echo "No trustworthy reliability run found under artifacts/reliability" >&2
      exit 1
    fi
    run_predictions "$RUN_ID"
    ;;
  weekly)
    run_reliability "configs/reliability_workflow.runtime.yaml"
    RUN_ID="$(find_latest_trustworthy_run)"
    run_predictions "$RUN_ID"
    ;;
  monthly)
    run_reliability "configs/reliability_workflow.default.yaml"
    RUN_ID="$(find_latest_trustworthy_run)"
    run_predictions "$RUN_ID"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac