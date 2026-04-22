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
Usage: scripts/run_cadence.sh <daily|weekly|monthly|shadow>

Cadences:
  daily    Run fresh predictions from the latest trustworthy reliability run.
  weekly   Run the runtime reliability workflow, then refresh predictions.
  monthly  Run the full default reliability workflow, then refresh predictions.
  shadow   Run direction-enhanced-relaxed-chop vs chop-suppression shadow profiles and archive a comparison artifact.
EOF
}

find_latest_trustworthy_run() {
  "$PYTHON_BIN" -m src.scripts.resolve_latest_trustworthy_reliability_run
}

require_latest_trustworthy_run() {
  local run_id=""
  if ! run_id="$(find_latest_trustworthy_run)"; then
    cat >&2 <<'EOF'
No trustworthy reliability run found under artifacts/reliability.

Cadence requires prebuilt reliability artifacts that are not tracked in git.
Restore the deployed artifacts bundle before running cadence.

For the checked-in self-hosted GitHub Actions workflow, preflight and bootstrap now run
before scripts/run_cadence.sh. If you are invoking cadence outside that workflow,
restore the required artifacts into this checkout first.
EOF
    return 1
  fi

  if [[ -z "$run_id" ]]; then
    echo "Resolved empty trustworthy reliability run id" >&2
    return 1
  fi
  printf '%s\n' "$run_id"
}

ensure_runtime_directories() {
  mkdir -p \
    data/spot_klines \
    artifacts/predictions \
    artifacts/monitoring
}

run_predictions() {
  local run_id="$1"
  echo "Using trustworthy run: $run_id"
  ensure_runtime_directories
  "$PYTHON_BIN" -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.shadow_simplified.yaml \
    --targets 0.25,1,4,8,12 \
    --thresholds-json "artifacts/reliability/${run_id}/summary/calibrated_thresholds.json" \
    --platt-calibration "artifacts/reliability/${run_id}/summary/platt_calibration.json" \
    --write-artifacts
}

run_shadow_comparison() {
  local reliable_run_id="$1"
  echo "Using trustworthy run for shadow comparison: $reliable_run_id"
  ensure_runtime_directories
  "$PYTHON_BIN" -m src.scripts.run_shadow_profile_comparison \
    --lhs-config configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml \
    --rhs-config configs/run_refresh_and_predict.shadow_chop_suppression.yaml \
    --lhs-label shadow_direction_enhanced_relaxed_chop \
    --rhs-label shadow_chop_suppression \
    --targets 0.25,1,4,8,12 \
    --thresholds-json "artifacts/reliability/${reliable_run_id}/summary/calibrated_thresholds.json" \
    --platt-calibration "artifacts/reliability/${reliable_run_id}/summary/platt_calibration.json" \
    --restore-latest-to rhs
}

run_reliability() {
  local config_path="$1"
  "$PYTHON_BIN" -m src.scripts.run_reliability_pipeline \
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
    RUN_ID="$(require_latest_trustworthy_run)"
    run_predictions "$RUN_ID"
    ;;
  weekly)
    run_reliability "configs/reliability_workflow.runtime.yaml"
    RUN_ID="$(require_latest_trustworthy_run)"
    run_predictions "$RUN_ID"
    ;;
  monthly)
    run_reliability "configs/reliability_workflow.default.yaml"
    RUN_ID="$(require_latest_trustworthy_run)"
    run_predictions "$RUN_ID"
    ;;
  shadow)
    RUN_ID="$(require_latest_trustworthy_run)"
    run_shadow_comparison "$RUN_ID"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac