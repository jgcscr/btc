#!/usr/bin/env bash
set -euo pipefail

# Simple helper to copy the trained model artifacts
# (regression and direction) into the FastAPI app
# directory for container builds.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REG_MODEL_DIR="$REPO_ROOT/artifacts/models/xgb_ret1h_v1"
DIR_MODEL_DIR="$REPO_ROOT/artifacts/models/xgb_dir1h_v1"
API_MODEL_DIR="$REPO_ROOT/src/api/model"

if [[ ! -d "$REG_MODEL_DIR" ]]; then
  echo "Regression model directory not found: $REG_MODEL_DIR" >&2
  exit 1
fi

if [[ ! -d "$DIR_MODEL_DIR" ]]; then
  echo "Direction model directory not found: $DIR_MODEL_DIR" >&2
  exit 1
fi

rm -rf "$API_MODEL_DIR"
mkdir -p "$API_MODEL_DIR"

cp "$REG_MODEL_DIR/xgb_ret1h_model.json" "$API_MODEL_DIR/"
cp "$REG_MODEL_DIR/model_metadata.json" "$API_MODEL_DIR/"
cp "$DIR_MODEL_DIR/xgb_dir1h_model.json" "$API_MODEL_DIR/"
cp "$DIR_MODEL_DIR/model_metadata_direction.json" "$API_MODEL_DIR/"

echo "Copied regression and direction models into $API_MODEL_DIR"