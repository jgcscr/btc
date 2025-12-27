#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-jc-financial-466902}
SECRET_NAME=${ALPHA_VANTAGE_SECRET_NAME:-alpha-vantage-api-key}
SECRET_VERSION=${ALPHA_VANTAGE_SECRET_VERSION:-latest}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required to load ${SECRET_NAME}" >&2
  exit 1
fi

export ALPHA_VANTAGE_API_KEY="$(gcloud secrets versions access "${SECRET_VERSION}" --secret "${SECRET_NAME}" --project "${PROJECT_ID}")"

echo "Loaded ALPHA_VANTAGE_API_KEY from ${SECRET_NAME} (${SECRET_VERSION}) in ${PROJECT_ID}."
