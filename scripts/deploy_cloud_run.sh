#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID or export PROJECT_ID before running}"
: "${SERVICE_NAME:=btc-trading-service}"
: "${REGION:=us-central1}"
: "${ARTIFACT_REPO:=btc-trading}"
: "${IMAGE_NAME:=btc-run-svc}"
: "${IMAGE_TAG:=$(date +%Y%m%d%H%M%S)}"
: "${GOVERNANCE_BUCKET:=gs://jc-btc-governance}"  # example bucket override

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

printf 'Building image %s\n' "${IMAGE_URI}"
gcloud builds submit --tag "${IMAGE_URI}" .

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 3 \
  --set-env-vars PYTHONPATH=/app,GOVERNANCE_OUTPUT_URI="${GOVERNANCE_BUCKET}/governance" \
  --set-secrets "CRYPTOCOMPARE_API_KEY=crypto-key:latest,ALPHA_VANTAGE_API_KEY=alpha-key:latest" \
  --timeout 900
