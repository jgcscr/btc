#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-jc-financial-466902}

ALPHA_VANTAGE_SECRET_NAME=${ALPHA_VANTAGE_SECRET_NAME:-alpha-vantage-api-key}
ALPHA_VANTAGE_SECRET_VERSION=${ALPHA_VANTAGE_SECRET_VERSION:-latest}
ALPHA_VANTAGE_PAID_SECRET_NAME=${ALPHA_VANTAGE_PAID_SECRET_NAME:-${ALPHA_VANTAGE_SECRET_NAME}}
ALPHA_VANTAGE_PAID_SECRET_VERSION=${ALPHA_VANTAGE_PAID_SECRET_VERSION:-${ALPHA_VANTAGE_SECRET_VERSION}}
ALPHA_VANTAGE_FREE_SECRET_NAME=${ALPHA_VANTAGE_FREE_SECRET_NAME:-alpha-vantage-free-api-key}
ALPHA_VANTAGE_FREE_SECRET_VERSION=${ALPHA_VANTAGE_FREE_SECRET_VERSION:-latest}
TWELVE_DATA_SECRET_NAME=${TWELVE_DATA_SECRET_NAME:-twelvedata-api-key}
TWELVE_DATA_SECRET_VERSION=${TWELVE_DATA_SECRET_VERSION:-latest}
TIINGO_SECRET_NAME=${TIINGO_SECRET_NAME:-tiingo-api-key}
TIINGO_SECRET_VERSION=${TIINGO_SECRET_VERSION:-latest}
FRED_SECRET_NAME=${FRED_SECRET_NAME:-fred-api-key}
FRED_SECRET_VERSION=${FRED_SECRET_VERSION:-latest}
KAIKO_API_KEY_SECRET_NAME=${KAIKO_API_KEY_SECRET_NAME:-kaiko-api-key}
KAIKO_API_KEY_SECRET_VERSION=${KAIKO_API_KEY_SECRET_VERSION:-latest}
KAIKO_API_SECRET_SECRET_NAME=${KAIKO_API_SECRET_SECRET_NAME:-kaiko-api-secret}
KAIKO_API_SECRET_SECRET_VERSION=${KAIKO_API_SECRET_SECRET_VERSION:-latest}
TWELVEDATA_PREMIUM_SECRET_NAME=${TWELVEDATA_PREMIUM_SECRET_NAME:-twelvedata-premium-api-key}
TWELVEDATA_PREMIUM_SECRET_VERSION=${TWELVEDATA_PREMIUM_SECRET_VERSION:-latest}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required to load macro provider secrets." >&2
  exit 1
fi

fetch_secret() {
  local secret_name=$1
  local secret_version=$2
  if [[ -z "${secret_name}" ]]; then
    echo "Secret name is required." >&2
    exit 1
  fi
  gcloud secrets versions access "${secret_version}" --secret "${secret_name}" --project "${PROJECT_ID}"
}

load_secret_to_envs() {
  local secret_name=$1
  local secret_version=$2
  shift 2
  local env_names=("$@")
  if [[ ${#env_names[@]} -eq 0 ]]; then
    echo "No environment variables supplied for ${secret_name}." >&2
    exit 1
  fi
  local value
  value="$(fetch_secret "${secret_name}" "${secret_version}")"
  for env_name in "${env_names[@]}"; do
    export "${env_name}"="${value}"
  done
  local env_list
  env_list=$(printf "%s " "${env_names[@]}")
  env_list=${env_list% }
  echo "Loaded ${env_list} from ${secret_name} (${secret_version}) in ${PROJECT_ID}."
}

copy_env_value() {
  local source_env=$1
  shift
  local env_names=("$@")
  local value=${!source_env:-}
  if [[ -z "${value}" ]]; then
    echo "Environment variable ${source_env} is empty; cannot duplicate into ${env_names[*]}." >&2
    exit 1
  fi
  for env_name in "${env_names[@]}"; do
    export "${env_name}"="${value}"
  done
  local env_list
  env_list=$(printf "%s " "${env_names[@]}")
  env_list=${env_list% }
  echo "Copied ${source_env} into ${env_list}."
}

maybe_load_secret_to_envs() {
  local secret_name=$1
  local secret_version=$2
  shift 2
  local env_names=("$@")
  if gcloud secrets describe "${secret_name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    load_secret_to_envs "${secret_name}" "${secret_version}" "${env_names[@]}"
  else
    local env_list
    env_list=$(printf "%s " "${env_names[@]}")
    env_list=${env_list% }
    echo "Secret ${secret_name} not found in ${PROJECT_ID}; skipping ${env_list} until credentials are provisioned." >&2
  fi
}

load_secret_to_envs "${ALPHA_VANTAGE_SECRET_NAME}" "${ALPHA_VANTAGE_SECRET_VERSION}" \
  "ALPHA_VANTAGE_API_KEY" "ALPHA_VANTAGE_KEYS"

if [[ "${ALPHA_VANTAGE_PAID_SECRET_NAME}" == "${ALPHA_VANTAGE_SECRET_NAME}" && \
      "${ALPHA_VANTAGE_PAID_SECRET_VERSION}" == "${ALPHA_VANTAGE_SECRET_VERSION}" ]]; then
  copy_env_value "ALPHA_VANTAGE_API_KEY" "ALPHA_VANTAGE_PAID_KEYS" "ALPHA_VANTAGE_PAID_API_KEY"
else
  load_secret_to_envs "${ALPHA_VANTAGE_PAID_SECRET_NAME}" "${ALPHA_VANTAGE_PAID_SECRET_VERSION}" \
    "ALPHA_VANTAGE_PAID_KEYS" "ALPHA_VANTAGE_PAID_API_KEY"
fi

load_secret_to_envs "${ALPHA_VANTAGE_FREE_SECRET_NAME}" "${ALPHA_VANTAGE_FREE_SECRET_VERSION}" \
  "ALPHA_VANTAGE_FREE_KEYS" "ALPHA_VANTAGE_FREE_API_KEY"

load_secret_to_envs "${TWELVE_DATA_SECRET_NAME}" "${TWELVE_DATA_SECRET_VERSION}" "TWELVE_DATA_API_KEY"
load_secret_to_envs "${TIINGO_SECRET_NAME}" "${TIINGO_SECRET_VERSION}" "TIINGO_API_KEY"
load_secret_to_envs "${FRED_SECRET_NAME}" "${FRED_SECRET_VERSION}" "FRED_API_KEY"
maybe_load_secret_to_envs "${KAIKO_API_KEY_SECRET_NAME}" "${KAIKO_API_KEY_SECRET_VERSION}" "KAIKO_API_KEY"
maybe_load_secret_to_envs "${KAIKO_API_SECRET_SECRET_NAME}" "${KAIKO_API_SECRET_SECRET_VERSION}" "KAIKO_API_SECRET"
maybe_load_secret_to_envs "${TWELVEDATA_PREMIUM_SECRET_NAME}" "${TWELVEDATA_PREMIUM_SECRET_VERSION}" "TWELVEDATA_PREMIUM_API_KEY"

echo "Loaded macro provider secrets successfully."
