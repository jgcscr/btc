#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_self_hosted_runner.sh --repo <owner/repo> --token <registration-token> [options]

Required:
  --repo <owner/repo>          GitHub repository, for example jgcscr/btc
  --token <registration-token> Repository self-hosted runner registration token

Optional:
  --name <runner-name>         Runner name (default: hostname)
  --labels <a,b,c>             Additional runner labels (default: self-hosted,Linux,X64,btc-local)
  --dir <path>                 Install directory (default: $HOME/actions-runner-btc)
  --version <runner-version>   Actions runner version (default: 2.333.0)
  --replace                    Replace an existing runner registration with the same name
  --install-service            Install and start the runner as a service

Example:
  scripts/setup_self_hosted_runner.sh \
    --repo jgcscr/btc \
    --token <token> \
    --name btc-local-runner \
    --dir $HOME/actions-runner-btc \
    --replace \
    --install-service
EOF
}

REPO=""
TOKEN=""
RUNNER_NAME="$(hostname)"
RUNNER_LABELS="self-hosted,Linux,X64,btc-local"
RUNNER_DIR="${HOME}/actions-runner-btc"
RUNNER_VERSION="2.333.0"
REPLACE_FLAG="false"
INSTALL_SERVICE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --name)
      RUNNER_NAME="$2"
      shift 2
      ;;
    --labels)
      RUNNER_LABELS="$2"
      shift 2
      ;;
    --dir)
      RUNNER_DIR="$2"
      shift 2
      ;;
    --version)
      RUNNER_VERSION="$2"
      shift 2
      ;;
    --replace)
      REPLACE_FLAG="true"
      shift
      ;;
    --install-service)
      INSTALL_SERVICE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REPO" || -z "$TOKEN" ]]; then
  usage >&2
  exit 1
fi

if [[ "$REPO" != */* ]]; then
  echo "--repo must be in <owner/repo> format" >&2
  exit 1
fi

ARCHIVE_NAME="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
DOWNLOAD_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${ARCHIVE_NAME}"

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [[ ! -f "$ARCHIVE_NAME" ]]; then
  curl -fsSL -o "$ARCHIVE_NAME" "$DOWNLOAD_URL"
fi

if [[ ! -x "$RUNNER_DIR/config.sh" ]]; then
  tar xzf "$ARCHIVE_NAME"
fi

CONFIG_ARGS=(
  --url "https://github.com/${REPO}"
  --token "$TOKEN"
  --name "$RUNNER_NAME"
  --labels "$RUNNER_LABELS"
  --unattended
)

if [[ "$REPLACE_FLAG" == "true" ]]; then
  CONFIG_ARGS+=(--replace)
fi

./config.sh "${CONFIG_ARGS[@]}"

if [[ "$INSTALL_SERVICE" == "true" ]]; then
  sudo ./svc.sh install
  sudo ./svc.sh start
  echo "Runner service installed and started in $RUNNER_DIR"
else
  cat <<EOF
Runner configured in $RUNNER_DIR.

To start it interactively:
  cd "$RUNNER_DIR"
  ./run.sh

To install it as a service later:
  cd "$RUNNER_DIR"
  sudo ./svc.sh install
  sudo ./svc.sh start
EOF
fi