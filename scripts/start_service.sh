#!/usr/bin/env sh
set -euo pipefail

APP_MODULE=${APP_MODULE:-src.service.main:app}
PORT=${PORT:-8080}
GUNICORN_WORKERS=${GUNICORN_WORKERS:-1}
GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-600}
GUNICORN_LOG_LEVEL=${GUNICORN_LOG_LEVEL:-info}

exec gunicorn \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${GUNICORN_WORKERS}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --bind "0.0.0.0:${PORT}" \
  --log-level "${GUNICORN_LOG_LEVEL}" \
  "${APP_MODULE}"
