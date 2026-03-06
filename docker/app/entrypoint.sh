#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8000}"

echo "[app-server] Starting on ${HOST}:${PORT}"
echo "[app-server] SIM_SERVER_URL=${SIM_SERVER_URL:-http://sim-server:8100}"
echo "[app-server] INFERENCE_SERVER_URL=${INFERENCE_SERVER_URL:-http://inference-server:8200/v1}"

exec uv run uvicorn dr_ai_palletizer.server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level info
