#!/usr/bin/env bash
# Run all dev services: sim-server (8100), api (8000), UI (5173).
# Usage: ./dev.sh [--no-ui] [--no-sim] [--robot]
#   --no-ui   skip the Vite dev server
#   --no-sim  skip the sim-server (and Isaac Sim)
#   --robot   pass --load-robot to the sim process

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp"

NO_UI=false
NO_SIM=false
ROBOT_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --no-ui)   NO_UI=true ;;
        --no-sim)  NO_SIM=true ;;
        --robot)   ROBOT_FLAG="--load-robot" ;;
    esac
done

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
PIDS=()
SIM_PID=""

cleanup() {
    echo ""
    echo "[dev.sh] Shutting down..."

    # SIGTERM all tracked children
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    # Isaac Sim takes time to shut down; wait up to 12s then force-kill
    if [[ -n "$SIM_PID" ]]; then
        local waited=0
        while kill -0 "$SIM_PID" 2>/dev/null && (( waited < 12 )); do
            sleep 1
            (( waited++ ))
        done
    fi

    # Kill any remaining Isaac Sim / kit_app processes by pattern
    local sim_procs
    sim_procs=$(pgrep -f "drp_sim.server\|kit_app\|isaacsim" 2>/dev/null || true)
    if [[ -n "$sim_procs" ]]; then
        echo "[dev.sh] Killing remaining Isaac Sim processes: $sim_procs"
        kill -9 $sim_procs 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    echo "[dev.sh] Done."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Helper: kill whatever is on a port
# ---------------------------------------------------------------------------
kill_port() {
    local port="$1"
    local pid
    pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        echo "[dev.sh] Killing existing process on port $port (PID $pid)"
        kill "$pid" 2>/dev/null || true
        sleep 0.5
    fi
}

# ---------------------------------------------------------------------------
# Helper: kill sim subprocess (dr_ai_palletizer.sim_process) if running
# ---------------------------------------------------------------------------
kill_sim_process() {
    local pids
    pids=$(pgrep -f "drp_sim.server\|dr_ai_palletizer.sim_process\|kit_app\|isaacsim" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "[dev.sh] Found running sim process(es): $pids — terminating..."
        # SIGTERM first, then SIGKILL if still alive after 8s
        kill $pids 2>/dev/null || true
        local waited=0
        while kill -0 $pids 2>/dev/null && (( waited < 8 )); do
            sleep 1
            (( waited++ ))
        done
        if kill -0 $pids 2>/dev/null; then
            echo "[dev.sh] Sim process did not exit cleanly — sending SIGKILL"
            kill -9 $pids 2>/dev/null || true
        else
            echo "[dev.sh] Sim process exited."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Kill stale processes
# ---------------------------------------------------------------------------
kill_sim_process
kill_port 8100
kill_port 8000
kill_port 5173

# ---------------------------------------------------------------------------
# Sim-server (port 8100) — Isaac Sim + drp_sim FastAPI
# ---------------------------------------------------------------------------
if [[ "$NO_SIM" == false ]]; then
    SIM_LOG="$LOG_DIR/sim_server.log"
    echo "[dev.sh] Starting sim-server on port 8100 -> $SIM_LOG"

    _NVRTC_LIB="$REPO/.venv/lib/python3.11/site-packages/nvidia/cu13/lib"
    _LD="${LD_LIBRARY_PATH:-}"
    [[ ":$_LD:" != *":$_NVRTC_LIB:"* ]] && _LD="$_NVRTC_LIB${_LD:+:$_LD}"

    SIM_HOST=0.0.0.0 \
    SIM_PORT=8100 \
    SIM_LOAD_ROBOT="${SIM_LOAD_ROBOT:-true}" \
    SIM_SPAWN_BOXES="${SIM_SPAWN_BOXES:-true}" \
    OMNI_KIT_ACCEPT_EULA=YES \
    LD_LIBRARY_PATH="$_LD" \
    uv run python -m drp_sim.server \
        >"$SIM_LOG" 2>&1 &
    SIM_PID=$!
    PIDS+=($SIM_PID)
fi

# ---------------------------------------------------------------------------
# App-server / orchestrator (port 8000) — dr_ai_palletizer FastAPI
# ---------------------------------------------------------------------------
API_LOG="$LOG_DIR/api_server.log"
echo "[dev.sh] Starting app-server on port 8000 -> $API_LOG"

SIM_SERVER_URL="http://localhost:8100" \
INFERENCE_SERVER_URL="${INFERENCE_SERVER_URL:-http://localhost:8200/v1}" \
INFERENCE_MODEL="${INFERENCE_MODEL:-nvidia/Cosmos-Reason2-2B}" \
APP_HOST=0.0.0.0 \
APP_PORT=8000 \
uv run uvicorn dr_ai_palletizer.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    >"$API_LOG" 2>&1 &
PIDS+=($!)

# ---------------------------------------------------------------------------
# UI (port 5173)
# ---------------------------------------------------------------------------
if [[ "$NO_UI" == false ]]; then
    UI_LOG="$LOG_DIR/ui.log"
    echo "[dev.sh] Starting UI on port 5173 -> $UI_LOG"

    (cd "$REPO/app/ui" && npm run dev >"$UI_LOG" 2>&1) &
    PIDS+=($!)
fi

# ---------------------------------------------------------------------------
# Wait for services to come up and print URLs
# ---------------------------------------------------------------------------
sleep 2

echo ""
echo "[dev.sh] Services:"
[[ "$NO_SIM" == false ]] && echo "  sim-server  http://localhost:8100   logs: $LOG_DIR/sim_server.log"
echo "  api         http://localhost:8000   logs: $LOG_DIR/api_server.log"
[[ "$NO_UI" == false ]]  && echo "  UI          http://localhost:5173   logs: $LOG_DIR/ui.log"
echo ""
echo "[dev.sh] Press Ctrl+C to stop all."

# ---------------------------------------------------------------------------
# Tail all logs to stdout
# ---------------------------------------------------------------------------
TAIL_ARGS=()
[[ "$NO_SIM" == false ]] && TAIL_ARGS+=("$LOG_DIR/sim_server.log")
TAIL_ARGS+=("$LOG_DIR/api_server.log")
[[ "$NO_UI" == false ]]  && TAIL_ARGS+=("$LOG_DIR/ui.log")

tail -F "${TAIL_ARGS[@]}" &
PIDS+=($!)

wait
