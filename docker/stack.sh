#!/usr/bin/env bash
# Run sim-server + app-server + frontend (everything except inference-server).
# inference-server can be started separately via inference.sh.
#
# Usage:
#   bash stack.sh           # start sim-server, app-server, frontend
#   bash stack.sh --down    # stop and remove containers
#   bash stack.sh --logs    # follow container logs
#   bash stack.sh --build   # force rebuild all images before start
#   bash stack.sh --ui      # rebuild and restart frontend only (fast UI update)
#   bash stack.sh --sim     # rebuild and restart sim-server only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Defaults -----------------------------------------------------------------
ACTION="up"
FORCE_BUILD=false
UI_ONLY=false
SIM_ONLY=false

# -- Parse flags --------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --down)   ACTION="down" ;;
        --logs)   ACTION="logs" ;;
        --build)  FORCE_BUILD=true ;;
        --ui)     UI_ONLY=true ;;
        --sim)    SIM_ONLY=true ;;
        --help|-h)
            sed -n '2,9s/^# //p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            exit 1
            ;;
    esac
done

SERVICES=(sim-server app-server frontend)

# -- Helpers ------------------------------------------------------------------
info()  { echo "[stack] $*"; }
error() { echo "[stack] ERROR: $*" >&2; }

_detect_cc_major() {
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ' || echo "8.9")
    echo "$cc" | cut -d. -f1
}

detect_sim_torch_backend() {
    local arch
    arch=$(uname -m)
    if [[ "$arch" == "aarch64" ]]; then
        local model
        model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
        if echo "$model" | grep -qi "thor"; then echo "cu130"; else echo "cu128"; fi
    else
        if (( $(_detect_cc_major) >= 10 )); then echo "cu130"; else echo "cu128"; fi
    fi
}

detect_sim_cuda_devel() {
    if [[ "$(detect_sim_torch_backend)" == "cu130" ]]; then
        echo "nvidia/cuda:13.0.0-devel-ubuntu24.04"
    else
        echo "nvidia/cuda:12.8.0-devel-ubuntu24.04"
    fi
}

ensure_env() {
    if [[ ! -f .env ]]; then
        info "No .env found -- copying from .env.example"
        cp .env.example .env
        info "Edit docker/.env to customise options."
    fi
}

check_ports() {
    local conflict=false
    local ports=("${SIM_PORT:-8100}" "${APP_PORT:-8000}" "${FRONTEND_PORT:-3000}")
    for port in "${ports[@]}"; do
        if ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .; then
            local owner
            owner=$(docker ps --format '{{.Names}}' --filter "publish=${port}" 2>/dev/null | head -1)
            if [[ -z "$owner" ]]; then
                error "Port ${port} is already in use by a non-Docker process."
            else
                error "Port ${port} is already in use by container '${owner}'."
            fi
            error "  Change the port in docker/.env and retry."
            conflict=true
        fi
    done
    [[ "$conflict" == false ]]
}

wait_healthy() {
    local service="$1" url="$2" timeout="${3:-300}"
    local elapsed=0
    info "Waiting for ${service} at ${url} (timeout: ${timeout}s)..."
    while (( elapsed < timeout )); do
        if curl -sf "$url" &>/dev/null; then
            info "${service} is healthy."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    error "${service} did not become healthy within ${timeout}s."
    return 1
}

# -- Actions ------------------------------------------------------------------

do_ui() {
    info "Rebuilding frontend only..."
    docker compose -f docker-compose.yml build frontend
    docker compose -f docker-compose.yml up -d --no-deps frontend
    info "Frontend updated. http://localhost:${FRONTEND_PORT:-3000}"
}

do_sim() {
    if [[ -z "${SIM_CUDA_DEVEL:-}" ]]; then
        SIM_CUDA_DEVEL=$(detect_sim_cuda_devel)
    fi
    if [[ -z "${SIM_TORCH_BACKEND:-}" ]]; then
        SIM_TORCH_BACKEND=$(detect_sim_torch_backend)
    fi
    export SIM_CUDA_DEVEL SIM_TORCH_BACKEND

    info "Rebuilding sim-server only..."
    docker compose -f docker-compose.yml build sim-server
    docker compose -f docker-compose.yml up -d --no-deps sim-server

    wait_healthy "sim-server" "http://localhost:${SIM_PORT:-8100}/sim/health" 300
    info "sim-server updated. http://localhost:${SIM_PORT:-8100}"
}

do_down() {
    info "Stopping ${SERVICES[*]}..."
    docker compose -f docker-compose.yml stop "${SERVICES[@]}"
    docker compose -f docker-compose.yml rm -f "${SERVICES[@]}"
    info "Done."
}

do_logs() {
    docker compose -f docker-compose.yml logs -f "${SERVICES[@]}"
}

do_up() {
    if ! command -v docker &>/dev/null || ! docker info &>/dev/null; then
        error "Docker is not running or not installed."
        exit 1
    fi
    if ! nvidia-smi &>/dev/null; then
        error "No NVIDIA GPU detected."
        exit 1
    fi

    ensure_env

    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a

    check_ports || exit 1

    if [[ -z "${SIM_CUDA_DEVEL:-}" ]]; then
        SIM_CUDA_DEVEL=$(detect_sim_cuda_devel)
        info "Auto-detected sim CUDA devel: ${SIM_CUDA_DEVEL}"
    fi
    if [[ -z "${SIM_TORCH_BACKEND:-}" ]]; then
        SIM_TORCH_BACKEND=$(detect_sim_torch_backend)
        info "Auto-detected sim torch backend: ${SIM_TORCH_BACKEND}"
    fi
    export SIM_CUDA_DEVEL SIM_TORCH_BACKEND

    # VLLM_IMAGE must be exported even though inference-server won't run,
    # because docker-compose.yml references it and compose validates all envs.
    export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

    local build_flag=""
    [[ "$FORCE_BUILD" == true ]] && build_flag="--build"

    # Start sim-server first (no dependencies)
    info "Starting sim-server..."
    docker compose -f docker-compose.yml up -d $build_flag sim-server

    wait_healthy "sim-server" "http://localhost:${SIM_PORT:-8100}/sim/health" 300

    # Start app-server without dependency check (skips inference-server)
    info "Starting app-server..."
    docker compose -f docker-compose.yml up -d --no-deps $build_flag app-server

    wait_healthy "app-server" "http://localhost:${APP_PORT:-8000}/api/health" 60

    # Start frontend
    info "Starting frontend..."
    docker compose -f docker-compose.yml up -d --no-deps $build_flag frontend

    echo ""
    echo "======================================"
    echo "  Stack is running (no inference)"
    echo "======================================"
    echo ""
    echo "  Service        URL"
    echo "  ---------      ---"
    echo "  sim-server     http://localhost:${SIM_PORT:-8100}/sim/health"
    echo "  app-server     http://localhost:${APP_PORT:-8000}/api/health"
    echo "  frontend       http://localhost:${FRONTEND_PORT:-3000}"
    echo ""
    echo "  To add inference: bash inference.sh"
    echo "  Logs: bash stack.sh --logs"
    echo "  Stop: bash stack.sh --down"
    echo "======================================"
}

# -- Main ---------------------------------------------------------------------
if [[ "$UI_ONLY" == true ]]; then
    do_ui
    exit 0
fi

if [[ "$SIM_ONLY" == true ]]; then
    set -a; source .env 2>/dev/null || true; set +a
    do_sim
    exit 0
fi

case "$ACTION" in
    up)   do_up   ;;
    down) do_down ;;
    logs) do_logs ;;
esac
