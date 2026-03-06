#!/usr/bin/env bash
# nvidia-zenith Docker pipeline launcher.
#
# Usage:
#   bash launch.sh            # full pipeline (build + start all services)
#   bash launch.sh --test     # test mode (real Isaac Sim + tiny model, no HF token needed)
#   bash launch.sh --down     # stop and remove containers
#   bash launch.sh --build    # force rebuild images
#   bash launch.sh --logs     # follow container logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Defaults -----------------------------------------------------------------
ACTION="up"
FORCE_BUILD=false
TEST_MODE=false

# -- Parse flags --------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --down)   ACTION="down" ;;
        --logs)   ACTION="logs" ;;
        --build)  FORCE_BUILD=true ;;
        --test)   TEST_MODE=true ;;
        --help|-h)
            sed -n '2,10s/^# //p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            exit 1
            ;;
    esac
done

# -- Helpers ------------------------------------------------------------------
info()  { echo "[zenith] $*"; }
error() { echo "[zenith] ERROR: $*" >&2; }

check_ports() {
    local conflict=false
    local ports=("${SIM_PORT:-8100}" "${INFERENCE_PORT:-8200}" "${APP_PORT:-8000}" "${FRONTEND_PORT:-3000}")

    for port in "${ports[@]}"; do
        if ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .; then
            # Check if the conflict is one of our own containers
            local owner
            owner=$(docker ps --format '{{.Names}}' --filter "publish=${port}" 2>/dev/null | head -1)
            if [[ -z "$owner" ]]; then
                error "Port ${port} is already in use by a non-Docker process."
            else
                error "Port ${port} is already in use by container '${owner}'."
            fi
            error "  Change the port in docker/.env (e.g. FRONTEND_PORT=3001) and retry."
            conflict=true
        fi
    done
    [[ "$conflict" == false ]]
}

check_prereqs() {
    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. See https://docs.docker.com/engine/install/"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        error "Docker daemon is not running or current user lacks permissions."
        exit 1
    fi
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        if ! command -v nvidia-container-toolkit &>/dev/null && \
           ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
            error "nvidia-container-toolkit not found. GPU containers will fail."
            error "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
    fi
    if ! nvidia-smi &>/dev/null; then
        error "nvidia-smi failed. No NVIDIA GPU detected or driver not loaded."
        exit 1
    fi
}

_detect_cc_major() {
    # Return the GPU compute capability major version (x86_64 only).
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ' || echo "8.9")
    echo "$cc" | cut -d. -f1
}

detect_vllm_image() {
    # Return the appropriate full vLLM image reference for this host.
    # x86_64:           select by host CUDA version (nvidia-smi header)
    # aarch64 (Jetson): select by board model (/proc/device-tree/model)
    local arch
    arch=$(uname -m)

    if [[ "$arch" == "aarch64" ]]; then
        local model
        model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
        if echo "$model" | grep -qi "thor"; then
            echo "nvcr.io/nvidia/vllm:26.01-py3"
        else
            # AGX Orin, Orin NX, Orin Nano
            echo "ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-24.04"
        fi
    else
        # Select based on GPU compute capability:
        #   < 10  (Ada sm_89, Hopper sm_90, Ampere sm_86, ...) -> CUDA 12 image
        #   >= 10 (Blackwell B100/B200 sm_100, RTX 5xxx sm_100/sm_120, ...) -> CUDA 13 image
        # Override via VLLM_IMAGE in docker/.env for heterogeneous GPU setups.
        if (( $(_detect_cc_major) >= 10 )); then
            echo "vllm/vllm-openai:latest-cu130"
        else
            echo "vllm/vllm-openai:latest"
        fi
    fi
}

detect_sim_torch_backend() {
    # Return cu128 or cu130 for the sim container based on the host GPU.
    # Mirrors detect_vllm_image() -- same compute capability threshold.
    # Override via SIM_TORCH_BACKEND in docker/.env.
    local arch
    arch=$(uname -m)

    if [[ "$arch" == "aarch64" ]]; then
        local model
        model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
        if echo "$model" | grep -qi "thor"; then
            echo "cu130"
        else
            echo "cu128"
        fi
    else
        if (( $(_detect_cc_major) >= 10 )); then
            echo "cu130"
        else
            echo "cu128"
        fi
    fi
}

detect_sim_cuda_devel() {
    # Return the nvidia/cuda devel base image for the sim-server container.
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
        info "Edit docker/.env to customise model, GPU devices, and ports."
    fi
    # Propagate HF_TOKEN from the shell environment into .env if not already set there.
    # This lets users pass the token via `HF_TOKEN=xxx bash launch.sh` without editing .env.
    if [[ -n "${HF_TOKEN:-}" ]] && ! grep -q '^HF_TOKEN=.' .env 2>/dev/null; then
        echo "HF_TOKEN=${HF_TOKEN}" >> .env
    fi
}

check_hf_token() {
    # Skip check in test mode (facebook/opt-125m is not gated)
    [[ "$TEST_MODE" == true ]] && return 0
    # Source .env to get the current token value
    local token="${HF_TOKEN:-}"
    if [[ -z "$token" ]]; then
        # Try reading from .env directly in case set -a hasn't run yet
        token=$(grep -E '^HF_TOKEN=.' .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
    fi
    if [[ -z "$token" ]]; then
        error "HF_TOKEN is not set. nvidia/Cosmos-Reason2-* models require HuggingFace authentication."
        error "  1. Get a token at https://huggingface.co/settings/tokens"
        error "  2. Accept the model licence at https://huggingface.co/nvidia/Cosmos-Reason2-2B"
        error "  3. Add HF_TOKEN=<your_token> to docker/.env and retry."
        exit 1
    fi
}

compose_files() {
    local files=("-f" "docker-compose.yml")
    if [[ "$TEST_MODE" == true ]]; then
        files+=("-f" "docker-compose.test.yml")
    fi
    echo "${files[@]}"
}

wait_healthy() {
    local service="$1" url="$2" timeout="${3:-300}"
    local elapsed=0
    info "Waiting for $service at $url (timeout: ${timeout}s)..."
    while (( elapsed < timeout )); do
        if curl -sf "$url" &>/dev/null; then
            info "$service is healthy."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    error "$service did not become healthy within ${timeout}s."
    return 1
}

print_status() {
    echo ""
    echo "======================================"
    echo "  nvidia-zenith pipeline is running"
    echo "======================================"
    if [[ "$TEST_MODE" == true ]]; then
        echo "  Mode:       TEST (real Isaac Sim + tiny model)"
    else
        echo "  Mode:       FULL"
    fi
    echo ""
    echo "  Service            URL"
    echo "  ---------          ---"
    echo "  sim-server         http://localhost:${SIM_PORT:-8100}/sim/health"
    echo "  sim-server camera  ws://localhost:${SIM_PORT:-8100}/sim/camera/stream"
    echo "  inference-server   http://localhost:${INFERENCE_PORT:-8200}/health"
    echo "  app-server         http://localhost:${APP_PORT:-8000}/api/health"
    echo "  frontend           http://localhost:${FRONTEND_PORT:-3000}"
    echo ""
    echo "  Logs:  cd docker && docker compose logs -f"
    echo "  Stop:  cd docker && bash launch.sh --down"
    echo "======================================"
}

# -- Actions ------------------------------------------------------------------

do_down() {
    info "Stopping containers..."
    # shellcheck disable=SC2046
    docker compose $(compose_files) down
    info "Done."
}

do_logs() {
    # shellcheck disable=SC2046
    docker compose $(compose_files) logs -f
}

do_up() {
    check_prereqs
    ensure_env

    # Source .env so we can read port variables for status output
    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a

    check_hf_token
    check_ports || exit 1

    # Resolve vLLM image: honour .env override, otherwise auto-detect
    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        VLLM_IMAGE=$(detect_vllm_image)
        info "Auto-detected vLLM image: ${VLLM_IMAGE}"
    else
        info "Using vLLM image from .env: ${VLLM_IMAGE}"
    fi
    export VLLM_IMAGE

    # Resolve sim CUDA base images and torch backend
    if [[ -z "${SIM_CUDA_DEVEL:-}" ]]; then
        SIM_CUDA_DEVEL=$(detect_sim_cuda_devel)
        info "Auto-detected sim CUDA devel: ${SIM_CUDA_DEVEL}"
    else
        info "Using sim CUDA devel from .env: ${SIM_CUDA_DEVEL}"
    fi
    if [[ -z "${SIM_TORCH_BACKEND:-}" ]]; then
        SIM_TORCH_BACKEND=$(detect_sim_torch_backend)
        info "Auto-detected sim torch backend: ${SIM_TORCH_BACKEND}"
    else
        info "Using sim torch backend from .env: ${SIM_TORCH_BACKEND}"
    fi
    export SIM_CUDA_DEVEL SIM_TORCH_BACKEND

    local build_flag=""
    if [[ "$FORCE_BUILD" == true ]]; then
        build_flag="--build"
    fi

    if [[ "$TEST_MODE" == true ]]; then
        info "Starting in TEST mode (real Isaac Sim + tiny model)..."
    else
        info "Starting full pipeline..."
    fi

    # shellcheck disable=SC2046
    docker compose $(compose_files) up -d $build_flag

    info "Containers started. Waiting for health checks..."

    local ok=true
    wait_healthy "sim-server"       "http://localhost:${SIM_PORT:-8100}/sim/health"    300 || ok=false
    wait_healthy "inference-server" "http://localhost:${INFERENCE_PORT:-8200}/health"  300 || ok=false
    wait_healthy "app-server"       "http://localhost:${APP_PORT:-8000}/api/health"     60 || ok=false

    if [[ "$ok" == true ]]; then
        print_status
    else
        error "Some services failed to start. Check logs: docker compose logs"
        exit 1
    fi
}

# -- Main ---------------------------------------------------------------------
case "$ACTION" in
    up)   do_up   ;;
    down) do_down ;;
    logs) do_logs ;;
esac
