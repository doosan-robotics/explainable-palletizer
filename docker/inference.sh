#!/usr/bin/env bash
# Run only the inference-server (nvidia/Cosmos-Reason2-2B) in Docker.
#
# Usage:
#   bash inference.sh           # start inference-server (2B model)
#   bash inference.sh --down    # stop and remove the container
#   bash inference.sh --logs    # follow container logs
#   bash inference.sh --build   # force rebuild image before start
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Defaults -----------------------------------------------------------------
ACTION="up"
FORCE_BUILD=false

# -- Parse flags --------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --down)   ACTION="down" ;;
        --logs)   ACTION="logs" ;;
        --build)  FORCE_BUILD=true ;;
        --help|-h)
            sed -n '2,8s/^# //p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            exit 1
            ;;
    esac
done

# -- Helpers ------------------------------------------------------------------
info()  { echo "[inference] $*"; }
error() { echo "[inference] ERROR: $*" >&2; }

_detect_cc_major() {
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ' || echo "8.9")
    echo "$cc" | cut -d. -f1
}

detect_vllm_image() {
    local arch
    arch=$(uname -m)

    if [[ "$arch" == "aarch64" ]]; then
        local model
        model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
        if echo "$model" | grep -qi "thor"; then
            echo "nvcr.io/nvidia/vllm:26.01-py3"
        else
            echo "ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-24.04"
        fi
    else
        if (( $(_detect_cc_major) >= 10 )); then
            echo "vllm/vllm-openai:latest-cu130"
        else
            echo "vllm/vllm-openai:latest"
        fi
    fi
}

ensure_env() {
    if [[ ! -f .env ]]; then
        info "No .env found -- copying from .env.example"
        cp .env.example .env
        info "Edit docker/.env to set HF_TOKEN and other options."
    fi
    if [[ -n "${HF_TOKEN:-}" ]] && ! grep -q '^HF_TOKEN=.' .env 2>/dev/null; then
        echo "HF_TOKEN=${HF_TOKEN}" >> .env
    fi
}

check_hf_token() {
    local token="${HF_TOKEN:-}"
    if [[ -z "$token" ]]; then
        token=$(grep -E '^HF_TOKEN=.' .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
    fi
    if [[ -z "$token" ]]; then
        error "HF_TOKEN is not set. nvidia/Cosmos-Reason2-2B requires HuggingFace authentication."
        error "  1. Get a token at https://huggingface.co/settings/tokens"
        error "  2. Accept the licence at https://huggingface.co/nvidia/Cosmos-Reason2-2B"
        error "  3. Add HF_TOKEN=<your_token> to docker/.env and retry."
        exit 1
    fi
}

wait_healthy() {
    local url="${1}" timeout="${2:-300}"
    local elapsed=0
    info "Waiting for inference-server at ${url} (timeout: ${timeout}s)..."
    while (( elapsed < timeout )); do
        if curl -sf "$url" &>/dev/null; then
            info "inference-server is healthy."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    error "inference-server did not become healthy within ${timeout}s."
    return 1
}

# -- Actions ------------------------------------------------------------------

do_down() {
    info "Stopping inference-server..."
    docker compose -f docker-compose.yml stop inference-server
    docker compose -f docker-compose.yml rm -f inference-server
    info "Done."
}

do_logs() {
    docker compose -f docker-compose.yml logs -f inference-server
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

    check_hf_token

    if [[ -z "${VLLM_IMAGE:-}" ]]; then
        VLLM_IMAGE=$(detect_vllm_image)
        info "Auto-detected vLLM image: ${VLLM_IMAGE}"
    else
        info "Using vLLM image from .env: ${VLLM_IMAGE}"
    fi
    export VLLM_IMAGE

    local model="${INFERENCE_MODEL:-nvidia/Cosmos-Reason2-2B}"
    local port="${INFERENCE_PORT:-8200}"
    info "Model : ${model}"
    info "Port  : ${port}"

    local build_flag=""
    [[ "$FORCE_BUILD" == true ]] && build_flag="--build"

    docker compose -f docker-compose.yml up -d $build_flag inference-server

    wait_healthy "http://localhost:${port}/health" 300

    echo ""
    echo "======================================"
    echo "  inference-server ready"
    echo "======================================"
    echo "  Model : ${model}"
    echo "  URL   : http://localhost:${port}"
    echo "  Health: http://localhost:${port}/health"
    echo ""
    echo "  Logs: bash inference.sh --logs"
    echo "  Stop: bash inference.sh --down"
    echo "======================================"
}

# -- Main ---------------------------------------------------------------------
case "$ACTION" in
    up)   do_up   ;;
    down) do_down ;;
    logs) do_logs ;;
esac
