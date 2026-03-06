#!/usr/bin/env bash
# Install the correct CUDA-enabled PyTorch into the current venv.
# Called by `make init` and `make sync`.
#
# Supports CUDA 12.x (cu128 backend) and CUDA 13.x (cu130 backend).
# Falls back to CPU torch (already installed by uv sync) when no GPU is found.
set -euo pipefail

# Version range matching the override in the root pyproject.toml.
TORCH_SPEC="torch>=2.9,<2.10"
TORCHVISION_SPEC="torchvision>=0.24,<0.25"
BACKEND_FILE=".torch-backend"

# ── Detect CUDA ───────────────────────────────────────────────────────────────

detect_backend() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"
        return
    fi

    local cuda_version
    cuda_version=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)

    if [[ -z "$cuda_version" ]]; then
        echo "cpu"
        return
    fi

    local major
    major=$(echo "$cuda_version" | cut -d. -f1)

    if (( major >= 13 )); then
        echo "cu130"
    elif (( major == 12 )); then
        echo "cu128"
    else
        echo "cpu"
    fi
}

# Allow manual override: TORCH_BACKEND=cu130 make init
BACKEND="${TORCH_BACKEND:-$(detect_backend)}"

if [[ "$BACKEND" == "cpu" ]]; then
    echo "No compatible CUDA driver found. Keeping CPU torch."
    echo "cpu" > "$BACKEND_FILE"
    exit 0
fi

# ── Install CUDA torch ───────────────────────────────────────────────────────

INDEX_URL="https://download.pytorch.org/whl/${BACKEND}"

echo "Installing torch+${BACKEND} from ${INDEX_URL} ..."

uv pip install \
    "$TORCH_SPEC" \
    "$TORCHVISION_SPEC" \
    --index-url "$INDEX_URL" \
    --reinstall-package torch \
    --reinstall-package torchvision

# Pin NVIDIA runtime libs to versions compatible with both torch 2.9 and isaacsim.
# Without --no-deps these would be dragged to incompatible versions.
# Ref: https://github.com/vesoma-main/holosoma/commit/7c57d951
uv pip install --no-deps \
    'nvidia-nccl-cu12==2.27.5' \
    'nvidia-nvshmem-cu12==3.3.20' \
    'nvidia-nvtx-cu12==12.8.90' \
    'triton==3.5.0'

INSTALLED_VERSION=$(uv pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}')
echo "$BACKEND" > "$BACKEND_FILE"
echo "Done: torch==${INSTALLED_VERSION} installed (backend: ${BACKEND})."
