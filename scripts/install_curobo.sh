#!/usr/bin/env bash
# Install cuRobo from source into the workspace venv.
# cuRobo is not on PyPI and cannot be managed by uv sync.
#
# Usage:
#   bash scripts/install_curobo.sh            # clone to ~/curobo (default)
#   CUROBO_DIR=/opt/curobo bash scripts/install_curobo.sh
#
# Prerequisites: make init must have been run first so that CUDA torch is
# already present in the venv (required for --no-build-isolation build).
set -euo pipefail

CUROBO_DIR="${CUROBO_DIR:-$HOME/curobo}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.7}"
CUROBO_REPO="https://github.com/NVlabs/curobo.git"

# ── Helpers ──────────────────────────────────────────────────────────────────

red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
info()  { echo "[install_curobo] $*"; }

die() { red "ERROR: $*" >&2; exit 1; }

# ── Preflight checks ─────────────────────────────────────────────────────────

info "Checking prerequisites..."

# 1. nvcc
if ! command -v nvcc &>/dev/null; then
    die "nvcc not found. Install CUDA toolkit 12.x and ensure nvcc is on PATH."
fi
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if (( CUDA_MAJOR < 12 )); then
    die "CUDA 12.x or 13.x required, found $CUDA_VER."
fi
info "nvcc $CUDA_VER OK"

# 2. venv exists (make init was run)
if [[ ! -d ".venv" ]]; then
    die ".venv not found. Run 'make init' first."
fi

# 3. CUDA torch installed
TORCH_VER=$(uv pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)
if [[ -z "$TORCH_VER" ]]; then
    die "torch not found in venv. Run 'make init' first."
fi
TORCH_LOCATION=$(uv pip show torch 2>/dev/null | grep "^Location:" | awk '{print $2}' || true)
if [[ "$TORCH_LOCATION" != *"cu"* ]] && ! python3 -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    info "Warning: CUDA torch may not be installed. Continuing anyway..."
fi
info "torch $TORCH_VER OK"

# ── Clone ────────────────────────────────────────────────────────────────────

if [[ -d "$CUROBO_DIR/.git" ]]; then
    info "cuRobo already cloned at $CUROBO_DIR — skipping clone."
else
    info "Cloning cuRobo ${CUROBO_TAG} into $CUROBO_DIR ..."
    git clone --branch "$CUROBO_TAG" --depth 1 "$CUROBO_REPO" "$CUROBO_DIR"
fi

# ── Install ──────────────────────────────────────────────────────────────────

info "Installing cuRobo from $CUROBO_DIR ..."
info "(This compiles CUDA extensions — expect 5-15 minutes on first run)"

# cuRobo's pyproject.toml declares build-system.requires including setuptools,
# setuptools_scm, wheel, and torch.  With --no-build-isolation these are NOT
# auto-installed, so we must provide them explicitly.
uv pip install setuptools setuptools_scm scikit-build-core cmake

# ninja speeds up CUDA extension compilation; fall back silently if unavailable.
uv pip install ninja 2>/dev/null || true

# TORCH_CUDA_ARCH_LIST tells PyTorch which GPU architectures to compile for.
# Required in GPU-less environments (Docker build, CI) where the physical GPU
# cannot be queried -- without it, _get_cuda_arch_flags() returns an empty list
# and crashes with IndexError: list index out of range.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    if (( CUDA_MAJOR >= 13 )); then
        # Hopper + Blackwell
        TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX"
    else
        # Volta, Turing, Ampere, Ada, Hopper
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    fi
    export TORCH_CUDA_ARCH_LIST
    info "Auto-set TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
fi

WORKSPACE_DIR="$PWD"
cd "$CUROBO_DIR"

# In Docker builds use a non-editable install without dev extras to keep the
# image lean and avoid fragile path-based references from editable installs.
if [[ "${DOCKER_BUILD:-}" == "1" ]]; then
    uv pip install "." --no-build-isolation
else
    uv pip install -e ".[dev]" --no-build-isolation
fi

# Return to workspace root before verifying: running `uv run` from CUROBO_DIR
# would pick up cuRobo's own pyproject.toml and resolve a different environment
# (not the workspace venv where curobo was just installed).
cd "$WORKSPACE_DIR"

# ── Verify ───────────────────────────────────────────────────────────────────

info "Verifying installation..."
VERSION=$(uv run python -c "import curobo; print(curobo.__version__)" 2>/dev/null || true)
if [[ -z "$VERSION" ]]; then
    die "cuRobo import failed after installation."
fi

green "cuRobo $VERSION installed successfully."
