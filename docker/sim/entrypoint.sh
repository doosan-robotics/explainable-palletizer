#!/usr/bin/env bash
set -euo pipefail

echo "[sim-server] Starting sim server on main thread..."
echo "[sim-server] SIM_HOST=${SIM_HOST:-0.0.0.0} SIM_PORT=${SIM_PORT:-8100}"
echo "[sim-server] SIM_LOAD_ROBOT=${SIM_LOAD_ROBOT:-true} SIM_SPAWN_BOXES=${SIM_SPAWN_BOXES:-true}"

# Isaac Sim requires both libgomp variants to be preloaded before Python starts
# to avoid a conflict between the system libgomp and PyTorch's bundled copy.
# This is required on all platforms where Isaac Sim runs with PyTorch.
_SYSGOMP="/lib/$(uname -m)-linux-gnu/libgomp.so.1"
_TORCHGOMP=$(find /workspace/.venv -name 'libgomp*.so*' -path '*/torch.libs/*' 2>/dev/null | head -1 || true)
if [[ -f "${_SYSGOMP}" ]]; then
    if [[ -n "${_TORCHGOMP}" ]]; then
        export LD_PRELOAD="${LD_PRELOAD:-}:${_SYSGOMP}:${_TORCHGOMP}"
    else
        export LD_PRELOAD="${LD_PRELOAD:-}:${_SYSGOMP}"
    fi
fi
unset _SYSGOMP _TORCHGOMP

# NVIDIA Vulkan requires a display for vkCreateInstance even in headless mode.
# Start a virtual X framebuffer so the RTX renderer can initialise.
Xvfb :99 -screen 0 1280x720x24 &>/dev/null &
export DISPLAY=:99
echo "[sim-server] Xvfb started on DISPLAY=$DISPLAY"

exec uv run --no-sync python -m drp_sim.server
