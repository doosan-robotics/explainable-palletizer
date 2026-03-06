# ── Single-stage build on devel image ─────────────────────────────────────────
# Isaac Sim extensions ship native .so plugins (libomni.usd.so, etc.) that
# depend on libraries only present in the devel image.  A multi-stage copy to
# cuda:runtime silently drops these, causing cascade import failures at startup.
# Using devel for runtime is larger (~4 GB overhead) but guarantees all native
# deps are present.
ARG CUDA_DEVEL=nvidia/cuda:12.8.0-devel-ubuntu24.04

FROM ${CUDA_DEVEL}

# TORCH_BACKEND must be declared after FROM to be available in RUN commands.
ARG TORCH_BACKEND=cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git \
    libgl1 libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# -- Dependency layer (10GB+, changes only when pyproject.toml or uv.lock change)
# Copy only pyproject.toml files and create stub packages so hatchling's
# build-system check passes. Real source code is copied AFTER uv sync so
# that code changes never invalidate the expensive dependency layer.
COPY pyproject.toml uv.lock ./
COPY sim/pyproject.toml sim/pyproject.toml
COPY motion/pyproject.toml motion/pyproject.toml
COPY app/pyproject.toml app/pyproject.toml
RUN mkdir -p sim/src/drp_sim motion/src/drp_motion app/src/dr_ai_palletizer && \
    touch sim/src/drp_sim/__init__.py motion/src/drp_motion/__init__.py app/src/dr_ai_palletizer/__init__.py

# Install sim package and app server with simulator and API deps.
# isaacsim is ~10GB -- Docker layer caching is critical here.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.11 --package drp-sim --extra sim --extra api --no-dev

# -- CUDA torch layer ----------------------------------------------------------
COPY scripts/install_cuda_torch.sh scripts/install_cuda_torch.sh
COPY scripts/install_curobo.sh scripts/install_curobo.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    TORCH_BACKEND=${TORCH_BACKEND} bash scripts/install_cuda_torch.sh

# -- cuRobo compilation layer (5-15 min, changes rarely) -----------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    CUROBO_DIR=/workspace/curobo DOCKER_BUILD=1 bash scripts/install_curobo.sh

# -- Runtime libs not in the base image ----------------------------------------
# Installed in a late layer to avoid invalidating the expensive dep layers above.
# libGLU: iray/neuraylib plugin (libneuray.so)
# libvulkan1: Vulkan ICD loader (RTX rendering)
# libxt6: MaterialX (libMaterialXRenderHw.so, libMaterialXRenderGlsl.so)
# xvfb + xauth: virtual X display — NVIDIA Vulkan requires a display for vkCreateInstance
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa libvulkan1 libxt6 xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

# Vulkan ICD for NVIDIA -- the nvidia-container-toolkit mounts the real
# driver libs at runtime, but the Vulkan loader still needs a JSON manifest
# telling it which library to dlopen.  The host ICD JSON is sometimes
# mounted by the toolkit, but not always; create a fallback so Isaac Sim's
# RTX renderer can find the GPU.
RUN mkdir -p /etc/vulkan/icd.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}\n' \
    > /etc/vulkan/icd.d/nvidia_icd.json

# -- Source code and assets (changes frequently, fast to copy) -----------------
COPY sim/src/ sim/src/
COPY sim/configs/ sim/configs/
COPY sim/robot/ sim/robot/
COPY sim/usd/ sim/usd/
COPY motion/src/ motion/src/
COPY app/src/ app/src/

COPY docker/sim/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8100

ENTRYPOINT ["/entrypoint.sh"]
