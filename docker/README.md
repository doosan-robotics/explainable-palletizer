# Docker Pipeline

## Quick Start

```bash
# One command -- checks prerequisites, builds, and starts everything:
bash launch.sh

# Or from the repo root:
make docker-up
```

The launch script will:
1. Verify Docker, nvidia-container-toolkit, and GPU availability
2. Copy `.env.example` to `.env` if not present
3. Build images and start all 4 services
4. Wait for health checks and print a status table

## Test Mode

Run the full pipeline with a tiny free model (no HuggingFace token needed):

```bash
bash launch.sh --test    # or: make docker-test
```

Test mode uses `facebook/opt-125m` (~250MB, ~0.2 GPU memory) for the inference
server while keeping the real Isaac Sim running. Use this to validate the full
stack end-to-end on any machine.

## Services

| Service | Port | Description |
|---------|------|-------------|
| sim-server | 8100 | Isaac Sim + FastAPI |
| inference-server | 8200 | vLLM (OpenAI-compatible) |
| app-server | 8000 | FastAPI orchestrator |
| frontend | 3000 | nginx reverse proxy |

## Health Checks

```bash
curl http://localhost:8000/api/health  # app-server
curl http://localhost:8100/sim/health  # sim-server
curl http://localhost:8200/health      # inference-server
curl http://localhost:3000/api/status  # full system status via frontend
```

## vLLM Image Selection

`launch.sh` automatically selects the right vLLM image for your hardware:

**x86_64** (selected by GPU compute capability):

| GPU architecture | Compute capability | Image selected |
|---|---|---|
| Ampere (A100, RTX 3xxx) | sm_80 – sm_87 | `vllm/vllm-openai:latest` |
| Ada Lovelace (RTX 4xxx) | sm_89 | `vllm/vllm-openai:latest` |
| Hopper (H100, H200) | sm_90 | `vllm/vllm-openai:latest` |
| Blackwell (B100, B200, RTX 5xxx) | sm_100+ | `vllm/vllm-openai:latest-cu130` |

**aarch64 / Jetson** (selected by board model from `/proc/device-tree/model`):

| Board | Image selected |
|---|---|
| Thor | `nvcr.io/nvidia/vllm:26.01-py3` |
| AGX Orin, Orin NX, Orin Nano | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-24.04` |

For mixed-GPU setups or boards not listed above, override in `docker/.env`:

```
VLLM_IMAGE=vllm/vllm-openai:latest-cu130
```

## GPU Configuration

Edit `.env` to assign GPUs per service:

```
SIM_GPU_DEVICE=0
INFERENCE_GPU_DEVICE=1
APP_GPU_DEVICE=0
```

Single-GPU dev: leave all at `0`.

## Launch Script Flags

| Flag | Description |
|------|-------------|
| (none) | Full launch: build + start all services |
| `--test` | Test mode: real Isaac Sim + tiny model, no HF token needed |
| `--build` | Force rebuild images |
| `--down` | Stop and remove containers |
| `--logs` | Follow container logs |

## Teardown

```bash
bash launch.sh --down    # or: make docker-down
docker compose down -v   # also removes volumes (model cache)
```
