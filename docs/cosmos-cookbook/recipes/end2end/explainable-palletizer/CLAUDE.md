# end2end

## What This Recipe Does
Runs the Doosan Robotics "explainable palletizer" — a closed-loop demo where Cosmos Reason 2 reasons about box contents, condition, and fragility from camera crops, then drives a virtual Doosan P3020 arm in Isaac Sim. cuRobo motion planning runs inside `sim-server`, and every decision is backed by an auditable reasoning trace plus structured action JSON.

## Model
`nvidia/Cosmos-Reason2-8B` (default) or `nvidia/Cosmos-Reason2-2B` (24 GB-class GPUs).
Optional LoRA adapter: `yurirocha15/Cosmos-Reason2-8B-palletizer-lora` (or `-2B-` variant).

## Entry Points
<!-- This recipe ships as a Docker Compose stack from the upstream repo, not a Python entry point. -->
```bash
git clone https://github.com/doosan-robotics/explainable-palletizer.git
cd explainable-palletizer
cp docker/.env.example docker/.env       # then edit docker/.env to set HF_TOKEN
make adapters                              # optional — pulls LoRA adapters
make docker-up                             # build and start all 4 services
make docker-logs                           # follow logs
# UI: http://localhost:3000
make docker-down                           # stop and remove containers
```

For a no-token smoke run (real Isaac Sim, tiny stand-in model):
```bash
make docker-test
```

## Data Source
**Access:** Gated
**Size:** ~16 GB (Cosmos-Reason2-8B) + ~150 MB (LoRA adapter) — fully cached after first run
**License:** [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Reason2-8B) (base model); LoRA adapters Apache-2.0 (per upstream repo)

```bash
huggingface-cli download nvidia/Cosmos-Reason2-8B
```

The base model is gated — accept the license at https://huggingface.co/nvidia/Cosmos-Reason2-8B before downloading. The palletizer LoRA adapters at `yurirocha15/Cosmos-Reason2-{2B,8B}-palletizer-lora` are public and pulled automatically by `make adapters`.

## Compute Requirements
<!--
  Agent: Before running any entry points, check whether the user has sufficient
  local GPU resources. If not, ask:
    "This recipe requires at least the compute listed here. You can provision
     a matching instance on NVIDIA Brev. Do you have a Brev API token? If so, please
     paste it and I will set up the environment for you."
  Note: the first Docker build is long-running (30+ min). The agent should launch
  the stack and inform the user how to monitor progress — not wait for completion.
-->
- **Minimum:** 1× NVIDIA RTX 4090 (24 GB) running the 2B model. NVIDIA driver ≥ 585, CUDA ≥ 12.8.
- **Recommended:** 1× H200 / RTX PRO 6000 / Jetson Thor — runs 8B comfortably with VLLM_GPU_MEMORY_UTILIZATION=0.5 alongside Isaac Sim.
- **Multi-GPU:** on data-center GPUs without RT cores, run cuRobo on a separate GPU from Isaac Sim to avoid CUDA context conflicts. Set `CUROBO_GPU_DEVICE=1`.
- **Disk:** ~30 GB for cached HF weights + Docker image layers. Set `HF_CACHE_DIR` to a host directory with adequate space.
- **Time:** first `make docker-up` takes 30+ minutes (CUDA extension compile + weight download); subsequent launches start in ~1–2 minutes.

## Dependencies

```
docker (with Compose V2)
nvidia-container-toolkit
uv >= 0.4
huggingface-hub[cli]
```

vLLM, Isaac Sim, cuRobo, and the Python workspace (`sim`, `motion`, `app`) are managed inside the Docker images by the upstream repo's `Makefile` and `docker-compose.yml`.

## Required Environment Variables
<!--
  List variable names and descriptions only. Never write credential values here.
  All values must be set in the user's docker/.env before running `make docker-up`.
-->
| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for gated Cosmos models — required for non-test launches |
| `INFERENCE_MODEL` | `nvidia/Cosmos-Reason2-2B` or `nvidia/Cosmos-Reason2-8B` — defaults to 8B |
| `LORA_ADAPTER_PATH` | LoRA path inside container (e.g. `/adapters/8B`) — leave empty for base model only |
| `LORA_MODEL` | LoRA name exposed by vLLM (e.g. `palletize`); must match between inference-server and app-server |
| `VLLM_MAX_MODEL_LEN` | Max context length — keep ≥ 5120 (the app sends ~2200 input + requests up to 2048 output tokens) |
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory for vLLM — default `0.5` |
| `VLLM_REASONING_PARSER` | vLLM reasoning parser — default `qwen3` |
| `HF_CACHE_DIR` | Host directory bind-mounted as the HF cache — strongly recommended to avoid re-downloading weights on every rebuild |
| `SIM_GPU_DEVICE` / `INFERENCE_GPU_DEVICE` / `CUROBO_GPU_DEVICE` | GPU device IDs for multi-GPU setups |
| `SIM_PORT` / `INFERENCE_PORT` / `APP_PORT` / `FRONTEND_PORT` | Host ports for `sim-server`, `inference-server`, `app-server`, and `frontend` |

## Setup Prerequisites
- [ ] NVIDIA driver ≥ 585 installed and `nvidia-smi` works on the host
- [ ] `nvidia-container-toolkit` installed and configured (`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` succeeds)
- [ ] Docker with Compose V2 (`docker compose version` works)
- [ ] HuggingFace license accepted at https://huggingface.co/nvidia/Cosmos-Reason2-8B (skip for `make docker-test`)
- [ ] `uv` installed (used by `make adapters`)
- [ ] `docker/.env` populated with at least `HF_TOKEN`

## Key Files

| File | Role |
|------|------|
| `Makefile` | Top-level orchestration (`make init`, `make adapters`, `make docker-up`, `make docker-test`) |
| `docker/.env.example` | All tunable environment variables — copy to `docker/.env` and edit |
| `docker/docker-compose.yml` | 4-service stack: `sim-server`, `inference-server`, `app-server`, `frontend` |
| `docker/launch.sh` | Build/run wrapper invoked by `make docker-up` (auto-detects vLLM image for x86 / Jetson) |
| `docker/inference/entrypoint.sh` | Inference-server entrypoint — wraps vLLM with the LoRA flags and reasoning parser |
| `app/src/dr_ai_palletizer/control_loop.py` | Async orchestrator that polls boxes, calls inference, parses actions, and executes pick/place |
| `app/src/dr_ai_palletizer/prompt_builder.py` | Builds OpenAI-compatible multimodal messages from box crops and pallet state |
| `app/src/dr_ai_palletizer/action_parser.py` | Extracts reasoning and JSON actions from model responses |
| `sim/src/drp_sim/server.py` | Isaac Sim service entry point — creates `SimulationApp`, starts uvicorn, and runs the sim loop |
| `sim/src/drp_sim/motion_interface.py` | cuRobo `MotionGen` wrapper used by `sim-server` |
| `app/ui/` | React UI showing camera, reasoning, parsed action, and execution |
| `adapters/` | Bind-mounted into the inference container at `/adapters/` |

## Code Structure
- `dr_ai_palletizer.server` — FastAPI routes and WebSocket event stream (`/api/health`, control lifecycle, UI updates)
- `dr_ai_palletizer.control_loop` — state machine for polling box crops, maintaining pallet state, converting grid cells to world poses, and calling `sim-server`
- `dr_ai_palletizer.prompt_builder` / `dr_ai_palletizer.action_parser` — prompt template + structured-output parsing for Cosmos Reason 2 responses
- `drp_sim.api` / `drp_sim.server` — Isaac Sim REST endpoints (`/sim/health`, camera, boxes, geometry, robot commands)
- `drp_sim.motion_interface` — cuRobo wrapper that plans and executes collision-free trajectories inside `sim-server`
- The frontend connects to `app-server` over WebSocket for live streaming of reasoning + execution state

## Expected Output

```
- Frontend at http://localhost:3000 showing live camera feed, streaming reasoning trace, parsed placement parameters, and the Doosan arm executing each placement in Isaac Sim
- inference-server logs: vLLM serving `nvidia/Cosmos-Reason2-8B` (+ optional LoRA `palletize`) on port 8200
- sim-server logs: Isaac Sim running headless, generating frames, and executing cuRobo trajectories on port 8100
- app-server logs: per-frame VLM call duration + parsed placement decisions on port 8000
```

## Monitoring
```bash
make docker-logs                                 # follow all services
cd docker && docker compose logs -f inference-server
cd docker && docker compose logs -f sim-server
curl http://localhost:8200/health                # vLLM ready
curl http://localhost:8100/sim/health            # Isaac Sim ready
curl http://localhost:8000/api/health            # orchestrator ready
curl http://localhost:3000/api/status            # frontend proxy ready
```

## Gotchas
- **First build is long.** vLLM compiles CUDA extensions and Isaac Sim is large — budget 30+ minutes for `make docker-up` the first time. Pre-downloading weights via `hf download nvidia/Cosmos-Reason2-8B` and setting `HF_CACHE_DIR=~/.cache/huggingface` in `docker/.env` saves the most time on rebuilds.
- **LoRA prompt format changed upstream.** The current `yurirocha15/...-palletizer-lora` weights were trained against the previous prompt format and produce degraded results until upstream republishes. Until then, run base-model only (leave `LORA_ADAPTER_PATH` empty).
- **`VLLM_MAX_MODEL_LEN` must be ≥ 5120.** The app sends ~2200 input tokens and requests up to 2048 output tokens; lowering this causes 400 errors from vLLM.
- **vLLM image varies by platform.** `launch.sh` auto-picks the right image for x86 CUDA 12.x, x86 CUDA 13.x (Blackwell), Jetson Thor, and Jetson Orin. If you override `VLLM_IMAGE` manually, match your driver and architecture or vLLM will fail to start.
- **CUDA context conflicts on data-center GPUs.** GPUs without RT cores (e.g. H100, A100) need cuRobo on a separate GPU from Isaac Sim. Set `CUROBO_GPU_DEVICE` to a different device than `SIM_GPU_DEVICE`.
- **cuRobo is not a separate container.** The current Compose stack runs cuRobo inside `sim-server`; `CUROBO_GPU_DEVICE` maps an extra host GPU into that container and sets `CUROBO_DEVICE=cuda:1`.
- **Driver shadowing inside vLLM ≥ 0.14.** vLLM's image runs `ldconfig` over `/usr/local/cuda-*/compat/` at build time, which shadows the host `libcuda.so.1` on drivers < 600. The compose file already sets `LD_LIBRARY_PATH` to prioritise host driver libs — do not remove that.
- **Gated model.** Without a HuggingFace token that has accepted the Cosmos Reason 2 license, the inference-server fails to start. Use `make docker-test` for a smoke run that does not require gated weights.
