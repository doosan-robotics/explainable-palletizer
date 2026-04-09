# See How It Thinks: Mixed Palletizing with Explainable Visual Reasoning

## Problem

In warehouse palletizing, robots handle boxes blindly — following fixed rules regardless of contents, condition, or fragility. Damaged boxes get stacked, fragile items get crushed, and when something goes wrong, there's no explanation why.

## Solution

We built an end-to-end palletizing system powered by **NVIDIA Cosmos Reason2 8B**, fine-tuned with LoRA on synthetic data. Given only a camera image, the model:

- **Infers box contents** without barcodes or labels
- **Detects damage** and routes unsafe boxes to human inspection
- **Decides placement parameters** — position, speed, grip strength — based on weight and fragility reasoning

Every decision includes a full chain-of-thought trace, making the system **fully auditable**.

## Full NVIDIA Stack

Our system leverages the **complete NVIDIA ecosystem**: **Cosmos Reason2** for visual reasoning, **Isaac Sim** for simulation and synthetic data, **cuRobo** for GPU-accelerated motion planning, **vLLM** for inference, and **Jetson Thor** for edge deployment. Four containerized services form a continuous control loop from perception to execution.

## Running

### Minimum Requirements

| Requirement | Version |
| --- | --- |
| NVIDIA Driver | 585+ |
| CUDA | 12.8+ |
| nvidia-container-toolkit | installed and configured |
| Docker | with Compose V2 |

### Tested Hardware

| Hardware | Architecture |
| --- | --- |
| NVIDIA RTX 4090 | Ada Lovelace |
| NVIDIA H200 | Hopper |
| NVIDIA RTX PRO 6000 / Jetson Thor | Blackwell |

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### How to Run

**1. Configure environment variables**

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env` to set your configuration. Key variables:

| Variable | Description | Default |
| --- | --- | --- |
| `HF_TOKEN` | HuggingFace token (required for gated models) | — |
| `INFERENCE_MODEL` | Model ID (`nvidia/Cosmos-Reason2-2B` or `nvidia/Cosmos-Reason2-8B`) | `nvidia/Cosmos-Reason2-8B` |
| `LORA_ADAPTER_PATH` | LoRA adapter path inside container (e.g. `/adapters/cosmos-reason2-8b`) | — |
| `LORA_MODEL` | LoRA model name for vLLM (e.g. `palletize`) | — |
| `VLLM_MAX_MODEL_LEN` | Max model context length | `5120` |
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory for vLLM | `0.5` |
| `SIM_GPU_DEVICE` / `INFERENCE_GPU_DEVICE` / `CUROBO_GPU_DEVICE` | GPU device IDs (for multi-GPU setups) | `0` |
| `HF_CACHE_DIR` | Host directory for HuggingFace model cache | Docker volume |

The first Docker build can take **30+ minutes** due to compiling CUDA extensions and downloading model weights. It is recommended to download the Cosmos Reason2 models beforehand and mount your host HuggingFace cache into Docker by setting `HF_CACHE_DIR` (e.g. `~/.cache/huggingface`) to avoid re-downloading models on every container rebuild.

```bash
# Pre-download the model (pick one)
hf download nvidia/Cosmos-Reason2-2B
hf download nvidia/Cosmos-Reason2-8B
```

**2. Download LoRA adapters**

> **Note:** The VLM prompt was recently redesigned with a new reasoning structure and output format. LoRA weights trained on the previous prompt are not compatible and will produce degraded results. We are currently retraining the adapters and will publish updated weights as soon as they are ready. In the meantime, the system runs correctly without a LoRA adapter using the base model.

```bash
make adapters
```

This downloads the fine-tuned LoRA adapters into `adapters/2B` and `adapters/8B`. Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) — see Prerequisites above. Skips automatically if adapters are already present.

**3. Launch the system**

```bash
make docker-up      # build and start all 4 services
make docker-logs    # follow logs
make docker-down    # stop and remove containers
```

**4. Access the UI**

Open `http://localhost:3000` in your browser.

| Service | Port | Health Check |
| --- | --- | --- |
| app-server | 8000 | `curl localhost:8000/api/health` |
| sim-server | 8100 | `curl localhost:8100/sim/health` |
| inference-server | 8200 | `curl localhost:8200/health` |
| frontend | 3000 | `curl localhost:3000/api/status` |

**Test mode** (no HuggingFace token needed, uses a tiny model):

```bash
make docker-test
```

## For Developers

To set up the project locally for development:

```bash
make init
```

This creates the virtual environment, installs all dependencies, and downloads the LoRA adapters.

## Team

Kyungchan Son, Minsoo Song, Yujeong Jeong, Yuri Rocha — Doosan Robotics

## Disclaimer

This is **not** an official Doosan Robotics product. It is a proof-of-concept created for the **NVIDIA Cosmos Cookoff** hackathon.
