#!/usr/bin/env bash
set -euo pipefail

MODEL="${INFERENCE_MODEL:-nvidia/Cosmos-Reason2-2B}"
PORT="${VLLM_PORT:-8200}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-}"

echo "[inference-server] Model: ${MODEL}"
echo "[inference-server] Port: ${PORT}"
echo "[inference-server] Max model length: ${MAX_MODEL_LEN}"
echo "[inference-server] GPU memory utilization: ${GPU_MEM}"

ARGS=(
    --model "${MODEL}"
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM}"
)

# Only add --reasoning-parser when explicitly configured (not all models support it)
if [[ -n "${REASONING_PARSER:-}" ]]; then
    ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

# Detect local model path vs HuggingFace ID
if [[ "${MODEL}" == /* ]] || [[ "${MODEL}" == /models/* ]]; then
    echo "[inference-server] Using local model path"
fi

# Enable LoRA if adapter path is set
if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
    LORA_NAME="${LORA_MODEL:-palletize}"
    echo "[inference-server] LoRA adapter: ${LORA_NAME}=${LORA_ADAPTER_PATH}"
    ARGS+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_ADAPTER_PATH}")
fi

echo "[inference-server] Launching vLLM..."
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
