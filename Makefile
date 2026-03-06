.DEFAULT_GOAL := help
SHELL := /bin/bash

# Packages managed outside the lockfile (CUDA torch installed by scripts/install_cuda_torch.sh).
# --inexact prevents uv sync from removing them; --no-install-package prevents overwriting them.
KEEP_TORCH := --inexact --no-install-package torch --no-install-package torchvision

# LoRA adapters hosted on Hugging Face
ADAPTER_2B_REPO := yurirocha15/Cosmos-Reason2-2B-palletizer-lora
ADAPTER_8B_REPO := yurirocha15/Cosmos-Reason2-8B-palletizer-lora
ADAPTER_2B_DIR  := adapters/2B
ADAPTER_8B_DIR  := adapters/8B

# ── Primary targets ───────────────────────────────────────────────────────────

.PHONY: init
init: adapters ## First-time setup: sync all packages + install CUDA torch
	uv sync --all-packages --all-extras $(KEEP_TORCH)
	@bash scripts/install_cuda_torch.sh

.PHONY: adapters
adapters: ## Download LoRA adapters from Hugging Face (skips if present)
	@uv pip install "huggingface-hub[cli]" --quiet
	@if [ ! -d "$(ADAPTER_2B_DIR)/." ] || [ -z "$$(ls -A $(ADAPTER_2B_DIR) 2>/dev/null)" ]; then \
		echo "Downloading 2B adapter from $(ADAPTER_2B_REPO)..."; \
		uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$(ADAPTER_2B_REPO)', local_dir='$(ADAPTER_2B_DIR)')"; \
	else \
		echo "2B adapter already present, skipping."; \
	fi
	@if [ ! -d "$(ADAPTER_8B_DIR)/." ] || [ -z "$$(ls -A $(ADAPTER_8B_DIR) 2>/dev/null)" ]; then \
		echo "Downloading 8B adapter from $(ADAPTER_8B_REPO)..."; \
		uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$(ADAPTER_8B_REPO)', local_dir='$(ADAPTER_8B_DIR)')"; \
	else \
		echo "8B adapter already present, skipping."; \
	fi

.PHONY: sync
sync: ## Re-sync dependencies without overwriting CUDA torch
	uv sync --all-packages --all-extras $(KEEP_TORCH)

.PHONY: install-curobo
install-curobo: ## Install cuRobo from source (run after make init)
	@bash scripts/install_curobo.sh

# ── Run wrappers (skip auto-sync to preserve CUDA torch) ─────────────────────

.PHONY: run
run: ## Run a uv command: make run CMD="drp-train --help"
	UV_NO_SYNC=1 uv run $(CMD)

# ── Quality ───────────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Lint and format
	uv run ruff check --fix
	uv run ruff format

.PHONY: test
test: ## Run tests
	UV_NO_SYNC=1 PYTHONPATH="" uv run pytest

.PHONY: check
check: lint test ## Lint + test

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up: ## Launch full Docker pipeline
	cd docker && bash launch.sh --build

.PHONY: docker-down
docker-down: ## Stop Docker pipeline
	cd docker && bash launch.sh --down

.PHONY: docker-test
docker-test: ## Launch test mode (real Isaac Sim + tiny model, no HF token needed)
	cd docker && bash launch.sh --test

.PHONY: docker-logs
docker-logs: ## Follow Docker container logs
	cd docker && docker compose logs -f

# ── Helpers ───────────────────────────────────────────────────────────────────

.PHONY: cuda-info
cuda-info: ## Show detected CUDA backend and installed torch variant
	@echo "--- Detected backend ---"
	@cat .torch-backend 2>/dev/null || echo "(not configured -- run 'make init')"
	@echo ""
	@echo "--- Installed torch ---"
	@uv pip show torch 2>/dev/null | grep -E "^(Name|Version|Location)" || echo "torch not installed"

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
