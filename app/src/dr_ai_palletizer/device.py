"""Compute device selection: CUDA > MPS > CPU."""

from __future__ import annotations


def get_device() -> str:
    """Return the best available compute backend as a torch device string.

    Priority:
        1. CUDA -- NVIDIA GPUs (training + inference)
        2. MPS  -- Apple Silicon (inference, local dev)
        3. CPU  -- fallback
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
