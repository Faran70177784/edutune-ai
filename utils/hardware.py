"""
Hardware detection and safety utilities for EduTune AI.

This module provides lightweight hardware diagnostics without loading
large models or allocating unnecessary GPU memory.
"""

from __future__ import annotations

from typing import Any

import torch


def is_cuda_available() -> bool:
    """Return whether CUDA is available to PyTorch."""
    return bool(torch.cuda.is_available())


def get_device() -> str:
    """
    Return the safest available PyTorch device.

    CUDA is preferred when available; otherwise CPU is used.
    """
    return "cuda" if is_cuda_available() else "cpu"


def get_gpu_name() -> str | None:
    """Return the name of the first CUDA GPU, if available."""
    if not is_cuda_available():
        return None

    try:
        return torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        return None


def get_cuda_device_count() -> int:
    """Return the number of CUDA devices available to PyTorch."""
    if not is_cuda_available():
        return 0

    try:
        return int(torch.cuda.device_count())
    except (RuntimeError, AssertionError):
        return 0


def get_hardware_summary() -> dict[str, Any]:
    """
    Return a safe hardware summary for logging, diagnostics, and UI.

    This function does not load a model and does not allocate model memory.
    """
    cuda_available = is_cuda_available()
    device = get_device()
    device_count = get_cuda_device_count()

    return {
        "device": device,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_name": get_gpu_name(),
        "cuda_version": torch.version.cuda if cuda_available else None,
        "pytorch_version": torch.__version__,
    }


def can_run_large_model() -> bool:
    """
    Return whether the current environment is suitable for CUDA-dependent
    large-model execution.
    """
    return is_cuda_available()


def require_cuda(operation: str = "this operation") -> None:
    """
    Raise a clear error when an operation requires CUDA but CUDA is absent.
    """
    if not is_cuda_available():
        raise RuntimeError(
            f"{operation} requires a CUDA-enabled GPU, "
            "but CUDA is unavailable in the current environment."
        )