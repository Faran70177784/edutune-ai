"""Hardware-aware model loading utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-root import handling
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_ID  # noqa: E402

from utils.hardware import (
    get_cuda_device_count,
    get_device,
    get_gpu_name,
    is_cuda_available,
)


def detect_device() -> str:
    """
    Detect the available PyTorch device.

    Returns:
        "cuda" when CUDA is available, otherwise "cpu".
    """
    return get_device()


def get_hardware_summary() -> dict[str, Any]:
    """
    Return a compact hardware and model-loading summary.

    This function never loads model weights.
    """
    cuda_available = is_cuda_available()

    gpu_memory_gb: float | None = None

    if cuda_available:
        try:
            import torch

            memory_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = round(memory_bytes / (1024**3), 2)
        except (ImportError, RuntimeError, AssertionError):
            gpu_memory_gb = None

    return {
        "model_id": MODEL_ID,
        "device": detect_device(),
        "cuda_available": cuda_available,
        "device_count": get_cuda_device_count(),
        "gpu_name": get_gpu_name(),
        "gpu_memory_gb": gpu_memory_gb,
    }


def can_load_model() -> bool:
    """
    Return whether the current environment can load Mistral-7B.

    No model weights are downloaded or loaded.
    """
    return is_cuda_available()


def load_model(
    model_id: str | None = None,
    *,
    device: str | None = None,
):
    """
    Load the configured causal language model.

    Mistral-7B QLoRA loading requires CUDA in the current architecture.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ImportError: If required dependencies are missing.
    """
    selected_model_id = model_id or MODEL_ID
    selected_device = device or detect_device()

    if selected_device != "cuda":
        raise RuntimeError(
            "CUDA is unavailable. EduTune AI uses "
            "Mistral-7B with 4-bit QLoRA and requires "
            "a CUDA-enabled GPU for model loading."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Model loading requires PyTorch and Transformers."
        ) from exc

    from training.qlora_config import create_bitsandbytes_config

    quantization_config = create_bitsandbytes_config()

    return AutoModelForCausalLM.from_pretrained(
        selected_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )


def print_hardware_summary() -> None:
    """Print the current model-loading hardware status."""
    summary = get_hardware_summary()

    print("EduTune AI hardware summary")
    print("-" * 32)
    print(f"Model ID: {summary['model_id']}")
    print(f"Device: {summary['device']}")
    print(f"CUDA available: {summary['cuda_available']}")
    print(f"CUDA devices: {summary['device_count']}")

    if summary["gpu_name"]:
        print(f"GPU: {summary['gpu_name']}")

    if summary["gpu_memory_gb"] is not None:
        print(f"GPU memory: {summary['gpu_memory_gb']} GB")


def main() -> None:
    """
    Display model-loading readiness without loading model weights.
    """
    print_hardware_summary()
    print()

    if can_load_model():
        print("Model loading status: READY")
        print("CUDA-enabled environment detected.")
    else:
        print("Model loading status: BLOCKED")
        print(
            "CUDA is unavailable. Mistral-7B weights will not "
            "be loaded on this CPU-only machine."
        )


if __name__ == "__main__":
    main()