"""Hardware-aware model loading utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-root import handling
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import MODEL_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """
    Detect the available PyTorch device.

    Returns
    -------
    str
        "cuda" when a CUDA-capable NVIDIA GPU is available,
        "cpu" otherwise, or "unavailable" when PyTorch is not installed.
    """

    try:
        import torch
    except ImportError:
        return "unavailable"

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_hardware_summary() -> dict[str, Any]:
    """Return a compact hardware and software summary."""

    device = detect_device()

    summary: dict[str, Any] = {
        "model_id": MODEL_ID,
        "device": device,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        summary["cuda_available"] = cuda_available

        if cuda_available:
            summary["gpu_name"] = torch.cuda.get_device_name(0)

            memory_bytes = (
                torch.cuda.get_device_properties(0).total_memory
            )

            summary["gpu_memory_gb"] = round(
                memory_bytes / (1024**3),
                2,
            )

    except ImportError:
        pass

    return summary


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_id: str | None = None,
    *,
    device: str | None = None,
):
    """
    Load the configured causal language model.

    The EduTune AI training workflow uses 4-bit QLoRA for
    Mistral-7B, which requires a CUDA-enabled environment.

    Parameters
    ----------
    model_id:
        Optional Hugging Face model identifier. Defaults to MODEL_ID.

    device:
        Optional target device. Defaults to automatically detected
        hardware.

    Returns
    -------
    transformers.PreTrainedModel
        Loaded model when a CUDA-enabled environment is available.

    Raises
    ------
    RuntimeError
        When CUDA is unavailable.

    ImportError
        When required model-loading dependencies are unavailable.
    """

    selected_model_id = model_id or MODEL_ID
    selected_device = device or detect_device()

    if selected_device == "unavailable":
        raise ImportError(
            "PyTorch is required to load the EduTune AI model."
        )

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

    # Import the project's QLoRA configuration only after
    # CUDA availability has been confirmed.
    from training.qlora_config import create_bitsandbytes_config

    quantization_config = create_bitsandbytes_config()

    model = AutoModelForCausalLM.from_pretrained(
        selected_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return model


# ---------------------------------------------------------------------------
# Safe model-loading status
# ---------------------------------------------------------------------------


def can_load_model() -> bool:
    """
    Return whether the current environment can load the configured model.

    This function does not download model weights.
    """

    return detect_device() == "cuda"


def print_hardware_summary() -> None:
    """Print the detected hardware configuration."""

    summary = get_hardware_summary()

    print("EduTune AI hardware summary")
    print("-" * 32)
    print(f"Model ID: {summary['model_id']}")
    print(f"Device: {summary['device']}")
    print(
        f"CUDA available: "
        f"{summary['cuda_available']}"
    )

    if summary["gpu_name"]:
        print(f"GPU: {summary['gpu_name']}")

    if summary["gpu_memory_gb"] is not None:
        print(
            f"GPU memory: "
            f"{summary['gpu_memory_gb']} GB"
        )


def main() -> None:
    """Display model-loading readiness without loading model weights."""

    print_hardware_summary()

    print()

    if can_load_model():
        print("Model loading status: READY")
        print(
            "A CUDA-enabled environment is available "
            "for Mistral-7B QLoRA loading."
        )
    else:
        print("Model loading status: BLOCKED")
        print(
            "CUDA is unavailable. Model weights will not "
            "be loaded on this CPU-only machine."
        )


if __name__ == "__main__":
    main()