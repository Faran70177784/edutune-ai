"""Inference model loading utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import MODEL_ID  # noqa: E402
from models.load_model import detect_device  # noqa: E402


def check_inference_hardware() -> dict[str, Any]:
    """Return hardware information relevant to inference."""

    device = detect_device()

    result: dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_id": MODEL_ID,
    }

    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["device_count"] = torch.cuda.device_count()
    else:
        result["gpu_name"] = None
        result["device_count"] = 0

    return result


def load_inference_model(
    model_id: str | None = None,
) -> PreTrainedModel:
    """
    Load the EduTune model for inference.

    The configured Mistral-7B QLoRA workflow requires CUDA.
    CPU-only environments are deliberately blocked so that
    model weights are not accidentally downloaded or loaded.
    """

    hardware = check_inference_hardware()

    if not hardware["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable. EduTune AI inference for "
            "Mistral-7B requires a CUDA-enabled GPU."
        )

    from models.load_model import load_model

    return load_model(model_id=model_id)


def inference_is_available() -> bool:
    """Return whether model inference can currently be executed."""

    return torch.cuda.is_available()


if __name__ == "__main__":
    hardware = check_inference_hardware()

    print("EduTune AI inference model loader")
    print("--------------------------------")
    print(f"Model ID: {hardware['model_id']}")
    print(f"Device: {hardware['device']}")
    print(
        f"CUDA available: "
        f"{hardware['cuda_available']}"
    )
    print(f"GPU count: {hardware['device_count']}")

    if hardware["gpu_name"]:
        print(f"GPU: {hardware['gpu_name']}")

    print("--------------------------------")

    if inference_is_available():
        print("Inference model loading: READY")
    else:
        print("Inference model loading: BLOCKED")
        print(
            "CUDA is unavailable. Model weights will not "
            "be loaded on this CPU-only machine."
        )