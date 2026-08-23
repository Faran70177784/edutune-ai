"""QLoRA quantization configuration for EduTune AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import torch
import yaml
from transformers import BitsAndBytesConfig


# Ensure the project root is importable when this file is
# executed directly with: python training/qlora_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROJECT_ROOT as SETTINGS_PROJECT_ROOT


CONFIG_PATH = SETTINGS_PROJECT_ROOT / "config" / "training_config.yaml"


@dataclass(frozen=True)
class QLoRAConfig:
    """Project-level QLoRA configuration."""

    enabled: bool
    method: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    compute_dtype: str


def load_training_config() -> dict[str, Any]:
    """Load the YAML training configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a YAML mapping.")

    return config


def create_qlora_config() -> QLoRAConfig:
    """Create the project-level QLoRA configuration."""

    config = load_training_config()

    quantization = config["model"]["quantization"]

    return QLoRAConfig(
        enabled=bool(quantization.get("enabled", True)),
        method=str(quantization.get("method", "4bit")),
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            quantization.get("quant_type", "nf4")
        ),
        bnb_4bit_use_double_quant=bool(
            quantization.get("use_double_quantization", True)
        ),
        compute_dtype=str(
            quantization.get("compute_dtype", "bfloat16")
        ),
    )


def get_compute_dtype(dtype_name: str) -> torch.dtype:
    """Convert configured dtype name to a PyTorch dtype."""

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    normalized = dtype_name.lower().strip()

    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported compute dtype: {dtype_name}. "
            f"Supported values: {sorted(dtype_map)}"
        )

    return dtype_map[normalized]


def create_bitsandbytes_config() -> BitsAndBytesConfig:
    """
    Create the Hugging Face BitsAndBytes configuration.

    This project uses 4-bit QLoRA, which requires CUDA on the
    current training setup. Therefore this function refuses to
    construct the configuration on CPU-only machines.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. EduTune AI QLoRA 4-bit training "
            "requires a CUDA-enabled GPU."
        )

    config = create_qlora_config()

    if not config.enabled:
        raise RuntimeError(
            "QLoRA quantization is disabled in training_config.yaml."
        )

    compute_dtype = get_compute_dtype(config.compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def summarize_qlora_config(config: QLoRAConfig) -> dict[str, Any]:
    """Return a concise serializable summary."""

    return {
        "enabled": config.enabled,
        "method": config.method,
        "load_in_4bit": config.load_in_4bit,
        "quant_type": config.bnb_4bit_quant_type,
        "double_quantization": config.bnb_4bit_use_double_quant,
        "compute_dtype": config.compute_dtype,
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    """Print the QLoRA configuration and hardware status."""

    config = create_qlora_config()

    print("EduTune AI QLoRA configuration")
    print("--------------------------------")
    print(f"Method: {config.method}")
    print(f"Enabled: {config.enabled}")
    print(f"4-bit: {config.load_in_4bit}")
    print(f"Quantization type: {config.bnb_4bit_quant_type}")
    print(
        f"Double quantization: "
        f"{config.bnb_4bit_use_double_quant}"
    )
    print(f"Compute dtype: {config.compute_dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("--------------------------------")

    if torch.cuda.is_available():
        print("QLoRA configuration: READY")
    else:
        print("QLoRA configuration: GPU REQUIRED")
        print(
            "Training will remain blocked until a CUDA-enabled "
            "environment is available."
        )


if __name__ == "__main__":
    main()