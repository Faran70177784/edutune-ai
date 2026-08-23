"""Model preparation utilities for EduTune AI QLoRA training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.lora_config import create_lora_config  # noqa: E402

TRAINING_CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.yaml"


def load_training_config() -> dict[str, Any]:
    """Load the YAML training configuration."""
    if not TRAINING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {TRAINING_CONFIG_PATH}"
        )

    with TRAINING_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a dictionary.")

    return config


def detect_device() -> str:
    """Return the available PyTorch device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_training_environment(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the training environment without loading model weights."""
    device = detect_device()
    quantization = config["model"]["quantization"]
    training = config["training"]

    warnings: list[str] = []

    if device == "cpu":
        warnings.append(
            "CUDA is unavailable. QLoRA 4-bit training of a 7B model "
            "must remain blocked on this machine."
        )

    if quantization.get("enabled", False) and device == "cpu":
        warnings.append(
            "4-bit bitsandbytes quantization requires a supported CUDA "
            "environment for the configured training workflow."
        )

    if training.get("bf16", False) and device == "cpu":
        warnings.append(
            "bfloat16 training is unavailable in the current CPU-only setup."
        )

    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_id": config["model"]["name"],
        "fine_tuning_method": config["fine_tuning"]["method"],
        "quantization_enabled": quantization.get("enabled", False),
        "quantization_method": quantization.get("method"),
        "warnings": warnings,
    }


def build_model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build model-loading arguments for the CUDA QLoRA environment."""
    quantization = config["model"]["quantization"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Model loading for QLoRA training "
            "has been intentionally blocked to prevent an impractical "
            "CPU training run."
        )

    if not quantization.get("enabled", False):
        return {"device_map": "auto"}

    from transformers import BitsAndBytesConfig

    compute_dtype_name = str(
        quantization.get("compute_dtype", "float16")
    ).lower()

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if compute_dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported compute dtype: {compute_dtype_name}. "
            f"Supported values: {sorted(dtype_map)}"
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            quantization.get("quant_type", "nf4")
        ),
        bnb_4bit_use_double_quant=bool(
            quantization.get("use_double_quantization", True)
        ),
        bnb_4bit_compute_dtype=dtype_map[compute_dtype_name],
    )

    return {
        "quantization_config": bnb_config,
        "device_map": "auto",
    }


def load_base_model(config: dict[str, Any]):
    """Load the quantized base causal language model."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Mistral-7B QLoRA model loading is blocked."
        )

    model_kwargs = build_model_kwargs(config)

    return AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        **model_kwargs,
    )


def prepare_lora_model(config: dict[str, Any]):
    """
    Load the quantized base model and attach a trainable LoRA adapter.

    The k-bit preparation step is deliberately performed before PEFT adapter
    creation, which is required for the standard QLoRA workflow.
    """
    model = load_base_model(config)

    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for QLoRA model preparation."
        ) from exc

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    return model, lora_config


def count_trainable_parameters(model: Any) -> dict[str, int | float]:
    """Return total, trainable, and trainable-percentage parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    percentage = (trainable / total * 100.0) if total else 0.0

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": round(percentage, 6),
    }


def summarize_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Return a complete model-preparation summary without loading weights."""
    environment = validate_training_environment(config)
    lora_config = create_lora_config(config)

    environment["lora"] = {
        "rank": lora_config.r,
        "alpha": lora_config.lora_alpha,
        "dropout": lora_config.lora_dropout,
        "target_modules": sorted(lora_config.target_modules),
    }

    return environment


if __name__ == "__main__":
    config = load_training_config()
    summary = summarize_environment(config)

    print("EduTune AI model preparation")
    print("--------------------------------")
    print(f"Model ID: {summary['model_id']}")
    print(f"Fine-tuning: {summary['fine_tuning_method']}")
    print(f"Device: {summary['device']}")
    print(f"CUDA available: {summary['cuda_available']}")
    print(
        f"Quantization: {summary['quantization_method']} "
        f"({summary['quantization_enabled']})"
    )
    print(f"LoRA rank: {summary['lora']['rank']}")
    print(f"LoRA alpha: {summary['lora']['alpha']}")
    print(f"LoRA dropout: {summary['lora']['dropout']}")
    print("Target modules: " + ", ".join(summary["lora"]["target_modules"]))

    if summary["warnings"]:
        print("--------------------------------")
        print("Environment warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")

    print("--------------------------------")
    print(
        "Model loading: "
        + ("READY" if summary["device"] == "cuda" else "BLOCKED")
    )
