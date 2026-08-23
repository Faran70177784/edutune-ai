"""Executable QLoRA training pipeline for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ADAPTERS_DIR, MODEL_ID  # noqa: E402
from training.data_collator import prepare_training_components  # noqa: E402
from training.dataset_loader import load_training_datasets  # noqa: E402
from training.lora_config import create_lora_config, summarize_lora_config  # noqa: E402
from training.model_setup import (  # noqa: E402
    load_training_config,
    prepare_lora_model,
    summarize_environment,
)
from training.qlora_config import create_qlora_config, summarize_qlora_config  # noqa: E402
from training.trainer import (  # noqa: E402
    create_trainer,
    create_training_arguments,
    save_trained_adapter,
    summarize_training_config,
    train_model,
)


def check_training_hardware() -> dict[str, Any]:
    """Validate the hardware required by the configured QLoRA workflow."""
    cuda_available = torch.cuda.is_available()

    return {
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "status": "ready" if cuda_available else "blocked",
    }


def prepare_training() -> dict[str, Any]:
    """Prepare datasets, model, tokenizer, trainer and training arguments."""
    hardware = check_training_hardware()
    if not hardware["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable. EduTune AI uses 4-bit QLoRA for Mistral-7B "
            "and requires a CUDA-enabled GPU."
        )

    config = load_training_config()

    print("Loading training datasets...")
    datasets = load_training_datasets()

    max_length = int(config["dataset"].get("max_sequence_length", 512))
    print("Preparing tokenizer, tokenized datasets, and data collator...")
    tokenizer, tokenized_datasets, data_collator = prepare_training_components(
        datasets,
        max_length=max_length,
    )

    print("Loading quantized base model and attaching LoRA adapter...")
    model, lora_config = prepare_lora_model(config)

    print("Creating training arguments...")
    training_args = create_training_arguments()

    print("Creating Trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        data_collator=data_collator,
        training_args=training_args,
    )

    return {
        "hardware": hardware,
        "config": config,
        "datasets": datasets,
        "tokenizer": tokenizer,
        "tokenized_datasets": tokenized_datasets,
        "data_collator": data_collator,
        "model": model,
        "lora_config": lora_config,
        "training_args": training_args,
        "trainer": trainer,
    }


def print_training_summary() -> None:
    """Print the complete training configuration without loading weights."""
    config = load_training_config()
    hardware = check_training_hardware()
    environment = summarize_environment(config)
    lora_summary = summarize_lora_config(create_lora_config(config))
    qlora_summary = summarize_qlora_config(create_qlora_config())
    training_summary = summarize_training_config()

    print()
    print("EduTune AI QLoRA training configuration")
    print("=" * 50)
    print(f"Model ID: {MODEL_ID}")
    print(f"Device: {hardware['device']}")
    print(f"CUDA available: {hardware['cuda_available']}")
    print(f"GPU: {hardware['device_name']}")
    print()
    print("QLoRA")
    print("-" * 50)
    print(f"Method: {qlora_summary['method']}")
    print(f"4-bit: {qlora_summary['load_in_4bit']}")
    print(f"Quantization: {qlora_summary['quant_type']}")
    print(f"Double quantization: {qlora_summary['double_quantization']}")
    print(f"Compute dtype: {qlora_summary['compute_dtype']}")
    print()
    print("LoRA")
    print("-" * 50)
    print(f"Rank: {lora_summary['rank']}")
    print(f"Alpha: {lora_summary['alpha']}")
    print(f"Dropout: {lora_summary['dropout']}")
    print(f"Target modules: {', '.join(lora_summary['target_modules'])}")
    print()
    print("Training")
    print("-" * 50)
    print(f"Epochs: {training_summary['epochs']}")
    print(f"Train batch size: {training_summary['train_batch_size']}")
    print(f"Gradient accumulation: {training_summary['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_summary['learning_rate']}")
    print(f"Optimizer: {training_summary['optimizer']}")
    print(f"Scheduler: {training_summary['scheduler']}")
    print(f"Checkpoint directory: {training_summary['output_dir']}")
    print(f"Adapter directory: {training_summary['adapter_dir']}")
    print()
    if environment["warnings"]:
        print("Environment warnings")
        print("-" * 50)
        for warning in environment["warnings"]:
            print(f"- {warning}")
    print("=" * 50)


def run_training() -> dict[str, Any]:
    """Execute QLoRA training and save the resulting adapter."""
    prepared = prepare_training()
    trainer = prepared["trainer"]

    print("Starting QLoRA training...")
    training_result = train_model(trainer)

    adapter_path = save_trained_adapter(
        trainer,
        prepared["tokenizer"],
        ADAPTERS_DIR,
    )

    metrics = dict(training_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return {
        "status": "completed",
        "model_id": MODEL_ID,
        "adapter_path": str(adapter_path),
        "metrics": metrics,
    }


def main() -> int:
    """Run the complete training workflow when CUDA is available."""
    print_training_summary()
    hardware = check_training_hardware()

    if not hardware["cuda_available"]:
        print()
        print("TRAINING BLOCKED")
        print("-" * 50)
        print("CUDA is unavailable on this machine.")
        print("No model weights will be loaded and no training will start.")
        return 0

    try:
        result = run_training()
    except Exception as exc:
        print()
        print("TRAINING FAILED")
        print("-" * 50)
        print(f"Error: {exc}")
        return 1

    print()
    print("TRAINING COMPLETED")
    print("-" * 50)
    print(f"Adapter: {result['adapter_path']}")
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
