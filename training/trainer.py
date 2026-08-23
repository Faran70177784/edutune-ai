"""Trainer construction and execution utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROJECT_ROOT as SETTINGS_PROJECT_ROOT  # noqa: E402

CONFIG_PATH = SETTINGS_PROJECT_ROOT / "config" / "training_config.yaml"


def load_training_config() -> dict[str, Any]:
    """Load the project training configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a YAML mapping.")

    return config


def create_training_arguments() -> TrainingArguments:
    """Create Hugging Face TrainingArguments from project configuration."""
    config = load_training_config()
    training = config["training"]
    output = config["output"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. EduTune AI training requires a CUDA-enabled "
            "GPU for the configured QLoRA workflow."
        )

    return TrainingArguments(
        output_dir=str(PROJECT_ROOT / output["checkpoint_dir"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        eval_steps=int(training["eval_steps"]),
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        optim=str(training["optim"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        max_grad_norm=float(training["max_grad_norm"]),
        fp16=bool(training["fp16"]),
        bf16=bool(training["bf16"]),
        report_to=(
            ["wandb"] if config.get("wandb", {}).get("enabled", False) else []
        ),
        remove_unused_columns=False,
    )


def create_trainer(
    model: Any,
    tokenizer: Any,
    tokenized_datasets: dict[str, Any],
    data_collator: Any,
    training_args: TrainingArguments | None = None,
) -> Trainer:
    """Construct the Hugging Face Trainer for QLoRA fine-tuning."""
    if model is None:
        raise ValueError("Model cannot be None.")
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None.")
    if "train" not in tokenized_datasets:
        raise ValueError("Tokenized datasets must contain a 'train' split.")
    if "validation" not in tokenized_datasets:
        raise ValueError("Tokenized datasets must contain a 'validation' split.")
    if data_collator is None:
        raise ValueError("Data collator cannot be None.")

    if training_args is None:
        training_args = create_training_arguments()

    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )


def train_model(trainer: Trainer) -> Any:
    """Execute training and return the Hugging Face training result."""
    if trainer is None:
        raise ValueError("Trainer cannot be None.")

    return trainer.train()


def save_trained_adapter(
    trainer: Trainer,
    tokenizer: Any,
    adapter_dir: str | Path,
) -> Path:
    """Save the trained PEFT adapter and tokenizer."""
    if trainer is None:
        raise ValueError("Trainer cannot be None.")
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None.")

    output_path = Path(adapter_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return output_path


def summarize_training_config() -> dict[str, Any]:
    """Return a hardware-independent training summary."""
    config = load_training_config()
    training = config["training"]
    output = config["output"]

    return {
        "epochs": training["num_train_epochs"],
        "train_batch_size": training["per_device_train_batch_size"],
        "eval_batch_size": training["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "optimizer": training["optim"],
        "scheduler": training["lr_scheduler_type"],
        "bf16": training["bf16"],
        "fp16": training["fp16"],
        "output_dir": str(PROJECT_ROOT / output["checkpoint_dir"]),
        "adapter_dir": str(PROJECT_ROOT / output["adapter_dir"]),
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    """Display the configured training environment."""
    summary = summarize_training_config()

    print("EduTune AI trainer configuration")
    print("--------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("--------------------------------")
    if not torch.cuda.is_available():
        print("Training: BLOCKED")
        print("Reason: CUDA is unavailable. Use a CUDA-enabled GPU environment.")
    else:
        print("Training configuration: READY")


if __name__ == "__main__":
    main()
