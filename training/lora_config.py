"""LoRA configuration utilities for EduTune AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.yaml"


def load_training_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the EduTune AI training configuration from YAML."""

    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return config


def create_lora_config(
    config: dict[str, Any] | None = None,
) -> LoraConfig:
    """Create a PEFT LoRA configuration from project settings."""

    if config is None:
        config = load_training_config()

    lora_settings = config["fine_tuning"]["lora"]

    if not lora_settings.get("enabled", True):
        raise ValueError("LoRA is disabled in the training configuration.")

    return LoraConfig(
        r=int(lora_settings["rank"]),
        lora_alpha=int(lora_settings["alpha"]),
        lora_dropout=float(lora_settings["dropout"]),
        bias=str(lora_settings["bias"]),
        target_modules=list(lora_settings["target_modules"]),
        task_type="CAUSAL_LM",
    )


def summarize_lora_config(
    lora_config: LoraConfig,
) -> dict[str, Any]:
    """Return a concise, serializable LoRA configuration summary."""

    return {
        "task_type": str(lora_config.task_type),
        "rank": lora_config.r,
        "alpha": lora_config.lora_alpha,
        "dropout": lora_config.lora_dropout,
        "bias": lora_config.bias,
        "target_modules": list(lora_config.target_modules),
    }


if __name__ == "__main__":
    config = load_training_config()
    lora_config = create_lora_config(config)

    print("EduTune AI LoRA configuration")
    print("--------------------------------")
    print("Method:", config["fine_tuning"]["method"])
    print("Rank:", lora_config.r)
    print("Alpha:", lora_config.lora_alpha)
    print("Dropout:", lora_config.lora_dropout)
    print("Bias:", lora_config.bias)
    print("Task type:", lora_config.task_type)
    print("Target modules:", ", ".join(lora_config.target_modules))