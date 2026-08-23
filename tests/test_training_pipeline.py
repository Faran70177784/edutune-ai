"""Tests for EduTune AI training-pipeline safety and configuration."""

from pathlib import Path

import pytest
import torch

from training.hyperparameter_search import build_search_space, generate_trials
from training.model_setup import (
    load_training_config,
    summarize_environment,
)
from training.train import check_training_hardware, prepare_training


def test_training_hardware_summary():
    hardware = check_training_hardware()

    assert "device" in hardware
    assert "cuda_available" in hardware
    assert "status" in hardware
    assert isinstance(hardware["cuda_available"], bool)


def test_training_config_loads():
    config = load_training_config()

    assert config["model"]["name"]
    assert config["fine_tuning"]["method"] == "QLoRA"
    assert config["model"]["quantization"]["enabled"] is True


def test_training_environment_summary_is_hardware_safe():
    config = load_training_config()
    summary = summarize_environment(config)

    assert summary["model_id"] == config["model"]["name"]
    assert summary["fine_tuning_method"] == "QLoRA"
    assert "lora" in summary
    assert summary["lora"]["rank"] > 0


def test_cpu_environment_blocks_training_before_model_loading():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; CPU safety-block test is not applicable.")

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        prepare_training()


def test_hyperparameter_search_space():
    config = load_training_config()
    space = build_search_space(config)

    assert set(space) == {
        "learning_rate",
        "lora_rank",
        "lora_dropout",
    }
    assert space["learning_rate"]
    assert space["lora_rank"]
    assert space["lora_dropout"]


def test_hyperparameter_trials_are_deterministic():
    config = load_training_config()
    trials = generate_trials(config)

    expected = (
        len(config["hyperparameter_search"]["learning_rates"])
        * len(config["hyperparameter_search"]["lora_ranks"])
        * len(config["hyperparameter_search"]["dropout_values"])
    )

    assert len(trials) == expected
    assert len(trials) == len({tuple(sorted(t.items())) for t in trials})
