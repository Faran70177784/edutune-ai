"""Tests for tokenizer and inference components."""

import pytest

from inference.model_loader import (
    check_inference_hardware,
    inference_is_available,
    load_inference_model,
)

from inference.prompt_templates import (
    build_chat_prompt,
    build_educational_prompt,
)

from models.tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_tokenizer_loads():
    tokenizer = load_tokenizer()

    assert tokenizer is not None
    assert tokenizer.eos_token is not None
    assert tokenizer.eos_token_id is not None


def test_tokenizer_encodes_text():
    tokenizer = load_tokenizer()

    text = "Explain Newton's second law."

    encoded = tokenizer(text)

    assert "input_ids" in encoded
    assert len(encoded["input_ids"]) > 0


def test_tokenizer_decodes_text():
    tokenizer = load_tokenizer()

    text = "Explain photosynthesis."

    encoded = tokenizer(text)

    decoded = tokenizer.decode(
        encoded["input_ids"]
    )

    assert isinstance(decoded, str)
    assert len(decoded.strip()) > 0


# ---------------------------------------------------------------------------
# Prompt-template tests
# ---------------------------------------------------------------------------


def test_educational_prompt_contains_question():
    prompt = build_educational_prompt(
        "What is gravity?"
    )

    assert "What is gravity?" in prompt
    assert "EduTune AI" in prompt


def test_educational_prompt_contains_optional_context():
    prompt = build_educational_prompt(
        "Explain force.",
        subject="Physics",
        difficulty="Beginner",
    )

    assert "Physics" in prompt
    assert "Beginner" in prompt
    assert "Explain force." in prompt


def test_chat_prompt_contains_message():
    prompt = build_chat_prompt(
        "Explain recursion."
    )

    assert "Explain recursion." in prompt
    assert "EduTune AI" in prompt


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        build_chat_prompt("")


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        build_educational_prompt("")


# ---------------------------------------------------------------------------
# Hardware tests
# ---------------------------------------------------------------------------


def test_inference_hardware_summary():
    hardware = check_inference_hardware()

    assert "device" in hardware
    assert "cuda_available" in hardware
    assert "model_id" in hardware


def test_inference_availability_is_boolean():
    assert isinstance(
        inference_is_available(),
        bool,
    )


def test_cpu_environment_blocks_model_loading():
    if inference_is_available():
        pytest.skip(
            "CUDA is available; CPU safety-block test is not applicable."
        )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        load_inference_model()