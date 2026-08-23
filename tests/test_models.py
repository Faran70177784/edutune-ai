import pytest

from models.load_model import (
    can_load_model,
    detect_device,
    get_hardware_summary,
)
from models.tokenizer import load_tokenizer


def test_detect_device_returns_valid_device():
    assert detect_device() in {"cpu", "cuda"}


def test_model_hardware_summary_contains_required_fields():
    summary = get_hardware_summary()

    assert "model_id" in summary
    assert "device" in summary
    assert "cuda_available" in summary
    assert "device_count" in summary
    assert "gpu_name" in summary
    assert "gpu_memory_gb" in summary


def test_can_load_model_is_boolean():
    assert isinstance(can_load_model(), bool)


def test_cpu_environment_blocks_model_loading():
    if detect_device() != "cpu":
        pytest.skip("CPU-specific test.")

    from models.load_model import load_model

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        load_model()


def test_tokenizer_loads():
    tokenizer = load_tokenizer()

    assert tokenizer is not None
    assert tokenizer.pad_token is not None


def test_tokenizer_has_expected_special_tokens():
    tokenizer = load_tokenizer()

    assert tokenizer.eos_token is not None
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token is not None


def test_tokenizer_can_encode_text():
    tokenizer = load_tokenizer()

    encoded = tokenizer(
        "Explain photosynthesis in simple terms.",
        return_tensors="pt",
    )

    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[1] > 0