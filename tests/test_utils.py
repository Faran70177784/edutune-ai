from pathlib import Path

import pytest

from utils.hardware import (
    get_device,
    get_hardware_summary,
    is_cuda_available,
)
from utils.helpers import (
    clean_text,
    normalize_text,
    safe_filename,
    validate_non_empty,
)
from utils.logger import get_logger
from utils.seed import set_seed


def test_hardware_summary_contains_required_fields():
    summary = get_hardware_summary()

    assert "device" in summary
    assert "cuda_available" in summary
    assert "device_count" in summary
    assert "gpu_name" in summary


def test_device_is_valid():
    assert get_device() in {"cpu", "cuda"}


def test_cuda_availability_is_boolean():
    assert isinstance(is_cuda_available(), bool)


def test_normalize_text():
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_clean_text():
    assert clean_text("  Hello   world  ") == "Hello world"


def test_validate_non_empty():
    assert validate_non_empty("  EduTune AI  ") == "EduTune AI"


def test_validate_non_empty_rejects_empty():
    with pytest.raises(ValueError):
        validate_non_empty("   ", "question")


def test_safe_filename():
    result = safe_filename("my report?.json")

    assert result == "my_report_.json"


def test_logger_returns_logger():
    logger = get_logger("test_edutune_utils")

    assert logger.name == "test_edutune_utils"
    assert logger.handlers


def test_seed_returns_requested_value():
    assert set_seed(42) == 42


def test_project_root_exists():
    from utils.helpers import get_project_root

    root = get_project_root()

    assert isinstance(root, Path)
    assert root.exists()