"""
General-purpose helper functions for EduTune AI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Returns:
        Path: Normalized directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read and return JSON content from a file."""
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Write JSON data to a file.

    Parent directories are created automatically.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
        )

    return file_path


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation and comparison.

    Operations:
    - Converts to string
    - Removes leading/trailing whitespace
    - Collapses repeated whitespace
    - Converts to lowercase
    """
    if text is None:
        return ""

    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def clean_text(text: str) -> str:
    """
    Clean general text while preserving capitalization.
    """
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text).strip())


def validate_non_empty(value: Any, field_name: str = "value") -> str:
    """
    Validate that a value contains meaningful text.

    Returns:
        str: Cleaned value.

    Raises:
        ValueError: If the value is empty.
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be empty.")

    cleaned = str(value).strip()

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    return cleaned


def safe_filename(filename: str, default: str = "file") -> str:
    """
    Convert a string into a filesystem-safe filename.
    """
    if not filename:
        return default

    name = Path(str(filename)).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    return name or default


def get_project_root() -> Path:
    """Return the EduTune AI project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return get_project_root().joinpath(*parts)