"""Tests for EduTune AI dataset integrity."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "education_seed.jsonl"

REQUIRED_FIELDS = {
    "instruction",
    "input",
    "response",
    "category",
    "difficulty",
    "source",
}


def load_records() -> list[dict]:
    """Load JSONL records from the seed dataset."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def test_dataset_exists() -> None:
    """The seed dataset must exist."""

    assert DATASET_PATH.exists()


def test_dataset_has_records() -> None:
    """The dataset must contain at least one record."""

    records = load_records()

    assert len(records) > 0


def test_required_fields_exist() -> None:
    """Every record must contain the required fields."""

    records = load_records()

    for record in records:
        assert REQUIRED_FIELDS.issubset(record.keys())


def test_instructions_are_not_empty() -> None:
    """Every instruction must contain meaningful text."""

    records = load_records()

    for record in records:
        assert isinstance(record["instruction"], str)
        assert record["instruction"].strip()


def test_responses_are_not_empty() -> None:
    """Every response must contain meaningful text."""

    records = load_records()

    for record in records:
        assert isinstance(record["response"], str)
        assert record["response"].strip()


def test_categories_are_not_empty() -> None:
    """Every record must have a category."""

    records = load_records()

    for record in records:
        assert isinstance(record["category"], str)
        assert record["category"].strip()

def test_curated_dataset_exists() -> None:
    """The curated dataset must exist after curation."""

    curated_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "curated_dataset.jsonl"
    )

    assert curated_path.exists()


def test_curated_dataset_has_no_duplicates() -> None:
    """Curated instructions should be unique."""

    curated_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "curated_dataset.jsonl"
    )

    with curated_path.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    instructions = [
        record["instruction"].strip().lower()
        for record in records
    ]

    assert len(instructions) == len(set(instructions))