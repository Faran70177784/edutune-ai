"""Dataset loading utilities for EduTune AI training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

# Allow direct execution:
# python training/dataset_loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EVALUATION_DATA_DIR


REQUIRED_FIELDS = {
    "prompt",
    "response",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {file_path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object at {file_path}:{line_number}"
                )

            missing = REQUIRED_FIELDS - set(record.keys())

            if missing:
                raise ValueError(
                    f"Missing fields at {file_path}:{line_number}: "
                    f"{sorted(missing)}"
                )

            records.append(record)

    return records


def load_dataset_split(
    filename: str,
) -> Dataset:
    """Load one EduTune AI dataset split as a Hugging Face Dataset."""

    path = EVALUATION_DATA_DIR / filename

    records = load_jsonl(path)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return Dataset.from_list(records)


def load_training_datasets() -> dict[str, Dataset]:
    """Load train, validation, and test datasets."""

    return {
        "train": load_dataset_split("train.jsonl"),
        "validation": load_dataset_split("validation.jsonl"),
        "test": load_dataset_split("test.jsonl"),
    }


def summarize_datasets(
    datasets: dict[str, Dataset],
) -> dict[str, Any]:
    """Return a compact summary of loaded datasets."""

    return {
        name: {
            "records": len(dataset),
            "columns": dataset.column_names,
        }
        for name, dataset in datasets.items()
    }


if __name__ == "__main__":
    datasets = load_training_datasets()

    print("EduTune AI dataset loader")
    print("--------------------------------")

    summary = summarize_datasets(datasets)

    for split_name, info in summary.items():
        print(f"{split_name.capitalize()}: {info['records']} records")
        print(f"Columns: {', '.join(info['columns'])}")

    print("--------------------------------")
    print("Dataset loading: PASSED")