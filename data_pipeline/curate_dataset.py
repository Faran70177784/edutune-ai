"""Curate and clean the EduTune AI education dataset."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "instruction",
    "input",
    "response",
    "category",
    "difficulty",
    "source",
}

VALID_CATEGORIES = {
    "mathematics",
    "physics",
    "biology",
    "chemistry",
    "computer_science",
    "economics",
    "history",
    "english",
}

VALID_DIFFICULTIES = {
    "beginner",
    "intermediate",
    "advanced",
}

MIN_INSTRUCTION_LENGTH = 10
MIN_RESPONSE_LENGTH = 20


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving the meaning of the text."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse repeated whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate one dataset record.

    Returns:
        A tuple containing:
        - whether the record is valid
        - a list of validation errors
    """

    errors: list[str] = []

    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        errors.append(
            f"Missing fields: {', '.join(sorted(missing_fields))}"
        )

    instruction = record.get("instruction", "")
    response = record.get("response", "")
    category = record.get("category", "")
    difficulty = record.get("difficulty", "")

    if not isinstance(instruction, str):
        errors.append("Instruction must be a string.")
    elif len(instruction.strip()) < MIN_INSTRUCTION_LENGTH:
        errors.append("Instruction is too short.")

    if not isinstance(response, str):
        errors.append("Response must be a string.")
    elif len(response.strip()) < MIN_RESPONSE_LENGTH:
        errors.append("Response is too short.")

    if category not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {category}")

    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"Invalid difficulty: {difficulty}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def create_record_key(record: dict[str, Any]) -> str:
    """Create a normalized key used for duplicate detection."""

    instruction = normalize_text(
        str(record.get("instruction", ""))
    ).lower()

    input_text = normalize_text(
        str(record.get("input", ""))
    ).lower()

    return f"{instruction}|{input_text}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_jsonl(input_file: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    records: list[dict[str, Any]] = []

    with input_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: Invalid JSON on line {line_number}: {exc}"
                )
                continue

            if not isinstance(record, dict):
                print(
                    f"Warning: Line {line_number} is not a JSON object."
                )
                continue

            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def curate_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean, validate, normalize, and deduplicate records."""

    curated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    statistics = {
        "input_records": len(records),
        "accepted_records": 0,
        "rejected_records": 0,
        "duplicate_records": 0,
    }

    for record in records:
        is_valid, errors = validate_record(record)

        if not is_valid:
            statistics["rejected_records"] += 1

            print(
                "Rejected record:",
                record.get("instruction", "<missing instruction>"),
            )

            for error in errors:
                print(f"  - {error}")

            continue

        record = {
            "instruction": normalize_text(record["instruction"]),
            "input": normalize_text(record.get("input", "")),
            "response": normalize_text(record["response"]),
            "category": record["category"].strip().lower(),
            "difficulty": record["difficulty"].strip().lower(),
            "source": record["source"].strip().lower(),
        }

        record_key = create_record_key(record)

        if record_key in seen_keys:
            statistics["duplicate_records"] += 1
            continue

        seen_keys.add(record_key)
        curated.append(record)

    statistics["accepted_records"] = len(curated)

    return curated, statistics


# ---------------------------------------------------------------------------
# Dataset saving
# ---------------------------------------------------------------------------


def save_jsonl(
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Save records as UTF-8 JSONL."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_curation(
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """Run the complete dataset curation pipeline."""

    if input_file is None:
        input_file = RAW_DATA_DIR / "education_seed.jsonl"

    if output_file is None:
        output_file = PROCESSED_DATA_DIR / "curated_dataset.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}"
        )

    records = load_jsonl(input_file)

    curated_records, statistics = curate_records(records)

    save_jsonl(curated_records, output_file)

    print("\nDataset curation completed.")
    print(f"Input records: {statistics['input_records']}")
    print(f"Accepted records: {statistics['accepted_records']}")
    print(f"Rejected records: {statistics['rejected_records']}")
    print(f"Duplicate records: {statistics['duplicate_records']}")
    print(f"Output: {output_file}")

    return output_file


def main() -> None:
    """Command-line entry point."""

    run_curation()


if __name__ == "__main__":
    main()