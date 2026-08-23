"""Validate EduTune AI datasets and generate a quality report."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import (  # noqa: E402
    EVALUATION_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "instruction",
    "input",
    "response",
    "category",
    "difficulty",
    "source",
}

SYNTHETIC_REQUIRED_FIELDS = REQUIRED_FIELDS | {
    "task_type",
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

VALID_SOURCES = {
    "seed",
    "synthetic",
}

VALID_TASK_TYPES = {
    "concept_explanation",
    "example_generation",
    "question_answering",
    "study_guidance",
}

MIN_INSTRUCTION_LENGTH = 15
MIN_RESPONSE_LENGTH = 40


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for quality checks."""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def load_jsonl(
    file_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load JSONL records and collect parsing errors."""

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if not file_path.exists():
        errors.append(
            {
                "type": "missing_file",
                "file": str(file_path),
            }
        )

        return records, errors

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "type": "invalid_json",
                        "line": line_number,
                        "message": str(exc),
                    }
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    {
                        "type": "invalid_record_type",
                        "line": line_number,
                    }
                )
                continue

            records.append(record)

    return records, errors


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------


def validate_record(
    record: dict[str, Any],
    index: int,
    synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Validate a single record."""

    errors: list[dict[str, Any]] = []

    required_fields = (
        SYNTHETIC_REQUIRED_FIELDS
        if synthetic
        else REQUIRED_FIELDS
    )

    missing_fields = required_fields - record.keys()

    if missing_fields:
        errors.append(
            {
                "record": index,
                "type": "missing_fields",
                "fields": sorted(missing_fields),
            }
        )

    instruction = record.get(
        "instruction",
        "",
    )

    response = record.get(
        "response",
        "",
    )

    category = record.get(
        "category",
        "",
    )

    difficulty = record.get(
        "difficulty",
        "",
    )

    source = record.get(
        "source",
        "",
    )

    if not isinstance(instruction, str):
        errors.append(
            {
                "record": index,
                "type": "invalid_instruction_type",
            }
        )

    elif len(instruction.strip()) < MIN_INSTRUCTION_LENGTH:
        errors.append(
            {
                "record": index,
                "type": "instruction_too_short",
            }
        )

    if not isinstance(response, str):
        errors.append(
            {
                "record": index,
                "type": "invalid_response_type",
            }
        )

    elif len(response.strip()) < MIN_RESPONSE_LENGTH:
        errors.append(
            {
                "record": index,
                "type": "response_too_short",
            }
        )

    if category not in VALID_CATEGORIES:
        errors.append(
            {
                "record": index,
                "type": "invalid_category",
                "value": category,
            }
        )

    if difficulty not in VALID_DIFFICULTIES:
        errors.append(
            {
                "record": index,
                "type": "invalid_difficulty",
                "value": difficulty,
            }
        )

    if source not in VALID_SOURCES:
        errors.append(
            {
                "record": index,
                "type": "invalid_source",
                "value": source,
            }
        )

    if synthetic:
        task_type = record.get(
            "task_type",
            "",
        )

        if task_type not in VALID_TASK_TYPES:
            errors.append(
                {
                    "record": index,
                    "type": "invalid_task_type",
                    "value": task_type,
                }
            )

    return errors


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def find_duplicates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find duplicate instructions."""

    seen: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        instruction = normalize_text(
            record.get(
                "instruction",
                "",
            )
        )

        if instruction in seen:

            duplicates.append(
                {
                    "record": index,
                    "duplicate_of": seen[instruction],
                    "instruction": record.get(
                        "instruction",
                        "",
                    ),
                }
            )

        else:
            seen[instruction] = index

    return duplicates


# ---------------------------------------------------------------------------
# Repeated phrase detection
# ---------------------------------------------------------------------------


def find_repeated_phrases(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect obvious consecutive phrase repetition."""

    findings: list[dict[str, Any]] = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        instruction = normalize_text(
            record.get(
                "instruction",
                "",
            )
        )

        words = instruction.split()

        if len(words) < 4:
            continue

        for phrase_length in (
            2,
            3,
            4,
        ):

            for position in range(
                len(words) - phrase_length * 2 + 1
            ):

                first = words[
                    position : position + phrase_length
                ]

                second = words[
                    position
                    + phrase_length : position
                    + phrase_length * 2
                ]

                if first == second:

                    findings.append(
                        {
                            "record": index,
                            "type": "repeated_phrase",
                            "phrase": " ".join(first),
                            "instruction": record.get(
                                "instruction",
                                "",
                            ),
                        }
                    )

                    break

            else:
                continue

            break

    return findings


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------


def distribution(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    """Return value distribution for a field."""

    counter = Counter(
        record.get(
            field,
            "<missing>",
        )
        for record in records
    )

    return dict(
        sorted(
            counter.items()
        )
    )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_dataset(
    file_path: Path,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Validate a dataset and return a structured report."""

    records, parsing_errors = load_jsonl(
        file_path
    )

    record_errors: list[dict[str, Any]] = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        record_errors.extend(
            validate_record(
                record,
                index,
                synthetic=synthetic,
            )
        )

    duplicates = find_duplicates(
        records
    )

    repeated_phrases = find_repeated_phrases(
        records
    )

    report = {
        "file": str(file_path),
        "records": len(records),
        "parsing_errors": parsing_errors,
        "record_errors": record_errors,
        "duplicates": duplicates,
        "repeated_phrases": repeated_phrases,
        "distributions": {
            "category": distribution(
                records,
                "category",
            ),
            "difficulty": distribution(
                records,
                "difficulty",
            ),
            "source": distribution(
                records,
                "source",
            ),
        },
    }

    if synthetic:
        report["distributions"]["task_type"] = distribution(
            records,
            "task_type",
        )

    total_issues = (
        len(parsing_errors)
        + len(record_errors)
        + len(duplicates)
        + len(repeated_phrases)
    )

    report["summary"] = {
        "valid": total_issues == 0,
        "total_issues": total_issues,
        "parsing_errors": len(
            parsing_errors
        ),
        "record_errors": len(
            record_errors
        ),
        "duplicates": len(
            duplicates
        ),
        "repeated_phrases": len(
            repeated_phrases
        ),
    }

    return report


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------


def save_report(
    report: dict[str, Any],
    output_file: Path,
) -> None:
    """Save validation report as JSON."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Validate both curated and synthetic datasets."""

    curated_file = (
        PROCESSED_DATA_DIR
        / "curated_dataset.jsonl"
    )

    synthetic_file = (
        SYNTHETIC_DATA_DIR
        / "synthetic_dataset.jsonl"
    )

    curated_report = validate_dataset(
        curated_file,
        synthetic=False,
    )

    synthetic_report = validate_dataset(
        synthetic_file,
        synthetic=True,
    )

    combined_report = {
        "project": "EduTune AI",
        "validation_version": "1.0",
        "curated_dataset": curated_report,
        "synthetic_dataset": synthetic_report,
    }

    output_file = (
        EVALUATION_DATA_DIR
        / "dataset_validation_report.json"
    )

    save_report(
        combined_report,
        output_file,
    )

    print(
        "Dataset validation completed."
    )

    print(
        f"Curated records: "
        f"{curated_report['records']}"
    )

    print(
        f"Synthetic records: "
        f"{synthetic_report['records']}"
    )

    print(
        f"Curated issues: "
        f"{curated_report['summary']['total_issues']}"
    )

    print(
        f"Synthetic issues: "
        f"{synthetic_report['summary']['total_issues']}"
    )

    print(
        f"Report: {output_file}"
    )


if __name__ == "__main__":
    main()