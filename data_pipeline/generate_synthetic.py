"""Generate synthetic educational instruction data for EduTune AI."""

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


from config.settings import (  # noqa: E402
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TASK_TEMPLATES: dict[str, list[str]] = {
    "concept_explanation": [
        "Explain {topic} in simple terms for a beginner.",
        "Give a clear educational explanation of {topic}.",
    ],
    "example_generation": [
        "Give a practical example that helps explain {topic}.",
        "Provide an everyday example of {topic}.",
    ],
    "study_guidance": [
        "Give three important points a student should remember about {topic}.",
        "What should a beginner focus on when studying {topic}?",
    ],
    "question_answering": [
        "What is the main purpose or importance of {topic}?",
        "Why is {topic} important in its field?",
    ],
}


RESPONSE_TEMPLATES: dict[str, str] = {
    "concept_explanation": (
        "{topic} is an important concept in {category}. "
        "A student should begin by understanding its definition "
        "and the main idea behind it. The concept becomes easier "
        "to remember when it is connected to examples and practical "
        "applications."
    ),
    "example_generation": (
        "A practical way to understand {topic} is to connect it "
        "with a real or familiar situation related to {category}. "
        "Students can identify the main idea, observe how the "
        "principle works, and then relate the example back to "
        "the original concept."
    ),
    "study_guidance": (
        "When studying {topic}, first learn the basic definition "
        "and identify the key ideas. Next, review examples and "
        "related concepts. Finally, practice applying the idea "
        "through questions or problems and explain it in your "
        "own words."
    ),
    "question_answering": (
        "The importance of {topic} comes from its role in "
        "{category}. Understanding this concept helps students "
        "connect foundational knowledge with related topics and "
        "recognize how the idea can be applied in academic or "
        "practical situations."
    ),
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_curated_dataset(
    input_file: Path,
) -> list[dict[str, Any]]:
    """Load the curated JSONL dataset."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Curated dataset not found: {input_file}"
        )

    records: list[dict[str, Any]] = []

    with input_file.open(
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
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


def extract_topic(
    instruction: str,
) -> str:
    """
    Extract a clean educational topic from a seed instruction.

    Examples
    --------
    Explain Newton's second law in simple terms.
        -> Newton's second law

    What is the purpose of a database?
        -> purpose of a database

    Explain photosynthesis.
        -> photosynthesis
    """

    topic = instruction.strip()

    # Remove punctuation first.
    topic = topic.rstrip(
        "?.!,:; "
    )

    # Remove instruction prefixes.
    prefixes = [
        "Explain what ",
        "Explain the ",
        "Explain ",
        "What is the ",
        "What is ",
        "What are the ",
        "What are ",
    ]

    for prefix in prefixes:
        if topic.lower().startswith(
            prefix.lower()
        ):
            topic = topic[
                len(prefix):
            ]
            break

    # Remove trailing educational qualifiers.
    suffixes = [
        " in simple terms for a beginner",
        " in simple terms",
        " in simple language for a beginner",
        " in simple language",
        " for a beginner",
        " for beginners",
    ]

    # Keep stripping until no suffix remains.
    while True:

        original = topic

        for suffix in suffixes:

            if topic.lower().endswith(
                suffix.lower()
            ):
                topic = topic[
                    : -len(suffix)
                ].rstrip(
                    "?.!,:; "
                )

        if topic == original:
            break

    # Defensive cleanup for accidental repeated phrases.
    repeated_patterns = [
        (
            r"\bin simple terms\s+in simple terms\b",
            "in simple terms",
        ),
        (
            r"\bin simple language\s+in simple language\b",
            "in simple language",
        ),
    ]

    for pattern, replacement in repeated_patterns:
        topic = re.sub(
            pattern,
            replacement,
            topic,
            flags=re.IGNORECASE,
        )

    return topic.strip(
        "?.!,:; "
    )


# ---------------------------------------------------------------------------
# Record quality
# ---------------------------------------------------------------------------


def is_quality_record(
    record: dict[str, Any],
) -> bool:
    """Check whether a generated record satisfies basic requirements."""

    required_fields = {
        "instruction",
        "input",
        "response",
        "category",
        "difficulty",
        "source",
        "task_type",
    }

    if not required_fields.issubset(
        record.keys()
    ):
        return False

    if len(
        record["instruction"].strip()
    ) < 15:
        return False

    if len(
        record["response"].strip()
    ) < 40:
        return False

    if not record["category"]:
        return False

    if not record["task_type"]:
        return False

    if record["source"] != "synthetic":
        return False

    return True


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_synthetic_records(
    seed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate synthetic educational records."""

    generated: list[dict[str, Any]] = []

    for seed_record in seed_records:

        topic = extract_topic(
            seed_record["instruction"]
        )

        category = seed_record[
            "category"
        ].replace(
            "_",
            " ",
        )

        for task_type, instruction_templates in (
            TASK_TEMPLATES.items()
        ):

            response_template = (
                RESPONSE_TEMPLATES[
                    task_type
                ]
            )

            for instruction_template in (
                instruction_templates
            ):

                instruction = (
                    instruction_template.format(
                        topic=topic
                    )
                )

                response = (
                    response_template.format(
                        topic=topic,
                        category=category,
                    )
                )

                record = {
                    "instruction": instruction,
                    "input": "",
                    "response": response,
                    "category": seed_record[
                        "category"
                    ],
                    "difficulty": "intermediate",
                    "source": "synthetic",
                    "task_type": task_type,
                }

                if is_quality_record(
                    record
                ):
                    generated.append(
                        record
                    )

    return generated


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate instruction/input combinations."""

    unique_records: list[dict[str, Any]] = []

    seen: set[str] = set()

    for record in records:

        key = (
            record["instruction"]
            .strip()
            .lower()
            + "|"
            + record["input"]
            .strip()
            .lower()
        )

        if key in seen:
            continue

        seen.add(key)
        unique_records.append(
            record
        )

    return unique_records


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_jsonl(
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Save records as UTF-8 JSONL."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run synthetic dataset generation."""

    input_file = (
        PROCESSED_DATA_DIR
        / "curated_dataset.jsonl"
    )

    output_file = (
        SYNTHETIC_DATA_DIR
        / "synthetic_dataset.jsonl"
    )

    seed_records = load_curated_dataset(
        input_file
    )

    generated_records = (
        generate_synthetic_records(
            seed_records
        )
    )

    generated_records = (
        deduplicate_records(
            generated_records
        )
    )

    save_jsonl(
        generated_records,
        output_file
    )

    print(
        "Synthetic dataset generated successfully."
    )

    print(
        f"Seed records: {len(seed_records)}"
    )

    print(
        f"Synthetic records: "
        f"{len(generated_records)}"
    )

    print(
        f"Output: {output_file}"
    )


if __name__ == "__main__":
    main()