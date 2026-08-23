"""Format EduTune AI records for instruction tuning."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import (  # noqa: E402
    EVALUATION_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    RANDOM_SEED,
)


INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "synthetic_dataset.jsonl"
)

OUTPUT_DIR = EVALUATION_DATA_DIR

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


def load_records(path: Path) -> list[dict]:
    """Load JSONL records from disk."""

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def build_prompt(record: dict) -> str:
    """Build a standard instruction prompt."""

    instruction = record[
        "instruction"
    ].strip()

    input_text = record.get(
        "input",
        "",
    ).strip()

    if input_text:
        instruction_block = (
            f"{instruction}\n\n"
            f"Additional context:\n"
            f"{input_text}"
        )
    else:
        instruction_block = instruction

    return (
        "### Instruction:\n"
        f"{instruction_block}\n\n"
        "### Response:\n"
    )


def format_record(record: dict) -> dict:
    """Convert a raw record into training format."""

    return {
        "prompt": build_prompt(record),
        "response": record[
            "response"
        ].strip(),
        "category": record[
            "category"
        ],
        "difficulty": record[
            "difficulty"
        ],
        "source": record[
            "source"
        ],
        "task_type": record[
            "task_type"
        ],
    }


def split_records(
    records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train, validation and test sets."""

    shuffled = records.copy()

    random.Random(
        RANDOM_SEED
    ).shuffle(shuffled)

    total = len(shuffled)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = (
        train_end
        + int(total * VALIDATION_RATIO)
    )

    train_records = shuffled[
        :train_end
    ]

    validation_records = shuffled[
        train_end:validation_end
    ]

    test_records = shuffled[
        validation_end:
    ]

    return (
        train_records,
        validation_records,
        test_records,
    )


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    """Write records to JSONL."""

    with path.open(
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


def main() -> None:
    """Build train/validation/test datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_records(
        INPUT_FILE
    )

    formatted_records = [
        format_record(record)
        for record in records
    ]

    (
        train_records,
        validation_records,
        test_records,
    ) = split_records(
        formatted_records
    )

    write_jsonl(
        TRAIN_FILE,
        train_records,
    )

    write_jsonl(
        VALIDATION_FILE,
        validation_records,
    )

    write_jsonl(
        TEST_FILE,
        test_records,
    )

    print(
        "Instruction dataset formatting completed."
    )

    print(
        f"Total records: {len(formatted_records)}"
    )

    print(
        f"Training records: {len(train_records)}"
    )

    print(
        f"Validation records: "
        f"{len(validation_records)}"
    )

    print(
        f"Test records: {len(test_records)}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()