"""Build the initial domain-specific Education dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Ensure the project root is available for absolute imports when this file
# is executed directly with: python datasets/build_dataset.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import RAW_DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Seed dataset
# ---------------------------------------------------------------------------

SAMPLE_DATA: list[dict[str, Any]] = [
    {
        "instruction": "Explain Newton's second law in simple terms.",
        "input": "",
        "response": (
            "Newton's second law states that the force acting on an object "
            "equals its mass multiplied by its acceleration. In simple terms, "
            "a larger force produces greater acceleration, while a heavier "
            "object requires more force to achieve the same acceleration."
        ),
        "category": "physics",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "What is the difference between a variable and a constant?",
        "input": "",
        "response": (
            "A variable is a value that can change during a program or "
            "calculation, while a constant represents a value that is "
            "intended to remain unchanged."
        ),
        "category": "computer_science",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "Explain photosynthesis.",
        "input": "",
        "response": (
            "Photosynthesis is the process by which green plants use "
            "sunlight, carbon dioxide, and water to produce glucose and "
            "release oxygen. Chlorophyll in plant cells captures the "
            "light energy needed for this process."
        ),
        "category": "biology",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "What is the purpose of a database?",
        "input": "",
        "response": (
            "A database is used to store, organize, and retrieve data "
            "efficiently. It allows applications and users to manage "
            "structured information and perform operations such as "
            "searching, updating, and deleting records."
        ),
        "category": "computer_science",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "Explain the concept of opportunity cost.",
        "input": "",
        "response": (
            "Opportunity cost is the value of the next-best alternative "
            "that is given up when a decision is made. It helps explain "
            "the trade-offs involved in choosing one option over another."
        ),
        "category": "economics",
        "difficulty": "intermediate",
        "source": "seed",
    },
    {
        "instruction": "What is the Pythagorean theorem?",
        "input": "",
        "response": (
            "The Pythagorean theorem applies to right-angled triangles. "
            "It states that the square of the hypotenuse equals the sum "
            "of the squares of the other two sides: a^2 + b^2 = c^2."
        ),
        "category": "mathematics",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "Explain what an algorithm is.",
        "input": "",
        "response": (
            "An algorithm is a finite sequence of clear steps used to "
            "solve a problem or accomplish a task. A good algorithm "
            "specifies what operations should be performed and in what "
            "order."
        ),
        "category": "computer_science",
        "difficulty": "beginner",
        "source": "seed",
    },
    {
        "instruction": "What is the role of mitochondria in a cell?",
        "input": "",
        "response": (
            "Mitochondria are organelles that play a major role in "
            "producing usable energy for cells. They generate ATP through "
            "cellular respiration, which provides energy for many cellular "
            "activities."
        ),
        "category": "biology",
        "difficulty": "beginner",
        "source": "seed",
    },
]


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------


def ensure_data_directory() -> None:
    """Create the raw dataset directory if it does not exist."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_seed_dataset(output_file: Path | None = None) -> Path:
    """Create the initial seed dataset as JSONL."""

    ensure_data_directory()

    if output_file is None:
        output_file = RAW_DATA_DIR / "education_seed.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for record in SAMPLE_DATA:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_file


def main() -> None:
    """Build the initial dataset."""

    output_file = build_seed_dataset()

    print("EduTune AI dataset created successfully.")
    print(f"Records: {len(SAMPLE_DATA)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()