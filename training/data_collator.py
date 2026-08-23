"""Training data preprocessing utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from transformers import DataCollatorForLanguageModeling

# ---------------------------------------------------------------------------
# Project-root import handling
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_ID
from models.tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def build_training_text(record: dict[str, Any]) -> str:
    """Combine prompt and response into a causal-LM training example."""

    prompt = str(record.get("prompt", "")).strip()
    response = str(record.get("response", "")).strip()

    if not prompt:
        raise ValueError("Training record contains an empty prompt.")

    if not response:
        raise ValueError("Training record contains an empty response.")

    return f"{prompt}{response}"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int = 512,
):
    """Tokenize a Hugging Face dataset."""

    def tokenize_record(record: dict[str, Any]) -> dict[str, Any]:
        text = build_training_text(record)

        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        tokenize_record,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


def create_data_collator(tokenizer):
    """Create a causal language-model data collator."""

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


# ---------------------------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------------------------


def prepare_training_components(
    datasets: dict[str, Any],
    max_length: int = 512,
):
    """
    Prepare tokenizer, tokenized datasets, and data collator.

    Returns:
        tuple:
            tokenizer
            tokenized datasets
            causal-LM data collator
    """

    tokenizer = load_tokenizer()

    tokenized = {
        split: tokenize_dataset(
            dataset,
            tokenizer,
            max_length=max_length,
        )
        for split, dataset in datasets.items()
    }

    collator = create_data_collator(tokenizer)

    return tokenizer, tokenized, collator


# ---------------------------------------------------------------------------
# Backward-compatible public API
# ---------------------------------------------------------------------------


def prepare_training_datasets(
    datasets: dict[str, Any],
    max_length: int = 512,
):
    """
    Prepare tokenized training datasets.

    This wrapper is intentionally kept as the public API expected
    by training/train.py.
    """

    _, tokenized, _ = prepare_training_components(
        datasets,
        max_length=max_length,
    )

    return tokenized


# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a preprocessing smoke test."""

    from training.dataset_loader import load_training_datasets

    datasets = load_training_datasets()

    tokenizer, tokenized, collator = prepare_training_components(
        datasets,
        max_length=512,
    )

    print("EduTune AI training preprocessing")
    print("--------------------------------")

    for split, dataset in tokenized.items():
        print(f"{split.capitalize()}: {len(dataset)} records")
        print(f"Columns: {dataset.column_names}")

    print("--------------------------------")
    print(f"Tokenizer: {MODEL_ID}")
    print("Maximum sequence length: 512")
    print("Data collator: Causal LM")
    print("Preprocessing: PASSED")


if __name__ == "__main__":
    main()