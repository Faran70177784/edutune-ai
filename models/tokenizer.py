"""Tokenizer utilities for EduTune AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import MODEL_ID  # noqa: E402


def load_tokenizer(
    model_id: str = MODEL_ID,
    **kwargs: Any,
):
    """
    Load the tokenizer associated with the base model.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.

    Returns
    -------
    AutoTokenizer
        Hugging Face tokenizer instance.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **kwargs,
    )

    # Mistral-family tokenizers normally have
    # an EOS token but no dedicated PAD token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer


def inspect_tokenizer(
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    """Return basic tokenizer information."""

    tokenizer = load_tokenizer(
        model_id
    )

    return {
        "model_id": model_id,
        "vocab_size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
    }


if __name__ == "__main__":
    information = inspect_tokenizer()

    for key, value in information.items():
        print(
            f"{key}: {value}"
        )