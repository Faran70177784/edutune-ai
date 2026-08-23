"""
Tokenizer utilities for EduTune AI.

Tokenizer loading is independent from model-weight loading, so tokenizer
inspection can be performed in CPU-only development environments.
"""

from __future__ import annotations

from typing import Any

from config.settings import MODEL_ID


def load_tokenizer(
    model_id: str = MODEL_ID,
    **kwargs: Any,
):
    """
    Load the tokenizer associated with the configured foundation model.

    Args:
        model_id: Hugging Face model identifier.
        **kwargs: Additional arguments forwarded to AutoTokenizer.

    Returns:
        A Hugging Face tokenizer instance.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tokenizer loading requires the Transformers package."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **kwargs,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise RuntimeError(
                "Tokenizer has neither a pad token nor an EOS token."
            )

    return tokenizer


def inspect_tokenizer(
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    """
    Return basic tokenizer metadata.
    """
    tokenizer = load_tokenizer(model_id)

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

    print("EduTune AI tokenizer information")
    print("-" * 36)

    for key, value in information.items():
        print(f"{key}: {value}")