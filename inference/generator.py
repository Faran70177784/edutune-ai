"""Text generation utilities for EduTune AI."""

from __future__ import annotations

from typing import Any


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response from a loaded causal language model.

    This function expects an already-loaded model and tokenizer.
    It does not load model weights itself.
    """

    if model is None:
        raise ValueError("Model cannot be None.")

    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None.")

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    if not 0 < top_p <= 1:
        raise ValueError(
            "top_p must be greater than 0 and at most 1."
        )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    device = next(model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_length = encoded["input_ids"].shape[1]

    generated_tokens = generated[
        0,
        prompt_length:,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def generate_response_safe(
    model: Any,
    tokenizer: Any,
    prompt: str,
    **generation_kwargs: Any,
) -> str:
    """
    Generate a response with a clean user-facing error.

    Model inference errors are converted to RuntimeError with
    additional context.
    """

    try:
        return generate_response(
            model,
            tokenizer,
            prompt,
            **generation_kwargs,
        )

    except (ValueError, TypeError):
        raise

    except Exception as exc:
        raise RuntimeError(
            "EduTune AI response generation failed."
        ) from exc


if __name__ == "__main__":
    print("EduTune AI generator")
    print("--------------------------------")
    print(
        "Generator module loaded successfully."
    )
    print(
        "A real generation test requires a loaded "
        "model and therefore a CUDA-enabled environment."
    )