"""Response formatting utilities for EduTune AI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FormattedResponse:
    """Represent a formatted educational response."""

    content: str
    subject: str
    difficulty: str
    task_type: str
    model_available: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary."""

        return {
            "content": self.content,
            "subject": self.subject,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "model_available": self.model_available,
            "error": self.error,
        }


def clean_response(text: str) -> str:
    """Clean and normalize generated response text."""

    if text is None:
        return ""

    cleaned = str(text).strip()

    # Remove accidental surrounding quotation marks.
    if (
        len(cleaned) >= 2
        and cleaned[0] == '"'
        and cleaned[-1] == '"'
    ):
        cleaned = cleaned[1:-1].strip()

    # Normalize excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Normalize trailing whitespace.
    cleaned = "\n".join(
        line.rstrip()
        for line in cleaned.splitlines()
    )

    return cleaned.strip()


def format_error(
    message: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    task_type: str = "Question Answering",
) -> FormattedResponse:
    """Create a structured error response."""

    return FormattedResponse(
        content=(
            "EduTune AI could not generate a response.\n\n"
            f"Reason: {str(message).strip()}"
        ),
        subject=subject,
        difficulty=difficulty,
        task_type=task_type,
        model_available=False,
        error=str(message).strip(),
    )


def format_response(
    response: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    task_type: str = "Question Answering",
) -> FormattedResponse:
    """Create a structured successful response."""

    cleaned = clean_response(response)

    if not cleaned:
        return format_error(
            "The model returned an empty response.",
            subject=subject,
            difficulty=difficulty,
            task_type=task_type,
        )

    return FormattedResponse(
        content=cleaned,
        subject=subject,
        difficulty=difficulty,
        task_type=task_type,
        model_available=True,
        error=None,
    )


def response_to_markdown(response: FormattedResponse) -> str:
    """Convert a structured response into Streamlit-friendly Markdown."""

    if response.error:
        return (
            f"**EduTune AI**\n\n"
            f"{response.content}"
        )

    return response.content


def extract_text_from_model_output(output: Any) -> str:
    """
    Extract response text from common model output structures.

    Supported forms include:
    - plain strings
    - dictionaries containing response/text/content
    - objects with response/text/content attributes
    """

    if output is None:
        return ""

    if isinstance(output, str):
        return output.strip()

    if isinstance(output, dict):
        for key in ("response", "text", "content", "generated_text"):
            value = output.get(key)

            if value is not None:
                return str(value).strip()

    for attribute in (
        "response",
        "text",
        "content",
        "generated_text",
    ):
        if hasattr(output, attribute):
            value = getattr(output, attribute)

            if value is not None:
                return str(value).strip()

    return str(output).strip()