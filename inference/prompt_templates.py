"""Educational prompt templates for EduTune AI."""

from __future__ import annotations


def build_educational_prompt(
    question: str,
    *,
    subject: str | None = None,
    difficulty: str | None = None,
) -> str:
    """
    Build a structured educational prompt.

    Parameters
    ----------
    question:
        Student's question.

    subject:
        Optional academic subject.

    difficulty:
        Optional difficulty level.
    """

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    context_parts: list[str] = []

    if subject:
        context_parts.append(
            f"Subject: {subject.strip()}"
        )

    if difficulty:
        context_parts.append(
            f"Difficulty: {difficulty.strip()}"
        )

    context = ""

    if context_parts:
        context = (
            "\n"
            + "\n".join(context_parts)
            + "\n"
        )

    return (
        "You are EduTune AI, an educational assistant.\n"
        "Provide a clear, accurate, and student-friendly explanation."
        f"{context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_chat_prompt(
    message: str,
) -> str:
    """Build a general educational chat prompt."""

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    return (
        "You are EduTune AI, an educational assistant.\n"
        "Help the student understand the topic clearly.\n\n"
        f"Student: {message}\n"
        "EduTune AI:"
    )


if __name__ == "__main__":
    print("EduTune AI prompt templates")
    print("--------------------------------")
    print(
        build_educational_prompt(
            "Explain Newton's second law.",
            subject="Physics",
            difficulty="Beginner",
        )
    )