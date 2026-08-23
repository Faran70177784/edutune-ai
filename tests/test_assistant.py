"""Tests for the EduTune AI assistant layer."""

from __future__ import annotations

import pytest

from assistant import (
    EducationalAssistant,
    EducationalRequest,
    build_educational_prompt,
    clean_response,
    extract_text_from_model_output,
    format_response,
)


def test_educational_request_validates_question() -> None:
    """Empty educational questions should be rejected."""

    with pytest.raises(ValueError):
        EducationalRequest(question="")


def test_educational_prompt_contains_question() -> None:
    """Generated prompts should contain the student's question."""

    prompt = build_educational_prompt(
        "What is photosynthesis?",
        subject="Biology",
        difficulty="Beginner",
    )

    assert "What is photosynthesis?" in prompt
    assert "Biology" in prompt
    assert "Beginner" in prompt


def test_educational_prompt_supports_context() -> None:
    """Optional context should appear in the generated prompt."""

    prompt = build_educational_prompt(
        "Explain recursion.",
        subject="Computer Science",
        context="The student already understands functions.",
    )

    assert "Explain recursion." in prompt
    assert "The student already understands functions." in prompt


def test_clean_response_normalizes_text() -> None:
    """Response formatting should remove excessive whitespace."""

    result = clean_response(
        "  Hello\n\n\nStudent!   "
    )

    assert result == "Hello\n\nStudent!"


def test_extract_text_from_string() -> None:
    """Plain string model outputs should be supported."""

    assert (
        extract_text_from_model_output("Hello")
        == "Hello"
    )


def test_extract_text_from_dictionary() -> None:
    """Dictionary model outputs should be supported."""

    result = extract_text_from_model_output(
        {"response": "Educational answer"}
    )

    assert result == "Educational answer"


def test_format_response() -> None:
    """Successful responses should contain metadata."""

    response = format_response(
        "Photosynthesis converts light energy.",
        subject="Biology",
        difficulty="Beginner",
    )

    assert response.content == (
        "Photosynthesis converts light energy."
    )
    assert response.subject == "Biology"
    assert response.difficulty == "Beginner"
    assert response.error is None
    assert response.model_available is True


def test_assistant_requires_generator() -> None:
    """Assistant should safely report missing inference backend."""

    assistant = EducationalAssistant()

    response = assistant.ask(
        "What is a variable?",
        subject="Computer Science",
    )

    assert response.error is not None
    assert response.model_available is False


def test_assistant_works_with_generator() -> None:
    """Assistant should work with an injected generator."""

    def fake_generator(prompt: str) -> str:
        assert "What is a variable?" in prompt
        return "A variable stores a value that can change."

    assistant = EducationalAssistant(
        generator=fake_generator
    )

    response = assistant.ask(
        "What is a variable?",
        subject="Computer Science",
        difficulty="Beginner",
    )

    assert response.content == (
        "A variable stores a value that can change."
    )
    assert response.error is None
    assert response.model_available is True


def test_chat_history_is_updated() -> None:
    """Successful assistant interactions should update history."""

    assistant = EducationalAssistant(
        generator=lambda prompt: "Test answer."
    )

    assistant.ask("What is a variable?")

    history = assistant.get_history()

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_reset_clears_history() -> None:
    """Reset should clear the current conversation."""

    assistant = EducationalAssistant(
        generator=lambda prompt: "Test answer."
    )

    assistant.ask("Explain gravity.")

    assert assistant.get_history()

    assistant.reset()

    assert assistant.get_history() == []