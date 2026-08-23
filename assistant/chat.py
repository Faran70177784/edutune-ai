"""Educational chat orchestration for EduTune AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from assistant.educational_tasks import (
    build_educational_prompt,
)
from assistant.response_formatter import (
    FormattedResponse,
    extract_text_from_model_output,
    format_error,
    format_response,
)


@dataclass
class ChatMessage:
    """Represent one chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(
                "Message role must be system, user, or assistant."
            )

        if not str(self.content).strip():
            raise ValueError(
                "Message content cannot be empty."
            )

    def to_dict(self) -> dict[str, str]:
        """Convert message to a dictionary."""

        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class ChatSession:
    """Maintain conversation state for an educational session."""

    system_message: str = (
        "You are EduTune AI, an educational assistant "
        "designed to help students learn clearly."
    )
    messages: list[ChatMessage] = field(default_factory=list)
    max_history: int = 10

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError(
                "max_history must be greater than zero."
            )

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Add a message to the conversation."""

        self.messages.append(
            ChatMessage(
                role=role,
                content=content,
            )
        )

        self._trim_history()

    def add_user_message(self, content: str) -> None:
        """Add a user message."""

        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""

        self.add_message("assistant", content)

    def _trim_history(self) -> None:
        """Keep the conversation within the configured history limit."""

        if len(self.messages) > self.max_history:
            self.messages = self.messages[
                -self.max_history:
            ]

    def clear(self) -> None:
        """Clear conversation history."""

        self.messages.clear()

    def get_messages(self) -> list[dict[str, str]]:
        """Return conversation messages."""

        return [
            message.to_dict()
            for message in self.messages
        ]

    def get_prompt_messages(self) -> list[dict[str, str]]:
        """Return messages suitable for chat-style model APIs."""

        return [
            {
                "role": "system",
                "content": self.system_message,
            },
            *self.get_messages(),
        ]


class EducationalAssistant:
    """High-level educational assistant service."""

    def __init__(
        self,
        generator: Optional[Callable[..., Any]] = None,
        *,
        max_history: int = 10,
    ) -> None:
        self.session = ChatSession(
            max_history=max_history,
        )
        self.generator = generator

    def set_generator(
        self,
        generator: Callable[..., Any],
    ) -> None:
        """Configure the model-generation callable."""

        if not callable(generator):
            raise TypeError(
                "generator must be callable."
            )

        self.generator = generator

    def reset(self) -> None:
        """Reset the conversation."""

        self.session.clear()

    def build_prompt(
        self,
        question: str,
        *,
        subject: str = "General",
        difficulty: str = "Beginner",
        task_type: str = "Question Answering",
        context: Optional[str] = None,
    ) -> str:
        """Build an educational prompt."""

        return build_educational_prompt(
            question,
            subject=subject,
            difficulty=difficulty,
            task_type=task_type,
            context=context,
        )

    def ask(
        self,
        question: str,
        *,
        subject: str = "General",
        difficulty: str = "Beginner",
        task_type: str = "Question Answering",
        context: Optional[str] = None,
    ) -> FormattedResponse:
        """Generate an educational response."""

        question = str(question).strip()

        if not question:
            return format_error(
                "Question cannot be empty.",
                subject=subject,
                difficulty=difficulty,
                task_type=task_type,
            )

        if self.generator is None:
            return format_error(
                (
                    "No inference generator is configured. "
                    "The assistant is ready, but a model "
                    "generation backend must be connected."
                ),
                subject=subject,
                difficulty=difficulty,
                task_type=task_type,
            )

        prompt = self.build_prompt(
            question,
            subject=subject,
            difficulty=difficulty,
            task_type=task_type,
            context=context,
        )

        self.session.add_user_message(question)

        try:
            output = self._generate(prompt)

            response_text = extract_text_from_model_output(
                output
            )

            response = format_response(
                response_text,
                subject=subject,
                difficulty=difficulty,
                task_type=task_type,
            )

            if response.error is None:
                self.session.add_assistant_message(
                    response.content
                )

            return response

        except Exception as exc:
            return format_error(
                str(exc),
                subject=subject,
                difficulty=difficulty,
                task_type=task_type,
            )

    def _generate(self, prompt: str) -> Any:
        """Call the configured generation backend."""

        if self.generator is None:
            raise RuntimeError(
                "Inference generator is not configured."
            )

        try:
            return self.generator(prompt)
        except TypeError:
            # Support generator implementations that expose
            # a keyword-based prompt parameter.
            return self.generator(prompt=prompt)

    def get_history(self) -> list[dict[str, str]]:
        """Return the current conversation history."""

        return self.session.get_messages()