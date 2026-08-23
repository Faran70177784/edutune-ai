"""Educational task orchestration for EduTune AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SUPPORTED_SUBJECTS = (
    "General",
    "Biology",
    "Computer Science",
    "Economics",
    "Mathematics",
    "Physics",
)

SUPPORTED_DIFFICULTIES = (
    "Beginner",
    "Intermediate",
    "Advanced",
)

SUPPORTED_TASKS = (
    "Concept Explanation",
    "Question Answering",
    "Example Generation",
    "Study Guidance",
)


@dataclass(frozen=True)
class EducationalRequest:
    """Represent a structured educational request."""

    question: str
    subject: str = "General"
    difficulty: str = "Beginner"
    task_type: str = "Question Answering"
    context: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.question).strip():
            raise ValueError("Question cannot be empty.")

        if self.subject not in SUPPORTED_SUBJECTS:
            raise ValueError(
                f"Unsupported subject: {self.subject}. "
                f"Supported subjects: {', '.join(SUPPORTED_SUBJECTS)}"
            )

        if self.difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(
                f"Unsupported difficulty: {self.difficulty}. "
                f"Supported difficulties: "
                f"{', '.join(SUPPORTED_DIFFICULTIES)}"
            )

        if self.task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task type: {self.task_type}. "
                f"Supported tasks: {', '.join(SUPPORTED_TASKS)}"
            )


def build_educational_prompt(
    question: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    task_type: str = "Question Answering",
    context: Optional[str] = None,
) -> str:
    """Build a structured educational prompt."""

    request = EducationalRequest(
        question=question,
        subject=subject,
        difficulty=difficulty,
        task_type=task_type,
        context=context,
    )

    task_instructions = {
        "Concept Explanation": (
            "Explain the concept clearly. Start with the core definition, "
            "then explain the main idea and provide a simple example."
        ),
        "Question Answering": (
            "Answer the student's question directly and accurately. "
            "Explain the reasoning where useful."
        ),
        "Example Generation": (
            "Provide a practical and educational example that helps "
            "the student understand the requested concept."
        ),
        "Study Guidance": (
            "Provide a structured study approach. Break the topic into "
            "manageable steps and suggest what the student should learn next."
        ),
    }

    difficulty_instructions = {
        "Beginner": (
            "Use simple language and explain foundational terminology."
        ),
        "Intermediate": (
            "Assume basic background knowledge and provide moderate detail."
        ),
        "Advanced": (
            "Use technically precise language and provide deeper reasoning."
        ),
    }

    parts = [
        "You are EduTune AI, an educational AI assistant.",
        "Your goal is to help students understand academic concepts clearly.",
        "",
        f"Subject: {request.subject}",
        f"Difficulty: {request.difficulty}",
        f"Task: {request.task_type}",
        "",
        task_instructions[request.task_type],
        difficulty_instructions[request.difficulty],
        "",
        "Response requirements:",
        "- Stay focused on the student's educational question.",
        "- Be accurate and avoid unsupported claims.",
        "- Use clear structure when it improves understanding.",
        "- Define technical terms when necessary.",
        "- Do not unnecessarily repeat the question.",
    ]

    if request.context and request.context.strip():
        parts.extend(
            [
                "",
                "Additional context:",
                request.context.strip(),
            ]
        )

    parts.extend(
        [
            "",
            "Student question:",
            request.question.strip(),
            "",
            "Educational response:",
        ]
    )

    return "\n".join(parts)


def build_concept_explanation_prompt(
    topic: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    context: Optional[str] = None,
) -> str:
    """Build a prompt specifically for concept explanation."""

    return build_educational_prompt(
        topic,
        subject=subject,
        difficulty=difficulty,
        task_type="Concept Explanation",
        context=context,
    )


def build_question_answering_prompt(
    question: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    context: Optional[str] = None,
) -> str:
    """Build a prompt specifically for question answering."""

    return build_educational_prompt(
        question,
        subject=subject,
        difficulty=difficulty,
        task_type="Question Answering",
        context=context,
    )


def build_example_prompt(
    topic: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    context: Optional[str] = None,
) -> str:
    """Build a prompt specifically for example generation."""

    return build_educational_prompt(
        topic,
        subject=subject,
        difficulty=difficulty,
        task_type="Example Generation",
        context=context,
    )


def build_study_guidance_prompt(
    topic: str,
    *,
    subject: str = "General",
    difficulty: str = "Beginner",
    context: Optional[str] = None,
) -> str:
    """Build a prompt specifically for study guidance."""

    return build_educational_prompt(
        topic,
        subject=subject,
        difficulty=difficulty,
        task_type="Study Guidance",
        context=context,
    )


def get_subjects() -> tuple[str, ...]:
    """Return supported educational subjects."""

    return SUPPORTED_SUBJECTS


def get_difficulties() -> tuple[str, ...]:
    """Return supported difficulty levels."""

    return SUPPORTED_DIFFICULTIES


def get_task_types() -> tuple[str, ...]:
    """Return supported educational task types."""

    return SUPPORTED_TASKS