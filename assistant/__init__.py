"""EduTune AI educational assistant package."""

from assistant.chat import (
    ChatMessage,
    ChatSession,
    EducationalAssistant,
)
from assistant.educational_tasks import (
    EducationalRequest,
    SUPPORTED_DIFFICULTIES,
    SUPPORTED_SUBJECTS,
    SUPPORTED_TASKS,
    build_concept_explanation_prompt,
    build_educational_prompt,
    build_example_prompt,
    build_question_answering_prompt,
    build_study_guidance_prompt,
    get_difficulties,
    get_subjects,
    get_task_types,
)
from assistant.response_formatter import (
    FormattedResponse,
    clean_response,
    extract_text_from_model_output,
    format_error,
    format_response,
    response_to_markdown,
)


__all__ = [
    "ChatMessage",
    "ChatSession",
    "EducationalAssistant",
    "EducationalRequest",
    "FormattedResponse",
    "SUPPORTED_DIFFICULTIES",
    "SUPPORTED_SUBJECTS",
    "SUPPORTED_TASKS",
    "build_concept_explanation_prompt",
    "build_educational_prompt",
    "build_example_prompt",
    "build_question_answering_prompt",
    "build_study_guidance_prompt",
    "clean_response",
    "extract_text_from_model_output",
    "format_error",
    "format_response",
    "get_difficulties",
    "get_subjects",
    "get_task_types",
    "response_to_markdown",
]