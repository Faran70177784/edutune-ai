"""EduTune AI inference package."""

from inference.generator import (
    generate_response,
    generate_response_safe,
)

from inference.model_loader import (
    check_inference_hardware,
    inference_is_available,
    load_inference_model,
)

from inference.prompt_templates import (
    build_chat_prompt,
    build_educational_prompt,
)


__all__ = [
    "build_chat_prompt",
    "build_educational_prompt",
    "check_inference_hardware",
    "generate_response",
    "generate_response_safe",
    "inference_is_available",
    "load_inference_model",
]