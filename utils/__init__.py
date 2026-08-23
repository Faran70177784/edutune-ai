"""
EduTune AI utility package.
"""

from .hardware import (
    can_run_large_model,
    get_cuda_device_count,
    get_device,
    get_gpu_name,
    get_hardware_summary,
    is_cuda_available,
    require_cuda,
)

from .helpers import (
    clean_text,
    ensure_directory,
    get_project_root,
    normalize_text,
    read_json,
    resolve_project_path,
    safe_filename,
    validate_non_empty,
    write_json,
)

from .logger import (
    configure_file_logging,
    get_logger,
)

from .seed import (
    disable_deterministic_mode,
    enable_deterministic_mode,
    set_seed,
)

__all__ = [
    "can_run_large_model",
    "get_cuda_device_count",
    "get_device",
    "get_gpu_name",
    "get_hardware_summary",
    "is_cuda_available",
    "require_cuda",
    "clean_text",
    "ensure_directory",
    "get_project_root",
    "normalize_text",
    "read_json",
    "resolve_project_path",
    "safe_filename",
    "validate_non_empty",
    "write_json",
    "configure_file_logging",
    "get_logger",
    "disable_deterministic_mode",
    "enable_deterministic_mode",
    "set_seed",
]