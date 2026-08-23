"""
Centralized logging utilities for EduTune AI.
"""

from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def get_logger(
    name: str = "edutune_ai",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured application logger.

    Repeated calls do not add duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def configure_file_logging(
    logger: logging.Logger,
    log_path: str | Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Add a file handler to an existing logger.

    The log directory is created automatically.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(level)

    resolved_path = path.resolve()

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved_path:
                    return logger
            except OSError:
                continue

    file_handler = logging.FileHandler(
        path,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    logger.addHandler(file_handler)

    return logger